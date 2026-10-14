"""Train the MiniDog diffusion transformer (pretraining or SFT) with flow matching on VAE latents.

Usage:
    uv run torchrun --standalone --nproc_per_node=4 -m minidog.train \\
        --config configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml --compile --wandb
    # SFT from a pretrained checkpoint:
    uv run torchrun --standalone --nproc_per_node=4 -m minidog.train \\
        --config configs/sft_e2e-invae_128tok_mse_norepa.yaml --ckpt ckpts/<pretrain-run>/checkpoints/ep-0000200.pt \\
        --init-weights-only --compile --wandb
"""

import argparse
import dataclasses
import math
import os

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from copy import deepcopy

import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from tqdm.auto import tqdm

from minidog.config import Stage2Config
from minidog.data import prepare_unified_dataloader
from minidog.dinov2 import DINOv2Encoder
from minidog.eval import prepare_eval_datasets
from minidog.vae import VAE
from minidog.engine import train_one_epoch
from minidog.transport import Sampler, create_transport
from minidog.transport import setup_text_encoder
from minidog.utils.checkpoint import load_stage2_checkpoint, load_stage2_weights_only, save_stage2_checkpoint
from minidog.utils.dist_utils import cleanup_distributed, main_process_first, setup_distributed
from minidog.utils.model_utils import instantiate_from_config
from minidog.utils.optim_utils import build_optimizer, build_scheduler
from minidog.utils.resume_utils import configure_experiment_dirs, find_resume_checkpoint, save_worktree
from minidog.utils.train_utils import center_crop_arr, get_autocast_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MiniDog diffusion transformer on VAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--results-dir", type=str, default="ckpts")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--init-weights-only", action="store_true",
                         help="Load only model+EMA weights from --ckpt; start a fresh optimizer/scheduler "
                              "with epoch/step at 0. Use when fine-tuning on a dataset whose steps-per-epoch "
                              "differs too much from the checkpoint's original run for its saved scheduler "
                              "state to make sense.")
    parser.add_argument("--compile", action="store_true", help="torch.compile the training loss function")
    return parser.parse_args()


def main():
    """Train Stage 2 model using config-driven hyperparameters."""
    args = parse_args()

    #########################################################
    # Distributed + config setup
    #########################################################
    rank, world_size, device = setup_distributed()
    config: Stage2Config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config)))
    config.post_process()

    seed = config.training.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    autocast_kwargs = get_autocast_kwargs(args)

    #########################################################
    # Data setup; train and eval
    #########################################################
    global_batch_size = config.training.global_batch_size or (config.training.batch_size * world_size * config.training.grad_accum_steps)
    assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    micro_batch_size = global_batch_size // (world_size * config.training.grad_accum_steps)
    stage2_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config.training.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    needs_transform = config.dataset.type not in ("hf", "wds")
    # train dataloader
    dataloader = prepare_unified_dataloader(
        config=dataclasses.asdict(config.dataset),
        image_size=config.training.image_size,
        batch_size=micro_batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
        transform=stage2_transform if needs_transform else None,
        condition_type=config.conditioning.type,
        virtual_epoch_steps=config.training.virtual_epoch_steps,
    )

    # eval setup
    eval_datasets, eval_dir = None, None
    if config.eval is not None:
        eval_datasets = prepare_eval_datasets(config.eval.datasets)
        eval_dir = config.eval.eval_dir

    #########################################################
    # Models setup
    #########################################################
    latent_size = tuple(config.misc.latent_size)

    # stage 1: frozen VAE
    vae: VAE = instantiate_from_config(config.stage_1).to(device)
    vae.eval()

    # repa target encoder
    repa_target_encoder = None
    if config.repa.use_repa:
        with main_process_first(rank):
            if config.repa.encoder == "dinov3":
                from minidog.dinov3 import DEFAULT_DINOV3_MODEL, DINOv3Encoder
                repa_target_encoder = DINOv3Encoder(
                    config.training.image_size, config.repa.dinov3_model or DEFAULT_DINOV3_MODEL
                ).to(device)
            elif config.repa.encoder == "pe_spatial":
                from minidog.pe_spatial import DEFAULT_PE_SPATIAL_MODEL, PESpatialEncoder
                repa_target_encoder = PESpatialEncoder(
                    config.training.image_size,
                    config.repa.pe_spatial_model or DEFAULT_PE_SPATIAL_MODEL,
                    layernorm_features=config.repa.pe_spatial_layernorm,
                ).to(device)
            elif config.repa.encoder == "eupe":
                from minidog.eupe import DEFAULT_EUPE_MODEL, EUPEEncoder
                repa_target_encoder = EUPEEncoder(
                    config.training.image_size,
                    config.repa.eupe_model or DEFAULT_EUPE_MODEL,
                    strip_norm_affine=config.repa.eupe_strip_norm_affine,
                ).to(device)
            elif config.repa.encoder == "dinov2":
                repa_target_encoder = DINOv2Encoder(config.training.image_size).to(device)
            else:
                raise ValueError(f"unknown repa.encoder: {config.repa.encoder!r}")
        config.repa.z_dim = repa_target_encoder.embed_dim
        logger.info(
            f"REPA target encoder: {config.repa.encoder}, embed_dim={repa_target_encoder.embed_dim}"
        )

    # text encoder for text conditioning; None if not using text conditioning
    text_encoder = setup_text_encoder(config, rank, device)

    # prepare model params (must be called before model instantiation so that
    # condition_type, text_feature_dim, context_dim, repa z_dim etc. are set)
    config.prepare_model_params()

    # stage2: model - trainable
    model: torch.nn.Module = instantiate_from_config(config.stage_2).to(device)
    model.requires_grad_(True)
    # stage2 ema model
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()

    # ddp wrapper for stage2 model
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    model = ddp_model.module
    ddp_model.train()
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    if args.wandb and rank == 0:
        import wandb
        wandb.config.update({
            "model_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "trainable_params_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1),
        }, allow_val_change=True)

    #########################################################
    # Optimizer + Scheduler setup
    #########################################################
    optimizer, optim_msg = build_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        config.training.optimizer,
    )

    #########################################################
    # Steps per epoch setup
    #########################################################
    steps_per_epoch = len(dataloader) // config.training.grad_accum_steps
    logger.info(f"Using {steps_per_epoch} steps per epoch (virtual={config.training.virtual_epoch_steps is not None})")

    # Build scheduler (needs steps_per_epoch)
    scheduler = None
    sched_msg = None
    if config.training.scheduler is not None:
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, config.training.scheduler)

    #########################################################
    # Transport + Sampler setup
    #########################################################
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base
    )
    time_dist_shift_base_eval = config.misc.time_dist_shift_base if config.misc.time_dist_shift_base_eval is None else config.misc.time_dist_shift_base_eval
    time_dist_shift_eval = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / time_dist_shift_base_eval
    )
    transport = create_transport(
        config=config.transport,
        time_dist_shift=time_dist_shift,
        time_dist_shift_eval=time_dist_shift_eval,
    )
    transport_sampler = Sampler(transport)
    eval_sampler = transport_sampler.sample_ode(**dataclasses.asdict(config.sampler))

    if args.compile:
        transport.training_losses = torch.compile(transport.training_losses)

    #########################################################
    # Resume setup
    #########################################################
    start_epoch = 0
    global_step = 0
    if args.init_weights_only:
        assert args.ckpt, "--init-weights-only requires --ckpt"
        # An own_ckpt already in this experiment's dir means we're resuming after an
        # interruption of THIS run, not warm-starting — resume it normally (optimizer,
        # scheduler, and epoch/step all included), same as the non-init-weights-only path.
        own_ckpt = find_resume_checkpoint(experiment_dir)
        if own_ckpt:
            start_epoch, global_step = load_stage2_checkpoint(own_ckpt, ddp_model, ema_model, optimizer, scheduler)
            logger.info(f"[Rank {rank}] Resumed from {own_ckpt} (epoch={start_epoch}, step={global_step}).")
        else:
            load_stage2_weights_only(args.ckpt, ddp_model, ema_model)
            logger.info(f"[Rank {rank}] Initialized weights only from {args.ckpt} "
                        f"(epoch=0, step=0, fresh optimizer/scheduler).")
            if rank == 0:
                save_worktree(experiment_dir, config)
                logger.info(f"Saved training worktree and config to {experiment_dir}.")
    else:
        ckpt_path = find_resume_checkpoint(experiment_dir, args.ckpt)
        if ckpt_path:
            start_epoch, global_step = load_stage2_checkpoint(ckpt_path, ddp_model, ema_model, optimizer, scheduler)
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            if rank == 0:
                save_worktree(experiment_dir, config)
                logger.info(f"Saved training worktree and config to {experiment_dir}.")

    # print exp config and training details
    if rank == 0:
        logger.info(f"Stage-1 VAE parameters: {sum(p.numel() for p in vae.parameters())/1e6:.2f}M")
        logger.info(f"Stage-2 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
        logger.info(f"Clipping gradients to max norm {config.training.clip_grad}." if config.training.clip_grad else "Not clipping gradients.")
        logger.info(optim_msg)
        logger.info(f"Transport: prediction={config.transport.prediction}, time_dist={config.transport.time_dist_type}, t_eps={config.transport.t_eps}")
        logger.info(f"Training for {config.training.epochs} epochs, batch size {micro_batch_size} per GPU.")
        logger.info(f"Dataset: {dataloader.dataset_size:,} samples, {steps_per_epoch} steps/epoch.")
        if config.guidance.use_cfg:
            logger.info(f"CFG scale: {config.guidance.cfg.scale}, interval: [{config.guidance.cfg.t_min}, {config.guidance.cfg.t_max}]")
        else:
            logger.info("No classifier-free guidance at sampling time.")

    total_steps = config.training.epochs * steps_per_epoch
    progress_bar = tqdm(total=total_steps, initial=global_step, desc="Training", disable=rank != 0)

    # fixed state for consistent visualization across epochs (populated from first batch)
    num_viz_samples = min(micro_batch_size, 32)
    viz_fixed = {
        'zs': torch.randn(num_viz_samples, *latent_size, device=device, dtype=torch.float32,
                           generator=torch.Generator(device=device).manual_seed(seed)),
        'context': None,
        'attn_mask': None,
        'prompts': None,
    }

    # On resume: restore fixed prompts from the saved CSV so visualizations stay
    # consistent with pre-crash samples. Only rank 0 needs this (it's the only
    # rank that generates fixed samples), and only for text conditioning.
    if rank == 0 and config.conditioning.type == "text" and text_encoder is not None:
        prompts_path = os.path.join(experiment_dir, "fixed_prompts.csv")
        if os.path.exists(prompts_path):
            import csv
            with open(prompts_path, newline='', encoding='utf-8') as f:
                prompts = [row['prompt'] for row in csv.DictReader(f)]
            prompts = prompts[:num_viz_samples]
            with torch.no_grad():
                enc_out = text_encoder(prompts)
            viz_fixed['context'] = enc_out['tokens']
            viz_fixed['attn_mask'] = enc_out['attention_mask']
            viz_fixed['prompts'] = prompts
            logger.info(f"Restored {len(prompts)} fixed viz prompts from {prompts_path}")

    #########################################################
    # Training loop
    #########################################################
    dist.barrier()
    for epoch in range(start_epoch, config.training.epochs):
        model.train()

        global_step = train_one_epoch(
            ddp_model=ddp_model,
            ema_model=ema_model,
            vae=vae,
            transport=transport,
            eval_sampler=eval_sampler,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            autocast_kwargs=autocast_kwargs,
            device=device,
            epoch=epoch,
            global_step=global_step,
            config=config,
            args=args,
            rank=rank,
            world_size=world_size,
            micro_batch_size=micro_batch_size,
            checkpoint_dir=checkpoint_dir,
            experiment_dir=experiment_dir,
            progress_bar=progress_bar,
            text_encoder=text_encoder,
            repa_target_encoder=repa_target_encoder,
            eval_datasets=eval_datasets,
            viz_fixed=viz_fixed,
        )
    progress_bar.close()

    #########################################################
    # final checkpoint setup and cleanup
    #########################################################
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {config.training.epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-{config.training.epochs:07d}.pt"
        save_stage2_checkpoint(ckpt_path, global_step, config.training.epochs, ddp_model, ema_model, optimizer, scheduler)

    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
