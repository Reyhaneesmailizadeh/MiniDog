"""Stage 2 training engine: train_one_epoch and helpers."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.distributed as dist
import wandb
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from minidog.config import Stage2Config
from minidog.transport import encode_text, get_fixed_viz_batch_conditions, get_model_forward_fn, get_null_cond, sample_and_decode
from minidog.utils import wandb_utils
from minidog.utils.checkpoint import save_stage2_checkpoint
from minidog.utils.logging import save_eval_to_csv
from minidog.utils.train_utils import update_ema

logger = logging.getLogger("minidog")


#########################################################
# Main training function
#########################################################
def train_one_epoch(
    *, # * forces all arguments to be passed as keyword arguments
    ddp_model: DDP,
    ema_model: torch.nn.Module,
    vae,
    transport,
    eval_sampler,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    autocast_kwargs: dict,
    device: torch.device,
    epoch: int,
    global_step: int,
    config: Stage2Config,
    args,
    rank: int,
    world_size: int,
    micro_batch_size: int,
    checkpoint_dir: str,
    experiment_dir: str,
    progress_bar,
    text_encoder=None,
    repa_target_encoder=None,
    eval_datasets: Optional[Dict] = None,
    viz_fixed: Optional[Dict] = None,
) -> int:
    """Run one epoch of Stage 2 training. Returns updated global_step.

    Args:
        viz_fixed: Mutable dict with keys 'zs', 'y', 'encoder_hidden_states',
            'encoder_attention_mask'. Populated from first batch, persists across epochs.
    """
    #########################################################
    # Setup
    #########################################################
    model = ddp_model.module

    # Guidance: derive model_fn / ema_model_fn / sample_kwargs from config
    model_fn, sample_model_kwargs = get_model_forward_fn(model, config.guidance)
    ema_model_fn, _ = get_model_forward_fn(ema_model, config.guidance)
    use_guidance = config.guidance.use_cfg

    # Eval settings
    do_eval = config.eval is not None and eval_datasets is not None
    if do_eval: eval_dir = config.eval.eval_dir
    experiment_name = os.environ.get("EXPERIMENT_NAME")

    # Get null conditions for CFG dropout
    model_kwargs_null = get_null_cond(text_encoder, micro_batch_size)

    # per-epoch state
    num_viz_samples = viz_fixed['zs'].shape[0] if viz_fixed is not None else 0
    epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
    num_batches = 0
    optimizer.zero_grad()

    # save checkpoint at epoch start
    if config.training.checkpoint_interval > 0 and epoch % config.training.checkpoint_interval == 0 and rank == 0:
        logger.info(f"Saving checkpoint at epoch {epoch}...")
        ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
        save_stage2_checkpoint(ckpt_path, global_step, epoch, ddp_model, ema_model, optimizer, scheduler)

    #########################################################
    # Training loop
    #########################################################
    use_precomputed = getattr(config.dataset, 'type', 'wds') == 'latents'

    dataloader.set_epoch(epoch)
    for step, batch in enumerate(dataloader):
        if use_precomputed:
            # Batch is (latent, tokens, attn_mask, dinov2) — all pre-encoded tensors
            z, context, context_attn_mask, z_clean = batch
            z = z.to(device)
            context = context.to(device)
            context_attn_mask = context_attn_mask.to(device)
            # dinov2 sentinel is [B,1]; real shape is [B, patches, dim] (3D)
            z_clean = z_clean.to(device) if z_clean.dim() == 3 else None
            # Populate viz_fixed directly from pre-encoded context on first batch
            if viz_fixed is not None and viz_fixed['context'] is None:
                n = viz_fixed['zs'].shape[0]
                viz_fixed['context'] = context[:n]
                viz_fixed['attn_mask'] = context_attn_mask[:n]
        else:
            images, y = batch
            images = images.to(device)

            # Encode images to latents and compute REPA targets
            with torch.no_grad():
                z = vae.encode(images)
                z_clean = None
                if repa_target_encoder is not None:
                    z_clean = repa_target_encoder(images * 255.0)

            # Capture fixed conditions from first batch
            if viz_fixed is not None:
                viz_fixed = get_fixed_viz_batch_conditions(viz_fixed, y, text_encoder)

            context, context_attn_mask = encode_text(text_encoder, y)

        #########################################################
        # Forward + backward
        #########################################################
        model_kwargs = dict(context=context, attn_mask=context_attn_mask)

        repa_on = config.repa.use_repa

        with autocast(**autocast_kwargs):
            loss_dict = transport.training_losses(
                ddp_model, z, model_kwargs, model_kwargs_null,
                z_clean=z_clean,
                repa_coeff=config.repa.repa_coeff if repa_on else None,
                repa_loss_type=config.repa.repa_loss_type,
                cfg_dropout_prob=config.conditioning.cfg_dropout_prob,
                irepa=config.repa.irepa,
                irepa_gamma=config.repa.irepa_gamma,
            )
            loss_diff = loss_dict["loss"].mean()
            loss_repa = loss_dict["loss_repa"].mean()
            loss = loss_diff + loss_repa if repa_on else loss_diff

        loss = loss / config.training.grad_accum_steps

        is_accum_step = (step + 1) % config.training.grad_accum_steps != 0

        if is_accum_step:

            with ddp_model.no_sync():
                loss.backward()
        else:
            loss.backward()  # DDP auto-syncs gradients on final micro-step

        # Step optimizer and scheduler at grad accumulation boundary
        if not is_accum_step:
            if config.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), config.training.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            update_ema(ema_model, ddp_model.module, decay=config.training.ema_decay)
            global_step += 1
            progress_bar.update(1)

        epoch_metrics['loss'] += loss_diff.detach()
        num_batches += 1

        # Skip logging/viz/eval on non-boundary micro-steps
        if is_accum_step:
            continue

        #########################################################
        # Logging and visualization
        #########################################################
        if config.training.log_interval > 0 and global_step % config.training.log_interval == 0 and rank == 0:
            cur_loss = loss_diff.item()
            stats = {"train/loss": cur_loss, "train/lr": optimizer.param_groups[0]["lr"]}
            if config.repa.use_repa:
                stats["train/loss_repa"] = loss_repa.item()
            logger.info(
                f"[Epoch {epoch} | Step {global_step}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
            )
            if args.wandb:
                wandb_utils.log(stats, step=global_step)
            progress_bar.set_postfix(loss=cur_loss, lr=optimizer.param_groups[0]["lr"])

        # Sampling visualization
        if global_step % config.training.sample_every == 0:
            model.eval()
            logger.info("Generating EMA samples...")
            sample_args = dict(
                eval_sampler=eval_sampler, model_fn=ema_model_fn,
                sample_model_kwargs=sample_model_kwargs, vae=vae,
                use_guidance=use_guidance, text_encoder=text_encoder,
                autocast_kwargs=autocast_kwargs,
            )
            if rank == 0:
                with torch.no_grad():
                    samples_dict = {}
                    # 1. Batch samples (from current batch conditions)
                    batch_n = min(num_viz_samples, context.shape[0])
                    zs_batch = torch.randn(batch_n, *config.misc.latent_size, device=device, dtype=torch.float32)
                    samples_dict["samples/batch"] = sample_and_decode(
                        zs_batch, context[:batch_n], context_attn_mask[:batch_n], **sample_args,
                    )
                    # 2. Fixed samples (consistent across epochs)
                    if viz_fixed is not None and viz_fixed['context'] is not None:
                        samples_dict["samples/fixed"] = sample_and_decode(
                            viz_fixed['zs'].clone(), viz_fixed['context'].clone(), viz_fixed['attn_mask'].clone(),
                            **sample_args,
                        )
                    # save samples to disk
                    from torchvision.utils import save_image
                    samples_out_dir = os.path.join(experiment_dir, "samples")
                    os.makedirs(samples_out_dir, exist_ok=True)
                    for name, samples in samples_dict.items():
                        tag = name.replace("/", "_")
                        save_image(samples * 0.5 + 0.5, os.path.join(samples_out_dir, f"step{global_step:07d}_{tag}.png"), nrow=8, value_range=(0, 1))
                    # save fixed prompts once (first time they're available)
                    prompts_path = os.path.join(experiment_dir, "fixed_prompts.csv")
                    if viz_fixed is not None and viz_fixed.get('prompts') is not None and not os.path.exists(prompts_path):
                        import csv
                        with open(prompts_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(['index', 'prompt'])
                            for i, p in enumerate(viz_fixed['prompts']):
                                writer.writerow([i + 1, p])
                    if args.wandb: # log samples to wandb
                        for name, samples in samples_dict.items():
                            grid = wandb_utils.array2grid(samples)
                            wandb.log({name: wandb.Image(grid)}, step=global_step)
            dist.barrier()
            logger.info("Generating EMA samples done.")
            model.train() # set model back to train mode

        #########################################################
        # Evaluation; distributed evaluation
        #########################################################
        if do_eval and config.eval.eval_interval > 0 and global_step % config.eval.eval_interval == 0:
            from minidog.eval import evaluate_generation_distributed
            logger.info("Starting evaluation...")
            model.eval()
            # eval ema or both ema and running model if eval_model is True
            eval_models = [(ema_model_fn, "ema")] if not config.eval.eval_model else [(ema_model_fn, "ema"), (model_fn, "model")]
            for fn, mod_name in eval_models:
                for ds_name, ds_info in eval_datasets.items():
                    logger.info(f"Evaluating {mod_name} on {ds_name}...")
                    eval_stats = evaluate_generation_distributed(
                        fn, eval_sampler, tuple(config.misc.latent_size), sample_model_kwargs,
                        use_guidance, vae, ds_info.dataset, len(ds_info.dataset),
                        rank=rank, world_size=world_size, device=device,
                        batch_size=micro_batch_size, global_step=global_step,
                        autocast_kwargs=autocast_kwargs,
                        reference_npz_path=ds_info.reference_npz, text_encoder=text_encoder,
                    )
                    if eval_stats is not None and rank == 0:
                        save_eval_to_csv(experiment_name, mod_name, global_step, {'dataset': ds_name, **eval_stats}, eval_dir)
                        if args.wandb:
                            wandb_utils.log({f"eval_{mod_name}/{k}_{ds_name}": v for k, v in eval_stats.items()}, step=global_step)
            model.train() # set model back to train mode
            logger.info("Evaluation done.")


    #########################################################
    # Epoch summary
    #########################################################
    if rank == 0 and num_batches > 0:
        avg_loss = epoch_metrics['loss'].item() / num_batches
        epoch_stats = {"epoch/loss": avg_loss}
        logger.info(f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items()))
        if args.wandb:
            wandb_utils.log(epoch_stats, step=global_step)

    return global_step
