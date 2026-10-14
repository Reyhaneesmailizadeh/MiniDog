"""FID / Inception Score of a saved checkpoint, using the same eval the training loop runs.

Usage:
    uv run torchrun --standalone --nproc_per_node=4 -m minidog.offline_eval \\
        --config configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml --checkpoint ckpts/pretrain/checkpoints/ep-0000200.pt
    # sweep the classifier-free-guidance scale on one checkpoint:
    ... --cfg-scale 1.5 2.0 3.0 6.0
Also runs on a single GPU without torchrun (slower: all 26k samples on one device).
"""
import os
import argparse
import dataclasses
import math

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from minidog.config import Stage2Config
from minidog.eval import evaluate_generation_distributed, prepare_eval_datasets
from minidog.transport import Sampler, create_transport, get_model_forward_fn, setup_text_encoder
from minidog.utils.dist_utils import cleanup_distributed, setup_distributed
from minidog.utils.model_utils import instantiate_from_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--cfg-scale", type=float, nargs="+", default=None, help="Override guidance.cfg.scale (one eval per value).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()
    if not dist.is_initialized():  # single process: the eval still uses collectives, so open a 1-rank group
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", rank=0, world_size=1)
    config: Stage2Config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config)))
    config.post_process()
    autocast_kwargs = dict(enabled=args.precision == "bf16", dtype=torch.bfloat16)

    vae = instantiate_from_config(config.stage_1).to(device).eval()
    text_encoder = setup_text_encoder(config, rank, device)
    state = torch.load(args.checkpoint, map_location="cpu")
    if config.repa.use_repa:  # train.py sets z_dim from the DINOv2 encoder; here read it off the saved projector
        config.repa.z_dim = state[args.weights]["repa_projector.weight"].shape[0]
    config.prepare_model_params()
    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.load_state_dict(state[args.weights])
    if rank == 0:
        print(f"Loaded {args.weights} weights from {args.checkpoint} (epoch {state.get('epoch', '?')}, step {state.get('step', '?')})")

    latent_size = tuple(config.misc.latent_size)
    shift_dim = config.misc.time_dist_shift_dim or math.prod(latent_size)
    shift_base_eval = config.misc.time_dist_shift_base if config.misc.time_dist_shift_base_eval is None else config.misc.time_dist_shift_base_eval
    transport = create_transport(config.transport, time_dist_shift=math.sqrt(shift_dim / config.misc.time_dist_shift_base),
                                 time_dist_shift_eval=math.sqrt(shift_dim / shift_base_eval))
    sampler = Sampler(transport).sample_ode(**dataclasses.asdict(config.sampler))
    eval_datasets = prepare_eval_datasets(config.eval.datasets)

    for scale in args.cfg_scale or [config.guidance.cfg.scale]:
        config.guidance.cfg.scale = scale
        model_fn, sample_kwargs = get_model_forward_fn(model, config.guidance)
        for name, ds in eval_datasets.items():
            metrics = evaluate_generation_distributed(
                model_fn, sampler, latent_size, sample_kwargs, config.guidance.use_cfg, vae, ds.dataset, len(ds.dataset),
                batch_size=args.batch_size, rank=rank, world_size=world_size, device=device, global_step=state.get("step", 0),
                autocast_kwargs=autocast_kwargs, reference_npz_path=ds.reference_npz, text_encoder=text_encoder,
            )
            if rank == 0:
                print(f"RESULT {name} cfg={scale}: fid={metrics['fid']:.4f} inception_score={metrics['inception_score']:.4f}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
