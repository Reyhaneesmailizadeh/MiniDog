"""Pre-compute and cache VAE latents, text embeddings, and EUPE features.

An EUPE twin of minidog/precompute_latents.py, which is left untouched. Same
inputs, same output layout, same shard naming -- the only difference is that the
REPA target features come from EUPE ViT-B/16 (last block) instead of DINOv2
ViT-B/14, and are stored under their own key.

Requires EUPE_REPO_DIR and EUPE_CKPT_DIR in the environment; see minidog/eupe.py.

Reads existing WDS shards (jpg + txt), encodes everything with frozen models,
and writes new WDS shards where each sample contains:
  - latent.npy   : VAE latent  [C, H, W]  float16
  - tokens.npy   : text token embeddings  [seq_len, dim]  float16
  - attn_mask.npy: attention mask          [seq_len]       bool
  - eupe.npy     : EUPE patch tokens     [num_patches, dim]  float16  (only if RePA)
  - txt          : original caption (kept for reference)

data.py reads whichever REPA key a shard carries, so all the precomputed sets are
interchangeable at training time: point dataset.data_dir at the one you want.

Single-GPU usage:
    uv run python -m minidog.precompute_latents_eupe \
        --config configs/pretrain_e2e-invae_128tok_mse_repa_eupe.yaml \
        --input-dir data/dog-t2i-diffusion-data/dogs_recaptioned_wds \
        --output-dir data/dog-t2i-diffusion-data/dogs_recaptioned_latents_e2e-invae_eupe \
        --batch-size 16

Multi-GPU usage (recommended, splits shards across GPUs):
    uv run torchrun --nproc_per_node=8 -m minidog.precompute_latents_eupe \
        --config configs/pretrain_e2e-invae_128tok_mse_repa_eupe.yaml \
        --input-dir data/dog-t2i-diffusion-data/dogs_recaptioned_wds \
        --output-dir data/dog-t2i-diffusion-data/dogs_recaptioned_latents_e2e-invae_eupe \
        --batch-size 16
"""

import argparse
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

from minidog.config import Stage2Config
from minidog.eupe import DEFAULT_EUPE_MODEL, EUPEEncoder
from minidog.precompute_latents import encode_batch_text, to_fp16_numpy
from minidog.transport import setup_text_encoder
from minidog.utils.dist_utils import main_process_first
from minidog.utils.model_utils import instantiate_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-shard", type=int, default=500)
    parser.add_argument("--eupe-model", default=None,
                        help=f"EUPE hub name; default {DEFAULT_EUPE_MODEL}")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Distributed setup — works for both single-GPU and torchrun
    # ------------------------------------------------------------------ #
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        # start clean: shards are named rank{r}-shard-{n}, so leftovers from an earlier run with a
        # different number of ranks would otherwise survive and duplicate samples
        stale = list(output_dir.glob("rank*-shard-*.tar")) + list(output_dir.glob(".shard_sample_counts.json"))
        for f in stale:
            f.unlink()
        if stale:
            print(f"Removed {len(stale)} existing files from {output_dir}")
    if world_size > 1:
        dist.barrier()

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    cfg: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config))
    )
    cfg.post_process()

    # ------------------------------------------------------------------ #
    # Load frozen models (each rank loads its own copy on its GPU)
    # ------------------------------------------------------------------ #
    if rank == 0:
        print("Loading VAE...")
    vae = instantiate_from_config(cfg.stage_1).to(device)
    vae.eval()
    vae.requires_grad_(False)

    if rank == 0:
        print("Loading text encoder...")
    text_encoder = setup_text_encoder(cfg, rank=rank, device=device)

    eupe_model_name = args.eupe_model or cfg.repa.eupe_model or DEFAULT_EUPE_MODEL
    eupe_encoder = None
    if cfg.repa.use_repa:
        if rank == 0:
            print(f"Loading EUPE {eupe_model_name} ...")
        with main_process_first(rank):  # rank 0 warms the torch.hub cache; other ranks reuse it
            eupe_encoder = EUPEEncoder(cfg.training.image_size, eupe_model_name,
                                       strip_norm_affine=cfg.repa.eupe_strip_norm_affine).to(device)
        if rank == 0:
            print(f"  EUPE embed_dim={eupe_encoder.embed_dim} "
                  f"patch_size={eupe_encoder.patch_size} "
                  f"input_size={eupe_encoder.input_size} "
                  f"-> {eupe_encoder.num_patch_tokens} patch tokens")

        # The DiT consumes (latent_h * latent_w) tokens at patch_size=1; the REPA
        # target must match exactly or the MSE in transport.py fails. Check now
        # rather than after an hour of encoding.
        _, latent_h, latent_w = cfg.misc.latent_size
        dit_patch = cfg.stage_2.params["patch_size"]
        dit_tokens = (latent_h // dit_patch) * (latent_w // dit_patch)
        if eupe_encoder.num_patch_tokens != dit_tokens:
            raise ValueError(
                f"token count mismatch: EUPE gives {eupe_encoder.num_patch_tokens} tokens "
                f"({eupe_encoder.input_size}/{eupe_encoder.patch_size} per side) but the DiT "
                f"expects {dit_tokens}"
            )

    # ------------------------------------------------------------------ #
    # Image transform
    # ------------------------------------------------------------------ #
    image_size = cfg.training.image_size
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # [0, 1]
    ])

    # ------------------------------------------------------------------ #
    # Input shards — pass all shards to every rank.
    # wds.split_by_node automatically assigns disjoint subsets per rank.
    # ------------------------------------------------------------------ #
    all_shards = sorted(str(p) for p in input_dir.glob("*.tar"))
    if not all_shards:
        raise ValueError(f"No .tar shards found in {input_dir}")
    if rank == 0:
        print(f"Total shards: {len(all_shards)} | Shards per rank: ~{math.ceil(len(all_shards)/world_size)}")

    def decode_sample(sample):
        image = sample.get("jpg") or sample.get("png") or sample.get("jpeg") or sample.get("webp")
        caption_raw = sample.get("txt", b"")
        caption = caption_raw.decode("utf-8").strip() if isinstance(caption_raw, bytes) else str(caption_raw).strip()
        key = sample.get("__key__", "")
        if image is None:
            return None
        return transform(image), caption, key

    dataset = (
        wds.WebDataset(all_shards, shardshuffle=False, nodesplitter=wds.split_by_node)
        .decode("pil", handler=wds.ignore_and_continue)
        .map(decode_sample, handler=wds.ignore_and_continue)
        .select(lambda x: x is not None)
    )
    loader = wds.WebLoader(dataset, batch_size=args.batch_size, num_workers=2, collate_fn=list)

    # ------------------------------------------------------------------ #
    # Each rank writes to its own uniquely named shards to avoid conflicts.
    # e.g. rank 0 → rank00-shard-00000.tar, rank 1 → rank01-shard-00000.tar
    # The dataloader globs *.tar so it picks all of them up automatically.
    # ------------------------------------------------------------------ #
    pattern = str(output_dir / f"rank{rank:02d}-shard-%05d.tar")
    total_written = 0

    with wds.ShardWriter(pattern, maxcount=args.samples_per_shard) as sink:
        iterator = tqdm(loader, desc=f"[Rank {rank}] Encoding", position=rank) if rank == 0 else loader
        for batch in iterator:
            images_list = [s[0] for s in batch]
            captions    = [s[1] for s in batch]
            keys        = [s[2] for s in batch]

            images = torch.stack(images_list).to(device)

            with torch.no_grad():
                latents = vae.encode(images)                                   # [B, C, H, W]
                tokens, attn_mask = encode_batch_text(text_encoder, captions)  # [B, seq, dim], [B, seq]
                eupe_feats = None
                if eupe_encoder is not None:
                    eupe_feats = eupe_encoder(images * 255.0)                  # [B, num_patches, dim]

            for i in range(len(batch)):
                sample = {
                    "__key__": keys[i],
                    "txt": captions[i].encode("utf-8"),
                }

                buf = io.BytesIO(); np.save(buf, to_fp16_numpy(latents[i]))
                sample["latent.npy"] = buf.getvalue()

                buf = io.BytesIO(); np.save(buf, to_fp16_numpy(tokens[i]))
                sample["tokens.npy"] = buf.getvalue()

                buf = io.BytesIO(); np.save(buf, attn_mask[i].cpu().bool().numpy())
                sample["attn_mask.npy"] = buf.getvalue()

                if eupe_feats is not None:
                    buf = io.BytesIO(); np.save(buf, to_fp16_numpy(eupe_feats[i]))
                    sample["eupe.npy"] = buf.getvalue()

                sink.write(sample)
                total_written += 1

    print(f"[Rank {rank}] Done: {total_written} samples written")

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        total_shards = len(list(output_dir.glob("*.tar")))
        (output_dir / "_manifest.json").write_text(json.dumps({
            "repa_target": "eupe",
            "repa_key": "eupe.npy",
            "repa_layer": "last block (forward_features)",
            "eupe_model": eupe_model_name if eupe_encoder is not None else None,
            "eupe_embed_dim": eupe_encoder.embed_dim if eupe_encoder is not None else None,
            "eupe_patch_tokens": eupe_encoder.num_patch_tokens if eupe_encoder is not None else None,
            "eupe_strip_norm_affine": cfg.repa.eupe_strip_norm_affine,
            "vae_type": cfg.stage_1.params.get("vae_type"),
            "text_encoder": cfg.conditioning.text_encoder.model_name,
            "max_length": cfg.conditioning.text_encoder.max_length,
            "image_size": image_size,
            "source_dir": str(input_dir),
        }, indent=2))
        print(f"\nAll ranks finished. Total output shards: {total_shards} in {output_dir}")
        print(f"Wrote {output_dir / '_manifest.json'}")
        dist.destroy_process_group() if world_size > 1 else None


if __name__ == "__main__":
    main()
