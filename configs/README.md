# Configs

All configs train the same model: a 12-layer, 384-wide LightningDiT (6 heads, patch size 1) on
16x16x32 latents from a frozen VAE, conditioned on Qwen3-0.6B caption embeddings, with
velocity-prediction flow matching and 4 timestep tokens. They differ only in the knobs below.
Every FID is a single run of 200 epochs (~20k steps), measured against the 26k pretraining images.

## Naming

`{stage}_{tokenizer}_{caption length}_{loss}_{alignment}_{target}.yaml`

| Field | Values |
|---|---|
| stage | `pretrain`, `sft` |
| tokenizer | `e2e-invae`, `e2e-vavae` |
| caption length | `128tok`, `64tok` |
| loss | `mse` (squared error) or `cosine`: the alignment distance. `norepa` runs use `mse`, the flow-matching loss |
| alignment | `irepa`, `repa`, `norepa` |
| target | `dinov2`, `dinov3`, `pe-spatial`, `eupe`; left out for `norepa` |

## The final recipe

| Config | VAE | Alignment | Caption tokens | Epochs | EMA | LR | Result |
|---|---|---|---|---|---|---|---|
| `pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml` | E2E-INVAE | iREPA, EUPE, squared error | 128 | 200 | 0.9995 | 1e-4 for 100 epochs, then linear decay | **FID 8.32** |
| `sft_e2e-invae_128tok_mse_norepa.yaml` | E2E-INVAE | none | 128 | 100 | 0.995 | 5e-5, constant | Fine-tunes the 8.32 checkpoint on the 2k synthetic dogs: HPSv2 0.1625 -> 0.2427, PickScore 18.42 -> 19.61. Launch with `--ckpt <pretrain ckpt> --init-weights-only` |

## Tokenizer, caption length and alignment (Figs. 3-4, Table 2)

Each config changes one setting of the final recipe and keeps everything else fixed.

| In the paper | Config | Changed setting | FID |
|---|---|---|---|
| Fig. 3, Table 2 | `pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml` | none (the final recipe) | **8.32** |
| Fig. 3 | `pretrain_e2e-vavae_128tok_mse_irepa_eupe.yaml` | E2E-VAVAE instead of E2E-INVAE | 12.23 |
| Table 2 | `pretrain_e2e-invae_64tok_mse_irepa_eupe.yaml` | 64-token instead of 128-token captions | 8.54 |
| Fig. 4 | `pretrain_e2e-invae_128tok_mse_repa_eupe.yaml` | REPA instead of iREPA | 9.71 |
| Fig. 4 | `pretrain_e2e-invae_128tok_mse_norepa.yaml` | no alignment | 11.47 |

The tokenizers' reconstruction quality (Table 1: PSNR, SSIM, LPIPS, rFID) needs no config; it is
computed with `python -m minidog.recon_eval`, as shown in the [top-level README](../README.md).

## Target representation and distance (Table 3)

Sixteen configs named `pretrain_e2e-invae_128tok_{mse,cosine}_{repa,irepa}_{target}.yaml`, identical
apart from the alignment. REPA uses a linear projector; iREPA uses a 3x3 convolution with spatial
normalisation (gamma 0.7). All use E2E-INVAE, 128 caption tokens and lambda 0.5. FID at 20k:

| Target representation | REPA, squared error | REPA, cosine | iREPA, squared error | iREPA, cosine |
|---|---|---|---|---|
| `dinov2` (ViT-B/14) | **8.80** | 9.14 | 8.68 | 8.89 |
| `pe-spatial` (B/16) | 9.56 | 9.89 | 9.65 | 9.99 |
| `eupe` (ViT-B/16) | 9.71 | 10.20 | **8.32** | 8.37 |
| `dinov3` (ViT-B/16) | 10.12 | 10.32 | 8.46 | 9.01 |

Bold: best in each method, across both distances. Each target needs its own latents folder, cached
with the matching `minidog.precompute_latents{,_dinov3,_pe_spatial,_eupe}` script.

## Extra configs (not reported in the paper)

| Config | VAE | Alignment | Caption tokens |
|---|---|---|---|
| `pretrain_e2e-invae_64tok_mse_norepa.yaml` | E2E-INVAE | none | 64 |
| `pretrain_e2e-invae_64tok_mse_repa_dinov2.yaml` | E2E-INVAE | REPA, DINOv2 | 64 |
| `pretrain_e2e-invae_64tok_mse_repa_eupe.yaml` | E2E-INVAE | REPA, EUPE | 64 |
| `pretrain_e2e-vavae_128tok_mse_norepa.yaml` | E2E-VAVAE | none | 128 |
| `pretrain_e2e-vavae_128tok_mse_repa_dinov2.yaml` | E2E-VAVAE | REPA, DINOv2 | 128 |
| `pretrain_e2e-vavae_64tok_mse_norepa.yaml` | E2E-VAVAE | none | 64 |
| `pretrain_e2e-vavae_64tok_mse_repa_dinov2.yaml` | E2E-VAVAE | REPA, DINOv2 | 64 |

They fill in the tokenizer x caption-length x alignment grid. Each `norepa` config reuses the latents
folder of its `repa_dinov2` sibling.

## Data paths

Every config expects the dataset under `data/dog-t2i-diffusion-data/`, which is where
`hf download reyhanehesi/minidog-data --local-dir data/dog-t2i-diffusion-data` puts it.
The three tarballs extract into flat folders of `shard-*.tar`; the latents folders and the FID
reference stats are produced locally:

| Path (under `data/dog-t2i-diffusion-data/`) | What | Produced by |
|---|---|---|
| `dogs_recaptioned_wds/` | 26k real dog photos + captions | `tar -xzf dogs_recaptioned_wds.tar.gz -C dogs_recaptioned_wds/` |
| `dogs_synthetic_2k_wds/` | 2k synthetic SFT images + captions | `tar -xzf dogs_synthetic_2k_wds.tar.gz -C dogs_synthetic_2k_wds/` |
| `dogs_recaptioned_64tok_wds/` | the same 26k photos, captions of about 30-40 words (fit 64 tokens) | `tar -xzf dogs_recaptioned_64tok_wds.tar.gz -C dogs_recaptioned_64tok_wds/` |
| `dogs_recaptioned_stats.npz` | InceptionV3 mu/sigma for FID | `python -m minidog.fid_stats --data-dir .../dogs_recaptioned_wds --output .../dogs_recaptioned_stats.npz` |
| `dogs_recaptioned_latents_e2e-invae_eupe/` | the final recipe and the other EUPE runs (E2E-INVAE + EUPE features) | `python -m minidog.precompute_latents_eupe --config configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml --input-dir .../dogs_recaptioned_wds --output-dir .../dogs_recaptioned_latents_e2e-invae_eupe` |
| `dogs_synthetic_2k_latents_e2e-invae_eupe/` | SFT latents (E2E-INVAE, no alignment features) | `python -m minidog.precompute_latents_eupe --config configs/sft_e2e-invae_128tok_mse_norepa.yaml` and the synthetic shards |
| `dogs_recaptioned_latents_e2e-vavae_eupe/` | the tokenizer comparison, Fig. 3 (E2E-VAVAE + EUPE features) | `python -m minidog.precompute_latents_eupe --config configs/pretrain_e2e-vavae_128tok_mse_irepa_eupe.yaml` and the 26k shards |
| `dogs_recaptioned_64tok_latents_e2e-invae_eupe/` | the caption-length comparison, Table 2 (64-token captions + EUPE features) | `python -m minidog.precompute_latents_eupe --config configs/pretrain_e2e-invae_64tok_mse_irepa_eupe.yaml --input-dir .../dogs_recaptioned_64tok_wds` |
| `dogs_recaptioned_latents_e2e-invae/` | the DINOv2 runs and the 128-token no-alignment run | `python -m minidog.precompute_latents --config configs/pretrain_e2e-invae_128tok_mse_repa_dinov2.yaml --input-dir .../dogs_recaptioned_wds --output-dir .../dogs_recaptioned_latents_e2e-invae` |
| `dogs_recaptioned_latents_e2e-invae_{dinov3,pe_spatial}/` | the other target representations | same, with the matching `precompute_latents_*` script and config |
| `dogs_recaptioned_latents_e2e-vavae/`, `dogs_recaptioned_64tok_latents_e2e-{invae,vavae}/` | the extra configs | `python -m minidog.precompute_latents` with the `*_repa_dinov2` config of that tokenizer and caption length |
| `captions_500.json` | 500 held-out eval captions for `minidog.generate` | shipped |
