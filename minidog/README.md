# The `minidog` package

The text-to-image pipeline: a 12-block, 22M-parameter LightningDiT trained with flow matching on frozen
E2E-INVAE latents, conditioned on Qwen3-0.6B captions, with representation alignment to a frozen vision
encoder.

**Setup, the commands that reproduce the paper, and the expected results are in the
[top-level README](../README.md).** Every entry point runs from the repo root as
`python -m minidog.<name>`.

| | Files | What |
|---|---|---|
| Model and objective | [`dit.py`](dit.py), [`transport.py`](transport.py) | LightningDiT; flow matching, Euler sampler, CFG |
| Frozen components | [`vae.py`](vae.py), [`text_encoder.py`](text_encoder.py), [`eupe.py`](eupe.py), [`dinov2.py`](dinov2.py), [`dinov3.py`](dinov3.py), [`pe_spatial.py`](pe_spatial.py) | tokenizer, Qwen3 captions, the four target representations |
| Training | [`engine.py`](engine.py), [`data.py`](data.py), [`eval.py`](eval.py), [`config.py`](config.py) | epoch loop, loaders, FID/IS, config dataclasses |
| Entry points | [`precompute_latents`](precompute_latents.py) (and `_eupe`, `_dinov3`, `_pe_spatial`), [`fid_stats`](fid_stats.py), [`train`](train.py), [`offline_eval`](offline_eval.py), [`generate`](generate.py), [`score_absolute`](score_absolute.py), [`score`](score.py), [`recon_eval`](recon_eval.py) | in pipeline order |
| Utilities | [`utils/`](utils/) | checkpoints, distributed, optimizer, resume, W&B |
