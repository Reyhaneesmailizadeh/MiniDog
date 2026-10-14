#!/usr/bin/env python
"""Absolute HPSv2 and PickScore for one or more sample folders.

minidog.score compares two folders pairwise and reports a win rate. That answers
"which is preferred", but not "what does each one score", which is what a
before/after table needs. This reports the per-folder means instead:

  HPSv2      cosine(image, text) under the HPSv2 v2.1 reward model, ~0.20-0.30.
  PickScore  logit_scale * cosine(text, image) under PickScore_v1, ~15-25.
             Note minidog.score softmaxes this across an image pair; here it is
             left as the raw score so folders can be compared directly.

Usage:
    uv run python -m minidog.score_absolute \
        --dirs results/samples/pretrain-eupe-8.32 results/samples/sft-eupe-norepa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from minidog.score import _ensure_hpsv2_vocab


def collect(folder: Path):
    """Yield (breed, caption, png) for every sample that has a caption beside it."""
    for png in sorted(folder.rglob("*.png")):
        txt = png.with_suffix(".txt")
        if txt.exists():
            yield png.parent.name, txt.read_text().strip(), png


@torch.no_grad()
def pickscore_absolute(items, device):
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
    out = []
    for breed, caption, png in tqdm(items, desc="PickScore"):
        image = processor(images=[Image.open(png)], return_tensors="pt").to(device)
        text = processor(text=caption, padding=True, truncation=True,
                         max_length=77, return_tensors="pt").to(device)
        img = torch.nn.functional.normalize(model.get_image_features(**image), dim=-1)
        txt = torch.nn.functional.normalize(model.get_text_features(**text), dim=-1)
        out.append((breed, float(model.logit_scale.exp() * (txt @ img.T)[0, 0])))
    return out


@torch.no_grad()
def hpsv2_absolute(items, device):
    import huggingface_hub
    _ensure_hpsv2_vocab()
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import hps_version_map

    model, _, preprocess = create_model_and_transforms(
        "ViT-H-14", "laion2B-s32B-b79K", precision="amp", device=device, output_dict=True)
    ckpt = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    model.eval()
    tokenizer = get_tokenizer("ViT-H-14")

    out = []
    for breed, caption, png in tqdm(items, desc="HPSv2"):
        image = preprocess(Image.open(png)).unsqueeze(0).to(device)
        text = tokenizer([caption]).to(device)
        with torch.autocast(device_type=device.type):
            feats = model(image, text)
            score = (feats["image_features"] @ feats["text_features"].T)[0, 0]
        out.append((breed, float(score)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON with per-image scores.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for folder in args.dirs:
        items = list(collect(folder))
        if not items:
            raise SystemExit(f"no captioned samples found in {folder}")
        print(f"\n=== {folder}  ({len(items)} images) ===", flush=True)
        hps = hpsv2_absolute(items, device)
        pick = pickscore_absolute(items, device)
        results[str(folder)] = {
            "n": len(items),
            "hpsv2": sum(s for _, s in hps) / len(hps),
            "pickscore": sum(s for _, s in pick) / len(pick),
            "per_breed": {
                b: {
                    "hpsv2": sum(s for bb, s in hps if bb == b) / sum(1 for bb, _ in hps if bb == b),
                    "pickscore": sum(s for bb, s in pick if bb == b) / sum(1 for bb, _ in pick if bb == b),
                }
                for b in sorted({b for b, _ in hps})
            },
        }

    print(f"\n{'folder':44s} {'n':>5s} {'HPSv2':>9s} {'PickScore':>11s}")
    for name, r in results.items():
        print(f"{Path(name).name:44s} {r['n']:5d} {r['hpsv2']:9.4f} {r['pickscore']:11.2f}")

    if len(results) == 2:
        a, b = results.values()
        print(f"\n{'delta (second - first)':44s} {'':5s} "
              f"{b['hpsv2'] - a['hpsv2']:+9.4f} {b['pickscore'] - a['pickscore']:+11.2f}")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
