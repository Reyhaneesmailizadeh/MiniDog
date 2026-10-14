"""Score two folders of generated images against each other with learned human-preference models.

Both folders must come from `python -m minidog.generate --group-by-breed` on the same captions
(layout: <dir>/<breed>/<name>.png + <name>.txt), e.g. a pretraining and an SFT checkpoint.

  PickScore  (always)      pairwise preference probability per caption -> win rate of B over A
  HPSv2                    per-image score (the hpsv2 wheel lacks its BPE vocab; it is fetched on first use)

Usage:
    uv run python -m minidog.score --a results/samples/pretrain --b results/samples/sft
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def load_pairs(dir_a: Path, dir_b: Path):
    """Yield (breed, caption, image_a, image_b) for every caption present in both folders."""
    for png_a in sorted(dir_a.glob("*/*.png")):
        png_b = dir_b / png_a.relative_to(dir_a)
        if not png_b.exists():
            continue
        yield png_a.parent.name, png_a.with_suffix(".txt").read_text().strip(), png_a, png_b


@torch.no_grad()
def pickscore(pairs, device):
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
    out = []  # (breed, prob_a, prob_b)
    for breed, caption, png_a, png_b in tqdm(pairs, desc="PickScore"):
        images = processor(images=[Image.open(png_a), Image.open(png_b)], return_tensors="pt").to(device)
        text = processor(text=caption, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        img = torch.nn.functional.normalize(model.get_image_features(**images), dim=-1)
        txt = torch.nn.functional.normalize(model.get_text_features(**text), dim=-1)
        probs = torch.softmax(model.logit_scale.exp() * (txt @ img.T)[0], dim=-1).tolist()
        out.append((breed, probs[0], probs[1]))
    return out


def _ensure_hpsv2_vocab():
    """The hpsv2 1.2.0 wheel omits the CLIP BPE vocab its tokenizer needs at import time; fetch it once."""
    import importlib.util
    import urllib.request
    spec = importlib.util.find_spec("hpsv2")
    if spec is None or not spec.submodule_search_locations:
        return
    vocab = Path(list(spec.submodule_search_locations)[0]) / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
    if not vocab.exists():
        print(f"Downloading CLIP BPE vocab for hpsv2 -> {vocab}")
        urllib.request.urlretrieve("https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz", vocab)


@torch.no_grad()
def hpsv2_scores(pairs, device):
    """Per-image HPSv2 scores for both folders; returns None if hpsv2 is not installed."""
    try:
        import huggingface_hub
        _ensure_hpsv2_vocab()
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.utils import hps_version_map
    except ImportError:
        return None
    model, _, preprocess = create_model_and_transforms("ViT-H-14", "laion2B-s32B-b79K", precision="amp", device=device, output_dict=True)
    ckpt = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    model.eval()
    tokenizer = get_tokenizer("ViT-H-14")
    out = []  # (breed, score_a, score_b)
    for breed, caption, png_a, png_b in tqdm(pairs, desc="HPSv2"):
        images = torch.stack([preprocess(Image.open(p)) for p in (png_a, png_b)]).to(device)
        text = tokenizer([caption]).to(device)
        with torch.autocast(device_type=device.type):
            feats = model(images, text)
            scores = (feats["image_features"] @ feats["text_features"].T)[:, 0]
        out.append((breed, float(scores[0]), float(scores[1])))
    return out


def summarize(name, rows, fmt, label_a, label_b):
    breeds = sorted({r[0] for r in rows})
    print(f"\n{name}  ({len(rows)} caption pairs)")
    print(f"{'breed':22s} {label_a:>12s} {label_b:>12s}")
    for breed in breeds:
        a = [r[1] for r in rows if r[0] == breed]; b = [r[2] for r in rows if r[0] == breed]
        print(f"{breed:22s} {fmt(a):>12s} {fmt(b):>12s}")
    a = [r[1] for r in rows]; b = [r[2] for r in rows]
    print(f"{'overall':22s} {fmt(a):>12s} {fmt(b):>12s}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, type=Path, help="Baseline folder, e.g. pretraining samples.")
    parser.add_argument("--b", required=True, type=Path, help="Comparison folder, e.g. SFT samples.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON with per-pair scores.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_a, label_b = args.a.name, args.b.name

    pairs = list(load_pairs(args.a, args.b))
    if not pairs:
        raise SystemExit(f"No matching <breed>/<name>.png pairs found under {args.a} and {args.b}")

    results = {"pickscore": pickscore(pairs, device)}
    mean = lambda xs: f"{sum(xs) / len(xs):.4f}"
    wins = lambda xs: f"{100 * sum(x > 0.5 for x in xs) / len(xs):.1f}% wins"
    summarize("PickScore mean preference probability", results["pickscore"], mean, label_a, label_b)
    summarize("PickScore win rate", results["pickscore"], wins, label_a, label_b)

    hps = hpsv2_scores(pairs, device)
    if hps is None:
        print("\nHPSv2 skipped: the hpsv2 package is not importable.")
    else:
        results["hpsv2"] = hps
        summarize("HPSv2 mean score", hps, mean, label_a, label_b)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({k: [dict(breed=b, a=x, b=y) for b, x, y in v] for k, v in results.items()}, indent=1))
        print(f"\nPer-pair scores written to {args.output}")


if __name__ == "__main__":
    main()
