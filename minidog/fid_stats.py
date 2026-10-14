"""Compute the InceptionV3 reference statistics (mu, sigma) that FID is measured against.

Point it at a folder of WebDataset shards of real images (the same format the training data
uses) and pass the resulting .npz as eval.datasets.dogs.reference_npz in a config.

Usage:
    uv run python -m minidog.fid_stats --data-dir data/dog-t2i-diffusion-data/dogs_recaptioned_wds --output data/dog-t2i-diffusion-data/dogs_recaptioned_stats.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torchvision import transforms
from tqdm import tqdm

from minidog.data import DogsWebDataset


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Folder of .tar shards with jpg + txt samples.")
    parser.add_argument("--output", required=True, help="Where to write the .npz (keys: mu, sigma).")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    dataset = DogsWebDataset(args.data_dir, transform=transform, image_size=args.image_size)
    loader = wds.WebLoader(dataset.create_pipeline(shuffle=False), batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"{dataset.estimated_size} images in {dataset.num_shards} shards under {args.data_dir}")

    inception = FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    feats = []
    for images, _ in tqdm(loader, desc="InceptionV3 features"):
        (f,) = inception(images.mul(255).clamp(0, 255).to(torch.uint8).to(device))
        feats.append(f.cpu())
    feats = torch.cat(feats).double().numpy()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mu=feats.mean(axis=0), sigma=np.cov(feats, rowvar=False))
    print(f"Saved reference stats for {len(feats)} images to {out}")


if __name__ == "__main__":
    main()
