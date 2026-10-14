"""Frozen EUPE ViT-B/16, an alternative REPA alignment target.

Meta's Efficient Universal Perception Encoder (facebookresearch/EUPE), a
distilled encoder trained on LVD-1689M. Mirrors DINOv2Encoder / DINOv3Encoder:
images in [0, 255] -> (B, N, embed_dim) patch tokens.

This reads the LAST block (via forward_features), matching how DINOv2, DINOv3
and PE-Spatial are used here, so a four-way teacher comparison stays controlled.
The multi-layer-sum variant (blocks 2, 5, 8, 11) is a separate option; see
diffusion-bench's EUPEMultiLayerSimpleAddEncoder if you want to try it.

Two things differ from the DINO encoders:
  * patch size is 16, so the encoder is fed the native 256 (256 / 16 = 16 per
    side -> the 256 tokens the DiT carries), not DINOv2's 224;
  * the checkpoint's final norm keeps learned affine parameters, which the
    diffusion-bench reference strips to match the DINOv2/v3 default. We do the
    same -- see strip_norm_affine.

Weights are NOT on HuggingFace Hub. Clone facebookresearch/EUPE and download
EUPE-ViT-B.pt, then set (same contract as diffusion-bench's eupe_loader):
    export EUPE_REPO_DIR=/path/to/EUPE
    export EUPE_CKPT_DIR=/path/to/checkpoints
"""
import os

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

DEFAULT_EUPE_MODEL = "eupe_vitb16"  # 86M params, 768-dim, 12 blocks, patch 16

CHECKPOINT_FILENAMES = {
    "eupe_vitt16": "EUPE-ViT-T.pt",
    "eupe_vits16": "EUPE-ViT-S.pt",
    "eupe_vitb16": "EUPE-ViT-B.pt",
}


class EUPEEncoder(nn.Module):
    def __init__(self, resolution: int = 256, model_name: str = DEFAULT_EUPE_MODEL,
                 strip_norm_affine: bool = True):
        super().__init__()
        if model_name not in CHECKPOINT_FILENAMES:
            raise ValueError(f"unknown EUPE model {model_name!r}; "
                             f"expected one of {sorted(CHECKPOINT_FILENAMES)}")

        repo_dir = os.environ.get("EUPE_REPO_DIR")
        ckpt_dir = os.environ.get("EUPE_CKPT_DIR")
        if not repo_dir or not ckpt_dir:
            raise ValueError(
                "EUPE_REPO_DIR and EUPE_CKPT_DIR must be set. Clone "
                "facebookresearch/EUPE and download the checkpoint, then point "
                "these at the repo and the checkpoint directory."
            )
        weights = os.path.join(ckpt_dir, CHECKPOINT_FILENAMES[model_name])
        if not os.path.exists(weights):
            raise FileNotFoundError(f"EUPE checkpoint not found: {weights}")

        self.model_name = model_name
        self.model = torch.hub.load(repo_dir, model_name, source="local", weights=weights)
        self.model.eval()

        self.embed_dim = self.model.embed_dim
        self.patch_size = 16
        if strip_norm_affine:
            # match the DINOv2/DINOv3 default, which normalizes without learned
            # scale/shift; keeping the affine leaves the features on a different
            # scale from the other REPA targets
            self.model.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False)

        self.input_size = resolution
        if self.input_size % self.patch_size != 0:
            raise ValueError(
                f"resolution {self.input_size} is not a multiple of patch size {self.patch_size}"
            )
        self.normalize = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.requires_grad_(False)

    @property
    def num_patch_tokens(self) -> int:
        return (self.input_size // self.patch_size) ** 2

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Images in [0, 255], (B, 3, H, W) -> patch tokens (B, N, embed_dim)."""
        x = torch.nn.functional.interpolate(x / 255.0, self.input_size, mode="bilinear",
                                            antialias=True)
        x = self.normalize(x)
        return self.model.forward_features(x)["x_norm_patchtokens"]
