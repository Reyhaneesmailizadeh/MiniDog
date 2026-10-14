"""Frozen PE-Spatial-B16-512, an alternative REPA alignment target.

Meta's Perception Encoder, spatial variant. The iREPA paper (arXiv 2512.10794)
uses it as its motivating example that spatial structure, not ImageNet accuracy,
predicts how good an alignment target is. Mirrors DINOv2Encoder /
DINOv3Encoder: images in [0, 255] -> (B, N, embed_dim) patch tokens.

Differences from the DINO encoders, all taken from the diffusion-bench reference
implementation (src/encoders/vision_encoder.py):
  * normalization is mean=std=0.5 per channel, NOT the ImageNet statistics;
  * the checkpoint is configured for 512x512 (a 32x32 grid), but PE interpolates
    its absolute position embedding, so feeding 256x256 gives the 16x16 = 256
    patch tokens the DiT needs;
  * PE-Spatial sets use_ln_post=False, so the model's ln_post is nn.Identity and
    forward_features returns the raw final-block output (std ~7.9, range +-96),
    unlike DINOv2/v3 whose patch tokens are post-LayerNorm (std ~1.8 / ~0.4).
    Feeding raw PE features to the REPA MSE would scale loss_repa ~20x above the
    DINOv2 target and swamp the diffusion loss, so layernorm_features applies a
    parameter-free LayerNorm over the feature dim by default. This mirrors the
    diffusion-bench reference, which always normalizes encoder features (via a
    normed target_encoder, or normalization_stat_path for PE) and keeps
    repa_coeff fixed at 0.5 across every encoder.
"""
import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from minidog.pe import VisionTransformer

DEFAULT_PE_SPATIAL_MODEL = "PE-Spatial-B16-512"  # patch 16, width 768, 12 layers


class PESpatialEncoder(nn.Module):
    def __init__(self, resolution: int = 256, model_name: str = DEFAULT_PE_SPATIAL_MODEL,
                 layernorm_features: bool = True):
        super().__init__()
        self.model_name = model_name
        self.model = VisionTransformer.from_config(model_name, pretrained=True)
        self.model.eval()

        self.embed_dim = self.model.width
        self.patch_size = self.model.patch_size
        self.layernorm_features = layernorm_features
        # same rule as the reference preprocess(): patch_size * (resolution // 16),
        # i.e. always 16 patches per side, matching the DiT's 16x16 token grid
        self.input_size = self.patch_size * (resolution // 16)
        self.normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.requires_grad_(False)

    @property
    def num_patch_tokens(self) -> int:
        return (self.input_size // self.patch_size) ** 2

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Images in [0, 255], (B, 3, H, W) -> patch tokens (B, N, embed_dim)."""
        x = torch.nn.functional.interpolate(x / 255.0, self.input_size, mode="bilinear")
        x = self.normalize(x)
        out = self.model.forward_features(x, norm=False, layer_idx=-1, strip_cls_token=False)
        if self.model.use_cls_token:
            out = out[:, 1:]  # drop CLS, keep patch tokens
        if self.layernorm_features:
            # parameter-free LayerNorm over the feature dim, putting PE on the same
            # scale as the post-LayerNorm DINOv2/v3 targets
            out = torch.nn.functional.layer_norm(out, (out.shape[-1],))
        return out
