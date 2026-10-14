"""Frozen DINOv3 ViT-B/16 (4 register tokens), an alternative REPA alignment target.

Mirrors DINOv2Encoder: images in [0, 255] -> (B, N, embed_dim) patch tokens.

Two differences from the DINOv2 path, both consequences of the patch size going
14 -> 16:
  * the DiT needs (16 x 16) = 256 tokens, so the encoder is fed the native
    resolution (256 / 16 = 16) rather than DINOv2's 224 (224 / 14 = 16);
  * DINOv3 uses RoPE, so there is no absolute pos_embed table to resample -- any
    input that is a multiple of the patch size just works.
"""
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from transformers import AutoModel

DEFAULT_DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # 768-dim, same as DINOv2 ViT-B/14


class DINOv3Encoder(nn.Module):
    def __init__(self, resolution: int = 256, model_name: str = DEFAULT_DINOV3_MODEL):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        # last_hidden_state is [CLS] + register tokens + patch tokens
        self.num_prefix_tokens = 1 + getattr(self.model.config, "num_register_tokens", 4)
        self.input_size = resolution
        if self.input_size % self.patch_size != 0:
            raise ValueError(
                f"resolution {self.input_size} is not a multiple of patch size {self.patch_size}"
            )
        self.normalize = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.requires_grad_(False)
        self.eval()

    @property
    def num_patch_tokens(self) -> int:
        return (self.input_size // self.patch_size) ** 2

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Images in [0, 255], (B, 3, H, W) -> patch tokens (B, N, embed_dim)."""
        x = torch.nn.functional.interpolate(self.normalize(x / 255.0), self.input_size, mode="bicubic")
        out = self.model(pixel_values=x).last_hidden_state
        return out[:, self.num_prefix_tokens:]
