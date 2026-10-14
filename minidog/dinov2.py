"""Frozen DINOv2 ViT-B/14 (no register tokens), the REPA alignment target."""
import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize


class DINOv2Encoder(nn.Module):
    def __init__(self, resolution: int = 256):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.model.head = nn.Identity()
        self.embed_dim = self.model.embed_dim
        self.input_size = 224 * resolution // 256
        grid = resolution // 16
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(self.model.pos_embed.data, [grid, grid])
        self.normalize = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Images in [0, 255], (B, 3, H, W) -> patch tokens (B, N, embed_dim)."""
        x = torch.nn.functional.interpolate(self.normalize(x / 255.0), self.input_size, mode="bicubic")
        return self.model.forward_features(x)["x_norm_patchtokens"]
