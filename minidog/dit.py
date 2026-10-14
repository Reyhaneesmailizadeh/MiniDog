from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class RoPE(nn.Module):
    def __init__(self, dim, vis_len, cond_len=0, theta=10000.,):
        super().__init__()
        # 2D RoPE for vision
        d, T = dim // 2, int(vis_len ** 0.5)
        vis_freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))  # [D//4]
        vis_base_angles = torch.outer(torch.arange(T).float(), vis_freqs)  # [T, D//4]
        vis_angles = torch.cat([
            vis_base_angles[:, None].expand(-1, T, -1),
            vis_base_angles[None, :].expand(T, -1, -1)
        ], dim=-1).reshape(vis_len, d)  # [T, T, D//2] -> [L', D//2]
        # no PE for extra (cls) or cond tokens
        cond_angles = torch.zeros(cond_len, dim // 2)
        angles = torch.cat([vis_angles, cond_angles], dim=0).repeat_interleave(2, dim=-1)  # [L, D]
        self.register_buffer("freqs_cos", angles.cos())
        self.register_buffer("freqs_sin", angles.sin())

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class NormAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads, self.dim, self.head_dim = num_heads, dim, dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope, attn_mask=None):
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = rope(q), rope(k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        return self.proj(out)


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, hidden_size, n_tokens=4, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.learnable_tokens = nn.Parameter(torch.normal(0, 1 / hidden_size**0.5, (n_tokens, hidden_size)))

    def forward(self, t):
        t = t[:, None] * self.W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        return self.learnable_tokens + t_embed.unsqueeze(1)


class ConditionEmbedder(nn.Module):
    """Projects text-encoder token embeddings into the DiT width."""
    def __init__(self, hidden_size, context_dim=768):
        super().__init__()
        self.norm = RMSNorm(context_dim)
        self.proj = nn.Linear(context_dim, hidden_size)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(y))


class LightningDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2/3 * hidden_size * mlp_ratio))

    def forward(self, x, rope, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LightningFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x):
        x = self.norm_final(x)
        return self.linear(x)


class LightningDiT(nn.Module):
    def __init__(
        self,
        input_size=16,
        in_channels=768,
        patch_size=1,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        enable_repa=False,
        irepa=False,
        repa_layer_depth=8,
        z_dim=None,
        context_dim=768,
        cond_arch=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.repa_layer_depth = repa_layer_depth

        self.x_embedder = PatchEmbed(input_size, self.patch_size, in_channels, hidden_size)

        self.num_cond_tokens = cond_arch.num_t_tokens + cond_arch.num_c_tokens
        self.num_c_tokens = cond_arch.num_c_tokens  # text/context tokens, always the last num_c_tokens of the sequence
        self.t_embedder = GaussianFourierEmbedding(self.hidden_size, cond_arch.num_t_tokens)
        self.ctx_embedder = ConditionEmbedder(self.hidden_size, context_dim)

        self.blocks = nn.ModuleList([
            LightningDiTBlock(self.hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.final_layer = LightningFinalLayer(self.hidden_size, self.patch_size, self.in_channels)
        self.rope = RoPE(self.hidden_size // num_heads, self.x_embedder.num_patches, self.num_cond_tokens)
        self.irepa = irepa
        if enable_repa:
            if irepa:
                # iREPA (arXiv:2512.10794) Alg. 1: a 3x3 conv over the patch grid in
                # place of the projection layer, so the projector can see each token's
                # neighbours and spatial structure survives the projection.
                self.repa_projector = nn.Conv2d(self.hidden_size, z_dim,
                                                kernel_size=3, padding=1)
            else:
                self.repa_projector = nn.Linear(self.hidden_size, z_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Patch embedders
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """[N, T, patch_size**2 * C] -> [N, C, H, W]"""
        h, c, p = int(x.shape[1] ** 0.5), self.in_channels, self.patch_size
        x = x.reshape(x.shape[0], h, h, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h*p, h*p)
        return x

    def _build_sequence(self, x, t, condition_kwargs):
        seq = []
        seq.append(self.x_embedder(x))
        seq.append(self.t_embedder(t))
        seq.append(self.ctx_embedder(condition_kwargs["context"]))
        seq = torch.cat(seq, dim=1)
        return seq

    def _build_attn_mask(self, seq, condition_kwargs):
        # Create multiplicative mask template
        attn_mask = torch.ones((seq.shape[0], seq.shape[1]), device=seq.device)
        cond_mask = condition_kwargs.get("attn_mask")
        if cond_mask is not None:
            attn_mask[:, -cond_mask.shape[1]:] = cond_mask
        # Convert to additive mask
        attn_mask = (1.0 - attn_mask[:, None, None, :]) * torch.finfo(seq.dtype).min
        return attn_mask

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        zt_intermediate = None
        x = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(x, condition_kwargs)
        n = self.x_embedder.num_patches
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                feats = x[:, :n, :]                                  # (B, N, C)
                if self.irepa:
                    g = int(n ** 0.5)                                # 16x16 patch grid
                    B, _, C = feats.shape
                    feats = feats.transpose(1, 2).reshape(B, C, g, g)
                    feats = self.repa_projector(feats)               # (B, z_dim, g, g)
                    zt_intermediate = feats.flatten(2).transpose(1, 2)
                else:
                    zt_intermediate = self.repa_projector(feats)

        x = self.final_layer(x[:, :n, :])
        x = self.unpatchify(x)

        if return_intermediate:
            return x, zt_intermediate
        return x
