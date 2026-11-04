"""Plain 3D ViT image backbone: the "classic ViT" counterpart of Atlas for the windowing study.

Non-overlapping p^3 patches embedded by one linear layer (no conv stem, so no 3D convolution at all),
fixed 3D sin-cos position embedding, a class token, ``depth`` pre-norm transformer blocks with global
attention (F.scaled_dot_product_attention: flash attention in bf16 on MI250X / MI300A), and the class
token -> LayerNorm -> linear to ``out_dim`` (1152, the Atlas readout size, so the text projection and
the CLIP head are unchanged). Defaults = ViT-B/16 (86 M parameters in the blocks): 384^3 -> 24^3 =
13 824 tokens of 768-d.

No stochastic depth / dropout: the gradient cache runs every micro-batch twice (no-grad, then with
grad) and both passes must be identical for the cached gradients to be exact.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sincos_3d(grid: int, dim: int) -> torch.Tensor:
    """(grid^3, dim) fixed embedding: dim / 3 channels per axis (z, y, x), half sin, half cos."""
    per = dim // 3
    if per % 2 or per * 3 != dim:
        raise ValueError(f"dim {dim} must be a multiple of 6")
    omega = 1.0 / (10000 ** (torch.arange(per // 2, dtype=torch.float64) / (per // 2)))
    pos = torch.arange(grid, dtype=torch.float64)
    axis = torch.cat([torch.sin(pos[:, None] * omega), torch.cos(pos[:, None] * omega)], 1)   # (grid, per)
    z, y, x = torch.meshgrid(pos.long(), pos.long(), pos.long(), indexing="ij")
    return torch.cat([axis[z.reshape(-1)], axis[y.reshape(-1)], axis[x.reshape(-1)]], 1).float()


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.fc1, self.fc2 = nn.Linear(dim, int(dim * mlp_ratio)), nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class ViT3D(nn.Module):
    def __init__(self, in_chans: int, image_size: int = 384, patch: int = 16, dim: int = 768, depth: int = 12,
                 heads: int = 12, mlp_ratio: float = 4.0, out_dim: int = 1152, grad_checkpointing: bool = False):
        super().__init__()
        if image_size % patch:
            raise ValueError(f"image size {image_size} not divisible by patch {patch}")
        self.p, self.g, self.grad_checkpointing = patch, image_size // patch, grad_checkpointing
        self.image_size, self.out_dim = (image_size,) * 3, out_dim
        self.patch_embed = nn.Linear(in_chans * patch ** 3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos", sincos_3d(self.g, dim)[None], persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim, eps=1e-6), nn.Linear(dim, out_dim)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        bound = math.sqrt(6.0 / (self.patch_embed.in_features + self.patch_embed.out_features))   # MAE: xavier
        nn.init.uniform_(self.patch_embed.weight, -bound, bound)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_chans, D, H, W) at ``image_size`` -> (B, out_dim)."""
        if tuple(x.shape[2:]) != self.image_size:
            raise ValueError(f"ViT3D is built for {self.image_size}, got {tuple(x.shape[2:])}")
        B, C = x.shape[:2]
        p, g = self.p, self.g
        x = x.reshape(B, C, g, p, g, p, g, p).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, g ** 3, C * p ** 3)
        x = self.patch_embed(x) + self.pos.to(x.dtype)
        x = torch.cat([self.cls.expand(B, -1, -1).to(x.dtype), x], dim=1)
        for blk in self.blocks:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.head(self.norm(x[:, 0]))
