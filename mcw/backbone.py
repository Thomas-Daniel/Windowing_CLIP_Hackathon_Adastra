"""Atlas image backbone: Pillar-0's ``MultiModalAtlas`` (vendored, unmodified) for one
modality, plus optional activation checkpointing.

Default sizes = Pillar-0's Merlin abdomen-CT config
(pillar-pretrain model_configs/Qwen3-Embedding-8B-Atlas_small_d2d2d8_multiscale__pillar0_multimodal_ucsf_abd_ct_multiwindow.json):
384^3 input, conv stem with 6x6x6 total stride (64^3 tokens of 384-d), 3 scales built by
4x4x4 max-pooling with 4x4x4 attention windows, stages of 2 / 2 / 8 multi-scale blocks.
The output is the per-scale max-pooled tokens concatenated: 3 x 384 = 1152-d.

Only ``in_chans`` changes between the windowing runs (1, 11 or K channels into the first
conv); everything else is identical.
"""
from __future__ import annotations

import copy
import types

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .atlas.atlas_encoders import ConvBlock3D
from .atlas.multimodal_atlas import MultiModalAtlas
from .atlas.multimodal_msa import MultiScaleAttentionBlock, RelativePosEmb
from .gemm_conv3d import swap_conv3d

MODALITY = "abdomen_ct"

PILLAR0_ATLAS_SMALL = {
    "image_size": [384, 384, 384],
    "patch_size": [6, 6, 6],
    "merge_ratio": [4, 4, 4],
    "local2global": [4, 4, 4],
    "num_features": 192,
    "stages": [2, 2, 8],
    "atlas_args": {
        "attn_drop": 0.0, "class_token": False, "depth": 8, "dim": 384, "drop": 0.0,
        "drop_path_rate": 0.3, "embed_dim": 384, "global_pool": "avg", "in_chans": 1, "in_dim": 64,
        "mlp_ratio": 4.0, "num_classes": 0, "num_features": 192, "num_heads": 6, "num_scales": 3,
        "pre_norm": False, "qk_scale": False, "qkv_bias": True,
    },
}
# NB: AtlasStage has no ``drop_path_rate`` argument (it takes ``drop_path``), so Pillar-0's
# 0.3 lands in **kwargs and stochastic depth is effectively off. Kept as-is for fidelity.


def _checkpointed(module: nn.Module) -> None:
    """Recompute ``module`` in backward. List arguments are copied on every call because
    MultiScaleAttentionBlock.forward_sequential overwrites the entries of its input list,
    which would otherwise make the recomputation see already-processed tensors."""
    orig = module.forward

    def run(*args, **kwargs):
        args = [list(a) if isinstance(a, list) else a for a in args]
        return orig(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(run, *args, use_reentrant=False, **kwargs)
        return run(*args, **kwargs)

    module.forward = types.MethodType(forward, module)


def _stem_forward_channels_last(self, x: torch.Tensor) -> torch.Tensor:
    """Conv stem forward in channels_last_3d (bound method: survives deepcopy)."""
    return type(self).forward(self, x.contiguous(memory_format=torch.channels_last_3d)).contiguous()


class AtlasBackbone(nn.Module):
    def __init__(self, in_chans: int, cfg: dict | None = None, grad_checkpointing: bool = False,
                 conv_impl: str = "gemm"):
        super().__init__()
        c = copy.deepcopy(PILLAR0_ATLAS_SMALL)
        for k, v in (cfg or {}).items():
            if k == "atlas_args":
                c["atlas_args"].update(v)
            else:
                c[k] = v
        model_config = {
            "modalities": {MODALITY: {
                "image_size": c["image_size"], "in_chans": in_chans, "local2global": c["local2global"],
                "merge_ratio": c["merge_ratio"], "num_features": c["num_features"], "patch_size": c["patch_size"],
            }},
            "model": {"atlas_args": c["atlas_args"], "stages": c["stages"]},
        }
        self.atlas = MultiModalAtlas(args=None, embed_dim=c["atlas_args"]["dim"] * len(c["stages"]),
                                     model_config=model_config)
        self.image_size = tuple(c["image_size"])
        self.out_dim = c["atlas_args"]["dim"] * len(c["stages"])
        if conv_impl == "gemm":            # MIOpen's 3D convs are unusably slow on MI250X (mcw/gemm_conv3d.py)
            swap_conv3d(self.atlas)
        elif conv_impl == "triton":        # prototype: implicit GEMM, no column buffer (mcw/triton_conv3d.py)
            from .triton_conv3d import swap_conv3d_triton
            swap_conv3d_triton(self.atlas)
        elif conv_impl != "native":
            raise ValueError(f"model.conv_impl must be gemm | triton | native, got {conv_impl!r}")
        if conv_impl in ("gemm", "triton"):
            # run the whole conv stem in channels_last_3d (GemmConv3d, BatchNorm, LayerNorm3d and
            # the activations all keep it), back to NCDHW for the token stages
            stem = self.atlas.patch_embeds[MODALITY]
            stem.forward = types.MethodType(_stem_forward_channels_last, stem)
        if grad_checkpointing:
            for m in self.atlas.modules():
                if isinstance(m, (ConvBlock3D, MultiScaleAttentionBlock)):
                    _checkpointed(m)

    def train(self, mode: bool = True):
        # RelativePosEmb caches its embedding while in eval mode and only drops the cache on
        # the next training forward: clear it on every mode switch so an eval never reuses
        # embeddings computed with older weights.
        for m in self.modules():
            if isinstance(m, RelativePosEmb):
                m.modality_grid_exists, m.posemb = {}, {}
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_chans, D, H, W) at ``image_size`` -> (B, out_dim)."""
        if tuple(x.shape[2:]) != self.image_size:
            raise ValueError(f"Atlas layout is built for {self.image_size}, got {tuple(x.shape[2:])}")
        return self.atlas({MODALITY: x})
