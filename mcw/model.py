"""CLIP model: windowing arm -> Atlas image tower, frozen-Qwen-embedding -> MLP text tower.

Image tower: HU clamped to [-1000, 1000] (the arms' common input) -> arm (mcw/windowing_arms.py:
clip (range configurable: win1 [-1000, 1000], win5 abdominal [-120, 240]) | pillar11 | learned | local |
local_shuf | dense) -> Pillar-0 normalisation (channel arms)
-> Atlas, 1152-d multi-scale readout with no extra projection. Channel arms feed K = 11
channels to the stem; `dense` feeds raw HU to a RampConv3d that replaces the stem's first conv.

Text tower: a 2-layer bias-free GELU MLP (4096 -> 2624 -> 1152) on the cached, L2-normalised
Qwen embedding (Pillar-0's CacheTextEncoder). Learnable logit scale (init 1/0.07).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .backbone import MODALITY, AtlasBackbone
from .gemm_conv3d import swap_conv3d
from .learned_window import LearnedWindow
from .patch_windows import PATCH_ARMS, build_patch_arm
from .vit3d import ViT3D
from .windowing_arms import HU_HI, HU_LO, PostNorm, build_arm, default_edges

# Geometry of the vendored 3D stem's first conv (ChestCTEmbed3D.conv_down[0]): Conv3d(in, 64, 3, stride=3).
STEM_OUT, STEM_STRIDE, STEM_PADDING = 64, 3, 0


class TextProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kind: str = "mlp"):
        super().__init__()
        if kind == "mlp":
            hidden = (in_dim + out_dim) // 2
            self.proj = nn.Sequential(nn.Linear(in_dim, hidden, bias=False), nn.GELU(),
                                      nn.Linear(hidden, out_dim, bias=False))
        elif kind == "linear":
            self.proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            raise ValueError(f"unknown text projection {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CTClip(nn.Module):
    def __init__(self, cfg: dict, text_dim: int):
        super().__init__()
        m, w = cfg["model"], cfg["windowing"]
        atlas_cfg = m.get("atlas") or {}
        backbone = m.get("backbone", "atlas")   # atlas (Pillar-0 Atlas small) | vit (mcw/vit3d.py, plain ViT-B/16)
        vit_cfg = m.get("vit") or {}
        if backbone == "vit":                   # per-patch arms default to the ViT token grid
            patch = int(vit_cfg.get("patch", 16))
        else:
            patch = int((atlas_cfg.get("patch_size") or [6, 6, 6])[0])     # stem grid = Atlas patch size
        if w["arm"] in PATCH_ARMS:              # per-patch windows from the HU density (mcw/patch_windows.py)
            adapter = build_patch_arm(w["arm"], w, int(w.get("patch") or patch))
            arm = dict(adapter=adapter, in_chans=getattr(adapter, "out_chans", adapter.k), first_conv=None, post_norm=True)
        else:
            arm = build_arm(w["arm"], edges=default_edges(float(w.get("edges_step", 25.0))), k=int(w.get("k", 11)),
                            patch=int(w.get("patch") or patch), stem_out=STEM_OUT, stem_stride=STEM_STRIDE,
                            stem_padding=STEM_PADDING, clip_range=tuple(w.get("clip_range") or (HU_LO, HU_HI)),
                            warp=bool(w.get("warp", False)), windows=w.get("windows"),
                            minmax=bool(w.get("minmax", True)), hu_hist=w.get("hu_hist"))
        self.arm, self.in_chans, self.post_norm = w["arm"], arm["in_chans"], arm["post_norm"]
        self.adapter = arm["adapter"]
        self.norm = PostNorm() if arm["post_norm"] else nn.Identity()
        conv_impl = m.get("conv_impl", "gemm")  # gemm | triton | native, the same for the arm's and Atlas's convs
        if conv_impl == "gemm":
            swap_conv3d(self.adapter)          # LocalWindow's 1x1x1 gate convs
        elif conv_impl == "triton":
            from .triton_conv3d import swap_conv3d_triton  # imports triton: only when asked for
            swap_conv3d_triton(self.adapter)
        elif conv_impl != "native":
            raise ValueError(f"model.conv_impl must be gemm | triton | native, got {conv_impl!r}")
        if backbone == "vit":
            if arm["first_conv"] is not None:
                raise ValueError("the dense arm replaces Atlas's first conv: not available with model.backbone=vit")
            self.visual = ViT3D(arm["in_chans"], grad_checkpointing=m.get("grad_checkpointing", False), **vit_cfg)
        elif backbone == "atlas":
            self.visual = AtlasBackbone(arm["in_chans"], atlas_cfg, grad_checkpointing=m.get("grad_checkpointing", False),
                                        conv_impl=conv_impl)
        else:
            raise ValueError(f"model.backbone must be atlas | vit, got {backbone!r}")
        if arm["first_conv"] is not None:      # dense: RampConv3d replaces the stem's first conv
            stem = self.visual.atlas.patch_embeds[MODALITY].conv_down
            old = stem[0]
            assert (old.out_channels, old.stride[0], old.padding[0]) == (STEM_OUT, STEM_STRIDE, STEM_PADDING), old
            stem[0] = arm["first_conv"]
        self.text = TextProjection(text_dim, self.visual.out_dim, m.get("text_proj", "mlp"))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / m.get("init_temperature", 0.07))))
        self.max_logit_scale = math.log(m.get("max_logit_scale", 100.0))
        self.autocast_dtype = torch.bfloat16 if m.get("precision", "bf16") == "bf16" else None

    def learned_window(self) -> LearnedWindow | None:
        """The arm's learned curves (learned / local / local_shuf), else None."""
        a = self.adapter
        if isinstance(a, LearnedWindow):
            return a
        return a.curves if isinstance(getattr(a, "curves", None), LearnedWindow) else None

    def set_shuffle_seed(self, seed: int | None) -> None:
        """local_shuf: fix the histogram permutation of the next forward (the gradient cache runs
        every micro-batch twice and both passes must see the same permutation)."""
        if hasattr(self.adapter, "shuffle_seed"):
            self.adapter.shuffle_seed = seed

    def window(self, hu: torch.Tensor) -> torch.Tensor:
        """(B, 1, D, H, W) HU -> backbone input: arm channels after PostNorm, or clamped HU (dense)."""
        return self.norm(self.adapter(hu.float().clamp(HU_LO, HU_HI)))

    def encode_image(self, hu: torch.Tensor) -> torch.Tensor:
        x = self.window(hu)
        use_ac = self.autocast_dtype is not None and x.is_cuda
        if use_ac and self.post_norm:
            x = x.to(self.autocast_dtype)      # the stem casts anyway; halves its channels-last copy
        with torch.autocast("cuda", dtype=self.autocast_dtype or torch.float32, enabled=use_ac):
            return self.visual(x).float()

    def encode_text(self, emb: torch.Tensor) -> torch.Tensor:
        return self.text(emb.float())

    def scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(max=self.max_logit_scale).exp()
