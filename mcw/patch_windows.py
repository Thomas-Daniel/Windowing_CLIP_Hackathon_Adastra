"""Per-patch windowing learned implicitly by CLIP, conditioned on the patch's HU density.

No preset windows. The patch grid is the Atlas token grid (6^3 voxels = 9 mm cubes, 64^3 patches
per 384^3 volume). For every patch, the HU density (80-bin histogram over [-1000, 1000], computed
upstream on raw HU, optionally pooled over the 3^3 neighbouring patches) feeds a small 1x1x1 conv
network that sets the patch's windows. The last layer of that network starts at zero, so every
patch starts with the same global windows, which are initialised from the data (global HU
distribution of body voxels, scripts/hu_histogram.py), not from presets.

Arms (K output channels to the stem, default 8):
  patch_bank       K parametric windows per patch: out_k = sigmoid(4 (x - c_k(patch)) / w_k(patch)),
                   c_k = c0_k + 300 tanh(r / 3) HU, w_k = w0_k exp(1.5 tanh(r' / 1.5)) (c0 learned in
                   units of 100 HU). The network input is the normalised log density.
                   Global c0 / w0 learned, initialised at the body-HU quantiles.
  patch_cdf        K free monotone transfer curves per patch (80 bins of 25 HU): logits_k(patch) =
                   theta_k + net_k(density) + lambda_k log(density); the CDF of softmax(logits)
                   is the curve. Global theta_k initialised as global histogram equalisation at
                   K sharpness levels (density^gamma_k, gamma from 0 = linear ramp to 1.5).
  patch_eq         patch_cdf without the network: only the learned local-equalisation weight
                   lambda_k (lambda = 1: the patch's own histogram equalisation), the simplest use
                   of the upstream density.
  global_bank      control: patch_bank without the per-patch network (same init, same K).
  patch_bank_shuf  control: patch_bank with the densities permuted across patches (same
                   parameters, no local information; permutation seeded per micro-batch).
  window_select    the 11 fixed pillar11 channels mixed by a softmax instead of the stem conv: J (1 or 3)
                   output channels, each a softmax-weighted sum of the 11; weights global, or per patch
                   from the HU density (router features), see WindowSelect.

Both explicit backwards avoid atomics on parameters: per-patch window gradients are sums over each
patch's 6^3 voxels (a reshape), per-patch bin-mass gradients are bincount histograms.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .windowing_arms import PILLAR_PRESETS, FixedWindows

HU_LO, HU_HI = -1000.0, 1000.0


def _patch_view(x: torch.Tensor, s: int) -> torch.Tensor:
    """(B, D, H, W) -> (B, d, s, h, s, w, s)."""
    B, D, H, W = x.shape
    return x.view(B, D // s, s, H // s, s, W // s, s)


def _patch_ids(shape, s: int, device) -> torch.Tensor:
    """Flat patch index of every voxel, (B, D, H, W) int64."""
    B, D, H, W = shape
    d, h, w = D // s, H // s, W // s
    b_ = torch.arange(B, device=device).view(B, 1, 1, 1) * (d * h * w)
    z_ = (torch.arange(D, device=device) // s).view(1, D, 1, 1) * (h * w)
    y_ = (torch.arange(H, device=device) // s).view(1, 1, H, 1) * w
    x_ = (torch.arange(W, device=device) // s).view(1, 1, 1, W)
    return b_ + z_ + y_ + x_


def _bins(x: torch.Tensor, lo: float, step: float, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    f = (x - lo) / step
    j = f.floor().clamp_(0, m - 1)
    return j.long(), f - j


def _box3(x: torch.Tensor) -> torch.Tensor:
    """Mean over the 3x3x3 neighbourhood (zero padding, count-normalised) of (B, C, d, h, w)."""
    ones = torch.ones_like(x[:, :1])
    for dim in (2, 3, 4):
        def shift_sum(t):
            pad = [0, 0] * (4 - dim) + [1, 1]
            tp = F.pad(t, pad)
            n = t.shape[dim]
            return tp.narrow(dim, 0, n) + tp.narrow(dim, 1, n) + tp.narrow(dim, 2, n)
        x, ones = shift_sum(x), shift_sum(ones)
    return x / ones


class PatchDensity(nn.Module):
    """Per-patch HU density: (B, 1, D, H, W) HU -> (B, M, d, h, w) bin fractions (sum 1)."""

    def __init__(self, patch: int = 6, step: float = 25.0, context: int = 3, method: str = "bincount"):
        super().__init__()
        self.s, self.step, self.context, self.method = patch, step, context, method
        self.m = int(round((HU_HI - HU_LO) / step))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().squeeze(1).clamp(HU_LO, HU_HI)
        B, D, H, W = x.shape
        s, m = self.s, self.m
        d, h, w = D // s, H // s, W // s
        j, _ = _bins(x, HU_LO, self.step, m)
        if self.method == "bincount":                 # default: 12-13 ms per 384^3 volume on MI250X / MI300A
            idx = (_patch_ids(x.shape, s, x.device) * m + j).reshape(-1)
            hist = torch.bincount(idx, minlength=B * d * h * w * m).float().view(B, d, h, w, m)
        else:                                         # "sort": atomic-free, identical counts, slower (15-41 ms)
            jp = _patch_view(j.to(torch.int32), s).permute(0, 1, 3, 5, 2, 4, 6).reshape(-1, s ** 3)
            jp = torch.sort(jp, dim=-1).values
            k = torch.arange(m + 1, device=x.device, dtype=torch.int32).expand(jp.shape[0], m + 1).contiguous()
            below = torch.searchsorted(jp, k)         # voxels with bin id < k, per patch
            hist = (below[:, 1:] - below[:, :-1]).float().view(B, d, h, w, m)
        hist = hist.permute(0, 4, 1, 2, 3).contiguous() / (s ** 3)
        if self.context == 3:
            hist = _box3(hist)
        return hist


def _shuffle_patches(t: torch.Tensor, seed: int | None, training: bool) -> torch.Tensor:
    B, C, d, h, w = t.shape
    flat = t.permute(0, 2, 3, 4, 1).reshape(-1, C)
    gen = torch.Generator(device=t.device).manual_seed(0 if (not training or seed is None) else int(seed))
    idx = torch.randperm(flat.shape[0], generator=gen, device=t.device)
    return flat[idx].view(B, d, h, w, C).permute(0, 4, 1, 2, 3).contiguous()


def _density_features(dens: torch.Tensor) -> torch.Tensor:
    """Network input: log density mapped to ~[-1, 2] (fraction 1e-3 -> -1, 1e-2 -> 0, 0.1 -> 1, 1 -> 2).
    Unnormalised log densities (down to -7) made each Adam step move the network output by several
    units: the per-patch windows saturated within a few steps (smoke job 5636232)."""
    return (torch.log(dens + 1e-3) + 4.6) / 2.3


# ----------------------------------------------------------------------------- parametric bank
class _BankFn(torch.autograd.Function):
    """out_k = sigmoid(4 (x - C_k[patch]) / W_k[patch]); C, W: (K, B, d, h, w)."""

    @staticmethod
    def forward(ctx, C, Wd, x, s):
        K = C.shape[0]
        B, D, H, Wv = x.shape
        x7 = _patch_view(x, s)
        out = torch.empty((K, B, D, H, Wv), device=x.device, dtype=torch.promote_types(x.dtype, torch.float32))
        for k in range(K):
            c7 = C[k][:, :, None, :, None, :, None]
            w7 = Wd[k][:, :, None, :, None, :, None]
            out[k] = torch.sigmoid(4.0 * (x7 - c7) / w7).reshape(B, D, H, Wv)
        ctx.save_for_backward(C, Wd, x)
        ctx.s = s
        return out

    @staticmethod
    def backward(ctx, g):
        C, Wd, x = ctx.saved_tensors
        s = ctx.s
        x7 = _patch_view(x, s)
        gC, gW = torch.zeros_like(C), torch.zeros_like(Wd)
        for k in range(C.shape[0]):
            c7 = C[k][:, :, None, :, None, :, None]
            w7 = Wd[k][:, :, None, :, None, :, None]
            z = 4.0 * (x7 - c7) / w7
            sg = torch.sigmoid(z)
            t = _patch_view(g[k].to(sg.dtype), s) * sg * (1 - sg)          # g * dsigmoid/dz
            gC[k] = -(t * (4.0 / w7)).sum(dim=(2, 4, 6))
            gW[k] = -(t * z / w7).sum(dim=(2, 4, 6))                     # dz/dW = -z / W
        return gC, gW, None, None


class PatchWindowBank(nn.Module):
    """K sigmoid windows, optionally shifted per patch (per_patch) and / or per scan (scan_context).

    Options added after the mix runs (CT-RATE, 2-3x noisier than the other sources, is the one source where the
    learned banks lose to clip / pillar11):
      width_min      widths floored at width_min HU (W = width_min + exp(log_width)), so no window amplifies noise;
      smooth         the windows read a box-smoothed HU (kernel ``smooth``); the passthrough ramp stays raw;
      scan_context   a per-scan shift of every window from the volume's HU histogram and soft-tissue noise level;
      init           "quantile" (body-HU quantiles) or "uniform" (evenly over [-1000, 1000]);
      fixed_windows  hybrid: the 11 fixed pillar11 channels (10 presets + min-max) come before the learned ones.
    """

    def __init__(self, k: int = 8, patch: int = 6, step: float = 25.0, context: int = 3, hidden: int = 64,
                 per_patch: bool = True, control: str | None = None, hu_hist: str | None = None,
                 shift_max_hu: float = 300.0, width_max_log: float = 1.5, passthrough: bool = False,
                 width_min: float = 0.0, smooth: int = 0, scan_context: bool = False, init: str = "quantile",
                 fixed_windows: bool = False):
        super().__init__()
        self.k, self.s, self.per_patch, self.control = k, patch, per_patch, control
        # per-patch bounds: centre within +-shift_max_hu (100 HU per network unit near 0), width x exp(+-width_max_log)
        self.shift_max_hu, self.width_max_log = float(shift_max_hu), float(width_max_log)
        self.passthrough = passthrough                   # V1: + the fixed [-1000, 1000] ramp (the clip arm) as a channel
        self.width_min, self.smooth = float(width_min), int(smooth)
        if self.smooth and self.smooth % 2 == 0:
            raise ValueError(f"smooth must be an odd kernel size, got {smooth}")
        self.fixed = FixedWindows(PILLAR_PRESETS) if fixed_windows else None
        self.out_chans = k + int(passthrough) + (len(PILLAR_PRESETS) + 1 if fixed_windows else 0)
        self.density = PatchDensity(patch, step, context)
        m = self.density.m
        if init == "uniform":                            # centres evenly over [-1000, 1000], widths = 2 x spacing
            sp = (HU_HI - HU_LO) / k
            c0, w0 = HU_LO + sp * (np.arange(k) + 0.5), np.full(k, 2 * sp)
        elif init == "quantile":                         # centres at the body-HU quantiles (k + 0.5) / K, widths = 2 x spacing
            edges = np.arange(HU_LO, HU_HI + step / 2, step)
            hb = np.load(hu_hist)["hist_body"] if hu_hist else np.ones(len(edges) - 1)
            cdf = np.concatenate([[0], np.cumsum(hb / hb.sum())])
            c0 = np.interp((np.arange(k) + 0.5) / k, cdf, edges)
            gap = np.diff(np.concatenate([[HU_LO], c0, [HU_HI]]))
            w0 = np.clip(np.maximum(gap[:-1], gap[1:]) * 2, 50, 1000)
        else:
            raise ValueError(f"init must be quantile | uniform, got {init!r}")
        # centres learned in units of 100 HU (Adam moves a parameter by ~lr per step, whatever its unit)
        self.center100 = nn.Parameter(torch.tensor(c0 / 100.0, dtype=torch.float32))
        self.log_width = nn.Parameter(torch.tensor(np.log(np.maximum(w0 - self.width_min, 1.0)), dtype=torch.float32))
        if per_patch:
            self.net = nn.Sequential(nn.Conv3d(m, hidden, 1), nn.GELU(), nn.Conv3d(hidden, 2 * k, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.scan_net = None
        if scan_context:
            self.scan_net = nn.Sequential(nn.Linear(m + 1, hidden), nn.GELU(), nn.Linear(hidden, 2 * k))
            nn.init.zeros_(self.scan_net[-1].weight)
            nn.init.zeros_(self.scan_net[-1].bias)
        self.shuffle_seed = None
        self.last_stats: dict = {}

    @property
    def center(self) -> torch.Tensor:
        return 100.0 * self.center100

    @property
    def width(self) -> torch.Tensor:
        return self.width_min + self.log_width.exp()

    @staticmethod
    @torch.no_grad()
    def scan_noise(x: torch.Tensor) -> torch.Tensor:
        """(B,) soft-tissue noise feature: log mean |difference| of neighbour voxels (along W) that are both in
        [-100, 150] HU, minus 3.3 (~0 at 27 HU: Merlin / INSPECT / CancerVerse about -0.5, noisy CT-RATE up to +0.6)."""
        hu = x.float().squeeze(1)
        a, b = hu[..., :-1], hu[..., 1:]
        ok = ((a > -100) & (a < 150) & (b > -100) & (b < 150)).float()
        mad = ((a - b).abs() * ok).sum((1, 2, 3)) / ok.sum((1, 2, 3)).clamp_min(1.0)
        return torch.log(mad.clamp_min(1.0)) - 3.3

    def window_maps(self, x: torch.Tensor):
        """Per-patch centres and widths, (K, B, d, h, w) each, in HU."""
        B = x.shape[0]
        d, h, w = (n // self.s for n in x.shape[2:])
        C = self.center.view(-1, 1, 1, 1, 1).expand(self.k, B, d, h, w)
        lw = self.log_width.view(-1, 1, 1, 1, 1).expand(self.k, B, d, h, w)
        r = None
        if self.per_patch or self.scan_net is not None:
            dens = self.density(x)                                       # (B, m, d, h, w)
        if self.per_patch:
            feat = _density_features(dens)
            if self.control == "shuffle":
                feat = _shuffle_patches(feat, self.shuffle_seed, self.training)
            r = self.net(feat).float()                                   # (B, 2K, d, h, w)
        if self.scan_net is not None:
            g = torch.cat([_density_features(dens.mean((2, 3, 4))), self.scan_noise(x)[:, None]], 1)   # (B, m + 1)
            rs = self.scan_net(g).float()[:, :, None, None, None]        # (B, 2K, 1, 1, 1): the same shift everywhere
            if self.training:
                self._scan_shift = float((self.shift_max_hu * torch.tanh(100.0 * rs[:, :self.k] / self.shift_max_hu)).abs().mean())
            r = rs if r is None else r + rs
        if r is not None:
            dc, dlw = r[:, :self.k].transpose(0, 1), r[:, self.k:].transpose(0, 1)
            S, L = self.shift_max_hu, self.width_max_log
            C = C + S * torch.tanh(100.0 * dc / S)                       # 100 HU per unit, bounded at +-S HU
            lw = lw + L * torch.tanh(dlw / L)                            # width x exp(+-L) at most
            sat = (((100.0 * dc / S).abs() > 5 / 3) | ((dlw / L).abs() > 5 / 3)).float().mean()
            self._sat = float(sat) if self.training else 0.0
        return C, self.width_min + lw.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, 1, D, H, W) HU
        with torch.autocast(device_type=x.device.type, enabled=False):
            C, Wd = self.window_maps(x)
            if self.training:
                with torch.no_grad():
                    dc = C - self.center.detach().view(-1, 1, 1, 1, 1)
                    self.last_stats = {                              # per-patch stats only when windows vary
                        "patch_center_std_hu": float(C.flatten(1).std(1).mean()),
                        "patch_center_dev_mean_hu": float(dc.abs().mean()),
                        "patch_active": float((dc.abs() > 25).float().mean()),
                        "patch_logwidth_std": float(Wd.log().flatten(1).std(1).mean()),
                        "patch_saturated": getattr(self, "_sat", 0.0)} if self.per_patch else {}
                    self.last_stats |= {
                        "bank_center_min": float(self.center.detach().min()),
                        "bank_center_max": float(self.center.detach().max()),
                        "bank_width_min": float(self.width.min()),
                        "bank_width_max": float(self.width.max())}
                    if self.scan_net is not None:
                        self.last_stats["scan_shift_abs_hu"] = getattr(self, "_scan_shift", 0.0)
            hu = x.float().squeeze(1).clamp(HU_LO, HU_HI)
            hw = hu
            if self.smooth:
                hw = F.avg_pool3d(hu[:, None], self.smooth, stride=1, padding=self.smooth // 2,
                                  count_include_pad=False)[:, 0]
            out = _BankFn.apply(C.contiguous(), Wd.contiguous(), hw.contiguous(), self.s).transpose(0, 1)   # (B, K, D, H, W)
        if self.passthrough:
            out = torch.cat([out, ((hu - HU_LO) / (HU_HI - HU_LO))[:, None].to(out.dtype)], dim=1)
        if self.fixed is not None:
            out = torch.cat([self.fixed(hu[:, None]).to(out.dtype), out], dim=1)
        return out



# ----------------------------------------------------------------------------- window routing
class _RouteFn(torch.autograd.Function):
    """out[p, v, k] = sum_m alpha[p, k, m] * sigmoid(4 (x[p, v] - C[m]) / W[m]) for patch p, voxel v of the patch.
    Per patch, the (V, M) window responses times the (M, K) routing weights, as batched matmuls over chunks of
    patches; the backward recomputes the responses chunk by chunk (nothing of size V x M is kept)."""

    @staticmethod
    def forward(ctx, alpha, C, Wd, xp, chunk):
        P, V = xp.shape
        out = torch.empty((P, V, alpha.shape[1]), device=xp.device, dtype=torch.promote_types(xp.dtype, alpha.dtype))
        for a in range(0, P, chunk):
            sig = torch.sigmoid(4.0 * (xp[a:a + chunk, :, None] - C) / Wd)               # (pc, V, M)
            out[a:a + chunk] = torch.bmm(sig.to(alpha.dtype), alpha[a:a + chunk].transpose(1, 2))
        ctx.save_for_backward(alpha, C, Wd, xp)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, g):
        alpha, C, Wd, xp = ctx.saved_tensors
        P = xp.shape[0]
        g = g.to(alpha.dtype)
        g_alpha = torch.empty_like(alpha)
        gC, gW = torch.zeros_like(C), torch.zeros_like(Wd)
        for a in range(0, P, ctx.chunk):
            z = 4.0 * (xp[a:a + ctx.chunk, :, None] - C) / Wd
            sig = torch.sigmoid(z).to(alpha.dtype)                                       # (pc, V, M)
            gc = g[a:a + ctx.chunk]                                                      # (pc, V, K)
            g_alpha[a:a + ctx.chunk] = torch.bmm(gc.transpose(1, 2), sig)                # (pc, K, M)
            t = torch.bmm(gc, alpha[a:a + ctx.chunk]) * sig * (1 - sig)                  # dL/dz, (pc, V, M)
            gC -= (t * (4.0 / Wd)).sum((0, 1)).to(gC.dtype)
            gW -= (t * z.to(t.dtype) / Wd).sum((0, 1)).to(gW.dtype)
        return g_alpha, gC, gW, None, None


class PatchWindowRouter(nn.Module):
    """Per-patch window routing: M learned windows shared by every patch (centres at the body-HU quantiles,
    no presets); each of the K output channels is, per patch, a softmax mixture of the M windows chosen by a
    1x1x1 network from the patch's HU density at two scales (its 3^3-patch neighbourhood and a wider
    route_context^3 one, so it can tell lung from abdomen). A channel can thus act as a lung window in the
    chest and a soft-tissue window in the abdomen, with no bound on how far its window moves.

    Init: channel k routes mostly (softmax logit ``init_logit``, ~57 % for 3 with 16 windows) to window
    (k + 0.5) M / K, and the per-patch network starts at zero, so every patch starts with the same mixture.
    A soft init keeps gradients on the other windows (a near one-hot softmax would lock them, as the
    eps-initialised learned curves did). ``passthrough`` appends the fixed [-1000, 1000] ramp channel.

    v2 options (mix_patch_route mixed windows softly and underfit): ``hard`` routes every channel to its top
    window only (straight-through: the forward uses the one-hot argmax, gradients go to the softmax);
    ``width_min`` floors the widths (W = width_min + exp(log_width)); ``init_windows`` = {"centers": [...],
    "widths": [...]} warm-starts the first windows (channel k starts on window k), the other ones on a
    uniform grid of [-1000, 1000] with widths = 2 x their spacing.
    """

    def __init__(self, k: int = 8, n_windows: int = 16, patch: int = 6, step: float = 25.0, hidden: int = 64,
                 route_context: int = 9, init_logit: float = 3.0, hu_hist: str | None = None,
                 passthrough: bool = True, hard: bool = False, width_min: float = 0.0,
                 init_windows: dict | None = None, chunk_elems: int = 2 ** 27):
        super().__init__()
        self.k, self.M, self.s, self.route_context = k, n_windows, patch, route_context
        self.passthrough, self.out_chans = passthrough, k + int(passthrough)
        self.hard, self.width_min = hard, float(width_min)
        self.density = PatchDensity(patch, step, context=1)
        m = self.density.m
        if init_windows:
            ci, wi = np.asarray(init_windows["centers"], float), np.asarray(init_windows["widths"], float)
            if not k <= len(ci) < n_windows:
                raise ValueError(f"init_windows: need {k} to {n_windows - 1} windows, got {len(ci)}")
            sp = (HU_HI - HU_LO) / (n_windows - len(ci))
            c0 = np.concatenate([ci, HU_LO + sp * (np.arange(n_windows - len(ci)) + 0.5)])
            w0 = np.concatenate([wi, np.full(n_windows - len(ci), 2 * sp)])
            first = torch.arange(k)
        else:
            edges = np.arange(HU_LO, HU_HI + step / 2, step)
            hb = np.load(hu_hist)["hist_body"] if hu_hist else np.ones(len(edges) - 1)
            cdf = np.concatenate([[0], np.cumsum(hb / hb.sum())])
            c0 = np.interp((np.arange(n_windows) + 0.5) / n_windows, cdf, edges)
            gap = np.diff(np.concatenate([[HU_LO], c0, [HU_HI]]))
            w0 = np.clip(np.maximum(gap[:-1], gap[1:]) * 2, 50, 1000)
            first = ((torch.arange(k) + 0.5) * n_windows / k).long()
        self.center100 = nn.Parameter(torch.tensor(c0 / 100.0, dtype=torch.float32))
        self.log_width = nn.Parameter(torch.tensor(np.log(np.maximum(w0 - self.width_min, 1.0)), dtype=torch.float32))
        base = torch.zeros(k, n_windows)
        base[torch.arange(k), first] = init_logit
        self.route_logits = nn.Parameter(base)                                   # (K, M), global routing
        self.net = nn.Sequential(nn.Conv3d(2 * m, hidden, 1), nn.GELU(), nn.Conv3d(hidden, k * n_windows, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.chunk_elems = chunk_elems
        self.last_stats: dict = {}

    @property
    def center(self) -> torch.Tensor:
        return 100.0 * self.center100

    @property
    def width(self) -> torch.Tensor:
        return self.width_min + self.log_width.exp()

    def routing(self, x: torch.Tensor) -> torch.Tensor:
        """Per-patch routing weights alpha, (B, K, M, d, h, w)."""
        h1 = self.density(x)                                                      # (B, m, d, h, w)
        r = self.route_context
        wide = F.avg_pool3d(h1, r, stride=1, padding=r // 2, count_include_pad=False)
        feat = torch.cat([_density_features(_box3(h1)), _density_features(wide)], dim=1)
        B, _, d, h, w = h1.shape
        logits = self.net(feat).float().view(B, self.k, self.M, d, h, w) + self.route_logits.view(1, self.k, self.M, 1, 1, 1)
        return torch.softmax(logits, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, 1, D, H, W) HU
        with torch.autocast(device_type=x.device.type, enabled=False):
            hu = x.float().squeeze(1).clamp(HU_LO, HU_HI)
            B, D, H, W = hu.shape
            s = self.s
            d, h, w = D // s, H // s, W // s
            alpha = self.routing(x)                                               # (B, K, M, d, h, w)
            top = alpha.argmax(2)                                                 # (B, K, d, h, w)
            if self.hard:                                                         # straight-through top-1
                used = F.one_hot(top, self.M).permute(0, 1, 5, 2, 3, 4).to(alpha.dtype) + alpha - alpha.detach()
            else:
                used = alpha
            ap = used.permute(0, 3, 4, 5, 1, 2).reshape(-1, self.k, self.M)      # (P, K, M)
            xp = _patch_view(hu, s).permute(0, 1, 3, 5, 2, 4, 6).reshape(-1, s ** 3)   # (P, V)
            chunk = max(1, self.chunk_elems // (s ** 3 * self.M))
            out = _RouteFn.apply(ap.contiguous(), self.center, self.width, xp.contiguous(), chunk)
            out = out.view(B, d, h, w, s, s, s, self.k).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, self.k, D, H, W)
            if self.training:
                with torch.no_grad():
                    ceff = (used * self.center.view(1, 1, -1, 1, 1, 1)).sum(2)    # effective centre per channel
                    glob = self.route_logits.argmax(1).view(1, -1, 1, 1, 1)
                    self.last_stats = {
                        "route_center_std_hu": float(ceff.flatten(2).std(2).mean()),
                        "route_maxprob_mean": float(alpha.max(2).values.mean()),
                        "route_switched": float((top != glob).float().mean()),
                        "bank_center_min": float(self.center.min()), "bank_center_max": float(self.center.max()),
                        "bank_width_min": float(self.width.min()),
                        "bank_width_max": float(self.width.max())}
        if self.passthrough:
            out = torch.cat([out, ((hu - HU_LO) / (HU_HI - HU_LO))[:, None].to(out.dtype)], dim=1)
        return out


# ----------------------------------------------------------------------------- softmax selection of fixed windows
class WindowSelect(nn.Module):
    """The 11 fixed pillar11 channels (10 presets + min-max), mixed by a softmax before the stem instead of by the
    stem conv: out_j = sum_c alpha[j, c] ch_c, J = n_out channels (the stem then sees J channels, not 11).

    alpha = softmax over the 11 channels of global logits (J, 11), plus, when per_patch, per-patch logits from a
    1x1x1 network on the patch's HU density at two scales (the router's features: 3^3-patch neighbourhood and a
    route_context^3 one). The network's last layer starts at zero, so every patch starts with the global mixture.
    Init: n_out = 1 starts at the uniform mean of the 11 channels; n_out = 3 starts with output j leaning to the
    lung / mediastinum / bone window (logit ``init_logit``, ~42 % for 2 with 11 channels).
    A global softmax gives one monotone transfer curve (a convex sum of ramps): the 1-channel test of whether
    the stem needs several windows at once.
    """

    LEAN = ("lung", "mediastinum", "bone")                    # n_out = 3 init, indices in PILLAR_PRESETS

    def __init__(self, n_out: int = 1, per_patch: bool = False, patch: int = 6, step: float = 25.0,
                 hidden: int = 64, route_context: int = 9, init_logit: float = 2.0):
        super().__init__()
        self.fixed = FixedWindows(PILLAR_PRESETS, minmax_channel=True)
        self.n_in, self.k, self.out_chans = len(PILLAR_PRESETS) + 1, n_out, n_out
        self.s, self.per_patch, self.route_context = patch, per_patch, route_context
        base = torch.zeros(n_out, self.n_in)
        if n_out == 3:
            base[torch.arange(3), torch.tensor([0, 1, 4])] = init_logit      # lung, mediastinum, bone
        elif n_out != 1:
            raise ValueError(f"window_select: n_out must be 1 or 3, got {n_out}")
        self.select_logits = nn.Parameter(base)                              # (J, 11), global mixture
        if per_patch:
            self.density = PatchDensity(patch, step, context=1)
            m = self.density.m
            self.net = nn.Sequential(nn.Conv3d(2 * m, hidden, 1), nn.GELU(), nn.Conv3d(hidden, n_out * self.n_in, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.last_stats: dict = {}

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """Mixture weights alpha, (B, J, 11, d, h, w) per patch, or (1, J, 11, 1, 1, 1) global."""
        logits = self.select_logits.view(1, self.k, self.n_in, 1, 1, 1)
        if self.per_patch:
            h1 = self.density(x)                                             # (B, m, d, h, w)
            r = self.route_context
            wide = F.avg_pool3d(h1, r, stride=1, padding=r // 2, count_include_pad=False)
            feat = torch.cat([_density_features(_box3(h1)), _density_features(wide)], dim=1)
            B, _, d, h, w = h1.shape
            logits = self.net(feat).float().view(B, self.k, self.n_in, d, h, w) + logits
        return torch.softmax(logits, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, 1, D, H, W) HU
        with torch.autocast(device_type=x.device.type, enabled=False):
            ch = self.fixed(x.float().clamp(HU_LO, HU_HI))                   # (B, 11, D, H, W)
            alpha = self.weights(x)
            B, C, D, H, W = ch.shape
            s = self.s
            if self.per_patch:                                               # one mixture per 6^3 patch
                chp = ch.view(B, C, D // s, s, H // s, s, W // s, s)
                out = torch.einsum("bcxiyjzk,bocxyz->boxiyjzk", chp, alpha).reshape(B, self.k, D, H, W)
            else:
                out = torch.einsum("bcxyz,oc->boxyz", ch, alpha.view(self.k, C))
            if self.training:
                with torch.no_grad():
                    mean_w = alpha.mean(dim=(0, 3, 4, 5))                    # (J, 11)
                    self.last_stats = {"select_maxw_mean": float(alpha.max(2).values.mean())}
                    for j in range(self.k):                                  # dominant channel of each output
                        self.last_stats[f"select_top{j}"] = float(mean_w[j].argmax())
                        self.last_stats[f"select_top{j}_w"] = float(mean_w[j].max())
                    if self.per_patch:
                        top = alpha.argmax(2)
                        self.last_stats["select_switched"] = float((top != mean_w.argmax(1).view(1, -1, 1, 1, 1)).float().mean())
        return out

# ----------------------------------------------------------------------------- free curves
class _PatchCDFFn(torch.autograd.Function):
    """out_k(v) = CDF of p[k, patch(v)] at x(v); p: (K, P, M) bin masses over [-1000, 1000]."""

    @staticmethod
    def forward(ctx, p, x, s, step):
        K, P, M = p.shape
        cum = F.pad(p.cumsum(-1), (1, 0))
        j, frac = _bins(x, HU_LO, step, M)
        pid = _patch_ids(x.shape, s, x.device)
        out = torch.empty((K,) + x.shape, device=x.device, dtype=torch.promote_types(x.dtype, p.dtype))
        ic, ip = pid * (M + 1) + j, pid * M + j
        for k in range(K):
            out[k] = cum[k].reshape(-1).take(ic) + p[k].reshape(-1).take(ip) * frac
        ctx.save_for_backward(p, x)
        ctx.conf = (s, step)
        return out

    @staticmethod
    def backward(ctx, g):
        p, x = ctx.saved_tensors
        s, step = ctx.conf
        K, P, M = p.shape
        j, frac = _bins(x, HU_LO, step, M)
        ip = (_patch_ids(x.shape, s, x.device) * M + j).reshape(-1)
        ff = frac.reshape(-1)
        gp = torch.empty_like(p)
        for k in range(K):
            gk = g[k].to(p.dtype).reshape(-1)
            S = torch.bincount(ip, weights=gk, minlength=P * M).to(p.dtype).view(P, M)
            Fh = torch.bincount(ip, weights=gk * ff, minlength=P * M).to(p.dtype).view(P, M)
            gp[k] = S.flip(-1).cumsum(-1).flip(-1) - S + Fh
        return gp, None, None, None


class PatchCDF(nn.Module):
    def __init__(self, k: int = 8, patch: int = 6, step: float = 25.0, context: int = 3, hidden: int = 64,
                 use_net: bool = True, local_eq: bool = True, local_eq_init: float = 0.0,
                 gamma_max: float = 1.5, hu_hist: str | None = None):
        super().__init__()
        self.k, self.s, self.step, self.use_net, self.local_eq = k, patch, step, use_net, local_eq
        self.density = PatchDensity(patch, step, context)
        m = self.density.m
        hb = np.load(hu_hist)["hist_body"] if hu_hist else np.ones(m)
        hb = hb / hb.sum() + 1e-4
        gammas = np.linspace(0.0, gamma_max, k)                     # 0 = linear ramp ... sharp equalisation
        self.theta = nn.Parameter(torch.tensor(np.stack([g * np.log(hb) for g in gammas]), dtype=torch.float32))
        if local_eq:
            init = np.full(k, local_eq_init) if np.isscalar(local_eq_init) else np.asarray(local_eq_init)
            self.lam = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        if use_net:
            self.net = nn.Sequential(nn.Conv3d(m, hidden, 1), nn.GELU(), nn.Conv3d(hidden, k * m, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.shuffle_seed = None
        self.last_stats: dict = {}

    def patch_masses(self, x: torch.Tensor) -> torch.Tensor:
        """Bin masses of every patch's curves, (K, P, M)."""
        m = self.density.m
        logits = self.theta[:, None, :]                              # (K, 1, M)
        if self.use_net or self.local_eq:
            dens = self.density(x)                                   # (B, M, d, h, w)
            B = dens.shape[0]
            if self.use_net:
                r = self.net(_density_features(dens)).float()        # (B, K*M, d, h, w)
                r = r.view(B, self.k, m, -1).permute(1, 0, 3, 2).reshape(self.k, -1, m)
                logits = logits + r
            if self.local_eq:
                ld = torch.log(dens + 1e-3).permute(0, 2, 3, 4, 1).reshape(1, -1, m)
                logits = logits + self.lam.view(-1, 1, 1) * ld
        else:
            B = x.shape[0]
            P = B * math.prod(n // self.s for n in x.shape[2:])
            logits = logits.expand(self.k, P, m)
        return logits.softmax(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, 1, D, H, W) HU
        with torch.autocast(device_type=x.device.type, enabled=False):
            p = self.patch_masses(x)
            if self.training:
                with torch.no_grad():
                    cum = p.cumsum(-1)
                    med = HU_LO + self.step * (cum < 0.5).sum(-1).float()   # median HU of every patch curve
                    self.last_stats = {"patch_median_std_hu": float(med.std(1).mean()),
                                       "patch_median_range_hu": float((med.amax(1) - med.amin(1)).mean())}
                    if self.local_eq:
                        self.last_stats |= {"local_eq_min": float(self.lam.min()), "local_eq_max": float(self.lam.max())}
            out = _PatchCDFFn.apply(p.contiguous(), x.float().squeeze(1).clamp(HU_LO, HU_HI), self.s, self.step)
        return out.transpose(0, 1)


PATCH_ARMS = ("patch_bank", "patch_cdf", "patch_eq", "global_bank", "patch_bank_shuf", "patch_route",
              "window_select")


def build_patch_arm(arm: str, w: dict, patch: int) -> nn.Module:
    """``w`` = the windowing config block."""
    common = dict(k=int(w.get("k", 8)), patch=patch, step=float(w.get("edges_step", 25.0)),
                  context=int(w.get("density_context", 3)), hidden=int(w.get("hidden", 64)),
                  hu_hist=w.get("hu_hist"))
    if arm in ("patch_bank", "global_bank", "patch_bank_shuf"):
        return PatchWindowBank(per_patch=arm != "global_bank", control="shuffle" if arm.endswith("_shuf") else None,
                               shift_max_hu=float(w.get("shift_max_hu", 300.0)),
                               width_max_log=float(w.get("width_max_log", 1.5)),
                               passthrough=bool(w.get("passthrough", False)),
                               width_min=float(w.get("width_min_hu", 0.0)), smooth=int(w.get("smooth", 0)),
                               scan_context=bool(w.get("scan_context", False)),
                               init=str(w.get("window_init", "quantile")),
                               fixed_windows=bool(w.get("fixed_windows", False)), **common)
    if arm == "patch_route":
        common.pop("context")
        return PatchWindowRouter(n_windows=int(w.get("n_windows", 16)), route_context=int(w.get("route_context", 9)),
                                 init_logit=float(w.get("route_init_logit", 3.0)),
                                 passthrough=bool(w.get("passthrough", True)), hard=bool(w.get("route_hard", False)),
                                 width_min=float(w.get("width_min_hu", 0.0)),
                                 init_windows=w.get("route_init_windows"), **common)
    if arm == "window_select":
        return WindowSelect(n_out=int(w.get("select_out", 1)), per_patch=bool(w.get("select_per_patch", False)),
                            patch=patch, step=common["step"], hidden=common["hidden"],
                            route_context=int(w.get("route_context", 9)),
                            init_logit=float(w.get("select_init_logit", 2.0)))
    if arm == "patch_cdf":
        return PatchCDF(use_net=True, local_eq=bool(w.get("local_eq", True)),
                        local_eq_init=w.get("local_eq_init", 0.0), **common)
    if arm == "patch_eq":
        return PatchCDF(use_net=False, local_eq=True, local_eq_init=w.get("local_eq_init", 0.5), **common)
    raise ValueError(arm)
