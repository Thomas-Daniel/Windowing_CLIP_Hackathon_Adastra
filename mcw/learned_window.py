"""Fenêtrage appris : K courbes de transfert HU -> [0, 1] monotones, linéaires par morceaux.

Même modèle que la version d'origine (out_k = CDF apprise sur une grille de bords HU),
mais avec un backward explicite : out est linéaire en p, donc
dL/dp[k, m] = sum_v g_k(v) * r_m(v), avec r_m(x) = clamp((x - e_m) / (e_{m+1} - e_m), 0, 1).
Testé contre la version d'origine : forward identique, gradient à 1e-6 près,
backward ~10x plus rapide (pas d'index_put accumulé sur des millions de voxels).

[mcw] Sur MI250X (rocBLAS), la réduction par blocs gf @ r.T (formes 11 x 2^20 @ 2^20 x 80, fp32)
tombe sur un noyau très lent : 2,5 s par volume 384^3 (scripts/bench_learned_window.py, job 5573052).
Le backward utilise donc la même somme écrite par bin : r_m(v) vaut 1 sous le bin j(v), frac(v)
dans j(v), 0 au-dessus, d'où
    dL/dp[k, m] = sum_{v : j(v) > m} g_k(v) + sum_{v : j(v) = m} g_k(v) * frac(v),
soit deux histogrammes pondérés par canal (torch.bincount) : 33 ms par volume. La version par
matmul est gardée (_grad_p_matmul) et sert de référence dans tests/test_cpu.py.

[mcw] warp=True : décalage et échelle HU appris par canal devant la courbe,
    out_k(x) = CDF_k(c_k + exp(-u_k) * (x - c_k - t_k)),  t_k = 100 * tau_k (HU),
c_k = centre de la fenêtre initiale (HU où la courbe initiale vaut 0.5). La fenêtre glisse de t_k
et s'élargit d'un facteur exp(u_k) autour de c_k ; tau = u = 0 à l'init (courbes identiques).
Raison : avec l'init eps = 1e-3, les bins hors fenêtre ont une masse ~1e-5 (logit -11) ; déplacer
un bord d'un bin demande ~+8 en logit, et le gradient softmax de ces bins est ~1000x plus faible
(win3b_learned_lr100, pas 150 : bornes 10-90 % inchangées à quelques HU près). t et u reçoivent le
gradient de tous les voxels situés dans la pente de la courbe. Backward explicite (_WarpedCDFFn) :
histogrammes par canal pour p (comme ci-dessus, mais avec les indices de bin propres au canal),
réductions de g * densité pour t et u ; pas d'index_put sur les paramètres.
"""
import math

import numpy as np  # [mcw]
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LearnedWindowFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, x, e, chunk):
        cum = F.pad(p.cumsum(-1), (1, 0))                    # (K, M+1), cum[:, 0] = 0, cum[:, -1] = 1
        xc = x.clamp(e[0], e[-1])
        j = torch.bucketize(xc, e[1:-1])                     # bin par voxel
        frac = (xc - e[j]) / (e[j + 1] - e[j])
        out = cum[:, j] + p[:, j] * frac                     # (K, *x.shape)
        ctx.save_for_backward(x, e)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, g):
        x, e = ctx.saved_tensors
        return _grad_p_hist(g, x, e), None, None, None      # [mcw] histogrammes au lieu de _grad_p_matmul


def _grad_p_matmul(g, x, e, chunk):
    """Backward d'origine : dL/dp = sum_v g(v) r(v)^T, par blocs de `chunk` voxels."""
    K = g.shape[0]
    xf, gf = x.reshape(-1), g.reshape(K, -1)
    lo, d = e[:-1, None], (e[1:] - e[:-1])[:, None]
    gp = torch.zeros(K, len(e) - 1, device=g.device, dtype=torch.float32)
    for s in range(0, xf.numel(), chunk):
        r = ((xf[None, s:s + chunk] - lo) / d).clamp_(0, 1)   # (M, n)
        gp += gf[:, s:s + chunk].float() @ r.T
    return gp


def _grad_p_hist(g, x, e):
    """[mcw] Même gradient par histogrammes : S[k, m] = sum g_k sur le bin m, F[k, m] = sum g_k * frac."""
    K, M = g.shape[0], len(e) - 1
    xc = x.reshape(-1).clamp(e[0], e[-1])
    j = torch.bucketize(xc, e[1:-1])
    frac = (xc - e[j]) / (e[j + 1] - e[j])
    if M <= 256:
        j = j.to(torch.uint8)                                # bincount ~1.5x plus rapide
    gf = g.reshape(K, -1).float()
    S = torch.stack([torch.bincount(j, weights=gf[k], minlength=M) for k in range(K)]).float()
    F_ = torch.stack([torch.bincount(j, weights=gf[k] * frac, minlength=M) for k in range(K)]).float()
    return S.flip(-1).cumsum(-1).flip(-1) - S + F_          # masse au-dessus du bin + part dans le bin


def _bins(xk, e, uniform):
    """Bin index (int64) and in-bin fraction of HU values already clamped to [e0, eM]."""
    if uniform:                                              # [mcw] grille régulière : pas de bucketize
        step = e[1] - e[0]
        f = (xk - e[0]) / step
        j = f.floor().clamp_(0, len(e) - 2)
        return j.long(), f - j
    j = torch.bucketize(xk, e[1:-1])
    return j, (xk - e[j]) / (e[j + 1] - e[j])


class _WarpedCDFFn(torch.autograd.Function):
    """[mcw] out_k = CDF_k(x'_k), x'_k = c_k + exp(-u_k) (x - c_k - 100 tau_k), canal par canal."""

    @staticmethod
    def forward(ctx, p, tau, u, x, e, c, uniform):
        K = p.shape[0]
        cum = F.pad(p.cumsum(-1), (1, 0))
        out = torch.empty((K,) + x.shape, device=x.device, dtype=torch.float32)
        for k in range(K):
            xk = (c[k] + torch.exp(-u[k]) * (x - c[k] - 100.0 * tau[k])).clamp(e[0], e[-1])
            j, frac = _bins(xk, e, uniform)
            out[k] = cum[k, j] + p[k, j] * frac
        ctx.save_for_backward(p, tau, u, x, e, c)
        ctx.uniform = uniform
        return out

    @staticmethod
    def backward(ctx, g):
        p, tau, u, x, e, c = ctx.saved_tensors
        K, M = p.shape
        width = e[1:] - e[:-1]
        gp = torch.zeros_like(p)
        gtau = torch.zeros_like(tau)
        gu = torch.zeros_like(u)
        for k in range(K):
            xr = c[k] + torch.exp(-u[k]) * (x - c[k] - 100.0 * tau[k])          # unclamped
            inside = (xr > e[0]) & (xr < e[-1])
            j, frac = _bins(xr.clamp(e[0], e[-1]), e, ctx.uniform)
            gk = g[k].float()
            jf = j.reshape(-1).to(torch.uint8) if M <= 256 else j.reshape(-1)
            S = torch.bincount(jf, weights=gk.reshape(-1), minlength=M).float()
            Fh = torch.bincount(jf, weights=(gk * frac).reshape(-1), minlength=M).float()
            gp[k] = S.flip(-1).cumsum(-1).flip(-1) - S + Fh
            gd = gk * (p[k] / width)[j] * inside                                 # g * dout/dx'
            gtau[k] = -100.0 * torch.exp(-u[k]) * gd.sum()                       # dx'/dtau = -100 e^-u
            gu[k] = -(gd * (xr - c[k])).sum()                                    # dx'/du = -(x' - c)
        return gp, gtau, gu, None, None, None, None


class LearnedWindow(nn.Module):
    def __init__(self, edges, n_channels=11, init_windows=None, eps=1e-3, chunk=2**20,
                 warp=False):  # [mcw] décalage / échelle appris par canal
        super().__init__()
        self.register_buffer("edges", edges.float())         # (M+1,) HU, croissant
        self.chunk = chunk
        theta = torch.zeros(n_channels, len(edges) - 1)
        if init_windows is not None:                         # [(lo, hi), ...] en HU, un par canal
            assert len(init_windows) == n_channels
            theta = torch.stack([self._window_logits(lo, hi, eps) for lo, hi in init_windows])
        self.theta = nn.Parameter(theta)
        self.warp = warp
        if warp:                                             # [mcw]
            e = self.edges
            cum = F.pad(theta.softmax(-1).cumsum(-1), (1, 0))
            # centre de chaque fenêtre initiale : HU où la courbe initiale vaut 0.5
            center = torch.tensor([float(np.interp(0.5, cum[k].numpy(), e.numpy())) for k in range(n_channels)])
            self.register_buffer("center", center)
            self.tau = nn.Parameter(torch.zeros(n_channels))      # décalage / 100 HU
            self.log_scale = nn.Parameter(torch.zeros(n_channels))
            d = e[1:] - e[:-1]
            self.uniform = bool(torch.allclose(d, d[0].expand_as(d)))

    def _window_logits(self, lo, hi, eps):
        # masse de chaque bin = recouvrement avec [lo, hi], plus eps réparti hors fenêtre
        e = self.edges
        overlap = (torch.minimum(e[1:], torch.tensor(hi)) - torch.maximum(e[:-1], torch.tensor(lo))).clamp(min=0)
        if overlap.sum() == 0:
            raise ValueError(f"fenêtre [{lo}, {hi}] hors de la grille [{e[0]:.0f}, {e[-1]:.0f}]")
        p = (1 - eps) * overlap / overlap.sum() + eps / len(overlap)
        return p.log()

    def curves(self, n=2001):
        """Courbes de transfert actuelles (K, n) sur [e0, eM], pour les figures."""
        with torch.no_grad():
            xs = torch.linspace(self.edges[0].item(), self.edges[-1].item(), n, device=self.edges.device)
            return xs, self(xs.view(1, 1, 1, 1, -1))[0, :, 0, 0]

    def window_shift_scale(self):
        """[mcw] (décalage en HU, facteur d'échelle) par canal ; (0, 1) sans warp."""
        if not self.warp:
            return None
        return 100.0 * self.tau.detach(), self.log_scale.detach().exp()

    def forward(self, x):                                    # x: (B, 1, D, H, W) en HU, float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            p = self.theta.float().softmax(-1)
            if self.warp:                                    # [mcw]
                out = _WarpedCDFFn.apply(p, self.tau.float(), self.log_scale.float(), x.float().squeeze(1),
                                         self.edges, self.center, self.uniform)
            else:
                out = _LearnedWindowFn.apply(p, x.float().squeeze(1), self.edges, self.chunk)
        return out.transpose(0, 1)                           # (B, K, D, H, W) dans [0, 1]
