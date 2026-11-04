"""Bras d'entrée du pilote fenêtrage (CLIP + Atlas 3D).

Entrée commune : x en HU, (B, 1, D, H, W), float32, déjà harmonisé à [-1000, 1000].

    bras        module                         canaux vers le stem
    clip        ClipChannels                   K copies de la rampe [-1000, 1000]
    pillar11    FixedWindows(presets, minmax)  10 presets + min-max par volume
    learned     LearnedWindow (global)         K courbes apprises
    dense       RampConv3d (remplace conv0[0]) M rampes fixes, fusionnées dans la 1re conv
    local       LocalWindow                    K courbes apprises x porte par patch (histogramme HU)
    local_shuf  LocalWindow(control="shuffle") idem, histogrammes permutés : mêmes paramètres, pas d'info locale
    windows     FixedWindows(noms, minmax)     [mcw] sous-ensemble nommé des presets (ex. win4 : lung, abdomen, liver, bone)
    cdf         HUCdf                          [mcw] K copies de la CDF des HU du corps (égalisation d'histogramme globale)

Test principal : local > dense ET local > local_shuf.
- learned ⊂ dense : courbes ponctuelles + 1re conv linéaire = combinaison des rampes de la base.
- local démarre exactement comme learned (porte = 1 à l'init) : l'écart local - learned est l'apport du local.

[mcw] Intégration merlin-clip-window : modifications locales marquées `# [mcw]`.
  - import relatif de LearnedWindow ;
  - PILLAR_PRESETS renseignés depuis rad-vision-engine (ANATOMICAL_WINDOWS["CT"], lo/hi = c -/+ w/2) ;
  - RampConv3d : conv via mcw.gemm_conv3d (MIOpen 3D inutilisable sur MI250X), rampes calculées
    en channels-last ;
  - LocalWindow : graine optionnelle pour la permutation du contrôle shuffle (le cache de gradient
    refait chaque forward : les deux passes doivent voir la même permutation) ;
  - build_arm : stride / padding de la 1re conv du stem en paramètres (3D pillar-pretrain : 3 / 0) ;
  - build_arm : bornes de la rampe du bras clip en paramètre (clip_range ; win5_abdo : [-120, 240]) ;
  - warp : décalage / échelle HU appris par canal devant les courbes (LearnedWindow(warp=True)),
    transmis par build_arm aux bras learned / local.
  - bras windows (WINDOW_PRESETS : presets nommés, canal min-max optionnel) et cdf (HUCdf, histogramme hu_hist).
"""
import math

import numpy as np  # [mcw] bras cdf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .gemm_conv3d import conv3d as gemm_conv3d  # [mcw]
from .learned_window import LearnedWindow  # [mcw] import relatif

HU_LO, HU_HI = -1000.0, 1000.0
PILLAR_MEAN, PILLAR_STD = 0.289, 0.198
# [mcw] [(lo, hi)] x 10, rad-vision-engine @ 20adeb8 vision_engine/utils/windowing_utils.py,
# ANATOMICAL_WINDOWS["CT"] dans l'ordre de get_available_windows("CT") : lo/hi = centre -/+ largeur / 2.
PILLAR_PRESETS = [
    (-1350.0, 150.0),    # lung          c=-600 w=1500
    (-150.0, 250.0),     # mediastinum   c=50   w=400
    (-160.0, 240.0),     # abdomen       c=40   w=400
    (5.0, 155.0),        # liver         c=80   w=150
    (-500.0, 1300.0),    # bone          c=400  w=1800
    (0.0, 80.0),         # brain         c=40   w=80
    (-32.5, 182.5),      # subdural      c=75   w=215
    (20.0, 60.0),        # stroke        c=40   w=40
    (-800.0, 2000.0),    # temporal_bone c=600  w=2800
    (-125.0, 225.0),     # soft_tissue   c=50   w=350
]
# [mcw] les mêmes presets par nom (bras windows)
WINDOW_PRESETS = dict(zip(["lung", "mediastinum", "abdomen", "liver", "bone", "brain", "subdural", "stroke",
                           "temporal_bone", "soft_tissue"], PILLAR_PRESETS))


def _ramp(x, lo, hi):
    return ((x - lo) / (hi - lo)).clamp(0, 1)


def default_edges(step=25.0):
    return torch.arange(HU_LO, HU_HI + step / 2, step)


class PostNorm(nn.Module):
    """Normalisation Pillar-0, identique pour les bras à canaux."""

    def __init__(self, mean=PILLAR_MEAN, std=PILLAR_STD):
        super().__init__()
        self.mean, self.std = mean, std

    def forward(self, x):
        return (x - self.mean) / self.std


class ClipChannels(nn.Module):
    """Bras clip : rampe [lo, hi] -> [0, 1], répétée K fois pour garder le même stem."""

    def __init__(self, k=11, lo=HU_LO, hi=HU_HI):
        super().__init__()
        self.k, self.lo, self.hi = k, lo, hi

    def forward(self, x):
        return _ramp(x.float(), self.lo, self.hi).expand(-1, self.k, -1, -1, -1)


class FixedWindows(nn.Module):
    """Bras pillar11 : fenêtres fixes + canal min-max par volume (comme Pillar-0)."""

    def __init__(self, windows, minmax_channel=True):
        super().__init__()
        if windows is None:
            raise ValueError("renseigner PILLAR_PRESETS (10 fenêtres du code RAVE)")
        self.register_buffer("lo", torch.tensor([w[0] for w in windows], dtype=torch.float32).view(1, -1, 1, 1, 1))
        self.register_buffer("hi", torch.tensor([w[1] for w in windows], dtype=torch.float32).view(1, -1, 1, 1, 1))
        self.minmax_channel = minmax_channel

    def forward(self, x):
        x = x.float()
        out = _ramp(x, self.lo, self.hi)
        if self.minmax_channel:
            mn = x.amin(dim=(2, 3, 4), keepdim=True)
            mx = x.amax(dim=(2, 3, 4), keepdim=True)
            out = torch.cat([out, (x - mn) / (mx - mn).clamp_min(1e-6)], 1)
        return out


class HUCdf(nn.Module):
    """[mcw] Bras cdf : HU -> CDF des voxels du corps du jeu (histogramme de scripts/hu_histogram.py, bins de
    25 HU, linéaire dans chaque bin), répétée K fois comme clip. Courbe fixe, monotone, sans écrêtage : la plage
    [0, 1] est répartie selon la fréquence des HU (beaucoup pour les tissus mous, peu pour l'os)."""

    def __init__(self, hu_hist, k=11):
        super().__init__()
        h = np.load(hu_hist)
        hist = np.asarray(h["hist_body"], dtype=np.float64)
        cdf = np.concatenate([[0.0], np.cumsum(hist)]) / hist.sum()
        self.register_buffer("edges", torch.as_tensor(h["edges"], dtype=torch.float32))
        self.register_buffer("cdf", torch.as_tensor(cdf, dtype=torch.float32))
        self.k = k

    def forward(self, x):
        e, c = self.edges, self.cdf
        x = x.float().clamp(e[0], e[-1])
        j = torch.bucketize(x, e[1:-1])
        y = c[j] + (c[j + 1] - c[j]) * (x - e[j]) / (e[j + 1] - e[j])
        return y.expand(-1, self.k, -1, -1, -1)


class RampConv3d(nn.Module):
    """Bras dense : M rampes fixes (une par bin de `edges`) comme canaux d'entrée de la 1re conv du stem.

    Équivaut à F.conv3d(cat_m r_m(x), W) sans matérialiser les M canaux : la conv est sommée
    par blocs de rampes, recalculés au backward (checkpoint). À mettre à la place de
    `patch_embed.conv0[0]`, avec l'entrée en HU brute (1 canal) et sans PostNorm : la BatchNorm
    qui suit absorbe toute normalisation affine des rampes.
    """

    def __init__(self, edges, out_channels, kernel_size=3, stride=2, padding=1, bias=False, block=8):
        super().__init__()
        self.register_buffer("edges", edges.float())
        m = len(edges) - 1
        self.weight = nn.Parameter(torch.empty(out_channels, m, *(kernel_size,) * 3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # init par défaut de nn.Conv3d
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride, self.padding, self.block = stride, padding, block

    def _block(self, x, w, lo, d):
        # [mcw] rampes calculées directement en channels-last (B, D, H, W, m) puis vues en NCDHW
        # (channels_last_3d) : la conv GEMM les consomme sans transposition. Même valeur que
        # F.conv3d(((x - lo) / d).clamp(0, 1), w, None, self.stride, self.padding).
        r = ((x.permute(0, 2, 3, 4, 1) - lo.view(-1)) / d.view(-1)).clamp(0, 1).permute(0, 4, 1, 2, 3)
        return gemm_conv3d(r, w, None, self.stride, self.padding)

    def forward(self, x):
        x, e = x.float(), self.edges
        out = None
        for s in range(0, len(e) - 1, self.block):
            lo = e[s:s + self.block].view(1, -1, 1, 1, 1)
            d = (e[s + 1:s + self.block + 1] - e[s:s + self.block]).view(1, -1, 1, 1, 1)
            w = self.weight[:, s:s + self.block]
            if torch.is_grad_enabled():
                y = checkpoint(self._block, x, w, lo, d, use_reentrant=False)
            else:
                y = self._block(x, w, lo, d)
            out = y if out is None else out + y
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
        return out


class LocalWindow(nn.Module):
    """Bras local : sortie_k(v) = g_k(patch(v)) * courbe_k(x_v).

    g = 2 * sigmoid(z), z prédit par patch depuis l'histogramme HU brut du patch ;
    dernière couche à zéro, donc g = 1 et le module vaut exactement LearnedWindow à l'init.
    La grille des patchs doit être celle du stem d'Atlas (pas 16).
    control="shuffle" : histogrammes permutés entre tous les patchs du batch (permutation fixe en eval).
    """

    def __init__(self, edges, n_channels=11, init_windows=None, patch=16, hist_bins=32,
                 hidden=64, context_kernel=1, control=None, warp=False):  # [mcw] warp
        super().__init__()
        self.curves = LearnedWindow(edges, n_channels, init_windows, warp=warp)  # [mcw] warp
        self.patch, self.hist_bins, self.control = patch, hist_bins, control
        self.register_buffer("hist_inner", torch.linspace(HU_LO, HU_HI, hist_bins + 1)[1:-1])
        self.gate = nn.Sequential(
            nn.Conv3d(hist_bins, hidden, context_kernel, padding=context_kernel // 2),
            nn.GELU(),
            nn.Conv3d(hidden, n_channels, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.last_gate = None
        self.shuffle_seed = None  # [mcw] fixé par l'entraîneur pour chaque micro-batch (train)

    @torch.no_grad()
    def patch_histograms(self, x):
        """(B, 1, D, H, W) HU -> (B, hist_bins, D/p, H/p, W/p), fréquences par patch."""
        B, _, D, H, W = x.shape
        p = self.patch
        if D % p or H % p or W % p:
            raise ValueError(f"taille {tuple(x.shape[2:])} non divisible par le patch {p}")
        d, h, w = D // p, H // p, W // p
        b = torch.bucketize(x.float().clamp(HU_LO, HU_HI), self.hist_inner)
        b = b.view(B, d, p, h, p, w, p).permute(0, 1, 3, 5, 2, 4, 6).reshape(B * d * h * w, p ** 3)
        hist = torch.zeros(b.shape[0], self.hist_bins, device=x.device)
        hist.scatter_add_(1, b, torch.ones(b.shape, device=x.device))
        return (hist / p ** 3).view(B, d, h, w, self.hist_bins).permute(0, 4, 1, 2, 3).contiguous()

    def _shuffle(self, hist):
        B, nb, d, h, w = hist.shape
        flat = hist.permute(0, 2, 3, 4, 1).reshape(-1, nb)
        gen = None if self.training else torch.Generator(device=hist.device).manual_seed(0)
        if self.training and self.shuffle_seed is not None:  # [mcw] même permutation dans les deux passes
            gen = torch.Generator(device=hist.device).manual_seed(int(self.shuffle_seed))
        idx = torch.randperm(flat.shape[0], generator=gen, device=hist.device)
        return flat[idx].view(B, d, h, w, nb).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x):
        ch = self.curves(x)                                   # (B, K, D, H, W), fp32
        hist = self.patch_histograms(x)
        if self.control == "shuffle":
            hist = self._shuffle(hist)
        g = 2 * torch.sigmoid(self.gate(hist))                # (B, K, d, h, w)
        self.last_gate = g.detach()
        p = self.patch
        g = g.float().repeat_interleave(p, 2).repeat_interleave(p, 3).repeat_interleave(p, 4)
        return ch * g


class WindowedEncoder(nn.Module):
    """Bras d'entrée + PostNorm + backbone. Pour `dense`, adapter = Identity et post_norm=False."""

    def __init__(self, adapter, backbone, post_norm=True):
        super().__init__()
        self.adapter, self.backbone = adapter, backbone
        self.norm = PostNorm() if post_norm else nn.Identity()

    def forward(self, x_hu):
        return self.backbone(self.norm(self.adapter(x_hu)))


def build_arm(arm, presets=PILLAR_PRESETS, edges=None, k=11, patch=16, stem_out=64,
              stem_stride=2, stem_padding=1,  # [mcw] géométrie de la 1re conv du stem
              clip_range=(HU_LO, HU_HI),      # [mcw] bornes de la rampe du bras clip
              warp=False,                     # [mcw] décalage / échelle appris des courbes
              windows=None, minmax=True,      # [mcw] bras windows : noms de WINDOW_PRESETS, canal min-max
              hu_hist=None):                  # [mcw] bras cdf : histogramme des HU du corps (.npz)
    """Renvoie dict(adapter, in_chans, first_conv, post_norm).

    first_conv n'est pas None pour `dense` : le substituer à la 1re conv du stem
    (2D local : model.atlas_models[0].patch_embed.conv0[0] ; vérifier le nom dans le 3D de pillar-pretrain).
    [mcw] 3D pillar-pretrain : atlas.patch_embeds["abdomen_ct"].conv_down[0], Conv3d(in, 64, 3, stride=3, padding=0).
    """
    edges = default_edges() if edges is None else edges
    init = None if presets is None else list(presets)[:k - 1] + [(HU_LO, HU_HI)]
    if arm == "clip":
        return dict(adapter=ClipChannels(k, *clip_range), in_chans=k, first_conv=None, post_norm=True)  # [mcw]
    if arm == "pillar11":
        return dict(adapter=FixedWindows(presets, minmax_channel=True), in_chans=k, first_conv=None, post_norm=True)
    if arm == "windows":  # [mcw]
        wins = [WINDOW_PRESETS[n] for n in windows]
        return dict(adapter=FixedWindows(wins, minmax_channel=minmax), in_chans=len(wins) + int(minmax),
                    first_conv=None, post_norm=True)
    if arm == "cdf":  # [mcw]
        return dict(adapter=HUCdf(hu_hist, k), in_chans=k, first_conv=None, post_norm=True)
    if arm == "learned":
        return dict(adapter=LearnedWindow(edges, k, init, warp=warp), in_chans=k, first_conv=None,  # [mcw] warp
                    post_norm=True)
    if arm == "dense":
        return dict(adapter=nn.Identity(), in_chans=1,
                    first_conv=RampConv3d(edges, stem_out, stride=stem_stride, padding=stem_padding),  # [mcw]
                    post_norm=False)
    if arm in ("local", "local_shuf"):
        control = "shuffle" if arm == "local_shuf" else None
        return dict(adapter=LocalWindow(edges, k, init, patch=patch, control=control, warp=warp),  # [mcw] warp
                    in_chans=k, first_conv=None, post_norm=True)
    raise ValueError(arm)


def param_groups(model, lr, weight_decay, window_lr=None):
    """Courbes et porte du bras d'entrée : sans weight decay (il tirerait vers le clip / vers g = 1)."""
    window, decay, no_decay = [], [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        if name.startswith("adapter.") and not isinstance(model.adapter, nn.Identity):
            window.append(prm)
        elif prm.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(prm)
        else:
            decay.append(prm)
    groups = [dict(params=decay, lr=lr, weight_decay=weight_decay),
              dict(params=no_decay, lr=lr, weight_decay=0.0)]
    if window:
        groups.append(dict(params=window, lr=window_lr or lr, weight_decay=0.0))
    return groups
