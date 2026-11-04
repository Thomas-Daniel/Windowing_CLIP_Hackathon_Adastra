"""Show the per-patch windows of a patch arm on one axial slice of a volume.

Row 1: the HU slice (soft-tissue display window) and the patch density median (HU).
Then for each channel: the per-patch window map (bank: centre deviation from the channel's global
centre, HU; curves: per-patch median of the curve, HU) and the windowed output slice.

    python scripts/viz_patch_windows.py --ckpt runs/win8_patch_bank/ckpt_final.pt [--study AC4213645] [--z 192]
    python scripts/viz_patch_windows.py --config configs/win8_patch_bank.yaml     # at init
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcw.config import load_config  # noqa: E402
from mcw.data import TextEmbeddings  # noqa: E402
from mcw.model import CTClip  # noqa: E402
from mcw.patch_windows import PatchCDF, PatchWindowBank, PatchWindowRouter  # noqa: E402

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None, help="instead of --ckpt: the arm at init")
    ap.add_argument("--study", default=None, help="study id (default: first val volume)")
    ap.add_argument("--z", type=int, default=None, help="axial slice (default: middle)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
    else:
        cfg = load_config(args.config)
    model = CTClip(cfg, TextEmbeddings(cfg["data"]["text_emb_dir"], cfg["data"]["text_pooling"]).dim)
    if args.ckpt:
        model.load_state_dict(ck["model"])
    ad = model.adapter.to(args.device).eval()
    if not isinstance(ad, (PatchWindowBank, PatchCDF, PatchWindowRouter)):
        raise SystemExit(f"{cfg['windowing']['arm']} is not a patch arm")

    man = pd.read_csv(cfg["data"]["manifest"])
    row = man[man["study_id"] == args.study].iloc[0] if args.study else man[man["split"] == "val"].iloc[0]
    hu = torch.from_numpy(np.load(row["path"]).astype(np.float32)).clamp(-1000, 1000)[None, None].to(args.device)
    z = args.z if args.z is not None else hu.shape[2] // 2
    s = ad.s
    with torch.no_grad():
        out = ad(hu)[0, :, z].cpu().numpy()                       # (K, H, W)
        dens = ad.density(hu)[0, :, z // s].cpu().numpy()          # (M, h, w)
        cdf_d = np.cumsum(dens, 0)
        dens_med = -1000 + 25 * (cdf_d < 0.5).sum(0)               # patch density median (HU)
        if isinstance(ad, PatchWindowRouter):                    # effective centre sum_m alpha_km c_m, per patch
            al = ad.routing(hu)[0, :, :, z // s]                    # (K, M, h, w)
            if ad.hard:                                           # top-1 routing: the window actually used
                al = torch.nn.functional.one_hot(al.argmax(1), ad.M).permute(0, 3, 1, 2).float()
                glob = ad.center[ad.route_logits.argmax(1)]
            else:
                glob = torch.softmax(ad.route_logits, 1) @ ad.center
            ceff = (al * ad.center.view(1, -1, 1, 1)).sum(1)
            maps = (ceff - glob.view(-1, 1, 1)).cpu().numpy()
            titles = [f"ch {k}: effective centre {glob[k]:.0f} HU (global), top window share "
                      f"{float(al[k].max(0).values.mean()):.2f}" for k in range(ad.k)]
            map_label = "centre deviation (HU)"
        elif isinstance(ad, PatchWindowBank):
            C, Wd = ad.window_maps(hu)
            maps = (C[:, 0, z // s] - ad.center.view(-1, 1, 1)).cpu().numpy()
            titles = [f"ch {k}: centre {ad.center[k]:.0f} HU, width {Wd[k].median():.0f}" for k in range(ad.k)]
            map_label = "centre deviation (HU)"
        else:
            p = ad.patch_masses(hu)                                # (K, P, M)
            d, h, w = (n // s for n in hu.shape[2:])
            med = (-1000 + 25 * (p.cumsum(-1) < 0.5).sum(-1).float()).view(ad.k, d, h, w)[:, z // s]
            maps = med.cpu().numpy()
            titles = [f"ch {k}: curve median per patch" for k in range(ad.k)]
            map_label = "median HU"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = out.shape[0]
    fig, axes = plt.subplots(k + 1, 2, figsize=(7, 3.2 * (k + 1)), facecolor=SURFACE, constrained_layout=True)
    sl = hu[0, 0, z].cpu().numpy()
    axes[0, 0].imshow(np.clip((sl + 160) / 400, 0, 1), cmap="gray")
    axes[0, 0].set_title(f"{row['study_id']} z={z} (display W400/L40)", fontsize=9, color=INK)
    im = axes[0, 1].imshow(dens_med, cmap="gray", vmin=-1000, vmax=300)
    axes[0, 1].set_title("patch HU density: median", fontsize=9, color=INK)
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    diverging = isinstance(ad, (PatchWindowBank, PatchWindowRouter))
    lim = max(1.0, float(np.abs(maps).max())) if diverging else None
    for i in range(k):
        if i >= len(maps):                                         # passthrough ramp channel: no per-patch map
            axes[i + 1, 0].axis("off")
            axes[i + 1, 1].imshow(out[i], cmap="gray", vmin=0, vmax=1)
            axes[i + 1, 1].set_title(f"ch {i}: fixed [-1000, 1000] ramp", fontsize=9, color=INK)
            continue
        if diverging:                                              # diverging: deviation around the global centre
            im = axes[i + 1, 0].imshow(maps[i], cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            im = axes[i + 1, 0].imshow(maps[i], cmap="viridis")
        fig.colorbar(im, ax=axes[i + 1, 0], fraction=0.046, label=map_label)
        axes[i + 1, 0].set_title(titles[i], fontsize=9, color=INK)
        axes[i + 1, 1].imshow(out[i], cmap="gray", vmin=0, vmax=1)
        axes[i + 1, 1].set_title(f"ch {i}: windowed output", fontsize=9, color=INK)
    for a in axes.flat:
        a.set_xticks([])
        a.set_yticks([])
    outp = Path(args.out) if args.out else (Path(args.ckpt).parent if args.ckpt else Path(".")) / f"patch_windows_{row['study_id']}_z{z}.png"
    fig.savefig(outp, dpi=90)
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
