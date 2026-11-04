"""Global HU histogram of training volumes, the data-driven (preset-free) init of the patch arms.

Histogram over [-1000, 1000] HU in 25-HU bins of a random subset of training volumes, also
restricted to body voxels (HU > -950, i.e. without the air around the patient).

    python scripts/hu_histogram.py --n 100 --out $SCRATCHDIR/merlin_clip_data/hu_hist_80.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.expandvars("$SCRATCHDIR/merlin_clip_data/manifest_384.csv"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=os.path.expandvars("$SCRATCHDIR/merlin_clip_data/hu_hist_80.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    man = pd.read_csv(args.manifest)
    paths = man[man["split"] == "train"].sample(n=args.n, random_state=args.seed)["path"].tolist()
    edges = np.arange(-1000, 1001, 25, dtype=np.float64)
    h_all = np.zeros(len(edges) - 1)
    h_body = np.zeros(len(edges) - 1)
    for i, p in enumerate(paths):
        v = np.load(p, mmap_mode="r")[::2, ::2, ::2].astype(np.float32).clip(-1000, 1000)   # 1 voxel in 8
        h_all += np.histogram(v, bins=edges)[0]
        h_body += np.histogram(v[v > -950], bins=edges)[0]
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(paths)}", flush=True)
    np.savez(args.out, edges=edges, hist_all=h_all / h_all.sum(), hist_body=h_body / h_body.sum(), n_volumes=len(paths))
    q = np.interp([0.1, 0.25, 0.5, 0.75, 0.9], np.concatenate([[0], np.cumsum(h_body / h_body.sum())]), edges)
    print(f"-> {args.out}; body-voxel HU quantiles 10/25/50/75/90%: {np.round(q).tolist()}")


if __name__ == "__main__":
    main()
