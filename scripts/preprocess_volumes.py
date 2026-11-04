"""Offline NIfTI -> fixed-geometry int16 .npy export, shared by all three windowing runs.

Per volume: nibabel ``as_closest_canonical`` (RAS) -> transpose to (Z, Y, X) -> trilinear
resample to ``--spacing`` mm isotropic -> centre pad/crop to ``--size``^3 (pad = -1024 HU,
air) -> clip HU to [``--hu-min``, ``--hu-max``] -> int16.

Unlike the [-1000, 1000]-clipped exports used by concerto-ct / Merlin_CT_JEPA
($SCRATCHDIR/merlin_npy_384), the default keeps the full CT range [-1024, 3071]: Pillar-0's
bone (up to 1300 HU) and temporal-bone (up to 2000 HU) windows, its per-volume minmax and
the learned window all need the HU above 1000. Every run reads the same files, and the
windowing is applied on the GPU inside the model.

Resumable: existing outputs are skipped, each file is written to a temp name then renamed.
Unreadable inputs (truncated gzip in the Merlin release) are reported and left out of
the output manifest.

Usage (CPU only; see scripts/submit_preprocess.slurm):
    python scripts/preprocess_volumes.py --manifest $D/manifest_nifti.csv \
        --out-dir $D/npy_384_1p5mm --out-manifest $D/manifest_384.csv --workers 64
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def center_pad_crop(vol, size: int, pad_value: float):
    import torch

    out = torch.full((size, size, size), pad_value, dtype=vol.dtype)
    src, dst = [], []
    for n in vol.shape:
        if n <= size:
            o = (size - n) // 2
            src.append(slice(0, n)); dst.append(slice(o, o + n))
        else:
            o = (n - size) // 2
            src.append(slice(o, o + size)); dst.append(slice(0, size))
    out[tuple(dst)] = vol[tuple(src)]
    return out


def process_one(job: tuple) -> dict:
    sid, nifti_path, out_path, size, spacing, hu_min, hu_max = job
    out_path = Path(out_path)
    rec = {"study_id": sid, "path": str(out_path)}
    if out_path.exists():
        return {**rec, "status": "skipped"}

    import nibabel as nib
    import torch
    import torch.nn.functional as F

    try:
        img = nib.as_closest_canonical(nib.load(nifti_path))
        data = np.asarray(img.get_fdata(dtype=np.float32))
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
    except Exception as exc:  # noqa: BLE001 -- truncated gzip etc.: report and skip
        return {**rec, "status": f"unreadable: {str(exc)[:160]}"}
    if data.ndim != 3:
        return {**rec, "status": f"bad ndim {data.ndim}"}

    vol = torch.from_numpy(np.ascontiguousarray(data.transpose(2, 1, 0)))     # (X,Y,Z) RAS -> (Z,Y,X)
    zooms_zyx = zooms[::-1]
    out_shape = [max(1, int(round(n * z / spacing))) for n, z in zip(vol.shape, zooms_zyx)]
    vol = F.interpolate(vol[None, None], size=out_shape, mode="trilinear", align_corners=False)[0, 0]
    vol = center_pad_crop(vol, size, pad_value=-1024.0).clamp_(hu_min, hu_max)
    arr = vol.round_().numpy().astype(np.int16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.stem}.tmp{os.getpid()}.npy")
    np.save(tmp, arr)
    tmp.replace(out_path)
    return {**rec, "status": "converted", "native_shape_zyx": "x".join(map(str, data.shape[::-1])),
            "native_spacing_zyx": "x".join(f"{z:.4f}" for z in zooms_zyx)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest_nifti.csv from build_manifest.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--hu-min", type=float, default=-1024.0)
    ap.add_argument("--hu-max", type=float, default=3071.0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--max-volumes", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["split"].isin(args.splits)]
    if args.max_volumes:
        df = df.head(args.max_volumes)
    out_dir = Path(args.out_dir)
    jobs = [(r.study_id, r.nifti_path, str(out_dir / f"{r.study_id}.npy"), args.size, args.spacing,
             args.hu_min, args.hu_max) for r in df.itertuples()]
    print(f"[preprocess] {len(jobs)} volumes -> {out_dir} ({args.size}^3 @ {args.spacing} mm, "
          f"HU [{args.hu_min:g}, {args.hu_max:g}], int16, {args.workers} workers)", flush=True)

    t0, results = time.time(), []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_one, jobs, chunksize=4), 1):
            results.append(r)
            if not r["status"] in ("converted", "skipped"):
                print(f"[preprocess] {r['study_id']}: {r['status']}", flush=True)
            if i % 500 == 0:
                print(f"[preprocess] {i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)

    res = pd.DataFrame(results)
    ok = res[res["status"].isin(["converted", "skipped"])][["study_id", "path"]]
    out = df.drop(columns=["nifti_path"]).merge(ok, on="study_id").sort_values("study_id")
    out.to_csv(args.out_manifest, index=False)
    print(f"[preprocess] {int((res.status == 'converted').sum())} converted, "
          f"{int((res.status == 'skipped').sum())} already present, {len(res) - len(ok)} failed; "
          f"{len(out)} rows -> {args.out_manifest} ({time.time() - t0:.0f}s)", flush=True)
    print(out["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
