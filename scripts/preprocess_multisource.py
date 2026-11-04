"""NIfTI -> fixed-geometry int16 .npy for the chest/abdomen mix (CancerVerse, CT-RATE, INSPECT; Merlin is reused).

Same geometry as scripts/preprocess_volumes.py (RAS -> (Z, Y, X), trilinear to --spacing mm, centre pad/crop to
--size^3 with -1024 HU, clip to [--hu-min, --hu-max], int16) so every source is on the Merlin grid. Per-source fixes:
  ctrate       the v1 NIfTI headers carry neither spacing (zooms = 1) nor the DICOM rescale: HU = raw * slope +
               intercept and the spacing come from train_metadata.csv (columns slope, intercept, xy0, xy1, z);
               the axial plane is also rotated by 180 deg (x and y flipped) to match the other sources
  cancerverse  HU already clipped to [-1000, 1000] upstream (float, real spacing), nothing to correct
  inspect      LAS, real HU up to 3071 (a few sources start at -3024), real spacing
  merlin       rows with an npy_path are not reprocessed; their existing path goes to the output manifest
Resumable (existing outputs skipped, temp name then rename); unreadable inputs are reported and left out.

    python scripts/preprocess_multisource.py --manifest $D/manifest_nifti.csv --out-dir $D/npy_384_1p5mm \
        --out-manifest $D/manifest_384.csv --workers 64
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_volumes import center_pad_crop  # noqa: E402


def load_ras(row: dict):
    """-> (X, Y, Z) float32 array in RAS order and its (x, y, z) spacing in mm."""
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(row["nifti_path"]))
    if row["source"] == "ctrate":
        data = np.asarray(img.dataobj).astype(np.float32) * np.float32(row["slope"]) + np.float32(row["intercept"])
        # the affine of the v1 files is the identity, so closest_canonical is a no-op and the axes stay (x, y, z);
        # the stored axial slices are rotated by 180 deg w.r.t. RAS (table on top, heart on the image right, checked on
        # the volumes against Merlin / CancerVerse / INSPECT): flip x and y, z (head up) is already right
        data = data[::-1, ::-1]
        zooms = [row["xy1"], row["xy0"], row["z"]]
    else:
        data = np.asarray(img.get_fdata(dtype=np.float32))
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
    return data, zooms


def process_one(job: dict) -> dict:
    import torch
    import torch.nn.functional as F

    sid, out_path = job["study_id"], Path(job["out_path"])
    rec = {"study_id": sid, "path": str(out_path)}
    if out_path.exists():
        return {**rec, "status": "skipped"}
    try:
        data, zooms = load_ras(job)
    except Exception as exc:  # noqa: BLE001 -- truncated gzip etc.: report and skip
        return {**rec, "status": f"unreadable: {str(exc)[:160]}"}
    if data.ndim != 3 or min(data.shape) < 2 or not all(np.isfinite(zooms)) or min(zooms) <= 0:
        return {**rec, "status": f"bad geometry shape={data.shape} zooms={zooms}"}

    vol = torch.from_numpy(np.ascontiguousarray(data.transpose(2, 1, 0)))     # (X,Y,Z) RAS -> (Z,Y,X)
    out_shape = [max(1, int(round(n * z / job["spacing"]))) for n, z in zip(vol.shape, zooms[::-1])]
    vol = F.interpolate(vol[None, None], size=out_shape, mode="trilinear", align_corners=False)[0, 0]
    vol = center_pad_crop(vol, job["size"], pad_value=-1024.0).clamp_(job["hu_min"], job["hu_max"])
    arr = vol.round_().numpy().astype(np.int16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.stem}.tmp{os.getpid()}.npy")
    np.save(tmp, arr)
    tmp.replace(out_path)
    return {**rec, "status": "converted", "native_shape_zyx": "x".join(map(str, data.shape[::-1])),
            "native_spacing_zyx": "x".join(f"{z:.4f}" for z in zooms[::-1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest_nifti.csv from build_manifest_multi.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--hu-min", type=float, default=-1024.0)
    ap.add_argument("--hu-max", type=float, default=3071.0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--sources", nargs="+", default=None, help="restrict to these sources")
    ap.add_argument("--max-per-source", type=int, default=None, help="debug: first N volumes of each source")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, keep_default_na=False)
    if args.sources:
        df = df[df.source.isin(args.sources)]
    if args.max_per_source:
        df = df.groupby("source").head(args.max_per_source)
    out_dir = Path(args.out_dir)
    todo = df[df.npy_path == ""]
    jobs = [dict(r._asdict(), out_path=str(out_dir / r.source / f"{r.study_id}.npy"), size=args.size,
                 spacing=args.spacing, hu_min=args.hu_min, hu_max=args.hu_max) for r in todo.itertuples(index=False)]
    print(f"[preprocess] {len(jobs)} volumes to process ({len(df) - len(jobs)} reused) -> {out_dir} "
          f"({args.size}^3 @ {args.spacing} mm, HU [{args.hu_min:g}, {args.hu_max:g}], {args.workers} workers)", flush=True)

    t0, results = time.time(), []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_one, jobs, chunksize=2), 1):
            results.append(r)
            if r["status"] not in ("converted", "skipped"):
                print(f"[preprocess] {r['study_id']}: {r['status']}", flush=True)
            if i % 500 == 0:
                print(f"[preprocess] {i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)

    res = pd.DataFrame(results, columns=["study_id", "path", "status"] if not results else None)
    ok = res[res["status"].isin(["converted", "skipped"])][["study_id", "path"]] if len(res) else pd.DataFrame(columns=["study_id", "path"])
    reused = df[df.npy_path != ""].assign(path=lambda d: d.npy_path)[["study_id", "path"]]
    paths = pd.concat([reused, ok])
    keep = ["study_id", "split", "few_shot", "source", "domain"]
    out = df[keep].merge(paths, on="study_id").sort_values("study_id")
    out.to_csv(args.out_manifest, index=False)
    if len(res):
        res.to_csv(Path(args.out_manifest).with_suffix(".status.csv"), index=False)
    print(f"[preprocess] {int((res.status == 'converted').sum()) if len(res) else 0} converted, "
          f"{int((res.status == 'skipped').sum()) if len(res) else 0} already present, "
          f"{len(res) - len(ok)} failed; {len(out)} rows -> {args.out_manifest} ({time.time() - t0:.0f}s)", flush=True)
    print(out.groupby(["domain", "source", "split"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
