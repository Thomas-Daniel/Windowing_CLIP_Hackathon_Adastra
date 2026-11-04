"""Expand the chest/abdomen draw (manifest_v2.csv) into the inputs of the preprocessing and embedding steps.

manifest_v2.csv (scripts of $SCRATCHDIR/chest_abdo_clip_data: pools.py + build_manifest_v2.py) has one row per
volume: study_id, patient, source, domain, split, text. Sources: merlin + cancerverse (abdomen), ctrate + inspect
(chest). Writes into --out-dir:

  manifest_nifti.csv  study_id, split, few_shot, source, domain, nifti_path, npy_path, slope, intercept, xy0, xy1, z
                      (merlin rows keep their existing 384^3 .npy and have no nifti_path; ctrate rows carry the
                      DICOM rescale and spacing of train_metadata.csv, since the v1 NIfTI headers have neither)
  reports.jsonl       {"study_id", "split", "text"}    (input of embed_reports.py; text as released, no cleaning)

    python scripts/build_manifest_multi.py --out-dir $SCRATCHDIR/chest_abdo_clip_data
    python scripts/build_manifest_multi.py --out-dir $SCRATCHDIR/chest_abdo_clip_data --manifest .../manifest_D.csv --tag _D
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import pandas as pd

CTRATE = "/lus/work/CT3/cad17796/SHARED/ibrahimhamamci/CT-RATE/dataset"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", default=None, help="default <out-dir>/manifest_v2.csv (variant D: manifest_D.csv)")
    ap.add_argument("--tag", default="", help="suffix of the outputs: manifest_nifti<tag>.csv, reports<tag>.jsonl")
    ap.add_argument("--merlin-manifest", default=os.path.expandvars("$SCRATCHDIR/merlin_clip_data/manifest_384.csv"))
    ap.add_argument("--inspect-dir", default=os.path.expandvars("$SCRATCHDIR/inspect/full/CTPA"))
    ap.add_argument("--cancerverse-dir", default=os.path.expandvars("$DSDIR/BodyMaps/CancerVerse/CancerVerse"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    m = pd.read_csv(args.manifest or out / "manifest_v2.csv")
    assert not m.study_id.duplicated().any(), "study ids must be unique across sources"

    merlin = pd.read_csv(args.merlin_manifest).set_index("study_id")["path"]
    meta = pd.concat([pd.read_csv(f"{CTRATE}/metadata/{n}_metadata.csv") for n in ("train", "validation")]).set_index("VolumeName")

    rows = []
    for r in m.itertuples():
        d = dict(study_id=r.study_id, split=r.split, few_shot=0, source=r.source, domain=r.domain,
                 nifti_path="", npy_path="", slope=1.0, intercept=0.0, xy0=0.0, xy1=0.0, z=0.0)
        if r.source == "merlin":
            d["npy_path"] = merlin[r.study_id]
        elif r.source == "cancerverse":
            d["nifti_path"] = f"{args.cancerverse_dir}/{r.study_id}/ct.nii.gz"
        elif r.source == "inspect":
            d["nifti_path"] = f"{args.inspect_dir}/{r.study_id}.nii.gz"
        elif r.source == "ctrate":
            pid = "_".join(r.study_id.split("_")[:2])
            folder = "valid" if r.study_id.startswith("valid_") else "train"        # official CT-RATE validation = test here
            d["nifti_path"] = f"{CTRATE}/{folder}/{pid}/{r.study_id}/{r.study_id}_1.nii.gz"
            v = meta.loc[f"{r.study_id}_1.nii.gz"]
            xy = ast.literal_eval(v.XYSpacing)
            d.update(slope=float(v.RescaleSlope), intercept=float(v.RescaleIntercept),
                     xy0=float(xy[0]), xy1=float(xy[1]), z=float(v.ZSpacing))
        else:
            raise ValueError(r.source)
        rows.append(d)
    df = pd.DataFrame(rows)
    miss = [p for p in df.nifti_path[df.nifti_path != ""] if not os.path.exists(p)]
    miss += [p for p in df.npy_path[df.npy_path != ""] if not os.path.exists(p)]
    print(f"{len(df)} volumes; missing files: {len(miss)} {miss[:3]}")
    assert df[df.source == "ctrate"].z.gt(0).all(), "ctrate rows need a z spacing"
    df.to_csv(out / f"manifest_nifti{args.tag}.csv", index=False)
    with open(out / f"reports{args.tag}.jsonl", "w") as f:
        for r in m.itertuples():
            f.write(json.dumps({"study_id": r.study_id, "split": r.split, "text": r.text}) + "\n")
    print(df.groupby(["domain", "source", "split"]).size().unstack().to_string())
    print(f"wrote {out / f'manifest_nifti{args.tag}.csv'} and {out / f'reports{args.tag}.jsonl'}")


if __name__ == "__main__":
    main()
