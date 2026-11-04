"""Pair every Merlin NIfTI with its report and official split.

Reads Merlin's ``reports_final.xlsx`` (columns: study id, Findings, Split, Few Shot) and the
``merlin_data/`` NIfTI directory, and writes into ``--out-dir``:

  manifest_nifti.csv  study_id, split, few_shot, nifti_path   (input of preprocess_volumes.py)
  reports.jsonl       {"study_id", "split", "text"}           (input of embed_reports.py)

The report text is the ``Findings`` cell exactly as released (no cleaning, no section
splitting). Only the pairing is cleaned:
  - a study id listed twice with the same text is kept once;
  - a study id listed with different texts (2 ids in the release) is dropped, since it is
    not known which report belongs to the volume;
  - reports shorter than ``--min-chars`` characters (1 report, "1`") are dropped;
  - studies without a NIfTI file are dropped.

Usage:
    python scripts/build_manifest.py \
        --reports /lus/work/CT3/cad17796/SHARED/merlinabdominalctdataset/reports_final.xlsx \
        --nifti-dir /lus/work/CT3/cad17796/SHARED/merlinabdominalctdataset/merlin_data \
        --out-dir $SCRATCHDIR/merlin_clip_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--nifti-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-chars", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_excel(args.reports)
    df = df.rename(columns={"study id": "study_id", "Findings": "text", "Split": "split", "Few Shot": "few_shot"})
    df["study_id"] = df["study_id"].astype(str)
    df["text"] = df["text"].astype(str)
    n0 = len(df)

    df = df.drop_duplicates(["study_id", "text"])
    n_texts = df.groupby("study_id")["text"].transform("nunique")
    conflicting = sorted(df.loc[n_texts > 1, "study_id"].unique())
    df = df[n_texts == 1]
    short = df[df["text"].str.len() < args.min_chars]
    df = df[df["text"].str.len() >= args.min_chars]

    nifti_dir = Path(args.nifti_dir)
    df["nifti_path"] = [str(nifti_dir / f"{sid}.nii.gz") for sid in df["study_id"]]
    exists = df["nifti_path"].map(lambda p: Path(p).exists())
    df = df[exists].sort_values("study_id").reset_index(drop=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df[["study_id", "split", "few_shot", "nifti_path"]].to_csv(out / "manifest_nifti.csv", index=False)
    with open(out / "reports.jsonl", "w") as f:
        for r in df.itertuples():
            f.write(json.dumps({"study_id": r.study_id, "split": r.split, "text": r.text}) + "\n")

    print(f"{n0} report rows -> {len(df)} studies "
          f"(conflicting duplicate ids dropped: {conflicting}; short reports dropped: "
          f"{short['study_id'].tolist()}; missing NIfTI: {int((~exists).sum())})")
    print(df["split"].value_counts().to_string())
    print(f"wrote {out / 'manifest_nifti.csv'} and {out / 'reports.jsonl'}")


if __name__ == "__main__":
    main()
