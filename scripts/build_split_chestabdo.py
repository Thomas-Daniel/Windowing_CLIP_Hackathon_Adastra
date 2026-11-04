"""Variant D of the chest / abdomen mix: one row per volume with its report and split, written to
<out-dir>/manifest_D.csv (study_id, patient, source, domain, split, text).

  abdomen = Merlin (official train+val -> train, official test -> test) + CancerVerse
  chest   = CT-RATE (official train -> train, official valid -> test) + INSPECT (official train+valid -> train,
            official test -> test)
Following the "all of the data" recipe: no cap on scans per patient. Volume filters (same as the frozen draw):
  - one volume per study: CT-RATE reconstruction _1, CancerVerse one series (portal_venous first);
  - CT-RATE scans listed in no_chest_*.txt dropped; CancerVerse studies with conflicting reports dropped;
  - reports shorter than 20 characters dropped; INSPECT volumes with unreadable header / 4D / < 100 slices /
    z spacing > 3 mm / z extent outside [150, 700] mm dropped ($SCRATCHDIR/inspect/headers.pkl of the audit).
CancerVerse has no official split: patients are drawn 2300 for the test (one study per patient), the rest train.
Train only: at most --text-cap volumes (5) with the same report text (false negatives in the CLIP batch).
A monitoring set --n-val per domain (1500) is carved by patient out of the train patients; tests are kept as released
(official Merlin test = 5121 as in the earlier runs).

    python scripts/build_split_chestabdo.py --out-dir $SCRATCHDIR/chest_abdo_clip_data
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

CTRATE = "/lus/work/CT3/cad17796/SHARED/ibrahimhamamci/CT-RATE/dataset"


def ctrate(split: str) -> pd.DataFrame:
    name, rep, nc = ("train", "train_reports.csv", "no_chest_train.txt") if split == "train" else \
                    ("valid", "validation_reports.csv", "no_chest_valid.txt")
    rp = pd.read_csv(f"{CTRATE}/radiology_text_reports/{rep}")
    nochest = {os.path.basename(x).rsplit("_", 1)[0] for x in open(f"{CTRATE}/metadata/{nc}").read().split()}
    rp = rp[rp.VolumeName.str.endswith("_1.nii.gz")].copy()
    rp["study_id"] = rp.VolumeName.str.replace("_1.nii.gz", "", regex=False)
    rp = rp[~rp.study_id.isin(nochest)]
    pid = rp.study_id.str.extract(r"_(\d+)_")[0]
    return pd.DataFrame({"study_id": rp.study_id, "patient": f"CTR{name[0]}_" + pid, "source": "ctrate",
                         "domain": "chest", "split": "train" if split == "train" else "test",
                         "text": rp.Findings_EN.fillna("")})


def inspect_pool(sdir: str) -> pd.DataFrame:
    I = f"{sdir}/inspect/"
    h = pd.read_pickle(I + "headers.pkl")
    sp = pd.read_csv(I + "full/splits_20250611.tsv", sep="\t")
    mp = pd.read_csv(I + "full/study_mapping_20250611.tsv", sep="\t")
    im = pd.read_csv(I + "full/impressions_20250611.tsv", sep="\t")
    ok = h[h.err.isna()].copy()
    ok["nz"] = ok["shape"].map(lambda s: s[2] if len(s) == 3 else 0)      # 4D volumes -> 0
    ok["z"] = ok.zooms.map(lambda z: z[2])
    ok["ext"] = ok.nz * ok.z
    ok = ok[(ok.nz >= 100) & (ok.z <= 3) & (ok.ext >= 150) & (ok.ext <= 700)]
    d = ok.merge(mp[["image_id", "impression_id", "person_id"]], on="image_id") \
          .merge(sp[["impression_id", "split"]], on="impression_id").merge(im, on="impression_id")
    return pd.DataFrame({"study_id": d.image_id, "patient": "INS_" + d.person_id.astype(str), "source": "inspect",
                         "domain": "chest", "split": np.where(d.split == "test", "test", "train"),
                         "text": d.impressions.fillna("")})


def cancerverse(seed: int, n_test: int) -> pd.DataFrame:
    cv = pd.read_csv(os.path.expandvars("$DSDIR/BodyMaps/CancerVerse/CancerVerse_dataset_metadata.csv"))
    nrep = cv.groupby("Encrypted Accession Number").report.nunique()
    cv = cv[~cv["Encrypted Accession Number"].isin(set(nrep[nrep > 1].index))]
    rng = np.random.default_rng(seed)
    pref = {"portal_venous": 0, "venous": 1, "arterial_late": 2, "arterial": 3, "arterial_early": 4, "non-contrast": 5}
    cv = cv.assign(rk=cv.phase.map(pref).fillna(6), u=rng.random(len(cv)), text=cv.report.fillna(""))
    cv = cv[cv.text.str.len() >= 20].sort_values(["Encrypted Accession Number", "rk", "u"]) \
           .drop_duplicates("Encrypted Accession Number")
    cv = cv.assign(patient="CV_" + cv["Patient ID"])
    pats = np.array(sorted(cv.patient.unique())); rng.shuffle(pats)
    test_p = set(pats[:n_test])                       # one random study per test patient, other studies dropped
    cv["split"] = np.where(cv.patient.isin(test_p), "test", "train")
    t = cv[cv.split == "test"].sample(frac=1, random_state=seed).drop_duplicates("patient")
    cv = pd.concat([cv[cv.split == "train"], t])
    return pd.DataFrame({"study_id": cv["CancerVerse ID"], "patient": cv.patient, "source": "cancerverse",
                         "domain": "abdomen", "split": cv.split, "text": cv.text})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv-test", type=int, default=2300, help="CancerVerse test patients")
    ap.add_argument("--n-val", type=int, default=1500, help="monitoring volumes per domain, taken from train patients")
    ap.add_argument("--text-cap", type=int, default=5, help="max train volumes sharing one report text (0 = no cap)")
    ap.add_argument("--merlin-manifest", default=os.path.expandvars("$SCRATCHDIR/merlin_clip_data/manifest_384.csv"))
    ap.add_argument("--merlin-reports", default=os.path.expandvars("$SCRATCHDIR/merlin_clip_data/reports.jsonl"))
    args = ap.parse_args()
    sdir = os.path.expandvars("$SCRATCHDIR")

    mm = pd.read_csv(args.merlin_manifest)
    rep = pd.DataFrame([json.loads(l) for l in open(args.merlin_reports)])[["study_id", "text"]]
    mer = mm.merge(rep, on="study_id")
    mer = pd.DataFrame({"study_id": mer.study_id, "patient": mer.study_id, "source": "merlin", "domain": "abdomen",
                        "split": np.where(mer.split == "test", "test", "train"), "text": mer.text})   # train + val -> train

    pools = [mer, cancerverse(args.seed, args.cv_test), ctrate("train"), ctrate("valid"), inspect_pool(sdir)]
    M = pd.concat(pools, ignore_index=True)
    assert not M.study_id.duplicated().any(), "study ids must be unique across sources"
    n0 = len(M)
    M = M[M.text.str.len() >= 20]
    print(f"{n0} volumes -> {len(M)} with a report of >= 20 characters")

    rng = np.random.default_rng(args.seed)
    tr = M[M.split == "train"].sample(frac=1, random_state=args.seed)
    if args.text_cap:
        n1 = len(tr)
        tr = tr[tr.groupby("text").cumcount() < args.text_cap]
        print(f"text cap {args.text_cap}: {n1 - len(tr)} train volumes removed")
    M = pd.concat([tr, M[M.split == "test"]])

    # monitoring split: whole patients out of train, n_val per domain, one volume per patient
    for dom in ("abdomen", "chest"):
        t = M[(M.domain == dom) & (M.split == "train")]
        p = t.groupby("patient").head(1).sample(n=args.n_val, random_state=int(rng.integers(1e9)))
        M.loc[M.patient.isin(set(p.patient)) & (M.domain == dom) & (M.split == "train"), "split"] = "val_all"
        keep = set(p.study_id)
        M.loc[(M.split == "val_all") & M.study_id.isin(keep), "split"] = "val"
        M = M[M.split != "val_all"]            # the other volumes of the val patients are dropped (no train/val leakage)
    assert not (set(M[M.split == "train"].patient) & set(M[M.split != "train"].patient)), "patient leakage"

    out = Path(args.out_dir)
    M.sort_values("study_id").to_csv(out / "manifest_D.csv", index=False)
    t = M.groupby(["domain", "source", "split"]).size().unstack(fill_value=0)
    print(t.to_string()); print(M.groupby(["domain", "split"]).size().unstack(fill_value=0).to_string())
    tr_n = M[M.split == "train"].groupby("domain").size(); print("abdomen share of train:", round(tr_n["abdomen"] / tr_n.sum(), 3))
    print(f"wrote {out / 'manifest_D.csv'}")


if __name__ == "__main__":
    main()
