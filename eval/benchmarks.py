"""Test tasks of the multi-source data (Merlin, CT-RATE, INSPECT, CancerVerse) and model-agnostic metrics.

Labels (binary, one row per study and task; splits are those of the multi-source manifest, so
probes train on a source's train split and are scored on its test split):
  merlin       30 findings (zero_shot_findings_disease_cls.csv; -1 = missing) and the 6 five-year
               diseases (five_years_disease_task.csv; 2 = positive, 0 = negative, 1 / 3 excluded)
  ctrate       18 abnormalities (multi_abnormality_labels/{train,valid}_predicted_labels.csv, recon _1)
  inspect      pe_positive_nlp; 1 / 6 / 12-month mortality and readmission, 12-month pulmonary
               hypertension (labels_20250611.tsv; "Censored" excluded), joined on study_mapping
  cancerverse  exam ICD-10 code (CancerVerse_dataset_metadata.csv): one task per code family with
               enough test positives, positives = that family, negatives = every other scan
               (including the 70% without a code); plus any C code. Clinical exam codes, not image-verified.
Kinds: "finding" (zero-shot and probe) or "outcome" (probe only: mortality, readmission, PH, five-year).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CTRATE_LABELS = ["/lus/work/CT3/cad17796/SHARED/ibrahimhamamci/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv",
                 "/lus/work/CT3/cad17796/SHARED/ibrahimhamamci/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"]
INSPECT_DIR = os.path.expandvars("$SCRATCHDIR/inspect/full")
CANCERVERSE_META = os.path.expandvars("$DSDIR/BodyMaps/CancerVerse/CancerVerse_dataset_metadata.csv")
MERLIN_FINDINGS = "/lus/work/CT3/cad17796/SHARED/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv"
MERLIN_FIVE_YEAR = "/lus/work/CT3/cad17796/SHARED/merlinabdominalctdataset/five_years_disease_task.csv"
FIVE_YEAR = ["cvd", "ihd", "htn", "dm", "ckd", "ost"]
INSPECT_TASKS = {"pe_positive_nlp": "finding", "1_month_mortality": "outcome", "6_month_mortality": "outcome",
                 "12_month_mortality": "outcome", "1_month_readmission": "outcome", "6_month_readmission": "outcome",
                 "12_month_readmission": "outcome", "12_month_PH": "outcome"}
CANCERVERSE_CODES = {"C79": "secondary_neoplasm_other_sites", "C78": "secondary_neoplasm_resp_digestive",
                     "C22": "liver_cancer", "C64": "kidney_cancer", "C25": "pancreas_cancer", "C16": "stomach_cancer"}


def _rows(src, ids, task, kind, y, valid):
    return pd.DataFrame({"study_id": np.asarray(ids)[valid], "source": src, "task": f"{src}:{task}", "kind": kind,
                         "y": np.asarray(y, dtype=int)[valid]})


def build_labels(manifest: pd.DataFrame) -> pd.DataFrame:
    """Long table study_id, source, task, kind, y for every manifest study that has the label."""
    out = []
    have = set(manifest["study_id"].astype(str))
    # merlin
    fi = pd.read_csv(MERLIN_FINDINGS).drop_duplicates("study id")
    fi = fi[fi["study id"].astype(str).isin(have)]
    for c in fi.columns:
        if c == "study id":
            continue
        y = fi[c].astype(int).to_numpy()
        out.append(_rows("merlin", fi["study id"].astype(str), c, "finding", y > 0, y != -1))
    fy = pd.read_csv(MERLIN_FIVE_YEAR).drop_duplicates("study id")
    fy = fy[fy["study id"].astype(str).isin(have)]
    for c in FIVE_YEAR:
        y = fy[c].astype(int).to_numpy()
        out.append(_rows("merlin", fy["study id"].astype(str), f"5y_{c}", "outcome", y == 2, np.isin(y, [0, 2])))
    # ct-rate (our study id = VolumeName without the reconstruction suffix; we use recon _1)
    ct = pd.concat([pd.read_csv(f) for f in CTRATE_LABELS], ignore_index=True)
    ct = ct[ct["VolumeName"].str.endswith("_1.nii.gz")].copy()
    ct["sid"] = ct["VolumeName"].str[: -len("_1.nii.gz")]
    ct = ct[ct["sid"].isin(have)]
    for c in ct.columns:
        if c in ("VolumeName", "sid"):
            continue
        out.append(_rows("ctrate", ct["sid"], c.lower().replace(" ", "_"), "finding", ct[c].astype(int), np.ones(len(ct), bool)))
    # inspect
    lab = pd.read_csv(Path(INSPECT_DIR) / "labels_20250611.tsv", sep="\t")
    mp = pd.read_csv(Path(INSPECT_DIR) / "study_mapping_20250611.tsv", sep="\t")[["impression_id", "image_id"]]
    lab = lab.merge(mp, on="impression_id").drop_duplicates("image_id")
    lab = lab[lab["image_id"].astype(str).isin(have)]
    for c, kind in INSPECT_TASKS.items():
        v = lab[c].astype(str).str.upper()
        out.append(_rows("inspect", lab["image_id"].astype(str), c, kind, v == "TRUE", v.isin(["TRUE", "FALSE"])))
    # cancerverse
    cv = pd.read_csv(CANCERVERSE_META, usecols=["CancerVerse ID", "icd10_code"])
    cv = cv[cv["CancerVerse ID"].astype(str).isin(have)].drop_duplicates("CancerVerse ID")
    code = cv["icd10_code"].fillna("").astype(str).str.upper()
    for c3, name in CANCERVERSE_CODES.items():
        out.append(_rows("cancerverse", cv["CancerVerse ID"].astype(str), name, "finding", code.str[:3] == c3,
                         np.ones(len(cv), bool)))
    out.append(_rows("cancerverse", cv["CancerVerse ID"].astype(str), "any_malignancy_code", "finding",
                     code.str.startswith("C"), np.ones(len(cv), bool)))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- metrics
def recall(img: torch.Tensor, txt: torch.Tensor, ks=(1, 5, 10)) -> dict:
    sim = img @ txt.T
    gt = torch.arange(len(img))
    ri = (sim > sim[gt, gt][:, None]).sum(1)
    rt = (sim.T > sim[gt, gt][:, None]).sum(1)
    return {**{f"R@{k} i2t": float((ri < k).float().mean()) for k in ks},
            **{f"R@{k} t2i": float((rt < k).float().mean()) for k in ks}}


def retrieval_suite(img: torch.Tensor, txt: torch.Tensor) -> dict:
    """R@k among all pairs, in consecutive chunks of 128 and in random pools of 100 (Jolia App. G)."""
    n = len(img)
    res = {f"{k} (all)": v for k, v in recall(img, txt).items()}
    if n >= 128:
        ch = [recall(img[s:s + 128], txt[s:s + 128]) for s in range(0, n - 127, 128)]
        res |= {f"{k} (128)": float(np.mean([c[k] for c in ch])) for k in ch[0]}
    if n >= 100:
        keys, p = list(recall(img[:2], txt[:2])), []
        for seed in range(10):                       # random pools of 100, the remainder dropped
            perm = np.random.RandomState(seed).permutation(n)
            rs = [recall(img[s], txt[s]) for s in (perm[i:i + 100] for i in range(0, n - 99, 100))]
            p.append({k: float(np.mean([q[k] for q in rs])) for k in keys})
        res |= {f"{k} (pool100)": float(np.mean([q[k] for q in p])) for k in keys}
    return res


def linear_probes(ids, X: np.ndarray, labels: pd.DataFrame, split_of: dict, min_pos: int = 10) -> pd.DataFrame:
    """StandardScaler + balanced logistic regression per task: train split -> test split (same source)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    row = {s: i for i, s in enumerate(ids)}
    res = []
    for (task, kind, src), g in labels.groupby(["task", "kind", "source"], sort=True):
        g = g[g["study_id"].isin(row)]
        sp = g["study_id"].map(split_of)
        tr, te = g[sp == "train"], g[sp == "test"]
        rec = {"task": task, "kind": kind, "source": src, "n_train": len(tr), "n_test": len(te),
               "n_test_pos": int(te["y"].sum()), "auroc": np.nan}
        if tr["y"].nunique() == 2 and te["y"].sum() >= min_pos and (len(te) - te["y"].sum()) >= min_pos:
            Xtr, Xte = X[[row[s] for s in tr["study_id"]]], X[[row[s] for s in te["study_id"]]]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), tr["y"])
            rec["auroc"] = float(roc_auc_score(te["y"], clf.predict_proba(sc.transform(Xte))[:, 1]))
        res.append(rec)
    return pd.DataFrame(res)


def zero_shot_long(img_of: dict, text_fn, labels: pd.DataFrame, split_of: dict, has_text: set,
                   r: int = 50, seeds: int = 5, min_pos: int = 10) -> pd.DataFrame:
    """Jolia Table 16 protocol: per task, prototypes = mean of r positive / r negative train reports of the
    same source (through the model's text head); score = cos(img, pos) - cos(img, neg); AUROC on test."""
    from sklearn.metrics import roc_auc_score
    res = []
    for (task, src), g in labels[labels["kind"] == "finding"].groupby(["task", "source"], sort=True):
        sp = g["study_id"].map(split_of)
        tr = g[(sp == "train") & g["study_id"].isin(has_text)]
        te = g[(sp == "test") & g["study_id"].isin(img_of)]
        P, N = tr[tr["y"] == 1]["study_id"].tolist(), tr[tr["y"] == 0]["study_id"].tolist()
        rec = {"task": task, "source": src, "n_test": len(te), "n_test_pos": int(te["y"].sum()), "auroc_long": np.nan}
        if len(P) >= r and len(N) >= r and te["y"].sum() >= min_pos and (len(te) - te["y"].sum()) >= min_pos:
            img = F.normalize(torch.from_numpy(np.stack([img_of[s] for s in te["study_id"]])).float(), dim=-1)
            aucs = []
            for seed in range(seeds):
                rng = np.random.RandomState(seed)
                ep = F.normalize(text_fn(list(rng.choice(P, r, replace=False))), dim=-1).mean(0)
                en = F.normalize(text_fn(list(rng.choice(N, r, replace=False))), dim=-1).mean(0)
                s = (img @ F.normalize(ep, dim=0) - img @ F.normalize(en, dim=0)).numpy()
                aucs.append(roc_auc_score(te["y"], s))
            rec["auroc_long"], rec["auroc_long_sd"] = float(np.mean(aucs)), float(np.std(aucs))
        res.append(rec)
    return pd.DataFrame(res)
