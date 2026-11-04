"""Paired bootstrap of the test metrics of all runs: 95% CI per run and for differences between runs.

Uses the embeddings saved by `eval.evaluate extract` (<run>/eval_<ckpt>/emb_shard*.npz). For each
run it recomputes per-sample retrieval hits and the linear-probe test predictions (same protocol as
eval/evaluate.py: StandardScaler + LogisticRegression(class_weight="balanced")), then resamples the
test studies with replacement, the same resample for every run (paired), B times.

    python scripts/bootstrap_evals.py [--B 1000] [--ref win2_pillar11]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.evaluate import FIVE_YEAR, load_embeddings  # noqa: E402

RUNS = ["win1_clip", "win2_pillar11", "win3_learned", "win4_local", "win5_abdo"]


def retrieval_hits(img: np.ndarray, txt: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    im = F.normalize(torch.from_numpy(img).float(), dim=-1)
    tx = F.normalize(torch.from_numpy(txt).float(), dim=-1)
    sim = im @ tx.T
    gt = torch.arange(sim.shape[0])
    i2t = ((sim > sim[gt, gt][:, None]).sum(1) < k).numpy()
    t2i = ((sim.T > sim[gt, gt][:, None]).sum(1) < k).numpy()
    return i2t, t2i


def probe_predictions(ids, X, manifest, findings_csv, five_year_csv) -> list[dict]:
    """[{task, label, test_ids, y, p}] for every label with both classes in train and test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    man = pd.read_csv(manifest)
    split_of = dict(zip(man["study_id"].astype(str), man["split"].astype(str)))
    fi = pd.read_csv(findings_csv)
    fi["sid"] = fi["study id"].astype(str)
    fi["_split"] = fi["sid"].map(split_of)
    fi_cols = [c for c in fi.columns if c not in ("study id", "sid", "_split")]
    fy = pd.read_csv(five_year_csv)
    fy["sid"] = fy["study id"].astype(str)
    fy["_split"] = fy["merlin_split"].astype(str).str.strip()
    fy = fy[fy["_split"].isin(["train", "val", "test"])]
    row = {s: i for i, s in enumerate(ids)}
    out = []
    for task, df, cols in (("findings", fi, fi_cols), ("five_year", fy, FIVE_YEAR)):
        df = df.drop_duplicates("sid")
        df = df[df["sid"].isin(row)]
        Xt = X[[row[s] for s in df["sid"]]]
        split = df["_split"].to_numpy()
        sids = df["sid"].to_numpy()
        for col in cols:
            y_raw = df[col].astype(int).to_numpy()
            valid, y = (y_raw != -1, (y_raw > 0).astype(int)) if task == "findings" else \
                (np.isin(y_raw, [0, 2]), (y_raw == 2).astype(int))
            tr, te = valid & (split == "train"), valid & (split == "test")
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            sc = StandardScaler().fit(Xt[tr])
            clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xt[tr]), y[tr])
            out.append({"task": task, "label": col, "test_ids": sids[te], "y": y[te],
                        "p": clf.predict_proba(sc.transform(Xt[te]))[:, 1]})
    return out


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    npos = y.sum()
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCHDIR/merlin_clip_runs"))
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--ckpt", default="ckpt_final")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--ref", default="win2_pillar11", help="differences are reported against this run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root)
    cfg = __import__("json").load(open(root / args.runs[0] / "config_resolved.json"))
    man = pd.read_csv(cfg["data"]["manifest"])
    split_of = dict(zip(man["study_id"].astype(str), man["split"].astype(str)))

    data = {}
    for run in args.runs:
        ids, img, txt = load_embeddings(str(root / run / f"eval_{args.ckpt}"))
        test = np.array([split_of.get(s) == "test" for s in ids])
        i2t, t2i = retrieval_hits(img[test], txt[test])
        probes = probe_predictions(ids, img, cfg["data"]["manifest"], cfg["eval"]["findings_csv"],
                                   cfg["eval"]["five_year_csv"])
        data[run] = {"test_ids": ids[test], "i2t": i2t, "t2i": t2i, "probes": probes}
        print(f"{run}: {test.sum()} test pairs, {len(probes)} probe labels", flush=True)

    test_ids = data[args.runs[0]]["test_ids"]
    for run in args.runs:
        assert np.array_equal(data[run]["test_ids"], test_ids), "runs must share the test set"
    n = len(test_ids)
    pos = {s: i for i, s in enumerate(test_ids)}
    rng = np.random.default_rng(args.seed)
    boots = [rng.integers(0, n, n) for _ in range(args.B)]

    def metric_samples(run: str) -> dict[str, np.ndarray]:
        d = data[run]
        out = {"R@10 i2t": [], "R@10 t2i": [], "probe findings": [], "probe 5-year": []}
        point = {"R@10 i2t": d["i2t"].mean(), "R@10 t2i": d["t2i"].mean()}
        # probes: each label's test samples, as positions in the shared test set
        lab = [(pr["task"], np.array([pos[s] for s in pr["test_ids"]]), pr["y"], pr["p"]) for pr in d["probes"]]
        for task, name in (("findings", "probe findings"), ("five_year", "probe 5-year")):
            point[name] = np.mean([auroc(y, p) for t, _, y, p in lab if t == task])
        for b in boots:
            out["R@10 i2t"].append(d["i2t"][b].mean())
            out["R@10 t2i"].append(d["t2i"][b].mean())
            counts = np.bincount(b, minlength=n)
            for task, name in (("findings", "probe findings"), ("five_year", "probe 5-year")):
                aucs = []
                for t, where, y, p in lab:
                    if t != task:
                        continue
                    w = counts[where]                      # resampled multiplicity of each labelled study
                    keep = np.repeat(np.arange(len(where)), w)
                    aucs.append(auroc(y[keep], p[keep]))
                out[name].append(np.nanmean(aucs))
        return {"point": point, **{k: np.array(v) for k, v in out.items()}}

    samples = {run: metric_samples(run) for run in args.runs}
    metrics = ["R@10 i2t", "R@10 t2i", "probe findings", "probe 5-year"]
    rows = []
    for run in args.runs:
        r = {"run": run}
        for m in metrics:
            s = samples[run][m]
            r[m] = f"{samples[run]['point'][m]:.3f} [{np.percentile(s, 2.5):.3f}, {np.percentile(s, 97.5):.3f}]"
        rows.append(r)
    print(f"\nTest metrics, point estimate [95% paired-bootstrap CI], B={args.B}, n_test={n}")
    print(pd.DataFrame(rows).to_string(index=False))
    refs = [args.ref] + ([RUNS[0]] if args.ref != RUNS[0] and RUNS[0] in args.runs else [])
    for ref in refs:
        rows = []
        for run in args.runs:
            if run == ref:
                continue
            r = {"run": f"{run} - {ref}"}
            for m in metrics:
                dlt = samples[run][m] - samples[ref][m]
                pt = samples[run]["point"][m] - samples[ref]["point"][m]
                r[m] = f"{pt:+.3f} [{np.percentile(dlt, 2.5):+.3f}, {np.percentile(dlt, 97.5):+.3f}] P>0={np.mean(dlt > 0):.2f}"
            rows.append(r)
        print(f"\nDifferences vs {ref} (P>0 = share of bootstrap resamples where the run is better)")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
