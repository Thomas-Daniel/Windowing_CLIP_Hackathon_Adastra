"""Evaluate trained CLIP checkpoints on Merlin, comparable across the windowing arms.

Phase ``extract`` (sharded over GCDs): one deterministic forward per volume (no
augmentation, eval-mode BN), saves per volume the 1152-d image feature (the CLIP embedding
before normalisation) and the projected report feature, for every volume of the
requested splits, to <out>/emb_shard*.npz.

Phase ``report``:
  - retrieval on the test split (image <-> report, R@1/5/10, among all test pairs and
    in chunks of 128, the training batch size), and test InfoNCE at batch 128;
  - zero-shot findings (Merlin's 30 findings, test split): AUROC of cos(img, pos) -
    cos(img, neg) with the Qwen-embedded prompts of mcw/prompts.py;
  - linear probe, the concerto-ct / Merlin_CT_JEPA protocol (eval/linear_probe.py there):
    StandardScaler + LogisticRegression(class_weight="balanced"), findings on the official
    split, five-year disease prediction on five_years_disease_task.csv's merlin_split.

    srun ... python -m eval.evaluate extract --ckpt runs/win2_pillar11/ckpt_final.pt --out-dir <dir>
    python -m eval.evaluate report --ckpt runs/win2_pillar11/ckpt_final.pt --out-dir <dir>

Chest / abdomen mix (config eval.benchmark: multi): ``report`` scores each test
set separately with eval/benchmarks.py: retrieval per source (all pairs, chunks of 128, random pools
of 100) and test InfoNCE, long-form zero-shot (prototypes of 50 + 50 train reports) on every finding
task, linear probes on every task (train -> test of the same source; labels in eval.labels), and
the short-prompt zero-shot above on the Merlin test. ``extract --max-train-per-source N`` embeds a
fixed random N train volumes per source (the probes' training sets) instead of all of them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mcw.data import MerlinClipDataset, TextEmbeddings, collate
from mcw.evaluate import batched_infonce, retrieval_metrics, zero_shot_findings
from mcw.model import CTClip

FIVE_YEAR = ["cvd", "ihd", "htn", "dm", "ckd", "ost"]


def load_model(ckpt: str, device) -> tuple[CTClip, dict]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    texts = TextEmbeddings(cfg["data"]["text_emb_dir"], cfg["data"]["text_pooling"])
    model = CTClip(cfg, texts.dim)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def extract(args) -> None:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    model, cfg = load_model(args.ckpt, device)
    texts = TextEmbeddings(cfg["data"]["text_emb_dir"], cfg["data"]["text_pooling"])
    keep = None
    if args.max_train_per_source:                    # probes only need a train subset: N per source
        man = pd.read_csv(cfg["data"]["manifest"], dtype={"study_id": str})
        man = man[man["split"].isin(args.splits)]
        tr = man[man["split"] == "train"]
        tr = tr.groupby("source", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), args.max_train_per_source), random_state=0))
        keep = set(man[man["split"] != "train"]["study_id"]) | set(tr["study_id"])
    ds = MerlinClipDataset(cfg["data"]["manifest"], texts, args.splits, max_items=args.max_items, seed=0, keep_ids=keep,
                           crop=cfg["data"].get("crop"))                 # long runs: centred crop^3, as in training val
    shard = list(range(len(ds)))[rank::world]
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, sampler=shard, num_workers=args.num_workers,
                                         collate_fn=collate)
    ids, img, txt, t0 = [], [], [], time.time()
    for batch in loader:
        hu = batch["volume"].to(device, non_blocking=True).float()
        img.append(model.encode_image(hu).cpu())
        txt.append(model.encode_text(batch["text"].to(device)).cpu())
        ids += batch["study_id"]
        if len(ids) % 200 < args.batch:
            print(f"[extract {rank}/{world}] {len(ids)}/{len(shard)} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"emb_shard{rank:03d}.npz", ids=np.array(ids), img=torch.cat(img).numpy(),
             txt=torch.cat(txt).numpy())
    print(f"[extract {rank}] {len(ids)} volumes in {time.time() - t0:.0f}s", flush=True)


def load_embeddings(out_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shards = sorted(glob.glob(str(Path(out_dir) / "emb_shard*.npz")))
    if not shards:
        raise FileNotFoundError(f"no emb_shard*.npz in {out_dir}")
    parts = [np.load(s) for s in shards]
    ids = np.concatenate([p["ids"] for p in parts])
    img = np.concatenate([p["img"] for p in parts])
    txt = np.concatenate([p["txt"] for p in parts])
    order = np.argsort(ids)
    return ids[order], img[order], txt[order]


def linear_probe(ids, X, manifest, findings_csv, five_year_csv) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
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
    results = []
    for task, df, cols in (("findings", fi, fi_cols), ("five_year", fy, FIVE_YEAR)):
        df = df.drop_duplicates("sid")
        df = df[df["sid"].isin(row)]
        Xt = X[[row[s] for s in df["sid"]]]
        split = df["_split"].to_numpy()
        for col in cols:
            y_raw = df[col].astype(int).to_numpy()
            if task == "findings":
                valid, y = y_raw != -1, (y_raw > 0).astype(int)
            else:
                valid, y = np.isin(y_raw, [0, 2]), (y_raw == 2).astype(int)
            tr, va, te = (valid & (split == s) for s in ("train", "val", "test"))
            rec = {"task": task, "label": col, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                   "auroc_val": np.nan, "auroc_test": np.nan}
            if len(np.unique(y[tr])) == 2:
                sc = StandardScaler().fit(Xt[tr])
                clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xt[tr]), y[tr])
                for key, m in (("auroc_val", va), ("auroc_test", te)):
                    if len(np.unique(y[m])) == 2:
                        rec[key] = float(roc_auc_score(y[m], clf.predict_proba(sc.transform(Xt[m]))[:, 1]))
            results.append(rec)
    return pd.DataFrame(results)


@torch.no_grad()
def report_multi(args, model, cfg, ids, img, txt) -> None:
    from eval.benchmarks import build_labels, linear_probes, retrieval_suite, zero_shot_long
    man = pd.read_csv(cfg["data"]["manifest"], dtype={"study_id": str})
    split_of, src_of = dict(zip(man["study_id"], man["split"])), dict(zip(man["study_id"], man["source"]))
    ev = cfg["eval"]
    labels = (pd.read_csv(ev["labels"], dtype={"study_id": str}) if ev.get("labels") and Path(ev["labels"]).exists()
              else build_labels(man))
    texts = TextEmbeddings(cfg["data"]["text_emb_dir"], cfg["data"]["text_pooling"])

    @torch.no_grad()
    def text_fn(sids):
        return model.encode_text(torch.from_numpy(np.stack([texts[s] for s in sids]))).float()

    ids = [str(s) for s in ids]
    row = {s: i for i, s in enumerate(ids)}
    res = {"ckpt": args.ckpt, "run_name": cfg["run_name"], "arm": cfg["windowing"]["arm"], "benchmark": "multi"}
    gb, scale = cfg["training"]["global_batch"], float(model.scale())
    for src in sorted(man["source"].unique()):
        t = [row[s] for s in ids if split_of.get(s) == "test" and src_of.get(s) == src]
        if len(t) < 2:
            continue
        it, tt = torch.from_numpy(img[t]).float(), torch.from_numpy(txt[t]).float()
        res[src] = {"n_test": len(t), **retrieval_suite(F.normalize(it, dim=-1), F.normalize(tt, dim=-1)),
                    "test InfoNCE@batch": batched_infonce(it, tt, scale, gb)}
    out = Path(args.out_dir)
    zs = zero_shot_long(dict(zip(ids, img)), text_fn, labels, split_of, set(texts.row))
    zs.to_csv(out / "zeroshot_long.csv", index=False)
    lp = None if args.skip_probe else linear_probes(ids, img, labels, split_of)
    if lp is not None:
        lp.to_csv(out / "linear_probe.csv", index=False)
    for src in [k for k, v in res.items() if isinstance(v, dict)]:
        res[src]["zero-shot long AUROC (findings)"] = float(zs[zs["source"] == src]["auroc_long"].mean())
        if lp is not None:
            l_ = lp[lp["source"] == src]
            res[src]["probe AUROC (findings)"] = float(l_[l_["kind"] == "finding"]["auroc"].mean())
            res[src]["probe AUROC (outcomes)"] = float(l_[l_["kind"] == "outcome"]["auroc"].mean())
    mt = [s for s in ids if split_of.get(s) == "test" and src_of.get(s) == "merlin"]
    if mt:                                           # our usual short-prompt zero-shot, Merlin findings
        zsm, per = zero_shot_findings(model, torch.from_numpy(img[[row[s] for s in mt]]), mt, ev["prompts"],
                                      ev["findings_csv"], cfg["data"]["text_pooling"])
        res["merlin"]["zero-shot short AUROC"] = zsm["zs_macro_auroc"]
        per.to_csv(out / "zeroshot_test.csv", index=False)
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def report(args) -> None:
    model, cfg = load_model(args.ckpt, "cpu")
    ids, img, txt = load_embeddings(args.out_dir)
    if cfg.get("eval", {}).get("benchmark") == "multi":
        return report_multi(args, model, cfg, ids, img, txt)
    man = pd.read_csv(cfg["data"]["manifest"])
    split_of = dict(zip(man["study_id"].astype(str), man["split"].astype(str)))
    test = np.array([split_of.get(s) == "test" for s in ids])
    res = {"ckpt": args.ckpt, "run_name": cfg["run_name"], "arm": cfg["windowing"]["arm"],
           "n_test": int(test.sum())}

    it, tt = torch.from_numpy(img[test]), torch.from_numpy(txt[test])
    res |= {f"test_{k}": v for k, v in retrieval_metrics(it, tt).items()}
    gb = cfg["training"]["global_batch"]
    res["test_loss_b"] = batched_infonce(it, tt, float(model.scale()), gb)
    chunks = [retrieval_metrics(it[s:s + gb], tt[s:s + gb]) for s in range(0, len(it) - gb + 1, gb)]
    res |= {f"test_b{gb}_{k}": float(np.mean([c[k] for c in chunks])) for k in chunks[0] if k != "n_candidates"}

    ev = cfg["eval"]
    zs, per = zero_shot_findings(model, it, [s for s, m in zip(ids, test) if m], ev["prompts"], ev["findings_csv"],
                                 cfg["data"]["text_pooling"])
    res |= {f"test_{k}": v for k, v in zs.items()}
    out = Path(args.out_dir)
    per.to_csv(out / "zeroshot_test.csv", index=False)

    if not args.skip_probe:
        lp = linear_probe(ids, img, cfg["data"]["manifest"], ev["findings_csv"], ev["five_year_csv"])
        lp.to_csv(out / "linear_probe.csv", index=False)
        for task in ("findings", "five_year"):
            res[f"probe_{task}_macro_auroc_test"] = float(lp[lp.task == task]["auroc_test"].mean())
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "report"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="extract: splits to embed (the probe needs train+val+test)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--max-items", type=int, default=None, help="debug: fixed random subset of the splits")
    ap.add_argument("--max-train-per-source", type=int, default=None,
                    help="extract: embed only N random train volumes per source (probe training sets)")
    args = ap.parse_args()
    extract(args) if args.phase == "extract" else report(args)


if __name__ == "__main__":
    main()
