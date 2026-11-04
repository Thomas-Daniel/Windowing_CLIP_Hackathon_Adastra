"""Gather the test evaluations of all runs into one table (runs/eval_summary.csv + terminal).

Reads <runs_root>/<run>/eval_<ckpt>/summary.json written by `eval.evaluate report`.

    python scripts/compare_evals.py                    # ckpt_final of every run
    python scripts/compare_evals.py --ckpt ckpt_epoch010
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

COLUMNS = [  # (summary key, short name)
    ("test_loss_b", "test InfoNCE@128"),
    ("test_b128_i2t_R@1", "R@1 i2t (128)"), ("test_b128_t2i_R@1", "R@1 t2i (128)"),
    ("test_b128_i2t_R@10", "R@10 i2t (128)"), ("test_b128_t2i_R@10", "R@10 t2i (128)"),
    ("test_i2t_R@10", "R@10 i2t (all)"), ("test_t2i_R@10", "R@10 t2i (all)"),
    ("test_zs_macro_auroc", "zero-shot AUROC"),
    ("probe_findings_macro_auroc_test", "probe findings AUROC"),
    ("probe_five_year_macro_auroc_test", "probe 5-year AUROC"),
]


MULTI = ["n_test", "R@1 i2t (pool100)", "R@1 t2i (pool100)", "R@10 i2t (pool100)", "R@10 t2i (pool100)",
         "R@10 i2t (all)", "R@10 t2i (all)", "test InfoNCE@batch", "zero-shot long AUROC (findings)",
         "zero-shot short AUROC", "probe AUROC (findings)", "probe AUROC (outcomes)"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCHDIR/merlin_clip_runs"))
    ap.add_argument("--ckpt", default="ckpt_final")
    args = ap.parse_args()
    root = Path(args.root)
    rows, multi = [], []
    for f in sorted(root.glob(f"*/eval_{args.ckpt}/summary.json")):
        if f.parts[-3].startswith("_"):
            continue
        s = json.loads(f.read_text())
        if s.get("benchmark") == "multi":            # chest / abdomen mix: one row per run and test source
            for src, v in s.items():
                if isinstance(v, dict):
                    multi.append({"run": s["run_name"], "arm": s["arm"], "source": src, **{k: v.get(k) for k in MULTI}})
            continue
        rows.append({"run": s["run_name"], "arm": s["arm"], "n_test": s["n_test"],
                     **{name: s.get(key) for key, name in COLUMNS}})
    if multi:
        dm = pd.DataFrame(multi).sort_values(["source", "run"])
        dm.to_csv(root / f"eval_summary_{args.ckpt}_multi.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
            print(dm.to_string(index=False))
        print(f"-> {root / f'eval_summary_{args.ckpt}_multi.csv'}")
    if not rows:
        if not multi:
            print(f"no eval_{args.ckpt}/summary.json under {root} yet")
        return
    df = pd.DataFrame(rows).sort_values("run")
    out = root / f"eval_summary_{args.ckpt}.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    print(f"\n(R@k (128): mean over chunks of 128 test pairs, the training batch; (all): among all "
          f"{int(df.n_test.max())} test pairs)\n-> {out}")


if __name__ == "__main__":
    main()
