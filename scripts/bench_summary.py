"""Summary of the scripts/bench_mi300.sh campaign: steady-state s/step, data wait, peak memory, projected cost
of a full mix run (4 840 steps), and loss agreement between runs with the same seed (MI300A vs MI250X).

    python scripts/bench_summary.py [--root $SCRATCHDIR/mix_clip_runs/_bench] [--steps 4840]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def records(d: Path) -> list[dict]:
    f = d / "metrics.jsonl"
    if not f.exists():
        return []
    return [r for r in map(json.loads, f.read_text().splitlines()) if "loss" in r and "type" not in r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCHDIR/mix_clip_runs/_bench"))
    ap.add_argument("--extra", nargs="*", default=[os.path.expandvars("$SCRATCHDIR/mix_clip_runs/_bench4/mix_patch_bank")])
    ap.add_argument("--steps", type=int, default=4840, help="steps of a full mix run (8 epochs)")
    ap.add_argument("--skip", type=int, default=5, help="first steps excluded from the steady state")
    args = ap.parse_args()
    dirs = sorted(p for p in Path(args.root).glob("*") if p.is_dir()) + [Path(p) for p in args.extra if Path(p).is_dir()]
    rows, losses = [], {}
    for d in dirs:
        rec = records(d)
        name = d.name if d.parent.name == "_bench" else "m250_4n_mb1 (_bench4)"
        m = re.match(r"m(300|250)_(\d+)n", name)
        if not rec or not m:
            rows.append({"run": name, "status": "no metrics yet"})
            continue
        gpu, nodes = ("MI300A" if m.group(1) == "300" else "MI250X"), int(m.group(2))
        st = [r for r in rec if r["step"] > args.skip] or rec
        sps = float(np.median([r["s_per_step"] for r in st]))
        rows.append({"run": name, "gpu": gpu, "nodes": nodes, "steps": rec[-1]["step"],
                     "s/step": sps, "data wait s": float(np.median([r["data_wait_s"] for r in st])),
                     "peak GB": max(r["mem_gb"] for r in rec),
                     "full run h": sps * args.steps / 3600, "node-h / run": sps * args.steps / 3600 * nodes,
                     "status": "ok"})
        losses[name] = {r["step"]: (r["loss"], r["gnorm"]) for r in rec}
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
        print(df.to_string(index=False))
    # same seed and global batch on both GPUs: losses should agree up to bf16 / reduction-order noise
    for a, b in (("m300_2n_mb1", "m250_2n_mb1"), ("m300_8n_mb1", "m250_8n_mb1"), ("m300_1n_mb1", "m300_2n_mb1")):
        if a in losses and b in losses:
            common = sorted(set(losses[a]) & set(losses[b]))
            dl = [abs(losses[a][s][0] - losses[b][s][0]) for s in common]
            print(f"loss |{a} - {b}| over {len(common)} steps: max {max(dl):.4f}, mean {np.mean(dl):.4f} "
                  f"(step 1: {losses[a][common[0]][0]:.4f} vs {losses[b][common[0]][0]:.4f})")


if __name__ == "__main__":
    main()
