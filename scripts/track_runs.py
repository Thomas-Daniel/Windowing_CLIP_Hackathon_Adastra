"""Track the windowing runs: status table in the terminal + a figure of the curves.

Reads <runs_root>/<run>/metrics.jsonl (written by train.py), works while the runs are going.
train.py also runs it in the background (rank 0, every training.track_every steps and after each
validation), so <runs_root>/progress.png and progress.txt stay current without anyone polling.

    python scripts/track_runs.py                       # all runs under <runs_root>: table + <runs_root>/progress.png
    python scripts/track_runs.py --runs win1_clip win4_local --out /tmp/p.png
    python scripts/track_runs.py --watch 30            # refresh every 30 min

Table, per run: SLURM state, step / total, train loss (last 10-step window and EMA), last
validation (InfoNCE at batch 128, R@10 image->report and report->image, zero-shot findings
AUROC), logit scale, s/step, ETA, and events (resumes, skipped non-finite steps).
Figure (one panel per measure, same colour per run everywhere): train InfoNCE, val InfoNCE,
val R@10 both ways, val zero-shot AUROC, logit scale, gradient norm, learned-curve drift
(learned / local).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

RUNS = ["win1_clip", "win2_pillar11", "win3_learned", "win4_local", "win5_abdo"]
# fixed colour per run (categorical slots 1..6 of the reference palette, light mode): identity
# follows the run, never its rank or the set of runs shown
COLORS = {"win1_clip": "#2a78d6", "win2_pillar11": "#eb6834", "win3_learned": "#1baf7a",
          "win4_local": "#eda100", "win5_abdo": "#e87ba4", "win6_dense": "#008300",
          "win3c_learned_warp": "#6250d6", "win4c_local_warp": "#e34948",
          "win8_patch_bank": "#2a78d6", "win9_patch_cdf": "#eb6834", "win10_patch_eq": "#1baf7a",
          "win11_global_bank": "#eda100", "win12_patch_bank_shuf": "#e87ba4"}
# channel names of the learned curves (init: the 10 Pillar-0 presets + the full ramp)
CURVE_NAMES = ["lung", "mediastinum", "abdomen", "liver", "bone", "brain", "subdural", "stroke",
               "temporal bone", "soft tissue", "full ramp"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CHANCE_LOSS = math.log(128)            # InfoNCE of a random model at batch 128


def load(run_dir: Path) -> dict:
    out = {"train": [], "val": [], "events": [], "cfg": None}
    f = run_dir / "metrics.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = r.get("type")
            (out["train"] if kind is None else out["val"] if kind == "val" else out["events"]).append(r)
    c = run_dir / "config_resolved.json"
    if c.exists():
        out["cfg"] = json.loads(c.read_text())
    return out


def ema(xs: list[float], alpha: float = 0.15) -> list[float]:
    out, m = [], None
    for x in xs:
        m = x if m is None else alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def slurm_states() -> dict[str, str]:
    """{job name: state} for the user's jobs (empty if squeue is unavailable)."""
    try:
        txt = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%T|%M|%L"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return {}
    states = {}
    for line in txt.splitlines():
        name, state, elapsed, left = line.split("|")
        if not name.endswith("_resume") or name not in states:
            states.setdefault(name, f"{state} {elapsed} (left {left})" if state == "RUNNING" else state)
    return states


def total_steps(d: dict) -> int | None:
    cfg = d["cfg"]
    if not cfg:
        return None
    t = cfg["training"]
    n_train = 15300 if not cfg["data"].get("max_train_items") else cfg["data"]["max_train_items"]
    return (n_train // t["global_batch"]) * t["epochs"]


def table(runs: list[str], data: dict, states: dict) -> str:
    hdr = (f"{'run':16s} {'state':28s} {'step':>11s} {'epoch':>6s} {'loss':>6s} {'ema':>6s} | "
           f"{'val@step':>8s} {'vloss':>6s} {'R@10 i2t':>8s} {'R@10 t2i':>8s} {'zs AUC':>6s} | "
           f"{'scale':>6s} {'s/step':>6s} {'ETA':>7s} | {'curve Δ':>7s} {'gate act':>8s}  events")
    lines = [hdr, "-" * len(hdr)]
    for run in runs:
        d = data[run]
        st = states.get(run, "not in queue")
        tr, va = d["train"], d["val"]
        if not tr:
            lines.append(f"{run:16s} {st:28s} {'no metrics yet':>11s}")
            continue
        last = tr[-1]
        tot = total_steps(d)
        e = ema([r["loss"] for r in tr])[-1]
        sps = sorted(r["s_per_step"] for r in tr[-20:])[len(tr[-20:]) // 2]
        eta = (tot - last["step"]) * sps / 3600 if tot else float("nan")
        v = va[-1] if va else {}
        n_res = sum(1 for r in d["events"] if r.get("type") == "resume")
        n_nf = sum(1 for r in d["events"] if r.get("type") == "nonfinite")
        chk = [r for r in d["events"] if r.get("type") == "window_check"]
        chk_s = ("window check OK" if chk[-1]["learning"] else "WINDOW NOT LEARNING") if chk else ""
        ev = ", ".join(x for x in (f"{n_res} resume(s)" if n_res else "", f"{n_nf} non-finite" if n_nf else "",
                                   chk_s) if x)
        cd = last.get("win_curve_drift_max")
        ga = last.get("gate_active")
        ps = last.get("patch_center_std_hu", last.get("patch_median_std_hu"))
        if cd is None and ps is not None:            # patch arms: spread of the windows across patches (HU)
            cd = ps
        lines.append(
            f"{run:16s} {st[:28]:28s} {last['step']:>5d}/{tot or '?':<5} {last['epoch']:6.2f} {last['loss']:6.3f} {e:6.3f} | "
            f"{v.get('step', '-'):>8} {v.get('val_loss_b', float('nan')):6.3f} {v.get('val_i2t_R@10', float('nan')):8.3f} "
            f"{v.get('val_t2i_R@10', float('nan')):8.3f} {v.get('val_zs_macro_auroc', float('nan')):6.3f} | "
            f"{last['logit_scale']:6.2f} {sps:6.1f} {eta:6.1f}h | "
            f"{(f'{cd:.3f}' if cd is not None else '-'):>7s} {(f'{ga:.3f}' if ga is not None else '-'):>8s}  {ev or '-'}")
    lines.append(f"(loss = global-batch InfoNCE, chance ln 128 = {CHANCE_LOSS:.3f}; val on {data[runs[0]]['val'][-1]['n_val'] if data[runs[0]]['val'] else 'the fixed'} "
                 f"val volumes; R@10 among them; curve Δ = max change of the learned curves since init (patch arms: "
                 f"std of the windows across patches, HU), "
                 f"output units; gate act = share of local gates with |g - 1| > 0.1)")
    return "\n".join(lines)


def figure(runs: list[str], data: dict, out: Path, steps_per_epoch: int = 119) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    panels = [
        ("Train InfoNCE (batch 128)", "train", "loss", True, CHANCE_LOSS),
        ("Val InfoNCE (batch 128)", "val", "val_loss_b", False, CHANCE_LOSS),
        ("Val R@10, image → report", "val", "val_i2t_R@10", False, None),
        ("Val R@10, report → image", "val", "val_t2i_R@10", False, None),
        ("Val zero-shot findings, macro AUROC", "val", "val_zs_macro_auroc", False, 0.5),
        ("Logit scale (1 / temperature)", "train", "logit_scale", False, None),
        ("Gradient norm (before clipping)", "train", "gnorm", True, None),
        ("Learned curves: max change since init (output units)", "train", "win_curve_drift_max", False, None),
        ("Local gate: share of gates with |g − 1| > 0.1", "train", "gate_active", False, None),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), constrained_layout=True)
    xmax = max([r["step"] for run in runs for r in data[run]["train"]] or [steps_per_epoch])
    for ax, (title, src, key, smooth, ref) in zip(axes.flat, panels):
        ax.set_xlim(0, xmax * 1.1)                     # same step axis on every panel, room for labels
        ax.set_title(title, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel(f"step ({steps_per_epoch} per epoch)")
        ends = []
        for run in runs:
            recs = [r for r in data[run][src] if key in r and r[key] is not None]
            if not recs:
                continue
            x = [r["step"] for r in recs]
            y = [r[key] for r in recs]
            c = COLORS.get(run, INK2)
            if smooth and len(y) > 3:
                ax.plot(x, y, color=c, linewidth=1, alpha=0.25)
                y = ema(y)
            ax.plot(x, y, color=c, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=run,
                    marker="o" if src == "val" else None, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ends.append((x[-1], y[-1], run))
        if ref is not None:
            ax.axhline(ref, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate("chance", (1, ref), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                        textcoords="offset points", fontsize=8, color=INK2, ha="right")
        # direct labels at the line ends (text in ink; the line beside it carries the colour)
        if 0 < len(ends) <= 5 and key != "gnorm":     # gnorm is noisy: legend only
            ys = sorted(ends, key=lambda e: e[1])
            lo, hi = ax.get_ylim()
            gap = (hi - lo) * 0.06
            placed = []
            for xe, ye, run in ys:
                yl = max(ye, placed[-1] + gap) if placed else ye
                placed.append(yl)
                ax.annotate(run.split("_", 1)[1], (xe, ye), xytext=(6, (yl - ye) / (hi - lo) * 200),
                            textcoords="offset points", fontsize=8, color=INK2, va="center")
        if not ends:
            ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", va="center", color=INK2)
    handles, labels = [], []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    if handles:
        fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False,
                   labelcolor=INK)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}{out.suffix}")   # atomic: several runs refresh it
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, out)


def curves_figure(run: str, d: dict, out: Path) -> bool:
    """Per learned channel: init curve (dashed) vs the latest logged one, for one run."""
    init = [r for r in d["events"] if r.get("type") == "win_init"]
    cur = [r for r in d["events"] + d["val"] if r.get("win_curves") and r.get("type") != "win_init"]
    if not init or not cur:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    last = max(cur, key=lambda r: r["step"])
    hu = init[0]["win_curves"]["hu"]
    y0, y1 = init[0]["win_curves"]["out"], last["win_curves"]["out"]
    c = COLORS.get(run, INK)
    k = len(y0)
    ncol = 4
    nrow = (k + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3 * nrow), constrained_layout=True, facecolor=SURFACE)
    for i, ax in enumerate(axes.flat):
        if i >= k:
            ax.axis("off")
            continue
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.plot(hu, y0[i], color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), label="init")
        ax.plot(hu, y1[i], color=c, linewidth=2, label=f"step {last['step']}")
        drift = max(abs(a - b) for a, b in zip(y0[i], y1[i]))
        name = CURVE_NAMES[i] if i < len(CURVE_NAMES) else f"channel {i}"
        warp = ""
        if "shift_hu" in last["win_curves"]:
            warp = f"\nshift {last['win_curves']['shift_hu'][i]:+.0f} HU, width ×{last['win_curves']['scale'][i]:.2f}"
        ax.set_title(f"{name}  (max Δ {drift:.3f}){warp}", loc="left", fontsize=9, color=INK)
        ax.set_xlim(hu[0], hu[-1])
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(colors=INK2, labelsize=8)
    axes.flat[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(f"{run}: learned transfer curves, HU → [0, 1] (init dashed, step {last['step']} solid)",
                 color=INK, fontsize=11)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}{out.suffix}")
    fig.savefig(tmp, dpi=100)
    plt.close(fig)
    os.replace(tmp, out)
    return True


def discover(root: Path) -> list[str]:
    """Run directories under root (with a metrics.jsonl; names starting with '_' are tests),
    win1..win4 first in study order, then any other."""
    found = sorted(p.parent.name for p in root.glob("*/metrics.jsonl") if not p.parent.name.startswith("_"))
    order = list(COLORS)
    return sorted(found, key=lambda r: (order.index(r) if r in order else len(order), r)) or RUNS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCHDIR/merlin_clip_runs"))
    ap.add_argument("--runs", nargs="+", default=None, help="default: every run found under --root")
    ap.add_argument("--out", default=None, help="figure path (default <root>/progress.png)")
    ap.add_argument("--table-out", default=None, help="also write the table to this file")
    ap.add_argument("--quiet", action="store_true", help="no terminal output (background refresh)")
    ap.add_argument("--watch", type=float, default=0, help="refresh every N minutes")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "progress.png"
    while True:
        runs = args.runs or discover(root)
        data = {r: load(root / r) for r in runs}
        text = time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + table(runs, data, slurm_states())
        if not args.quiet:
            print(text)
        if args.table_out:
            tp = Path(args.table_out)
            tmp = tp.with_name(f".{tp.name}.{os.getpid()}")
            tmp.write_text(text + "\n")
            os.replace(tmp, tp)
        try:
            figure(runs, data, out)
            if not args.quiet:
                print(f"figure: {out}")
            for r in runs:                       # learned runs: their curves, init vs now
                if curves_figure(r, data[r], root / r / "curves.png") and not args.quiet:
                    print(f"curves: {root / r / 'curves.png'}")
        except Exception as exc:  # noqa: BLE001 -- the table is still useful without the figure
            print(f"figure not written: {exc}", file=sys.stderr)
        if not args.watch:
            break
        time.sleep(args.watch * 60)


if __name__ == "__main__":
    main()
