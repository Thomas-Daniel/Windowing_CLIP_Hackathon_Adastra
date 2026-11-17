"""CLIP training of Atlas on Merlin volumes against frozen Qwen3-Embedding report embeddings.

One run per windowing arm (configs/win*.yaml); everything else comes from
configs/base.yaml and is identical across runs: data, backbone, global batch, steps,
optimiser, augmentation, seed (hence data order).

Global batch = world x accum_steps x micro_batch volumes per optimizer step. The
contrastive loss is computed over the whole global batch with a gradient cache
(Gao et al. 2021): (1) forward every micro-batch without grad, (2) compute the global
InfoNCE on the gathered features and backprop to the features only, (3) re-forward each
micro-batch with grad and backprop the cached feature gradient. Equivalent to one
forward/backward of the full batch; costs one extra forward. Gradients are then summed
over ranks by hand (no DDP wrapper).

Checkpoints and resume (output_dir is stable, derived from run_name):
  ckpt_last.pt   every training.ckpt_every steps, at every epoch end, at the stop step and on
                 SIGUSR1 (sent by SLURM 15 min before the time limit, see submit_train.slurm);
                 the previous one is kept as ckpt_prev.pt (fallback if ckpt_last.pt is unreadable)
  ckpt_epochXXX.pt  every training.keep_every_epochs epochs;  ckpt_final.pt  at the last step
A checkpoint holds the model (BN running stats, arm curves / gate, logit scale), the AdamW state
and the step. Data order, LR and augmentation are functions of the step, so resuming continues
the exact same run. On resume, metrics.jsonl records after the checkpoint step (steps redone
after a crash) move to metrics_dropped.jsonl, and a {"type": "resume"} record is added.
Resubmitting the same command resumes; a finished run exits at once.

Profiling (training.profile=true, off by default): at every log step, a "prof" log line and a
{"type": "profile"} record in metrics.jsonl with the time of each phase of a step (max / min over
ranks, slowest rank) and the node memory pressure (free memory, page cache, reclaim counters).
See StepProfiler.

Usage (one GPU per process, see scripts/submit_train.slurm):
    python train.py --config configs/win2_pillar11.yaml [--set training.epochs=2 ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

torch.backends.cuda.preferred_blas_library("cublas")

from mcw import dist as D
from mcw.config import load_config, set_by_dotted
from mcw.data import BalancedShardSampler, EpochShardSampler, MerlinClipDataset, TextEmbeddings, collate, random_axial_rotation
from mcw.evaluate import batched_infonce, encode_dataset, eval_sampler_indices, retrieval_metrics, zero_shot_findings
from mcw.loss import clip_loss
from mcw.model import CTClip
from mcw.windowing_arms import param_groups


def log(msg: str) -> None:
    if D.rank() == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Profiling (training.profile=true)
# ---------------------------------------------------------------------------
PROFILE_PHASES = ("data", "h2d_aug", "fwd_nograd", "loss", "fwd_bwd", "allreduce", "optim", "misc")
VM_KEYS = ("pgscan_direct", "pgscan_kswapd", "allocstall", "compact_stall")


def _read_proc(path: str) -> dict[str, int]:
    """/proc/meminfo ("MemFree:  123 kB") or /proc/vmstat ("pgscan_direct 123") -> {key: int}."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    out[parts[0].rstrip(":")] = int(parts[1])
    except (OSError, ValueError):
        pass
    return out


def _vm_counters() -> list[float]:
    vm = _read_proc("/proc/vmstat")
    return [float(vm.get("pgscan_direct", 0)), float(vm.get("pgscan_kswapd", 0)),
            float(sum(v for k, v in vm.items() if k.startswith("allocstall"))),    # allocstall_{dma,normal,movable,...}
            float(vm.get("compact_stall", 0))]


class StepProfiler:
    """Time of each phase of a training step, the GPU synchronised at every phase boundary (a few
    ms per step), averaged over the log window on each rank, then gathered over ranks:
      data        wait for the DataLoader            h2d_aug     host -> GPU copy + rotation
      fwd_nograd  (1) forwards without grad           loss        (2) global InfoNCE + backward to the features
      fwd_bwd     (3) re-forwards + backwards         allreduce   gradient all-reduce
      optim       clip, AdamW, idx_sum                misc        logging, stop check, val / ckpt
    A rank waiting for a slower one has a large 'allreduce': compare its max and min over ranks.
    Node memory over the window (ranks of one node see the same values; max / min over nodes):
    free memory and page cache (/proc/meminfo, GB), and the increase of the /proc/vmstat reclaim
    counters: pgscan_direct / allocstall / compact_stall > 0 = processes held up to free memory."""

    def __init__(self, enabled: bool, device: torch.device):
        self.on, self.device = enabled, device
        self.sums = dict.fromkeys(PROFILE_PHASES, 0.0)
        self.n, self.t = 0, 0.0
        self.vm_prev = _vm_counters() if enabled else None

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self, data_wait: float) -> None:
        if self.on:
            self.sums["data"] += data_wait
            self._sync()
            self.t = time.perf_counter()

    def mark(self, phase: str, end_of_step: bool = False) -> None:
        if self.on:
            self._sync()
            now = time.perf_counter()
            self.sums[phase] += now - self.t
            self.t = now
            self.n += end_of_step

    def report(self, step: int) -> dict | None:
        """Collective: call on every rank (log steps are the same on every rank)."""
        if not self.on or self.n == 0:
            return None
        mem = _read_proc("/proc/meminfo")
        vm = _vm_counters()
        dvm = [a - b for a, b in zip(vm, self.vm_prev)]
        self.vm_prev = vm
        phases = [self.sums[p] / self.n for p in PROFILE_PHASES]
        local = torch.tensor(phases + [sum(phases), mem.get("MemFree", 0) / 2**20, mem.get("Cached", 0) / 2**20] + dvm,
                             dtype=torch.float64, device=self.device)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, local)
            allv = torch.stack(parts).cpu()
        else:
            allv = local.cpu()[None]
        self.sums, self.n = dict.fromkeys(PROFILE_PHASES, 0.0), 0

        rec = {"type": "profile", "step": step, "n_ranks": int(allv.shape[0])}
        names = (*PROFILE_PHASES, "step")
        for k, name in enumerate(names):
            col = allv[:, k]
            rec |= {f"t_{name}_max": float(col.max()), f"t_{name}_min": float(col.min()),
                    f"t_{name}_argmax": int(col.argmax())}
        rec["t_step_per_rank"] = [round(float(x), 3) for x in allv[:, len(names) - 1]]
        rec["t_allreduce_per_rank"] = [round(float(x), 3) for x in allv[:, PROFILE_PHASES.index("allreduce")]]
        j = len(names)
        rec["node_free_gb_min"] = float(allv[:, j].min())
        rec["node_cache_gb_max"] = float(allv[:, j + 1].max())
        for i, key in enumerate(VM_KEYS):
            rec[f"{key}_max"] = float(allv[:, j + 2 + i].max())
        return rec


def profile_line(rec: dict) -> str:
    ph = " ".join(f"{p}={rec[f't_{p}_max']:.3f}/{rec[f't_{p}_min']:.3f}(r{rec[f't_{p}_argmax']})"
                  for p in (*PROFILE_PHASES, "step"))
    return (f"prof [s/step max/min(slowest rank)] {ph} | node free_gb_min={rec['node_free_gb_min']:.0f} "
            f"cache_gb_max={rec['node_cache_gb_max']:.0f} " + " ".join(f"{k}={rec[k + '_max']:.0f}" for k in VM_KEYS))


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    e = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, e)))


def build_optimizer(model: CTClip, cfg: dict) -> torch.optim.Optimizer:
    """windowing_arms.param_groups (adapter = curves / gate: no weight decay, own LR), with each
    group's LR expressed as a multiple of the base LR for the cosine schedule."""
    o, w = cfg["optim"], cfg["windowing"]
    groups = param_groups(model, o["lr"], o["weight_decay"], window_lr=w.get("window_lr"))
    gate = [p for n, p in model.named_parameters() if n.startswith(("adapter.gate.", "adapter.net.", "adapter.scan_net."))]
    if w.get("gate_lr") and gate:            # local / patch arms: the per-patch network gets its own LR
        ids = {id(p) for p in gate}
        for g in groups:
            g["params"] = [p for p in g["params"] if id(p) not in ids]
        groups = [g for g in groups if g["params"]] + [dict(params=gate, lr=w["gate_lr"], weight_decay=0.0)]
    for g in groups:
        g["lr_mult"] = g["lr"] / o["lr"]
    return torch.optim.AdamW(groups, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])


def window_stats(model: CTClip, ref: dict) -> dict:
    """Is the arm learning? Learned curves:
      win_curve_drift_max / _mean  largest / mean change of the transfer curves since init, in
                                   output units ([0, 1]; 0.1 = a tenth of the dynamic range)
      win_dtheta_max               largest logit change since init
      win_dtheta_rate              mean |logit change| per step since the previous log
      win_entropy_mean / _min      normalised entropy of the bin masses (1 = uniform)
    Local arms (last gate g, 1 everywhere at init): mean / std / min / max and gate_active, the
    share of patch x channel gates with |g - 1| > 0.1."""
    out = {}
    lw = model.learned_window()
    if lw is not None:
        th = lw.theta.detach()
        p = th.float().softmax(-1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(p.shape[-1])
        _, ys = lw.curves(n=161)
        drift = (ys.float() - ref["curves0"].to(ys.device)).abs()
        steps = max(1, ref["step"] - ref.get("prev_step", 0))
        out |= {"win_curve_drift_max": float(drift.max()), "win_curve_drift_mean": float(drift.mean()),
                "win_dtheta_max": float((th - ref["theta0"].to(th.device)).abs().max()),
                "win_dtheta_rate": float((th - ref["theta_prev"].to(th.device)).abs().mean()) / steps,
                "win_entropy_mean": float(ent.mean()), "win_entropy_min": float(ent.min())}
        ss = lw.window_shift_scale()
        if ss is not None:                           # warp: per-channel HU shift and width factor
            out |= {"win_shift_max_hu": float(ss[0].abs().max()), "win_scale_min": float(ss[1].min()),
                    "win_scale_max": float(ss[1].max())}
        ref["theta_prev"], ref["prev_step"] = th.clone(), ref["step"]
    out |= getattr(model.adapter, "last_stats", {}) or {}        # patch arms: spread of the per-patch windows
    g = getattr(model.adapter, "last_gate", None)
    if g is not None:
        out |= {"gate_mean": float(g.mean()), "gate_std": float(g.std()), "gate_min": float(g.min()),
                "gate_max": float(g.max()), "gate_active": float(((g - 1).abs() > 0.1).float().mean())}
    return out


def window_bounds(model: CTClip) -> list | None:
    """Per learned channel, the HU where its curve crosses 0.1 and 0.9 (the effective window)."""
    lw = model.learned_window()
    if lw is None:
        return None
    xs, ys = lw.curves(n=2001)
    xs, ys = xs.cpu(), ys.float().cpu()
    return [[float(xs[(y >= 0.1).nonzero()[0, 0]]), float(xs[(y >= 0.9).nonzero()[0, 0]])] for y in ys]


def window_curves(model: CTClip) -> dict:
    """Current transfer curves of the learned channels on [-1000, 1000] HU (161 points)."""
    lw = model.learned_window()
    if lw is None:
        return {}
    xs, ys = lw.curves(n=161)
    out = {"hu": xs.cpu().tolist(), "out": ys.float().cpu().tolist()}
    ss = lw.window_shift_scale()
    if ss is not None:
        out |= {"shift_hu": ss[0].cpu().tolist(), "scale": ss[1].cpu().tolist()}
    return out


STOP = {"signal": None}
TRACKER: dict[str, subprocess.Popen | None] = {"proc": None}


def refresh_tracking(out_dir: Path) -> None:
    """Rank 0: refresh <runs root>/progress.png and progress.txt (all runs) with
    scripts/track_runs.py in a detached background process; skipped if one is still running."""
    p = TRACKER["proc"]
    if p is not None and p.poll() is None:
        return
    root = out_dir.parent
    script = Path(__file__).resolve().parent / "scripts" / "track_runs.py"
    try:
        TRACKER["proc"] = subprocess.Popen(
            [sys.executable, str(script), "--root", str(root), "--out", str(root / "progress.png"),
             "--table-out", str(root / "progress.txt"), "--quiet"],
            stdout=subprocess.DEVNULL, stderr=open(out_dir / "tracker.err", "a"), start_new_session=True)
    except OSError as exc:
        log(f"WARNING: progress tracking not refreshed ({exc})")


def _request_stop(signum, frame) -> None:      # SIGUSR1 from SLURM (--signal=USR1@900): save and exit
    STOP["signal"] = signum


def save_ckpt(path: Path, model, opt, step, epoch, cfg, keep_previous: bool = False) -> None:
    """Atomic write (temp file + rename). keep_previous: the replaced file becomes ckpt_prev.pt."""
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "step": step, "epoch": epoch,
                "cfg": cfg, "world": D.world_size(), "job_id": os.environ.get("SLURM_JOB_ID"),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, tmp)
    if keep_previous and path.exists():
        path.replace(path.with_name("ckpt_prev.pt"))
    tmp.replace(path)


def load_resume_ckpt(out_dir: Path) -> tuple[dict | None, Path | None]:
    for name in ("ckpt_last.pt", "ckpt_prev.pt"):
        p = out_dir / name
        if p.exists():
            try:
                return torch.load(p, map_location="cpu", weights_only=False), p
            except Exception as exc:  # noqa: BLE001 -- unreadable: fall back to the previous one
                log(f"WARNING: cannot read {p} ({str(exc)[:120]}), trying the previous checkpoint")
    return None, None


def truncate_metrics(path: Path, step: int) -> int:
    """Keep the metrics records up to `step` (the resumed checkpoint); the others, logged by a
    job that died after its last checkpoint and about to be redone, go to metrics_dropped.jsonl."""
    if not path.exists():
        return 0
    keep, drop = [], []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:                  # line cut by a kill
            drop.append(line)
            continue
        (keep if rec.get("step", 0) <= step else drop).append(line)
    if drop:
        with open(path.with_name("metrics_dropped.jsonl"), "a") as f:
            f.write("\n".join(drop) + "\n")
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(line + "\n" for line in keep))
        tmp.replace(path)
    return len(drop)


@torch.no_grad()
def validate(model, loader, cfg, device, step, out_dir) -> dict:
    t0 = time.time()
    ids, img, txt = encode_dataset(model, loader, device)
    res = {"step": step, "n_val": len(ids)}
    res |= {f"val_{k}": v for k, v in retrieval_metrics(img, txt).items()}
    res["val_loss_b"] = batched_infonce(img, txt, float(model.scale()), cfg["training"]["global_batch"])
    ev = cfg.get("eval", {})
    if ev.get("prompts") and ev.get("findings_csv") and Path(ev["prompts"]).exists():
        zs, per = zero_shot_findings(model, img, ids, ev["prompts"], ev["findings_csv"], cfg["data"]["text_pooling"])
        res |= {f"val_{k}": v for k, v in zs.items()}
        if D.rank() == 0:
            per.to_csv(out_dir / f"val_zeroshot_step{step:07d}.csv", index=False)
    curves = window_curves(model)
    if curves:
        res["win_curves"] = curves
        res["win_bounds"] = window_bounds(model)
    res["val_time_s"] = round(time.time() - t0, 1)
    model.train()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. training.epochs=2")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_by_dotted(cfg, k, v)
    if os.environ.get("MCW_CUDNN_BENCHMARK") == "1":  # model.conv_impl native: MIOpen Find instead of immediate mode
        torch.backends.cudnn.benchmark = True
    signal.signal(signal.SIGUSR1, _request_stop)     # before any DataLoader worker is forked
    rank, world, local = D.setup()
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    t, dcfg = cfg["training"], cfg["data"]
    out_dir = Path(args.output_dir or Path(cfg["output_root"]) / cfg["run_name"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_resolved.json").write_text(json.dumps(cfg, indent=2))
    D.barrier()

    mb = t["micro_batch"]
    accum = t.get("accum_steps") or t["global_batch"] // (world * mb)     # null = fill the global batch
    if world * mb * accum != t["global_batch"]:
        raise ValueError(f"world {world} x micro_batch {mb} x accum_steps {accum} != global_batch {t['global_batch']}: "
                         f"set training.accum_steps = {t['global_batch'] // (world * mb)}")
    local_batch = mb * accum
    torch.manual_seed(cfg["seed"])                       # same init on every rank (no broadcast needed)
    np.random.seed(cfg["seed"])

    texts = TextEmbeddings(dcfg["text_emb_dir"], dcfg["text_pooling"])
    crop = dict(crop=dcfg.get("crop"), crop_seed=cfg["seed"],          # long runs: crop^3 cubes (val: centred)
                downsample=dcfg.get("downsample"))                     # or the whole volume resized by block mean
    train_ds = MerlinClipDataset(dcfg["manifest"], texts, dcfg["train_splits"], max_items=dcfg.get("max_train_items"),
                                 crop_jitter=int(dcfg.get("crop_jitter", 0)), **crop)
    val_ds = MerlinClipDataset(dcfg["manifest"], texts, dcfg["val_splits"], max_items=dcfg["val_max_items"], seed=cfg["seed"],
                               **crop)
    if t.get("balance_domains"):                      # chest / abdomen mix: same share of each domain in every batch
        if train_ds.domains is None:
            raise ValueError("training.balance_domains needs a 'domain' column in the manifest")
        sampler = BalancedShardSampler(train_ds.domains, t["global_batch"], rank, world, cfg["seed"])
    else:
        sampler = EpochShardSampler(len(train_ds), t["global_batch"], rank, world, cfg["seed"])
    steps_per_epoch = sampler.steps_per_epoch
    total_steps = t["epochs"] * steps_per_epoch
    stop_at = t.get("stop_at_step") or total_steps

    model = CTClip(cfg, texts.dim).to(device)
    opt = build_optimizer(model, cfg)
    log("optimizer groups: " + "; ".join(f"{sum(p.numel() for p in g['params'])} params lr {g['lr']:.2e} "
                                         f"wd {g['weight_decay']}" for g in opt.param_groups))
    lw = model.learned_window()
    win_ref = {}                                     # init curves (before any resume), for the drift stats
    if lw is not None:
        win_ref = {"theta0": lw.theta.detach().clone(), "theta_prev": lw.theta.detach().clone(),
                   "curves0": lw.curves(n=161)[1].float().clone(), "step": 0}
    params = [p for p in model.parameters() if p.requires_grad]
    n_vis = sum(p.numel() for p in model.visual.parameters())
    n_txt = sum(p.numel() for p in model.text.parameters())
    log(f"run {cfg['run_name']}: arm={model.arm} ({model.in_chans} ch to the stem"
        f"{', RampConv3d first conv' if model.arm == 'dense' else ''}), "
        f"Atlas {n_vis / 1e6:.2f}M params, text proj {n_txt / 1e6:.2f}M; train {len(train_ds)} vols, val {len(val_ds)}; "
        f"global batch {t['global_batch']} = {world} ranks x {accum} accum x {mb}; "
        f"{steps_per_epoch} steps/epoch x {t['epochs']} epochs = {total_steps} steps; out {out_dir}")

    step = 0
    ck_path = out_dir / "ckpt_last.pt"
    ck, ck_src = load_resume_ckpt(out_dir)
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        step = int(ck["step"])
        del ck
        if rank == 0:
            n_drop = truncate_metrics(out_dir / "metrics.jsonl", step)
            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"type": "resume", "step": step, "from": ck_src.name, "world": world,
                                    "job_id": os.environ.get("SLURM_JOB_ID"), "dropped_records": n_drop,
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
            log(f"resumed from {ck_src} at step {step} (epoch {step / steps_per_epoch:.2f}); "
                f"{n_drop} metrics records after that step moved to metrics_dropped.jsonl")
    D.barrier()
    if step >= stop_at:
        log("nothing to do (already at stop step)")
        D.cleanup()
        return

    loader_kw = dict(num_workers=t["num_workers"], collate_fn=collate, persistent_workers=False,
                     prefetch_factor=t.get("prefetch_factor", 2) if t["num_workers"] else None)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=t.get("eval_batch", 2), sampler=eval_sampler_indices(len(val_ds), rank, world), **loader_kw)

    metrics_f = open(out_dir / "metrics.jsonl", "a") if rank == 0 else None

    def emit(rec: dict) -> None:
        if metrics_f:
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()

    if lw is not None:
        win_ref["prev_step"] = step
        win_ref["theta_prev"] = lw.theta.detach().clone()
        if step == 0:                                # init curves, for the before / after plots
            emit({"type": "win_init", "step": 0, "win_curves": window_curves(model), "win_bounds": window_bounds(model)})
    wcfg = cfg["windowing"]

    prof = StepProfiler(str(t.get("profile", False)).lower() in ("1", "true", "yes"), device)
    if prof.on:
        log("profiling on: GPU synchronised at every phase boundary; 'prof' line and profile record at every log step")

    aug = cfg["augment"]
    model.train()
    t_last, data_wait, acc_n = time.time(), 0.0, 0
    acc_stats: dict[str, float] = {}
    n_nonfinite, stopped = 0, False
    while step < stop_at and not stopped:
        epoch, in_epoch = divmod(step, steps_per_epoch)
        sampler.set_epoch(epoch, in_epoch)
        train_ds.crop_epoch = epoch                      # forked into this epoch's workers
        loader = torch.utils.data.DataLoader(train_ds, batch_size=local_batch, sampler=sampler, drop_last=True, **loader_kw)
        t_wait = time.time()
        for batch in loader:
            wait = time.time() - t_wait
            data_wait += wait
            prof.start(wait)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg["optim"]["lr"], cfg["optim"]["warmup_steps"], total_steps) * g["lr_mult"]

            # ---- augmentation on raw HU (same RNG stream in every run: seed, step, rank)
            # per-sample seeds (seed, step, dataset index): the augmentation and the local_shuf
            # permutation of a volume do not depend on the GPU / node layout
            sample_seed = [(cfg["seed"] * 1_000_003 + step) * 100_003 + int(i) for i in batch["idx"]]

            def shuffle_seed(j: int) -> int:                     # local_shuf permutation of micro-batch j
                return sample_seed[j * mb]
            hu = batch["volume"].to(device, non_blocking=True)
            micro = [random_axial_rotation(hu[j * mb:(j + 1) * mb].float(), aug["rotate_deg"], aug["rotate_p"],
                                           sample_seed[j * mb:(j + 1) * mb]).half()
                     for j in range(accum)]
            del hu
            txt_emb = batch["text"].to(device, non_blocking=True)
            prof.mark("h2d_aug")

            # ---- (1) features without grad, (2) global loss -> feature gradients
            with torch.no_grad():
                feats = []
                for j, x in enumerate(micro):
                    model.set_shuffle_seed(shuffle_seed(j))
                    feats.append(model.encode_image(x.float()))
                feats = torch.cat(feats)
            prof.mark("fwd_nograd")
            feats.requires_grad_(True)
            loss, st = clip_loss(feats, model.encode_text(txt_emb), model.scale())
            loss.backward()
            fgrad = feats.grad
            prof.mark("loss")
            # ---- (3) re-forward each micro-batch with grad, inject the cached feature gradient
            for j, x in enumerate(micro):
                model.set_shuffle_seed(shuffle_seed(j))
                model.encode_image(x.float()).backward(fgrad[j * mb:(j + 1) * mb])
            model.set_shuffle_seed(None)
            del micro, feats, fgrad
            prof.mark("fwd_bwd")

            D.all_reduce_grads(params)
            if model.logit_scale.grad is not None:
                model.logit_scale.grad /= world
            prof.mark("allreduce")
            gnorm = torch.nn.utils.clip_grad_norm_(params, t["grad_clip"] if t.get("grad_clip") else float("inf"))
            finite = bool(torch.isfinite(gnorm)) and bool(torch.isfinite(loss))   # same on every rank
            if finite:
                opt.step()
                n_nonfinite = 0
            else:                                           # skip the update: never write NaN weights
                n_nonfinite += 1
                log(f"WARNING: non-finite loss/gradient at step {step + 1}, update skipped ({n_nonfinite} in a row)")
                emit({"type": "nonfinite", "step": step + 1, "loss": float(loss), "gnorm": float(gnorm)})
                if n_nonfinite > t.get("max_nonfinite_steps", 20):
                    raise RuntimeError(f"{n_nonfinite} non-finite steps in a row: stopping (last checkpoint kept)")
            opt.zero_grad(set_to_none=True)
            step += 1
            idx_sum = D.sum_int(int(batch["idx"].sum()), device)      # fingerprint of the step's samples
            prof.mark("optim", end_of_step=True)

            if finite:
                for k, v in {"loss": loss.detach(), "gnorm": gnorm, **st}.items():
                    acc_stats[k] = acc_stats.get(k, 0.0) + float(v)
                acc_n += 1
            if step % t["log_every"] == 0 or step == 1:
                dt = time.time() - t_last
                win_ref["step"] = step
                ws = window_stats(model, win_ref)
                rec = {"step": step, "epoch": round(step / steps_per_epoch, 3),
                       **{k: v / max(acc_n, 1) for k, v in acc_stats.items()}, "idx_sum": idx_sum,
                       "lr": opt.param_groups[0]["lr"], "logit_scale": float(model.scale()),
                       "s_per_step": dt / max(acc_n, 1), "data_wait_s": data_wait / max(acc_n, 1),
                       "mem_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                       **ws}
                log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
                emit(rec)
                prec = prof.report(step)                     # collective: every rank
                if prec is not None:
                    log(profile_line(prec))
                    emit(prec)
                if lw is not None and t.get("track_every") and step % t["track_every"] == 0:
                    emit({"type": "win_curves", "step": step, "win_curves": window_curves(model)})
                # learning check: the curves (or the per-patch windows) must have moved by `check_at_step`
                spread = rec.get("patch_center_std_hu", rec.get("patch_median_std_hu", rec.get("route_center_std_hu")))
                if (lw is not None or spread is not None) and wcfg.get("check_at_step") \
                        and step >= wcfg["check_at_step"] and not win_ref.get("checked") and step % t["log_every"] == 0:
                    win_ref["checked"] = True
                    if lw is not None:
                        d = rec.get("win_curve_drift_max", 0.0)
                        ok = d >= wcfg.get("min_curve_drift", 0.02)
                    else:                                    # patch arms: spread of the windows across patches (HU)
                        d = spread
                        ok = d >= wcfg.get("min_patch_spread_hu", 5.0)
                    emit({"type": "window_check", "step": step, "curve_drift_max": d, "learning": ok,
                          "gate_active": rec.get("gate_active")})
                    log(("window check OK" if ok else "WARNING: windowing NOT learning") +
                        (f" at step {step}: max curve change {d:.4f} (threshold {wcfg.get('min_curve_drift', 0.02)})"
                         if lw is not None else
                         f" at step {step}: per-patch window spread {d:.1f} HU (threshold {wcfg.get('min_patch_spread_hu', 5.0)})")
                        + (f", gate active {rec['gate_active']:.3f}" if "gate_active" in rec else ""))
                if rank == 0 and t.get("track_every") and step % t["track_every"] == 0:
                    refresh_tracking(out_dir)
                t_last, data_wait, acc_n, acc_stats = time.time(), 0.0, 0, {}

            end_of_epoch = step % steps_per_epoch == 0
            stop_req = D.any_true(STOP["signal"] is not None, device)      # SIGUSR1 seen by any rank
            do_val = (end_of_epoch and (step // steps_per_epoch) % t["val_every_epochs"] == 0) or step >= stop_at
            do_ckpt = step % t["ckpt_every"] == 0 or end_of_epoch or step >= stop_at or stop_req
            if do_val or do_ckpt:
                D.average_buffers(model)
            if do_val:                                       # before the checkpoint: a kill during
                res = validate(model, val_loader, cfg, device, step, out_dir)   # validation redoes it
                log("val " + " ".join(f"{k}={v:.4g}" for k, v in res.items() if isinstance(v, float)))
                emit({"type": "val", **res})
                if rank == 0:
                    refresh_tracking(out_dir)
            if do_ckpt:
                if rank == 0:
                    save_ckpt(ck_path, model, opt, step, step // steps_per_epoch, cfg, keep_previous=True)
                    if end_of_epoch and (step // steps_per_epoch) % t["keep_every_epochs"] == 0:
                        save_ckpt(out_dir / f"ckpt_epoch{step // steps_per_epoch:03d}.pt", model, opt, step,
                                  step // steps_per_epoch, cfg)
                D.barrier()
            if stop_req:
                log(f"stop requested (signal) at step {step}: checkpoint saved, exiting; "
                    f"resubmit the same command to resume")
                emit({"type": "stopped_by_signal", "step": step, "job_id": os.environ.get("SLURM_JOB_ID")})
                stopped = True
                break
            if step >= stop_at:
                break
            prof.mark("misc")
            t_wait = time.time()

    if rank == 0 and step >= total_steps:
        save_ckpt(out_dir / "ckpt_final.pt", model, opt, step, step // steps_per_epoch, cfg)
    log(f"done at step {step}")
    D.cleanup()


if __name__ == "__main__":
    main()