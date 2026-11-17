"""Communication checker (MCW_COMM_CHECK=1): verifies the RCCL collectives of training through a second,
independent channel, logs every error with a diagnosis, and repairs it so the run stays usable.

Why (2026-11-17): with torch 2.14.0+rocm10.1.0 (RCCL 2.30) on 4 MI300A nodes, the end-of-epoch average of the
BatchNorm buffers (mcw.dist.average_buffers: 16 small all_reduces in a row) sometimes returned, for buffer k,
the right value + 0.25 x buffer k+9: the 4 ranks of one node had contributed another collective's data.

The second channel is a gloo process group (TCP sockets, CPU tensors): it does not use RCCL, libfabric or the GPU.
Checked, on every rank:
  bn_avg     average_buffers: each rank's local buffers are gathered over gloo, the exact average is computed in
             float64 and compared with what RCCL returned on every rank. An error is explained by searching which
             rank set (a node, a single rank) sent which other buffer (or zeros) instead. Repaired with the exact
             average.
  grads      the flat gradient all_reduce: position-weighted checksums per chunk (64 chunks) before / after,
             expected sums over gloo. Repaired by redoing the reduction over gloo from a copy of the local gradients.
  gather     the all_gathers of the loss features and of validation: exact comparison with the same gather over
             gloo; a wrong slice is compared with the other ranks' data and with the previous gathers. Repaired.
  scalar     sum_int / any_true: compared with gloo.
  stress     every `stress_every` steps and at each epoch end: bursts of 16 all_reduces with the shapes of the BN
             buffers (persistent tensors, as the real buffers) on integer data known to every rank, each followed by
             the same division as average_buffers; variants: 16 in a row (as average_buffers), one flat all_reduce,
             16 with a GPU sync after each. Exact check, same diagnosis.
Output: <out_dir>/commcheck.jsonl (rank 0: every error and a summary every `summary_every` steps), and
<out_dir>/commcheck/rank<r>.log (each rank: its own errors in detail). Errors also go to the job log ([commcheck]).
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

TOL = 2e-5                       # relative tolerance of the float checks (fp32 sums of 16 values: ~1e-6)


def _rccl_version() -> str | None:
    try:
        v = torch.cuda.nccl.version()
        return ".".join(map(str, v)) if isinstance(v, tuple) else str(v)
    except Exception:                                    # noqa: BLE001 (diagnostic only)
        return None


class CommCheck:
    def __init__(self, rank: int, world: int, device: torch.device, out_dir: Path, stress_every: int = 100,
                 summary_every: int = 100, n_chunks: int = 64):
        self.rank, self.world, self.device = rank, world, device
        self.stress_every, self.summary_every, self.n_chunks = stress_every, summary_every, n_chunks
        t0 = time.time()
        self.gloo = dist.new_group(backend="gloo")
        hosts = [None] * world
        dist.all_gather_object(hosts, socket.gethostname(), group=self.gloo)
        self.hosts = hosts
        nodes = sorted(set(hosts), key=hosts.index)
        self.groups = {f"node {h} (ranks {[r for r in range(world) if hosts[r] == h]})":
                       [r for r in range(world) if hosts[r] == h] for h in nodes}
        self.groups |= {f"rank {r} ({hosts[r]})": [r] for r in range(world)}
        self.step, self.epoch, self.loader_t = 0, 0, time.time()
        self.counts: dict[str, list[int]] = {}           # kind -> [checked, bad]
        self.history: list[tuple[int, str, list[np.ndarray]]] = []   # recent gathers: (seq, what, per-rank data)
        self.seq = 0
        self._w = None
        self._sbufs = None
        (out_dir / "commcheck").mkdir(parents=True, exist_ok=True)
        self.f_rank = open(out_dir / "commcheck" / f"rank{rank:02d}.log", "a")
        self.f_json = open(out_dir / "commcheck.jsonl", "a") if rank == 0 else None
        self._rank_log(f"commcheck on: host {socket.gethostname()}, gloo group ready in {time.time() - t0:.1f}s, "
                       f"GLOO_SOCKET_IFNAME={os.environ.get('GLOO_SOCKET_IFNAME')}")
        self._emit({"type": "commcheck_start", "hosts": hosts, "job_id": os.environ.get("SLURM_JOB_ID"),
                    "env": {k: os.environ.get(k) for k in ("NCCL_NET_GDR_LEVEL", "NCCL_PROTO", "NCCL_ALGO",
                                                           "CXI_FORK_SAFE", "FI_PROVIDER", "NCCL_SOCKET_IFNAME",
                                                           "RCCL_OFI", "MCW_NET_GDR_LEVEL")},
                    "rccl": _rccl_version(), "torch": torch.__version__, "hip": torch.version.hip})

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------ logging
    def _rank_log(self, msg: str) -> None:
        self.f_rank.write(f"[{time.strftime('%H:%M:%S')}] step {self.step} {msg}\n")
        self.f_rank.flush()

    def _emit(self, rec: dict) -> None:
        if self.f_json:
            rec = {"step": self.step, "epoch": self.epoch, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
            self.f_json.write(json.dumps(rec) + "\n")
            self.f_json.flush()

    def _count(self, kind: str, bad: bool) -> None:
        c = self.counts.setdefault(kind, [0, 0])
        c[0] += 1
        c[1] += int(bad)

    def _report(self, kind: str, local_errors: list[dict], extra: dict | None = None) -> bool:
        """Collective (gloo): counts every rank's errors (and gathers them if any); rank 0 logs them. Returns True if
        any rank had one."""
        cnt = torch.tensor([len(local_errors)], dtype=torch.int64)
        dist.all_reduce(cnt, group=self.gloo)
        n = int(cnt)
        self._count(kind, n > 0)
        if not n:
            return False
        allerr = [None] * self.world
        dist.all_gather_object(allerr, local_errors, group=self.gloo)
        if self.rank == 0:
            since_fork = time.time() - self.loader_t
            print(f"[commcheck] ERROR {kind} at step {self.step} (epoch {self.epoch}, {since_fork:.0f}s after the "
                  f"DataLoader fork): {n} wrong result(s) on ranks {[r for r in range(self.world) if allerr[r]]}",
                  flush=True)
            for r, errs in enumerate(allerr):
                for e in errs[:6]:
                    print(f"[commcheck]   rank {r} ({self.hosts[r]}): {e.get('summary', e)}", flush=True)
            self._emit({"type": "comm_error", "kind": kind, "since_fork_s": round(since_fork, 1),
                        "errors": {r: e for r, e in enumerate(allerr) if e}, **(extra or {})})
        return n > 0

    def summary(self) -> None:
        if self.step % self.summary_every == 0 and self.rank == 0:
            self._emit({"type": "commcheck_summary", "counts": self.counts})
            print("[commcheck] summary " + " ".join(f"{k}={c[1]}/{c[0]} bad" for k, c in self.counts.items()),
                  flush=True)

    def on_new_loader(self, epoch: int) -> None:
        self.epoch, self.loader_t = epoch, time.time()

    # ------------------------------------------------------------------ diagnosis
    def _explain(self, dev: np.ndarray, data: np.ndarray, slices: list[tuple[int, int, str]], k: int) -> dict:
        """dev = observed SUM - expected SUM for buffer k; data = (world, n) local contributions. Best explanation:
        the ranks of a group sent buffer j (or zeros) instead of buffer k."""
        o, n, _ = slices[k]
        scale = float(np.abs(dev).max()) or 1.0
        best = None
        for gname, ranks in self.groups.items():
            own = data[ranks, o:o + n].sum(0)
            cands = [("zeros", -own)] + [(f"#{j} {nm}", data[ranks, oj:oj + nj].sum(0) - own)
                                         for j, (oj, nj, nm) in enumerate(slices) if nj == n and j != k]
            for jname, pred in cands:
                res = float(np.abs(dev - pred).max()) / scale
                if best is None or res < best[0]:
                    best = (res, gname, jname)
        return {"fit": f"{best[1]} sent {best[2]} instead of #{k} {slices[k][2]}", "fit_residual": round(best[0], 6),
                "offset": (int(best[2].split()[0][1:]) - k) if best[2] != "zeros" else None}

    # ------------------------------------------------------------------ BN buffers (mcw.dist.average_buffers)
    def average_buffers(self, module: torch.nn.Module) -> None:
        names, bufs = zip(*[(n, b) for n, b in module.named_buffers() if b.is_floating_point()])
        slices, o = [], 0
        for n, b in zip(names, bufs):
            slices.append((o, b.numel(), n.replace("visual.atlas.patch_embeds.abdomen_ct.", "")))
            o += b.numel()
        local = torch.cat([b.detach().flatten().double() for b in bufs]).cpu()
        t0 = time.time()
        for b in bufs:                                   # exactly mcw.dist.average_buffers
            dist.all_reduce(b)
            b /= self.world
        self._sync()
        t_rccl = time.time() - t0
        got = torch.cat([b.detach().flatten().double() for b in bufs]).cpu()
        data = [torch.empty_like(local) for _ in range(self.world)]
        dist.all_gather(data, local, group=self.gloo)
        data = torch.stack(data).numpy()
        expected = data.mean(0)
        errs = []
        for k, (o, n, nm) in enumerate(slices):
            g, e = got.numpy()[o:o + n], expected[o:o + n]
            tol = TOL * (np.abs(data[:, o:o + n]).max() + 1e-30)
            if np.abs(g - e).max() > tol:
                ex = self._explain((g - e) * self.world, data, slices, k)
                err = {"buffer": k, "name": nm, "max_abs_dev": float(np.abs(g - e).max()),
                       "max_dev_over_scale": float(np.abs(g - e).max() / (np.abs(e).max() + 1e-30)), **ex,
                       "summary": f"buffer #{k} {nm}: max |dev| {np.abs(g - e).max():.4g} -> {ex['fit']} "
                                  f"(residual {ex['fit_residual']:.2g})"}
                errs.append(err)
                self._rank_log("bn_avg ERROR " + json.dumps(err) + f" got[:4]={g[:4].tolist()} exp[:4]={e[:4].tolist()}")
        bad = self._report("bn_avg", errs, {"t_rccl_s": round(t_rccl, 4)})
        if bad or errs:                                  # repair on every rank with the exact average (float64 -> fp32)
            for b, (o, n, _) in zip(bufs, slices):
                b.copy_(torch.from_numpy(expected[o:o + n]).view_as(b).to(b.device, b.dtype))
            self._rank_log("bn_avg repaired with the gloo average")

    # ------------------------------------------------------------------ gradients (mcw.dist.all_reduce_grads)
    def _weights(self, n: int) -> torch.Tensor:
        if self._w is None or self._w.numel() != n:
            idx = torch.arange(n, dtype=torch.float64, device=self.device)
            self._w = ((idx * 0.6180339887498949) % 1.0 * 2 - 1).float()
            del idx
        return self._w

    def _chunk_sums(self, flat: torch.Tensor) -> torch.Tensor:
        v = flat.float() * self._weights(flat.numel())
        pad = (-v.numel()) % self.n_chunks
        v = torch.nn.functional.pad(v, (0, pad)).view(self.n_chunks, -1)
        return torch.stack([v.sum(1, dtype=torch.float64), v.abs().sum(1, dtype=torch.float64)])   # (2, C)

    def grad_pre(self, flat: torch.Tensor):
        return flat.detach().clone(), self._chunk_sums(flat)

    def grad_post(self, flat: torch.Tensor, pre) -> None:
        local, sums = pre
        got = self._chunk_sums(flat)[0].cpu().numpy()
        exp = sums.cpu()
        dist.all_reduce(exp, group=self.gloo)
        exp = exp.numpy()
        dev = np.abs(got - exp[0])
        badc = np.flatnonzero(dev > TOL * exp[1] + 1e-30)
        errs = []
        if len(badc):
            errs.append({"chunks": badc.tolist(), "n_chunks": self.n_chunks, "numel": flat.numel(),
                         "max_dev_over_abs": float((dev / (exp[1] + 1e-30)).max()),
                         "summary": f"gradient all_reduce wrong in {len(badc)}/{self.n_chunks} chunks "
                                    f"{badc[:12].tolist()} (max |dev| / sum|w g| {(dev / (exp[1] + 1e-30)).max():.3g})"})
            self._rank_log("grads ERROR " + json.dumps(errs[-1]))
        if self._report("grads", errs):                  # redo the reduction over gloo, exact for every rank
            t0 = time.time()
            x = local.cpu()
            dist.all_reduce(x, group=self.gloo)
            flat.copy_(x.to(flat.device))
            self._rank_log(f"grads repaired over gloo in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ all_gathers (loss features, validation)
    def gather(self, local: torch.Tensor, parts: list[torch.Tensor], what: str) -> None:
        self.seq += 1
        L = local.detach().cpu().contiguous()
        ref = [torch.empty_like(L) for _ in range(self.world)]
        dist.all_gather(ref, L, group=self.gloo)
        refn = [x.float().numpy() for x in ref]
        errs = []
        for r in range(self.world):
            p = parts[r].detach().cpu()
            if not torch.equal(p, ref[r]):
                pn = p.float().numpy()
                same_as = [s for s in range(self.world) if s != r and np.array_equal(pn, refn[s])]
                past = [(self.seq - q, w, s) for q, w, d in self.history for s in range(self.world)
                        if d[s].shape == pn.shape and np.array_equal(pn, d[s])]
                diff = np.abs(pn - refn[r])
                errs.append({"slice_of_rank": r, "what": what, "n_wrong": int((diff > 0).sum()), "size": int(pn.size),
                             "equals_rank": same_as, "equals_past_gather": past[:3],
                             "summary": f"{what}: slice of rank {r} wrong ({int((diff > 0).sum())}/{pn.size} values)"
                                        f"{', = rank ' + str(same_as) + ' data' if same_as else ''}"
                                        f"{', = data of gather -' + str(past[0][0]) + ' (' + past[0][1] + ')' if past else ''}"})
                parts[r].copy_(ref[r].to(parts[r].device))   # repair
        if errs:
            self._rank_log("gather ERROR " + json.dumps(errs))
        self.history = (self.history + [(self.seq, what, refn)])[-32:]
        self._report("gather_" + what, errs)

    # ------------------------------------------------------------------ small all_reduces (sum_int, any_true)
    def scalar(self, got: torch.Tensor, local: torch.Tensor, op, what: str) -> torch.Tensor:
        x = local.detach().cpu().clone()
        dist.all_reduce(x, op=op, group=self.gloo)
        g = got.detach().cpu()
        errs = [] if torch.equal(g, x) else [{"what": what, "got": g.tolist(), "expected": x.tolist(),
                                               "summary": f"{what}: got {g.tolist()} expected {x.tolist()}"}]
        if errs:
            self._rank_log("scalar ERROR " + json.dumps(errs))
        self._report("scalar_" + what, errs)
        return x.to(got.device)

    # ------------------------------------------------------------------ stress bursts
    def stress(self, module: torch.nn.Module, n_loop: int = 6, n_flat: int = 2, n_sync: int = 2) -> None:
        shapes = [(n.replace("visual.atlas.patch_embeds.abdomen_ct.", ""), b.shape, b.dtype)
                  for n, b in module.named_buffers() if b.is_floating_point()]
        if self._sbufs is None:                          # persistent tensors, as the real buffers
            self._sbufs = [torch.zeros(s, dtype=dt, device=self.device) for _, s, dt in shapes]
        slices, o = [], 0
        for nm, s, _ in shapes:
            n = int(np.prod(s))
            slices.append((o, n, nm))
            o += n
        tot = o
        e = np.arange(tot)
        errs = []
        modes = ["loop"] * n_loop + ["flat"] * n_flat + ["loop_sync"] * n_sync
        for b_i, mode in enumerate(modes):
            data = np.stack([(r * 131 + e * 7 + b_i * 29 + self.step * 3) % 1000 for r in range(self.world)]) \
                + np.concatenate([np.full(n, k * 17) for k, (_, n, _) in enumerate(slices)])[None]
            data = (data % 1000).astype(np.float64)       # integers < 1000: fp32 sums exact
            mine = torch.from_numpy(data[self.rank]).float().to(self.device)
            for k, (oo, n, _) in enumerate(slices):
                self._sbufs[k].view(-1).copy_(mine[oo:oo + n])
            self._sync()
            if mode == "flat":
                flat = torch.cat([x.view(-1) for x in self._sbufs])
                dist.all_reduce(flat)
                flat /= self.world
                self._sync()
                got = flat.double().cpu().numpy()
            else:
                for x in self._sbufs:
                    dist.all_reduce(x)
                    x /= self.world
                    if mode == "loop_sync":
                        self._sync()
                self._sync()
                got = torch.cat([x.view(-1) for x in self._sbufs]).double().cpu().numpy()
            exp = data.sum(0) / self.world
            for k, (oo, n, nm) in enumerate(slices):
                d = got[oo:oo + n] - exp[oo:oo + n]
                if np.abs(d).max() > 0:
                    ex = self._explain(d * self.world, data, slices, k)
                    errs.append({"burst": b_i, "mode": mode, "buffer": k, "name": nm, **ex,
                                 "summary": f"stress burst {b_i} ({mode}) buffer #{k}: {ex['fit']} "
                                            f"(residual {ex['fit_residual']:.2g})"})
        if errs:
            self._rank_log("stress ERROR " + json.dumps(errs))
        self._report("stress", errs, {"bursts": len(modes)})
