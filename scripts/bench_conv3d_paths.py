"""Compare the ways to compute the Atlas stem's 3D convolutions on one GPU (bf16, channels-last,
micro-batch 1, as in training):

  gemm       mcw/gemm_conv3d.py, the current path: im2col column buffer + rocBLAS GEMM
  gemm_pipe  same, with the column copy of slab i+1 on a second stream while slab i is in the
             GEMM (forward only: tests whether copies and GEMMs overlap on the GPU)
  triton     mcw/triton_conv3d.py: implicit GEMM, no column buffer (first call = autotuning)
  native     F.conv3d through MIOpen, in a subprocess with a timeout (a naive MIOpen solver
             cannot be interrupted). --miopen-find sets torch.backends.cudnn.benchmark=True:
             PyTorch then calls MIOpen Find (benchmarks the solvers) instead of immediate mode.

For each conv: forward and forward+backward time, TFLOP/s, peak memory, and the difference to
the gemm output. --breakdown splits the gemm forward of the 192->192 conv into pad / im2col
copies / GEMMs.

    python scripts/bench_conv3d_paths.py --breakdown
    python scripts/bench_conv3d_paths.py --impl gemm gemm_pipe triton
    MIOPEN_FIND_MODE=NORMAL python scripts/bench_conv3d_paths.py --impl native --miopen-find --timeout 1800
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcw import gemm_conv3d as G  # noqa: E402

# (name, C_in, C_out, stride, padding, input size): the convs of ChestCTEmbed3D at 384^3
CONVS = {
    "first_s3_11to64": (11, 64, 3, 0, 384),
    "c64to192_128": (64, 192, 1, 1, 128),
    "c192to192_128": (192, 192, 1, 1, 128),     # x6 per forward: ~85 % of the stem FLOPs
    "down_192to384_s2": (192, 384, 2, 1, 128),
}
DT = torch.bfloat16


def sync_time(fn, reps: int) -> float:
    fn()                                         # warm-up (autotuning, solver search, allocations)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    return best


def conv_cl_pipelined(xcl, wm, k, s, out_dhw):
    """gemm_conv3d._conv_cl with two column buffers: copy slab i+1 on a side stream during the
    GEMM of slab i. Same result, bit for bit (same GEMMs on the same columns)."""
    B, C = xcl.shape[0], xcl.shape[4]
    Do, Ho, Wo = out_dhw
    kc, O = wm.shape
    out = torch.empty((B, Do, Ho, Wo, O), device=xcl.device, dtype=xcl.dtype)
    rows = max(1, min(Do, G.CHUNK_ELEMS // max(1, Ho * Wo * kc)))
    cols = [torch.empty((rows, Ho, Wo, k, k, k, C), device=xcl.device, dtype=xcl.dtype) for _ in range(2)]
    main, side = torch.cuda.current_stream(), torch.cuda.Stream()
    copied = [torch.cuda.Event(), torch.cuda.Event()]
    used = [torch.cuda.Event(), torch.cuda.Event()]
    side.wait_stream(main)
    jobs = [(b, d0) for b in range(B) for d0 in range(0, Do, rows)]
    for i, (b, d0) in enumerate(jobs):
        j, nd = i % 2, min(rows, Do - d0)
        with torch.cuda.stream(side):
            if i >= 2:
                side.wait_event(used[j])         # the GEMM that read this buffer is done
            G._copy(cols[j][:nd], G._windows(xcl[b], d0, nd, Ho, Wo, k, s))
            copied[j].record(side)
        main.wait_event(copied[j])
        torch.mm(cols[j][:nd].view(-1, kc), wm, out=out[b, d0:d0 + nd].view(-1, O))
        used[j].record(main)
    return out


def make(name: str):
    cin, cout, s, p, n = CONVS[name]
    torch.manual_seed(0)
    x = torch.randn(1, cin, n, n, n, device="cuda", dtype=DT).contiguous(memory_format=torch.channels_last_3d)
    w = (torch.randn(cout, cin, 3, 3, 3, device="cuda") / (27 * cin) ** 0.5).to(DT)
    return x, w, s, p


def run_impl(impl: str, name: str, reps: int) -> dict:
    x, w, s, p = make(name)
    k = 3
    out_dhw = tuple(G._out_size(n, k, s, p) for n in x.shape[2:])
    flops = 2 * x.shape[1] * w.shape[0] * 27 * out_dhw[0] * out_dhw[1] * out_dhw[2]
    if impl == "native":
        f = lambda xx, ww: F.conv3d(xx, ww, None, s, p)
    elif impl == "gemm":
        f = lambda xx, ww: G.conv3d(xx, ww, None, s, p)
    elif impl == "triton":
        from mcw import triton_conv3d as T
        f = lambda xx, ww: T.conv3d(xx, ww, None, s, p)
    elif impl == "gemm_pipe":
        f = lambda xx, ww: G._from_cl(conv_cl_pipelined(G._to_cl(xx, p), G._wmat(ww), k, s, out_dhw), True)
    else:
        raise ValueError(impl)
    res = {}
    with torch.no_grad():
        y = f(x, w)
        ref = G.conv3d(x, w, None, s, p)
        res["max_rel_diff_vs_gemm"] = float((y.float() - ref.float()).abs().max() / ref.float().abs().max())
        del y, ref
        torch.cuda.reset_peak_memory_stats()
        res["fwd_ms"] = 1e3 * sync_time(lambda: f(x, w), reps)
        res["fwd_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    res["fwd_tflops"] = flops / res["fwd_ms"] * 1e-9
    if impl != "gemm_pipe":
        xg = x.detach().requires_grad_(name != "first_s3_11to64")   # the first conv's input only needs grad for learned arms
        wg = w.detach().requires_grad_(True)
        gy = torch.randn_like(f(x, w))

        def fb():
            xg.grad = wg.grad = None
            f(xg, wg).backward(gy)
        torch.cuda.reset_peak_memory_stats()
        res["fwd_bwd_ms"] = 1e3 * sync_time(fb, reps)
        res["fwd_bwd_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        res["fwd_bwd_tflops"] = (3 if xg.requires_grad else 2) * flops / res["fwd_bwd_ms"] * 1e-9
        fb()                                     # invariant 3 (bit-reproducible runs): no atomics in backward
        g1 = wg.grad.clone()
        fb()
        res["wgrad_repeatable"] = bool(torch.equal(g1, wg.grad))
    return res


def breakdown(reps: int) -> None:
    """gemm forward of the 192->192 conv at 128^3, split into its three parts."""
    x, w, s, p = make("c192to192_128")
    k, (B, C, D, H, W) = 3, x.shape
    wm = G._wmat(w)
    xcl = G._to_cl(x, p)
    kc, O = wm.shape
    rows = max(1, min(D, G.CHUNK_ELEMS // (H * W * kc)))
    col = torch.empty((rows, H, W, k, k, k, C), device="cuda", dtype=DT)
    out = torch.empty((D, H, W, O), device="cuda", dtype=DT)

    def copies():
        for d0 in range(0, D, rows):
            nd = min(rows, D - d0)
            G._copy(col[:nd], G._windows(xcl[0], d0, nd, H, W, k, s))

    def gemms():                                 # same GEMM shapes, on whatever is in the buffer
        for d0 in range(0, D, rows):
            nd = min(rows, D - d0)
            torch.mm(col[:nd].view(-1, kc), wm, out=out[d0:d0 + nd].view(-1, O))

    t_pad = sync_time(lambda: G._to_cl(x, p), reps)
    t_cp = sync_time(copies, reps)
    t_mm = sync_time(gemms, reps)
    t_all = sync_time(lambda: G.conv3d(x, w, None, s, p), reps)
    gb = D * H * W * kc * 2 / 1e9
    flops = 2 * D * H * W * kc * O
    print(f"gemm forward, 192->192 3^3 conv at 128^3 ({D // rows + 1} slabs of {rows} slices):")
    print(f"  pad + channels-last copy  {t_pad * 1e3:7.1f} ms")
    print(f"  im2col copies             {t_cp * 1e3:7.1f} ms   {gb:.1f} GB of columns, {2 * gb / t_cp:.0f} GB/s (write + read)")
    print(f"  GEMMs                     {t_mm * 1e3:7.1f} ms   {flops / t_mm / 1e12:.0f} TFLOP/s")
    print(f"  whole conv3d forward      {t_all * 1e3:7.1f} ms   {flops / t_all / 1e12:.0f} TFLOP/s effective", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", nargs="+", default=["gemm", "gemm_pipe", "triton"],
                    choices=["gemm", "gemm_pipe", "triton", "native"])
    ap.add_argument("--convs", nargs="+", default=list(CONVS), choices=list(CONVS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--miopen-find", action="store_true", help="torch.backends.cudnn.benchmark=True (native)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per native case")
    ap.add_argument("--child", nargs=2, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.miopen_find:
        torch.backends.cudnn.benchmark = True
    if args.child:                               # native case in its own process
        print("RESULT " + json.dumps(run_impl(args.child[0], args.child[1], args.reps)), flush=True)
        return
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | "
          f"MIOPEN_FIND_MODE={os.environ.get('MIOPEN_FIND_MODE')} cudnn.benchmark={args.miopen_find}", flush=True)
    if args.breakdown:
        breakdown(args.reps)
    for name in args.convs:
        for impl in args.impl:
            t0 = time.time()
            try:
                if impl == "native":
                    cmd = [sys.executable, __file__, "--child", impl, name, "--reps", str(args.reps)]
                    if args.miopen_find:
                        cmd.append("--miopen-find")
                    out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                    seen = []                    # MIOpen's solver / MIOpenDriver lines (MIOPEN_LOG_LEVEL, MIOPEN_ENABLE_LOGGING_CMD)
                    for ln in (out.stdout + out.stderr).splitlines():
                        if any(t in ln for t in ("MIOpenDriver", "olver", "Algorithm")) and ln not in seen:
                            seen.append(ln)
                    for ln in seen[:40]:
                        print("    " + ln[:300], flush=True)
                    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
                    if not line:
                        raise RuntimeError((out.stderr.strip().splitlines() or ["?"])[-1][:200])
                    r = json.loads(line[0][7:])
                else:
                    r = run_impl(impl, name, args.reps)
                msg = (f"fwd {r['fwd_ms']:8.1f} ms {r['fwd_tflops']:5.1f} TF/s {r['fwd_peak_gb']:5.1f} GB")
                if "fwd_bwd_ms" in r:
                    msg += (f" | fwd+bwd {r['fwd_bwd_ms']:8.1f} ms {r['fwd_bwd_tflops']:5.1f} TF/s {r['fwd_bwd_peak_gb']:5.1f} GB"
                            f" | grad_W repeatable {r['wgrad_repeatable']}")
                msg += f" | rel diff vs gemm {r['max_rel_diff_vs_gemm']:.1e}"
            except subprocess.TimeoutExpired:
                msg = f"TIMEOUT > {args.timeout} s"
            except Exception as e:                # report and go on with the next case
                msg = f"FAILED {type(e).__name__}: {str(e)[:200]}"
            print(f"{name:17s} {impl:9s} {msg}  [{time.time() - t0:.0f} s]", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
