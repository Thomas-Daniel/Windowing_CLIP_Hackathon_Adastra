"""Speed of the Atlas conv stem's 3D convolutions under MIOpen, per dtype / memory format.

Each case runs in a subprocess with a timeout, because a slow MIOpen solver (e.g. the naive
fallback) cannot be interrupted from Python. Example:
    python scripts/bench_conv3d.py --dtypes bf16 fp16 fp32 --formats ncdhw ndhwc --timeout 240
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# (name, C_in, C_out, stride, padding, input spatial size) -- the convs of ChestCTEmbed3D at 384^3
CONVS = [
    ("stem_s3_11to64", 11, 64, 3, 0, 384),
    ("c64to192_128", 64, 192, 1, 1, 128),
    ("c192to192_128", 192, 192, 1, 1, 128),
    ("down_192to384_s2", 192, 384, 2, 1, 128),
]


def child(name: str, dtype: str, fmt: str, batch: int, reps: int) -> None:
    import torch

    cin, cout, s, p, n = next((c[1], c[2], c[3], c[4], c[5]) for c in CONVS if c[0] == name)
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    mf = torch.channels_last_3d if fmt == "ndhwc" else torch.contiguous_format
    conv = torch.nn.Conv3d(cin, cout, 3, stride=s, padding=p, bias=False).cuda().to(dt).to(memory_format=mf)
    x = torch.randn(batch, cin, n, n, n, device="cuda", dtype=dt).to(memory_format=mf).requires_grad_(True)
    res = {}
    for phase in ("fwd", "bwd"):
        times = []
        for _ in range(reps + 1):                       # first = warm-up / solver search
            torch.cuda.synchronize()
            t0 = time.time()
            y = conv(x)
            if phase == "bwd":
                y.backward(torch.ones_like(y))
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        res[phase] = min(times[1:])
        res[f"{phase}_first"] = times[0]
    flops = 2 * cin * cout * 27 * (n // s) ** 3 * batch
    res["tflops_fwd"] = flops / res["fwd"] / 1e12
    print("RESULT " + json.dumps(res), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp16", "fp32"])
    ap.add_argument("--formats", nargs="+", default=["ncdhw", "ndhwc"])
    ap.add_argument("--convs", nargs="+", default=[c[0] for c in CONVS])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--child", nargs=3, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        child(*args.child, batch=args.batch, reps=args.reps)
        return
    print(f"env: MIOPEN_FIND_MODE={os.environ.get('MIOPEN_FIND_MODE')} "
          f"PYTORCH_MIOPEN_SUGGEST_NHWC={os.environ.get('PYTORCH_MIOPEN_SUGGEST_NHWC')}", flush=True)
    for name in args.convs:
        for dtype in args.dtypes:
            for fmt in args.formats:
                t0 = time.time()
                try:
                    out = subprocess.run([sys.executable, __file__, "--child", name, dtype, fmt, "--batch", str(args.batch),
                                          "--reps", str(args.reps)], capture_output=True, text=True, timeout=args.timeout)
                    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
                    if line:
                        r = json.loads(line[0][7:])
                        msg = (f"fwd {r['fwd'] * 1e3:8.1f} ms (first {r['fwd_first']:6.1f} s, {r['tflops_fwd']:5.1f} TFLOP/s) | "
                               f"fwd+bwd {r['bwd'] * 1e3:8.1f} ms (first {r['bwd_first']:6.1f} s)")
                    else:
                        msg = "FAILED " + (out.stderr.strip().splitlines() or ["?"])[-1][:200]
                except subprocess.TimeoutExpired:
                    msg = f"TIMEOUT > {args.timeout} s"
                print(f"{name:18s} {dtype:5s} {fmt:6s}: {msg}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
