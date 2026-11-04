"""grad_W GEMM orientations for the 192->192 conv (bf16)."""
import time
import torch


def t(fn, reps=10):
    fn(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize(); return (time.time() - t0) / reps


for M in (98304, 196608):
    col = torch.randn(M, 5184, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(M, 192, device="cuda", dtype=torch.bfloat16)
    fl = 2 * M * 5184 * 192
    for name, fn in (("col.t() @ g", lambda: col.t() @ g), ("(g.t() @ col).t()", lambda: (g.t() @ col).t()),
                     ("fwd col @ w", lambda: col @ g[:5184].t().contiguous() if False else col @ torch.empty(5184, 192, device='cuda', dtype=torch.bfloat16))):
        dt = t(fn)
        print(f"M={M:6d} {name:20s} {dt * 1e3:6.2f} ms {fl / dt / 1e12:5.1f} TF/s", flush=True)
