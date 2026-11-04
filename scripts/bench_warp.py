"""Learned-window front-end cost at 384^3 on one GCD: plain curves vs curves + per-channel warp."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcw.windowing_arms import build_arm  # noqa: E402


def t(fn, reps=3):
    fn(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize(); return (time.time() - t0) / reps


x = torch.randint(-1000, 1000, (1, 1, 384, 384, 384), device="cuda").float()
g = torch.randn(1, 11, 384, 384, 384, device="cuda")
for arm, warp in (("learned", False), ("learned", True), ("local", True)):
    m = build_arm(arm, warp=warp, patch=6)["adapter"].cuda()
    def fwd():
        with torch.no_grad():
            m(x)
    def fb():
        m(x).backward(g)
    torch.cuda.reset_peak_memory_stats()
    print(f"{arm:8s} warp={warp!s:5s} fwd {t(fwd) * 1e3:6.1f} ms | fwd+bwd {t(fb) * 1e3:6.1f} ms | "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
