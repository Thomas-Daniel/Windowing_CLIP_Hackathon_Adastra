"""Cost of the patch arms at 384^3 on one GCD: forward / forward+backward and peak memory."""
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcw.gemm_conv3d import swap_conv3d  # noqa: E402
from mcw.patch_windows import build_patch_arm  # noqa: E402


def t(fn, reps=3):
    fn(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize(); return (time.time() - t0) / reps


hh = os.path.expandvars("$SCRATCHDIR/merlin_clip_data/hu_hist_80.npz")
x = torch.randint(-1000, 1000, (1, 1, 384, 384, 384), device="cuda").float()
for arm in ("global_bank", "patch_bank", "patch_eq", "patch_cdf"):
    m = build_patch_arm(arm, {"k": 8, "hu_hist": hh}, 6).cuda().train()
    swap_conv3d(m)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01 * torch.randn_like(p))
    g = torch.randn(1, 8, 384, 384, 384, device="cuda")
    def fwd():
        with torch.no_grad():
            m(x)
    def fb():
        m(x).backward(g)
    torch.cuda.reset_peak_memory_stats()
    tf, tfb = t(fwd), t(fb)
    print(f"{arm:12s} fwd {tf * 1e3:6.0f} ms | fwd+bwd {tfb * 1e3:6.0f} ms | peak {torch.cuda.max_memory_allocated() / 2**30:5.1f} GB",
          flush=True)
    del m, g
    torch.cuda.empty_cache()
