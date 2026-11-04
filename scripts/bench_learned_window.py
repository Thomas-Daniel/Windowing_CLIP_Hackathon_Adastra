"""Speed of the learned-window front-end at 384^3 (one GCD): forward, backward, and variants
of the dL/dp reduction (chunk size, GEMM orientation, bincount histograms)."""
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


dev = "cuda"
x = torch.randint(-1000, 1000, (1, 1, 384, 384, 384), device=dev).float()
for arm in ():
    lw = build_arm(arm, patch=6)["adapter"].to(dev)
    g = torch.randn(1, 11, 384, 384, 384, device=dev)
    def fwd():
        with torch.no_grad():
            lw(x)
    def fb():
        lw(x).backward(g)
    print(f"{arm:8s} fwd {t(fwd) * 1e3:7.1f} ms | fwd+bwd {t(fb) * 1e3:7.1f} ms", flush=True)

lw = build_arm("learned")["adapter"].to(dev)
e = lw.edges
K, M = 11, len(e) - 1
xf = x.reshape(-1)
gf = torch.randn(K, xf.numel(), device=dev)
lo, d = e[:-1, None], (e[1:] - e[:-1])[:, None]
for chunk in ():
    def a():
        gp = torch.zeros(K, M, device=dev)
        for s in range(0, xf.numel(), chunk):
            r = ((xf[None, s:s + chunk] - lo) / d).clamp_(0, 1)
            gp += gf[:, s:s + chunk] @ r.T
    def b():
        gp = torch.zeros(M, K, device=dev)
        for s in range(0, xf.numel(), chunk):
            r = ((xf[None, s:s + chunk] - lo) / d).clamp_(0, 1)
            gp += r @ gf[:, s:s + chunk].T
    print(f"chunk 2^{chunk.bit_length() - 1}: g @ r.T {t(a) * 1e3:7.1f} ms | r @ g.T {t(b) * 1e3:7.1f} ms", flush=True)

j = torch.bucketize(xf.clamp(e[0], e[-1]), e[1:-1])
frac = (xf.clamp(e[0], e[-1]) - e[j]) / (e[j + 1] - e[j])
def hist():
    S = torch.stack([torch.bincount(j, weights=gf[k], minlength=M) for k in range(K)])
    F_ = torch.stack([torch.bincount(j, weights=gf[k] * frac, minlength=M) for k in range(K)])
    return S.flip(-1).cumsum(-1).flip(-1) - S + F_
ju8 = j.to(torch.uint8)
def hist_u8():
    S = torch.stack([torch.bincount(ju8, weights=gf[k], minlength=M) for k in range(K)])
    F_ = torch.stack([torch.bincount(ju8, weights=gf[k] * frac, minlength=M) for k in range(K)])
    return S.flip(-1).cumsum(-1).flip(-1) - S + F_
def scat():
    S = torch.zeros(K, M, device=dev).index_add_(1, j, gf)
    return S
print(f"bincount histograms int64 {t(hist) * 1e3:7.1f} ms | uint8 {t(hist_u8) * 1e3:7.1f} ms", flush=True)
print(f"index_add (atomics)      {t(scat, 1) * 1e3:7.1f} ms", flush=True)
