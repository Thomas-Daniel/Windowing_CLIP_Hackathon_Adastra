"""Micro-benchmark of im2col copy strategies for the 192->192 3x3x3 conv at 128^3 (bf16),
and of 2-D MIOpen convs as an alternative (3D conv = sum over kz of 2-D convs)."""
from __future__ import annotations

import itertools
import time

import torch
import torch.nn.functional as F


def t(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


dev = "cuda"
C, O, n, k = 192, 192, 128, 3
x = torch.randn(1, C, n, n, n, device=dev, dtype=torch.bfloat16)
xcl = F.pad(x, (1,) * 6).permute(0, 2, 3, 4, 1).contiguous()[0]          # (130,130,130,C)
rows = 6
col = torch.empty((rows, n, n, k, k, k, C), device=dev, dtype=torch.bfloat16)
sD, sH, sW, sC = xcl.stride()
win = xcl.as_strided((rows, n, n, k, k, k, C), (sD, sH, sW, sD, sH, sW, sC))
gb = col.numel() * 2 / 1e9
print(f"col slab {tuple(col.shape)} = {gb:.2f} GB", flush=True)

print(f"pad+permute to channels-last: {t(lambda: F.pad(x, (1,) * 6).permute(0, 2, 3, 4, 1).contiguous()) * 1e3:.1f} ms")
print(f"permute back to NCDHW:        {t(lambda: xcl[:n, :n, :n].permute(3, 0, 1, 2).contiguous()) * 1e3:.1f} ms")
dt = t(lambda: col.copy_(win))
print(f"7-D as_strided copy bf16:     {dt * 1e3:.1f} ms  ({2 * gb / dt:.0f} GB/s)")
wc, cc = win.view(torch.complex128), col.view(torch.complex128)
dt = t(lambda: cc.copy_(wc))
print(f"7-D as_strided copy c128:     {dt * 1e3:.1f} ms  ({2 * gb / dt:.0f} GB/s)")


def shifts(dtype_view):
    src = xcl.view(dtype_view) if dtype_view else xcl
    dst = col.view(dtype_view) if dtype_view else col
    for dz, dy, dx in itertools.product(range(k), repeat=3):
        dst[:, :, :, dz, dy, dx, :] = src[dz:dz + rows, dy:dy + n, dx:dx + n, :]


dt = t(lambda: shifts(None))
print(f"27 x 4-D copies bf16:         {dt * 1e3:.1f} ms  ({2 * gb / dt:.0f} GB/s)")
dt = t(lambda: shifts(torch.complex128))
print(f"27 x 4-D copies c128:         {dt * 1e3:.1f} ms  ({2 * gb / dt:.0f} GB/s)")
w = torch.randn(k ** 3 * C, O, device=dev, dtype=torch.bfloat16)
dt = t(lambda: col.view(-1, k ** 3 * C) @ w)
print(f"slab GEMM ({rows * n * n} x {k ** 3 * C} @ {k ** 3 * C} x {O}): {dt * 1e3:.1f} ms "
      f"({2 * rows * n * n * k ** 3 * C * O / dt / 1e12:.0f} TFLOP/s); x {n // rows + 1} slabs per conv")

# 2-D convs: (B*D, C, H, W) with a 3x3 kernel, MIOpen
for fmt in ("nchw", "nhwc"):
    for bd in (16, 64):
        x2 = torch.randn(bd, C, n, n, device=dev, dtype=torch.bfloat16)
        conv = torch.nn.Conv2d(C, O, 3, padding=1, bias=False).to(dev, torch.bfloat16)
        if fmt == "nhwc":
            x2, conv = x2.to(memory_format=torch.channels_last), conv.to(memory_format=torch.channels_last)
        x2.requires_grad_(True)
        try:
            dt_f = t(lambda: conv(x2), 3)
            dt_b = t(lambda: conv(x2).backward(torch.ones(bd, O, n, n, device=dev, dtype=torch.bfloat16)), 3)
            fl = 2 * C * O * 9 * n * n * bd
            print(f"conv2d {fmt} batch {bd}: fwd {dt_f * 1e3:.1f} ms ({fl / dt_f / 1e12:.0f} TFLOP/s), "
                  f"fwd+bwd {dt_b * 1e3:.1f} ms ({3 * fl / dt_b / 1e12:.0f} TFLOP/s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"conv2d {fmt} batch {bd}: failed {e}")
