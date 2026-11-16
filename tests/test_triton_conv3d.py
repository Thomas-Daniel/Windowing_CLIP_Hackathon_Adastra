"""TritonConv3d == nn.Conv3d (output, grad_x, grad_W) on the stem's stride / padding cases.

Runs on CPU through the Triton interpreter (no GPU needed), or on a GPU:
    TRITON_INTERPRET=1 python -m tests.test_triton_conv3d      # CPU, ~1-2 min
    python -m tests.test_triton_conv3d                         # GPU (fp32 then bf16)
"""
from __future__ import annotations

import copy
import os

import torch

from mcw import gemm_conv3d as G
from mcw import triton_conv3d as T

DEV = "cpu" if os.environ.get("TRITON_INTERPRET", "0") == "1" else "cuda"

# (C_in, C_out, stride, padding, bias, spatial): stem-like cases, C_in multiple / not multiple of BLOCK_K
CASES = [
    (3, 5, 1, 1, True, (9, 10, 11)),        # generic stride 1, tiny C (masked channel blocks)
    (32, 24, 1, 1, False, (8, 9, 10)),      # stride 1, C a multiple of BLOCK_K (ConvBlock3D-like)
    (11, 16, 3, 0, False, (12, 12, 15)),    # first conv: stride = kernel = 3, no padding
    (20, 12, 2, 1, False, (9, 10, 11)),     # Downsample3D-like: stride 2, padding 1
]


def check(dtype: torch.dtype, atol: float, rtol: float) -> None:
    torch.manual_seed(0)
    for cin, cout, s, p, bias, dhw in CASES:
        ref = torch.nn.Conv3d(cin, cout, 3, stride=s, padding=p, bias=bias).to(DEV)
        m = torch.nn.Sequential(copy.deepcopy(ref))
        G.swap_conv3d(m)                                    # nn.Conv3d -> GemmConv3d (as in the model)
        assert T.swap_conv3d_triton(m) == 1 and isinstance(m[0], T.TritonConv3d)
        assert list(m.state_dict()) == ["0." + k for k in ref.state_dict()]
        for fmt in (torch.contiguous_format, torch.channels_last_3d):
            x0 = torch.randn(2, cin, *dhw, device=DEV)
            x = x0.clone().to(dtype).contiguous(memory_format=fmt).requires_grad_(True)
            xr = x0.clone().requires_grad_(True)
            m.zero_grad(set_to_none=True)
            ref.zero_grad(set_to_none=True)
            mt = m.to(dtype)
            y = mt(x)
            yr = ref(xr)
            assert y.is_contiguous(memory_format=fmt), (cin, s, fmt)
            gy = torch.randn_like(yr)
            y.backward(gy.to(dtype).contiguous(memory_format=fmt))
            yr.backward(gy)
            err = lambda a, b: f"max |diff| {float((a.float() - b).abs().max()):.2e}"
            assert torch.allclose(y.float(), yr, atol=atol, rtol=rtol), ("out", cin, s, fmt, err(y, yr))
            assert torch.allclose(x.grad.float(), xr.grad, atol=atol, rtol=rtol), ("grad_x", cin, s, fmt, err(x.grad, xr.grad))
            gw = mt[0].weight.grad.float()
            assert torch.allclose(gw, ref.weight.grad, atol=atol * 10, rtol=rtol), ("grad_w", cin, s, fmt, err(gw, ref.weight.grad))
            m = mt.float()
        print(f"  {dtype} C_in={cin:2d} C_out={cout:2d} stride={s} padding={p}: OK", flush=True)


def check_wgrad_deterministic() -> None:
    torch.manual_seed(1)
    xcl = torch.randn(1, 10, 10, 10, 16, device=DEV)
    gcl = torch.randn(1, 8, 8, 8, 8, device=DEV)
    a = T.wgrad_cl(xcl, gcl, 3, 1, split=4)
    b = T.wgrad_cl(xcl, gcl, 3, 1, split=4)
    assert torch.equal(a, b)                                # same split -> bit-identical
    c = T.wgrad_cl(xcl, gcl, 3, 1, split=1)
    assert torch.allclose(a, c, atol=1e-4, rtol=1e-4)       # split only changes the summation order
    print("  wgrad deterministic, split-invariant: OK", flush=True)


if __name__ == "__main__":
    print(f"device: {DEV}", flush=True)
    check(torch.float32, atol=1e-4, rtol=1e-4)
    if DEV == "cuda":
        check(torch.bfloat16, atol=0.1, rtol=0.05)
    check_wgrad_deterministic()
    print("triton conv3d OK")
