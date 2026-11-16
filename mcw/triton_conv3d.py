"""3D convolution as implicit GEMM in Triton: the im2col columns are never materialised.

Prototype alternative to mcw/gemm_conv3d.py (same layout conventions, same parameters).
GemmConv3d copies, for every output voxel, its k^3 x C_in input window into a column buffer
(27x the input) and multiplies it with the (k^3*C_in, C_out) weight matrix. Here each Triton
program loads its input tiles straight from the padded channels-last input: for output rows m
and tap t = (dz, dy, dx), the A tile is x[base(m) + offset(t) + c], a contiguous run of
channels. The column buffer, and its write + read traffic, disappear.

Kernels (bf16 / fp16 / fp32 inputs, fp32 accumulation):
  _fwd_kernel     y[m, o]      = sum_{t, c} x[base(m) + off(t) + c] * W[t*C + c, o]
                  also used for grad_x at stride 1 (grad_out padded by k-1-p, flipped kernel),
                  as GemmConv3d does.
  _wgrad_kernel   dW[t*C + c, o] = sum_m x[base(m) + off(t) + c] * g[m, o]
                  split over m into SPLIT partial sums written to separate buffers and summed
                  afterwards: deterministic (no atomics), so runs stay bit-reproducible.
grad_x for stride != 1 (first conv, stride 3; Downsample3D, stride 2) keeps GemmConv3d's path:
those convs are a small share of the FLOPs.

Status: numerics validated on CPU with the Triton interpreter (tests/test_triton_conv3d.py).
Speed on MI250X / MI300A still to be measured (scripts/bench_conv3d_paths.py).
"""

import itertools
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from . import gemm_conv3d as G

_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HIP = torch.version.hip is not None


def _cfg(bm, bn, bk, warps, stages, **amd):
    kw = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}
    if _HIP:
        kw.update(amd)                      # AMD-only meta-parameters (unknown elsewhere)
    return triton.Config(kw, num_warps=warps, num_stages=stages)


if _INTERPRET:                              # CPU interpreter: one small config, no tuning
    _FWD_CONFIGS = [triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16})]
    _WGRAD_CONFIGS = [triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16})]
else:                                       # starting points; extend after the first benchmarks
    _FWD_CONFIGS = [
        _cfg(256, 64, 64, 8, 2, waves_per_eu=2, matrix_instr_nonkdim=16),
        _cfg(256, 128, 64, 8, 2, waves_per_eu=2, matrix_instr_nonkdim=16),
        _cfg(128, 64, 64, 4, 2, waves_per_eu=2, matrix_instr_nonkdim=16),
        _cfg(128, 128, 32, 4, 2, waves_per_eu=2, matrix_instr_nonkdim=16),
        _cfg(256, 256, 32, 8, 2, waves_per_eu=1, matrix_instr_nonkdim=16),
        _cfg(128, 256, 64, 8, 2, waves_per_eu=1),
    ]
    _WGRAD_CONFIGS = [
        _cfg(128, 64, 64, 4, 2, waves_per_eu=2, matrix_instr_nonkdim=16),
        _cfg(256, 64, 64, 8, 2, waves_per_eu=2, matrix_instr_nonkdim=16),
        _cfg(128, 128, 64, 8, 2, waves_per_eu=2, matrix_instr_nonkdim=16),
        _cfg(64, 64, 64, 4, 2, waves_per_eu=2, matrix_instr_nonkdim=16),
    ]


@triton.autotune(configs=_FWD_CONFIGS, key=["C", "O", "Ho", "Wo", "S", "K"])
@triton.jit
def _fwd_kernel(x_ptr, w_ptr, y_ptr,
                M, C, O, Do, Ho, Wo, Dp, Hp, Wp, S,
                K: tl.constexpr, DOT_IEEE: tl.constexpr, C_ALIGNED: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """x: padded channels-last (B, Dp, Hp, Wp, C); w: (K^3*C, O); y: (B, Do, Ho, Wo, O), contiguous."""
    pid = tl.program_id(0)
    n_pid_n = tl.cdiv(O, BLOCK_N)
    pid_m = pid // n_pid_n                  # consecutive programs share the same rows of x
    pid_n = pid % n_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < O

    ow = offs_m % Wo                        # output row m -> (b, d, h, w)
    t = offs_m // Wo
    oh = t % Ho
    t = t // Ho
    od = t % Do
    ob = t // Do
    base = (((ob * Dp + od * S) * Hp + oh * S) * Wp + ow * S).to(tl.int64) * C
    if C_ALIGNED:                           # C % 8 == 0: rows start on 16-byte boundaries (vector loads)
        base = tl.multiple_of(base, 8)

    n_cb = tl.cdiv(C, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, K * K * K * n_cb):    # flattened (tap, channel block) loop: long, pipelinable
        tap = i // n_cb
        c0 = (i % n_cb) * BLOCK_K
        dz = tap // (K * K)
        dy = (tap // K) % K
        dx = tap % K
        koff = ((dz * Hp + dy) * Wp + dx) * C + c0
        if C_ALIGNED:
            koff = tl.multiple_of(koff, 8)
        ck = c0 + offs_k
        mask_k = ck < C
        a = tl.load(x_ptr + base[:, None] + koff + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(w_ptr + (tap * C + ck)[:, None] * O + offs_n[None, :],
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        if DOT_IEEE:                         # fp32 inputs: exact products (no TF32)
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc += tl.dot(a, b)

    y_off = offs_m.to(tl.int64)[:, None] * O + offs_n[None, :]
    tl.store(y_ptr + y_off, acc.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(configs=_WGRAD_CONFIGS, key=["C", "O", "Ho", "Wo", "S", "K", "SPLIT"])
@triton.jit
def _wgrad_kernel(x_ptr, g_ptr, p_ptr,
                  M, C, O, Do, Ho, Wo, Dp, Hp, Wp, S, M_PER_SPLIT,
                  K: tl.constexpr, SPLIT: tl.constexpr, DOT_IEEE: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Partial weight gradients. x: padded channels-last input; g: (B, Do, Ho, Wo, O) grad_out;
    p: (SPLIT, K^3*C, O) fp32, p[s] = sum over rows m of split s."""
    pid_kc = tl.program_id(0)               # (tap, channel block)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    n_cb = tl.cdiv(C, BLOCK_K)
    tap = pid_kc // n_cb
    c0 = (pid_kc % n_cb) * BLOCK_K
    dz = tap // (K * K)
    dy = (tap // K) % K
    dx = tap % K
    koff = ((dz * Hp + dy) * Wp + dx) * C + c0

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = (c0 + offs_k) < C
    mask_n = offs_n < O

    m_start = pid_s * M_PER_SPLIT
    m_end = tl.minimum(m_start + M_PER_SPLIT, M)
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for m0 in range(m_start, m_end, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < m_end
        ow = offs_m % Wo
        t = offs_m // Wo
        oh = t % Ho
        t = t // Ho
        od = t % Do
        ob = t // Do
        base = (((ob * Dp + od * S) * Hp + oh * S) * Wp + ow * S).to(tl.int64) * C
        a = tl.load(x_ptr + base[None, :] + koff + offs_k[:, None],
                    mask=mask_k[:, None] & mask_m[None, :], other=0.0)          # (BLOCK_K, BLOCK_M)
        g = tl.load(g_ptr + offs_m.to(tl.int64)[:, None] * O + offs_n[None, :],
                    mask=mask_m[:, None] & mask_n[None, :], other=0.0)          # (BLOCK_M, BLOCK_N)
        if DOT_IEEE:
            acc += tl.dot(a, g, input_precision="ieee")
        else:
            acc += tl.dot(a, g)

    rows = tap * C + c0 + offs_k
    p_off = (pid_s * K * K * K * C + rows).to(tl.int64)[:, None] * O + offs_n[None, :]
    tl.store(p_ptr + p_off, acc, mask=mask_k[:, None] & mask_n[None, :])


def conv_cl(xcl: torch.Tensor, wm: torch.Tensor, k: int, s: int, out_dhw) -> torch.Tensor:
    """Drop-in for gemm_conv3d._conv_cl: valid conv of a padded channels-last input
    (B, Dp, Hp, Wp, C) with wm (k^3*C, O); returns channels-last (B, Do, Ho, Wo, O)."""
    xcl, wm = xcl.contiguous(), wm.contiguous()
    B, Dp, Hp, Wp, C = xcl.shape
    Do, Ho, Wo = out_dhw
    O = wm.shape[1]
    M = B * Do * Ho * Wo
    y = torch.empty((B, Do, Ho, Wo, O), device=xcl.device, dtype=xcl.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(O, meta["BLOCK_N"]),)
    _fwd_kernel[grid](xcl, wm, y, M, C, O, Do, Ho, Wo, Dp, Hp, Wp, s, K=k,
                      DOT_IEEE=xcl.dtype == torch.float32, C_ALIGNED=C % 8 == 0)
    return y


def wgrad_cl(xcl: torch.Tensor, gcl: torch.Tensor, k: int, s: int, split: int | None = None) -> torch.Tensor:
    """(k^3*C, O) fp32 weight gradient, rows ordered (dz, dy, dx, c) like gemm_conv3d._wmat."""
    xcl, gcl = xcl.contiguous(), gcl.contiguous()
    B, Dp, Hp, Wp, C = xcl.shape
    _, Do, Ho, Wo, O = gcl.shape
    M = B * Do * Ho * Wo
    if split is None:                       # enough programs to fill the GPU; partials stay small
        split = 1 if _INTERPRET else max(1, min(64, M // 16384))
    m_per = triton.cdiv(M, split)
    part = torch.empty((split, k ** 3 * C, O), device=xcl.device, dtype=torch.float32)
    grid = lambda meta: (k ** 3 * triton.cdiv(C, meta["BLOCK_K"]), triton.cdiv(O, meta["BLOCK_N"]), split)
    _wgrad_kernel[grid](xcl, gcl, part, M, C, O, Do, Ho, Wo, Dp, Hp, Wp, s, m_per, K=k, SPLIT=split,
                        DOT_IEEE=xcl.dtype == torch.float32)
    return part.sum(0)                      # fixed reduction order: deterministic


def _grad_x_strided(g: torch.Tensor, w: torch.Tensor, x_shape, s: int, p: int, cl: bool) -> torch.Tensor:
    """grad_x for stride > 1: GemmConv3d's path (GEMM to grad columns + scatter / tiled write)."""
    B, C, D, H, W = x_shape
    O, k = w.shape[0], w.shape[2]
    Do, Ho, Wo = g.shape[2:]
    kc = k ** 3 * C
    gcl = G._to_cl(g, 0)
    wm = G._wmat(w)
    Dp, Hp, Wp = D + 2 * p, H + 2 * p, W + 2 * p
    tiled = s == k
    gx = torch.zeros((B, Dp, Hp, Wp, C), device=g.device, dtype=g.dtype if tiled else torch.float32)
    rows = max(1, min(Do, G.CHUNK_ELEMS // max(1, Ho * Wo * kc)))
    for b in range(B):
        for d0 in range(0, Do, rows):
            nd = min(rows, Do - d0)
            gcol = torch.mm(gcl[b, d0:d0 + nd].reshape(-1, O), wm.t()).view(nd, Ho, Wo, k, k, k, C)
            if tiled:
                G._copy(G._windows(gx[b], d0, nd, Ho, Wo, k, s), gcol)
            else:
                for dz, dy, dx in itertools.product(range(k), repeat=3):
                    z0 = d0 * s + dz
                    gx[b, z0:z0 + s * (nd - 1) + 1:s, dy:dy + s * (Ho - 1) + 1:s,
                       dx:dx + s * (Wo - 1) + 1:s, :] += gcol[:, :, :, dz, dy, dx, :]
    return G._from_cl(gx[:, p:p + D, p:p + H, p:p + W, :].to(g.dtype).contiguous(), cl)


class _TritonConv3dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, stride: int, padding: int):
        k, s, p = w.shape[2], stride, padding
        out_dhw = tuple(G._out_size(n, k, s, p) for n in x.shape[2:])
        cl = x.is_contiguous(memory_format=torch.channels_last_3d)
        out = conv_cl(G._to_cl(x, p), G._wmat(w), k, s, out_dhw)
        ctx.save_for_backward(x, w)
        ctx.conf = (s, p, cl)
        return G._from_cl(out, cl)

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        s, p, cl = ctx.conf
        k = w.shape[2]
        g = grad_out.to(x.dtype)
        grad_x = grad_w = None
        if ctx.needs_input_grad[1]:
            gw = wgrad_cl(G._to_cl(x, p), G._to_cl(g, 0), k, s)
            grad_w = gw.view(k, k, k, w.shape[1], w.shape[0]).permute(4, 3, 0, 1, 2).to(w.dtype).contiguous()
        if ctx.needs_input_grad[0]:
            if s == 1:                      # full correlation with the flipped kernel, C_in <-> C_out
                w_t = w.flip(2, 3, 4).transpose(0, 1)
                gx = conv_cl(G._to_cl(g, k - 1 - p), G._wmat(w_t), k, 1, tuple(x.shape[2:]))
                grad_x = G._from_cl(gx, cl)
            else:
                grad_x = _grad_x_strided(g, w, x.shape, s, p, cl)
        return grad_x, grad_w, None, None


def conv3d(x, w, bias=None, stride: int = 1, padding: int = 0):
    """F.conv3d (cubic kernel, isotropic int stride / padding, groups=1) through Triton."""
    if torch.is_autocast_enabled() and x.is_cuda:
        dt = torch.get_autocast_gpu_dtype()
        x, w = x.to(dt), w.to(dt)
    else:
        w = w.to(x.dtype)
    y = _TritonConv3dFn.apply(x, w, int(stride), int(padding))
    if bias is not None:
        y = y + bias.to(y.dtype).view(1, -1, 1, 1, 1)
    return y


class TritonConv3d(nn.Conv3d):
    """nn.Conv3d computed with the implicit-GEMM Triton kernels (same parameters, same state_dict)."""

    def forward(self, x):
        if (len(set(self.kernel_size)) != 1 or len(set(self.stride)) != 1 or len(set(self.padding)) != 1
                or self.groups != 1 or self.dilation != (1, 1, 1) or self.padding_mode != "zeros"):
            return super().forward(x)
        return conv3d(x, self.weight, self.bias, self.stride[0], self.padding[0])


def swap_conv3d_triton(module: nn.Module) -> int:
    """Replace every nn.Conv3d / GemmConv3d under ``module`` by a TritonConv3d sharing its parameters."""
    n = 0
    for name, child in list(module.named_children()):
        if type(child) in (nn.Conv3d, G.GemmConv3d):
            new = TritonConv3d(child.in_channels, child.out_channels, child.kernel_size, child.stride,
                               child.padding, child.dilation, child.groups, child.bias is not None,
                               child.padding_mode, device="meta")
            new.weight = child.weight
            new.bias = child.bias
            setattr(module, name, new)
            n += 1
        else:
            n += swap_conv3d_triton(child)
    return n
