"""3D convolution as chunked im2col + GEMM, a drop-in for nn.Conv3d on ROCm.

MIOpen (torch 2.5.1+rocm6.2, MI250X) has no usable kernel for the Atlas conv stem: a
192->192 3x3x3 conv on a 128^3 volume takes minutes per forward/backward in bf16 and
fp16, whether the layout is NCDHW or NDHWC (scripts/bench_conv3d.py, job 5563098). This
module computes the same convolution with rocBLAS GEMMs (~100 TFLOP/s for these shapes
in bf16).

Forward: pad the input once and lay it out channels-last, (B, Dp, Hp, Wp, C). For each slab
of output depth slices, a single ``as_strided`` view (slab, Ho, Wo, k, k, k, C) of it is
copied into a column buffer and multiplied by the (k^3 * C_in, C_out) weight matrix.

Backward:
  grad_W  the column buffer is recomputed per slab and multiplied with grad_out
          (col^T @ grad_out), accumulated in fp32.
  grad_x  depends on the stride:
          stride 1    a forward convolution of grad_out with the flipped, transposed
                      kernel (padding k - 1 - p), through the same GEMM path;
          stride = k  windows tile without overlap, so grad_col is written back through
                      the same as_strided view;
          otherwise   grad_col is scatter-added with k^3 shifted adds, in fp32.

Only x and W are saved for backward, and the column buffer stays around ``CHUNK_ELEMS``
elements. Window copies move 16-byte groups of channels (viewed as complex128), which
runs ~4x faster than element-wise bf16 copies on MI250X.

Memory format: a ``channels_last_3d`` input is used as is and gives a ``channels_last_3d``
output, so a stem run in that format never transposes. NCDHW <-> channels-last copies cost
~40 ms per 128^3 x 192 tensor (scripts/bench_im2col.py), more than the GEMMs.
AtlasBackbone does this for the conv stem. Under autocast the inputs are cast to the autocast dtype, as for a native conv.
``GemmConv3d`` subclasses nn.Conv3d: same parameters, same state_dict keys, same numerics
up to summation order (tests/test_cpu.py).
"""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

CHUNK_ELEMS = 1 << 29            # column buffer size (elements): 1 GB in bf16


def _out_size(n: int, k: int, s: int, p: int) -> int:
    return (n + 2 * p - k) // s + 1


def _windows(xcl_b: torch.Tensor, d0: int, nd: int, Ho: int, Wo: int, k: int, s: int) -> torch.Tensor:
    """(nd, Ho, Wo, k, k, k, C) view of the windows of output slices d0 .. d0+nd-1.
    xcl_b: one padded channels-last volume (Dp, Hp, Wp, C), contiguous."""
    sD, sH, sW, sC = xcl_b.stride()
    return xcl_b.as_strided((nd, Ho, Wo, k, k, k, xcl_b.shape[3]),
                            (s * sD, s * sH, s * sW, sD, sH, sW, sC),
                            xcl_b.storage_offset() + d0 * s * sD)


def _copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """dst.copy_(src), moving 16-byte groups of channels when the layout allows it."""
    r = 16 // dst.element_size()
    if (dst.shape[-1] % r == 0 and src.stride(-1) == 1 and dst.stride(-1) == 1
            and all(st % r == 0 for st in src.stride()[:-1]) and src.storage_offset() % r == 0
            and all(st % r == 0 for st in dst.stride()[:-1]) and dst.storage_offset() % r == 0):
        dst.view(torch.complex128).copy_(src.view(torch.complex128))
    else:
        dst.copy_(src)


def _conv_cl(xcl: torch.Tensor, wm: torch.Tensor, k: int, s: int, out_dhw) -> torch.Tensor:
    """Valid convolution of a padded channels-last input (B, Dp, Hp, Wp, C) with the weight
    matrix wm (k^3*C, O): returns channels-last (B, Do, Ho, Wo, O)."""
    B, C = xcl.shape[0], xcl.shape[4]
    Do, Ho, Wo = out_dhw
    kc, O = wm.shape
    out = torch.empty((B, Do, Ho, Wo, O), device=xcl.device, dtype=xcl.dtype)
    rows = max(1, min(Do, CHUNK_ELEMS // max(1, Ho * Wo * kc)))
    col = torch.empty((rows, Ho, Wo, k, k, k, C), device=xcl.device, dtype=xcl.dtype)
    for b in range(B):
        for d0 in range(0, Do, rows):
            nd = min(rows, Do - d0)
            c = col[:nd]
            _copy(c, _windows(xcl[b], d0, nd, Ho, Wo, k, s))
            torch.mm(c.view(-1, kc), wm, out=out[b, d0:d0 + nd].view(-1, O))
    return out


def _wmat(w: torch.Tensor) -> torch.Tensor:
    """(O, C, k, k, k) -> (k^3 * C, O), rows ordered (dz, dy, dx, c) like the windows view."""
    O, C, k = w.shape[0], w.shape[1], w.shape[2]
    return w.permute(2, 3, 4, 1, 0).reshape(k ** 3 * C, O)


def _to_cl(x: torch.Tensor, p: int) -> torch.Tensor:
    """(B, C, D, H, W) in any memory format -> padded channels-last (B, Dp, Hp, Wp, C), contiguous.
    Free of transposition when x is channels_last_3d."""
    xcl = x.permute(0, 2, 3, 4, 1)
    if p:
        return F.pad(xcl, (0, 0, p, p, p, p, p, p))           # contiguous output
    return xcl.contiguous()


def _from_cl(ycl: torch.Tensor, channels_last: bool) -> torch.Tensor:
    """(B, D, H, W, C) contiguous -> (B, C, D, H, W): a channels_last_3d view, or an NCDHW copy."""
    y = ycl.permute(0, 4, 1, 2, 3)
    return y if channels_last else y.contiguous()


class _GemmConv3dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, stride: int, padding: int) -> torch.Tensor:
        k, s, p = w.shape[2], stride, padding
        out_dhw = tuple(_out_size(n, k, s, p) for n in x.shape[2:])
        cl = x.dim() == 5 and x.is_contiguous(memory_format=torch.channels_last_3d)
        out = _conv_cl(_to_cl(x, p), _wmat(w), k, s, out_dhw)
        ctx.save_for_backward(x, w)
        ctx.conf = (s, p, cl)
        return _from_cl(out, cl)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w = ctx.saved_tensors
        s, p, cl = ctx.conf
        B, C, D, H, W = x.shape
        O, k = w.shape[0], w.shape[2]
        Do, Ho, Wo = grad_out.shape[2:]
        kc = k ** 3 * C
        g = grad_out.to(x.dtype)
        grad_x = grad_w = None

        if ctx.needs_input_grad[1]:
            xcl = _to_cl(x, p)
            gcl = _to_cl(g, 0)                                             # (B, Do, Ho, Wo, O)
            gw = torch.zeros((kc, O), device=x.device, dtype=torch.float32)
            rows = max(1, min(Do, CHUNK_ELEMS // max(1, Ho * Wo * kc)))
            col = torch.empty((rows, Ho, Wo, k, k, k, C), device=x.device, dtype=x.dtype)
            for b in range(B):
                for d0 in range(0, Do, rows):
                    nd = min(rows, Do - d0)
                    c = col[:nd]
                    _copy(c, _windows(xcl[b], d0, nd, Ho, Wo, k, s))
                    gw.add_(torch.mm(c.view(-1, kc).t(), gcl[b, d0:d0 + nd].reshape(-1, O)))
            del xcl, col
            grad_w = gw.view(k, k, k, C, O).permute(4, 3, 0, 1, 2).to(w.dtype).contiguous()

        if ctx.needs_input_grad[0]:
            if s == 1:
                # full correlation of grad_out with the flipped kernel, C_in <-> C_out swapped
                w_t = w.flip(2, 3, 4).transpose(0, 1)                      # (C, O, k, k, k)
                gx = _conv_cl(_to_cl(g, k - 1 - p), _wmat(w_t), k, 1, (D, H, W))
                grad_x = _from_cl(gx, cl)
            else:
                gcl = _to_cl(g, 0)
                wm = _wmat(w)
                Dp, Hp, Wp = D + 2 * p, H + 2 * p, W + 2 * p
                tiled = s == k
                gx = torch.zeros((B, Dp, Hp, Wp, C), device=x.device, dtype=x.dtype if tiled else torch.float32)
                rows = max(1, min(Do, CHUNK_ELEMS // max(1, Ho * Wo * kc)))
                for b in range(B):
                    for d0 in range(0, Do, rows):
                        nd = min(rows, Do - d0)
                        gcol = torch.mm(gcl[b, d0:d0 + nd].reshape(-1, O), wm.t()).view(nd, Ho, Wo, k, k, k, C)
                        if tiled:
                            _copy(_windows(gx[b], d0, nd, Ho, Wo, k, s), gcol)
                        else:
                            for dz, dy, dx in itertools.product(range(k), repeat=3):
                                z0 = d0 * s + dz
                                gx[b, z0:z0 + s * (nd - 1) + 1:s, dy:dy + s * (Ho - 1) + 1:s,
                                   dx:dx + s * (Wo - 1) + 1:s, :] += gcol[:, :, :, dz, dy, dx, :]
                grad_x = _from_cl(gx[:, p:p + D, p:p + H, p:p + W, :].to(x.dtype).contiguous(), cl)
        return grad_x, grad_w, None, None


def conv3d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None, stride: int = 1,
           padding: int = 0) -> torch.Tensor:
    """F.conv3d (cubic kernel, isotropic int stride/padding, groups=1) through the GEMM path.
    Under autocast, x and w are cast to the autocast dtype like a native conv."""
    if torch.is_autocast_enabled() and x.is_cuda:
        dt = torch.get_autocast_gpu_dtype()
        x, w = x.to(dt), w.to(dt)
    else:
        w = w.to(x.dtype)
    y = _GemmConv3dFn.apply(x, w, int(stride), int(padding))
    if bias is not None:
        y = y + bias.to(y.dtype).view(1, -1, 1, 1, 1)
    return y


class GemmConv3d(nn.Conv3d):
    """nn.Conv3d (cubic kernel, isotropic stride/padding, groups=1) computed with GEMMs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (len(set(self.kernel_size)) != 1 or len(set(self.stride)) != 1 or len(set(self.padding)) != 1
                or self.groups != 1 or self.dilation != (1, 1, 1) or self.padding_mode != "zeros"):
            return super().forward(x)
        return conv3d(x, self.weight, self.bias, self.stride[0], self.padding[0])


def swap_conv3d(module: nn.Module) -> int:
    """Replace every nn.Conv3d under ``module`` by a GemmConv3d sharing its parameters."""
    n = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Conv3d:
            new = GemmConv3d(child.in_channels, child.out_channels, child.kernel_size, child.stride,
                             child.padding, child.dilation, child.groups, child.bias is not None,
                             child.padding_mode, device="meta")
            new.weight = child.weight
            new.bias = child.bias
            setattr(module, name, new)
            n += 1
        else:
            n += swap_conv3d(child)
    return n
