"""Minimal torch.distributed helpers (SLURM-native launch through scripts/adastra_bind.sh)."""
from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def setup() -> tuple[int, int, int]:
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if world > 1:
        # 30 min instead of the 10 min default: on 2026-10-23, 2 of 8 runs started together on 32 nodes hit the
        # NCCL watchdog on an early collective while one rank was still slow to start (no step done yet)
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", local),
                                timeout=datetime.timedelta(minutes=int(os.environ.get("NCCL_TIMEOUT_MIN", 30))))
    return rank, world, local


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def gather_cat(x: torch.Tensor) -> torch.Tensor:
    """All-gather along dim 0, no gradient (same size on every rank)."""
    if not is_dist():
        return x
    out = [torch.empty_like(x) for _ in range(world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out)


def gather_with_local_grad(x: torch.Tensor) -> torch.Tensor:
    """All-gather along dim 0; this rank's slice keeps its autograd graph."""
    if not is_dist():
        return x
    parts = [torch.empty_like(x) for _ in range(world_size())]
    dist.all_gather(parts, x.detach().contiguous())
    parts[rank()] = x
    return torch.cat(parts)


def all_reduce_grads(params) -> None:
    """Sum the gradients over ranks: one all-reduce per dtype on a flat buffer instead of one per
    tensor. Params whose .grad is None are skipped (same set on every rank: same model, same graph)."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return
    by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
    for p in params:
        if p.grad is not None:
            by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
    for grads in by_dtype.values():                    # same order on every rank (parameter order)
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat)                          # SUM, as before: train.py divides only logit_scale
        for g, f in zip(grads, _unflatten_dense_tensors(flat, grads)):
            g.copy_(f)


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if is_dist():
        x = x.clone()
        dist.all_reduce(x)
        x /= world_size()
    return x


def barrier() -> None:
    if is_dist():
        dist.barrier()


def cleanup() -> None:
    if is_dist():
        dist.destroy_process_group()


def average_buffers(module: torch.nn.Module) -> None:
    """Average floating-point buffers (BatchNorm running stats) over ranks: each rank's BN
    statistics come from its own micro-batches, so sync them before eval / checkpointing."""
    if not is_dist():
        return
    for b in module.buffers():
        if b.is_floating_point():
            dist.all_reduce(b)
            b /= world_size()


def any_true(flag: bool, device) -> bool:
    """True on every rank if ``flag`` is True on at least one rank."""
    if not is_dist():
        return flag
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def sum_int(x: int, device) -> int:
    """Sum of an integer over ranks."""
    if not is_dist():
        return int(x)
    t = torch.tensor([int(x)], device=device, dtype=torch.int64)
    dist.all_reduce(t)
    return int(t.item())
