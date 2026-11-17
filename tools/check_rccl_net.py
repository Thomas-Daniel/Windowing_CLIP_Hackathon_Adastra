"""RCCL network check: all-reduce bus bandwidth inside each node (4 GPUs) then over all ranks
(between nodes), plus a correctness check. Launched by tools/check_rccl_net.sh (srun, one
process per GPU); the RCCL INFO logs (NCCL_DEBUG_FILE) say which network was used."""
import os
import socket
import time

import torch
import torch.distributed as dist


def bench(group, n_elems: int, iters: int = 20) -> tuple[float, float]:
    x = torch.zeros(n_elems, dtype=torch.float32, device="cuda")
    for _ in range(5):                                    # warm-up (communicator setup, channels)
        dist.all_reduce(x, group=group)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x, group=group)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    n = dist.get_world_size(group)
    busbw = x.numel() * x.element_size() * 2 * (n - 1) / n / dt / 1e9     # GB/s, rccl-tests convention
    return dt, busbw


def main() -> None:
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    local, per_node = int(os.environ["SLURM_LOCALID"]), int(os.environ["SLURM_NTASKS_PER_NODE"].split("(")[0])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    # correctness: sum of the ranks
    y = torch.full((1024,), float(rank), device="cuda")
    dist.all_reduce(y)
    ok = bool((y == world * (world - 1) / 2).all())

    nodes = world // per_node
    node_groups = [dist.new_group(list(range(k * per_node, (k + 1) * per_node))) for k in range(nodes)]
    my_node = node_groups[rank // per_node]

    if rank == 0:
        print(f"world={world} ({nodes} nodes x {per_node}) host0={socket.gethostname()} all_reduce correct={ok}", flush=True)
        print(f"{'size':>8} {'intra-node GB/s':>16} {'all ranks GB/s':>15} {'all ranks ms':>13}", flush=True)
    for mb in (1, 16, 256, 1024):
        n = mb * 2**20 // 4
        _, bw_in = bench(my_node, n)
        dt, bw_all = bench(None, n)
        if rank == 0:
            print(f"{mb:>6}MB {bw_in:>16.1f} {bw_all:>15.1f} {dt * 1e3:>13.2f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if not ok:
        raise SystemExit("all_reduce result wrong")


if __name__ == "__main__":
    main()