"""Where does the Atlas step time go? Times (one GCD, synthetic 384^3, micro-batch 1):
the conv stem vs the attention stages, forward and backward, plus a raw GEMM rate for
the stem's column GEMM shape, and a torch.profiler table of the top kernels.

    python scripts/profile_model.py --windowing pillar11
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcw.config import load_config  # noqa: E402
from mcw.model import CTClip  # noqa: E402


def sync_time(fn, reps=2):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windowing", default="pillar11")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    cfg = load_config({"clip": "configs/win1_clip.yaml", "pillar11": "configs/win2_pillar11.yaml",
                       "learned": "configs/win3_learned.yaml", "local": "configs/win4_local.yaml",
                       "abdo": "configs/win5_abdo.yaml", "dense": "configs/win6_dense.yaml"}[args.windowing])
    model = CTClip(cfg, 4096).to(dev).train()
    atlas = model.visual.atlas
    pe = atlas.patch_embeds["abdomen_ct"]

    for M, K, N in ((16384, 5184, 192), (65536, 5184, 192), (16384, 192, 5184), (5184, 16384, 192)):
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        b = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
        t = sync_time(lambda: a @ b, 5)
        print(f"GEMM bf16 {M}x{K} @ {K}x{N}: {t * 1e3:6.2f} ms, {2 * M * K * N / t / 1e12:5.1f} TFLOP/s", flush=True)
    x = torch.randn(16384 * 27 * 192 // 192, 192, device=dev, dtype=torch.bfloat16)
    t = sync_time(lambda: x.clone(), 5)
    print(f"copy {x.numel() * 2 / 2**30:.2f} GB: {t * 1e3:.1f} ms, {2 * x.numel() * 2 / t / 1e12:.2f} TB/s", flush=True)

    hu = torch.randint(-1024, 2000, (1, 1, 384, 384, 384), device=dev).float()
    with torch.no_grad():
        xin = model.window(hu)
    ac = dict(device_type="cuda", dtype=torch.bfloat16)

    def stem_fwd():
        with torch.no_grad(), torch.autocast(**ac):
            pe(xin)

    def stem_fb():
        with torch.autocast(**ac):
            y = pe(xin)
        y.float().sum().backward()

    with torch.no_grad(), torch.autocast(**ac):
        tok = pe(xin).detach()
    print(f"stem output {tuple(tok.shape)}", flush=True)

    def body(t_in, grad):
        scales, _ = atlas._build_multiscale_tokens(t_in, merge_ratio=[4, 4, 4], local2global=[4, 4, 4])
        layout = atlas.prepare_multiscale_layout(img_size=[384] * 3, merge_ratio=[4, 4, 4], local2global=[4, 4, 4],
                                                 patch_size=[6, 6, 6])
        grids = [l["grid_size"] for l in layout]
        x = scales
        for level, st in enumerate(atlas.atlas_models):
            x = st(x, grid_sizes=grids[level:], multiscale_layout=layout[level:], merge_ratio=[4, 4, 4],
                   local2global=[4, 4, 4], modality="abdomen_ct")
            if level < len(atlas.atlas_models) - 1:
                x = x[1:]
        return x[0]

    def body_fwd():
        with torch.no_grad(), torch.autocast(**ac):
            body(tok, False)

    def body_fb():
        t_in = tok.clone().requires_grad_(True)
        with torch.autocast(**ac):
            y = body(t_in, True)
        y.float().sum().backward()

    for name, fn in (("stem fwd", stem_fwd), ("stem fwd+bwd", stem_fb), ("attention fwd", body_fwd),
                     ("attention fwd+bwd", body_fb)):
        torch.cuda.reset_peak_memory_stats(dev)
        t = sync_time(fn, 2)
        print(f"{name:20s} {t:6.2f} s  peak {torch.cuda.max_memory_allocated(dev) / 2**30:5.1f} GB", flush=True)
        model.zero_grad(set_to_none=True)

    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            stem_fb()
            body_fb()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    main()
