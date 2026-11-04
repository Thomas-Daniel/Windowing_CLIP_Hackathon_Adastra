"""Single-GCD speed / memory benchmark of the training step pieces, on synthetic 384^3 HU.

For each windowing and micro-batch size: phase-1 forward (no grad) and phase-2
forward+backward times per micro-batch, peak memory, with and without activation
checkpointing. One optimizer step costs ~ accum_steps x (fwd_nograd + fwd_bwd).

    python scripts/bench_step.py --windowing clip pillar11 learned local --micro-batch 1 2 4
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

CONFIGS = {"clip": "configs/win1_clip.yaml", "pillar11": "configs/win2_pillar11.yaml",
           "learned": "configs/win3_learned.yaml", "local": "configs/win4_local.yaml",
           "abdo": "configs/win5_abdo.yaml", "dense": "configs/win6_dense.yaml",
           "local_shuf": "configs/win7_local_shuf.yaml", "patch_bank": "configs/win8_patch_bank.yaml",
           "global_bank": "configs/win11_global_bank.yaml", "patch_cdf": "configs/win9_patch_cdf.yaml",
           "vit_clip": "configs/mixvit_clip.yaml", "vit_patch_bank": "configs/mixvit_patch_bank.yaml",
           "mix_patch_bank": "configs/mix_patch_bank.yaml", "mix_patch_bank_v2": "configs/mix_patch_bank_v2.yaml",
           "mix_patch_route": "configs/mix_patch_route.yaml"}


def timed(fn, reps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windowing", nargs="+", default=list(CONFIGS))
    ap.add_argument("--micro-batch", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--ckpt", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--conv-impl", nargs="+", default=["gemm"], help="gemm (mcw/gemm_conv3d.py) and/or native (MIOpen)")
    ap.add_argument("--sdpa", action="store_true", help="also time the attention backends on Atlas-like shapes")
    ap.add_argument("--density", action="store_true", help="also time the patch-density histogram (bincount vs sort)")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    print(f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).gcnArchName}), torch {torch.__version__}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB", flush=True)
    if args.sdpa:
        sdpa_bench(dev, args.reps)
    if args.density:
        density_bench(dev, args.reps)

    for kind, ck, conv in ((k, c, v) for k in args.windowing for c in args.ckpt for v in args.conv_impl):
        if True:
            cfg = load_config(CONFIGS[kind])
            cfg["model"]["grad_checkpointing"] = bool(ck)
            cfg["model"]["conv_impl"] = conv
            torch.manual_seed(0)
            model = CTClip(cfg, text_dim=4096).to(dev).train()
            for mb in args.micro_batch:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                hu = torch.randint(-1024, 2000, (mb, 1, 384, 384, 384), device=dev).half()
                try:
                    w = timed(lambda: model.window(hu.float()), args.reps)

                    def fwd():
                        with torch.no_grad():
                            model.encode_image(hu.float())

                    def fwd_bwd():
                        f = model.encode_image(hu.float())
                        f.backward(torch.randn_like(f))

                    fwd()                                   # warm-up (MIOpen kernel search)
                    fwd_bwd()
                    torch.cuda.reset_peak_memory_stats(dev)
                    t_f = timed(fwd, args.reps)
                    t_fb = timed(fwd_bwd, args.reps)
                    mem = torch.cuda.max_memory_allocated(dev) / 2**30
                    print(f"{kind:11s} conv={conv:6s} ckpt={ck} mb={mb}: window {w * 1e3:6.0f} ms | fwd(no grad) {t_f:6.2f} s | "
                          f"fwd+bwd {t_fb:6.2f} s | per volume {(t_f + t_fb) / mb:5.2f} s | peak {mem:5.1f} GB", flush=True)
                except torch.OutOfMemoryError:
                    print(f"{kind:11s} conv={conv:6s} ckpt={ck} mb={mb}: OOM", flush=True)
                except Exception as e:                       # e.g. MIOpen errors: report and go on
                    print(f"{kind:11s} conv={conv:6s} ckpt={ck} mb={mb}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
                model.zero_grad(set_to_none=True)
                del hu
            del model


def density_bench(dev, reps: int) -> None:
    from mcw.patch_windows import PatchDensity
    x = torch.randint(-1024, 2000, (1, 1, 384, 384, 384), device=dev).float()
    ref = None
    for method in ("bincount", "sort"):
        pdn = PatchDensity(6, 25.0, 3, method)
        h = pdn(x)
        ref = h if ref is None else ref
        t = timed(lambda: pdn(x), reps)
        print(f"density {method:8s}: {t * 1e3:7.1f} ms per 384^3 volume (max |diff| vs bincount {float((h - ref).abs().max()):.1e})", flush=True)


def sdpa_bench(dev, reps: int) -> None:
    """Atlas attention shapes: windows of 4096 tokens (16^3), 6 heads x 32 / 64 dims, bf16, fwd + bwd."""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for bw, heads, seq, dim in ((64, 6, 4096, 32), (8, 6, 4096, 64), (1, 6, 4096, 64)):
        q, k, v = (torch.randn(bw, heads, seq, dim, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
        for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                         ("math", SDPBackend.MATH)):
            try:
                with sdpa_kernel(be):
                    def step():
                        F.scaled_dot_product_attention(q, k, v).sum().backward()
                    step()
                    t = timed(step, reps)
                print(f"sdpa {name:9s} (B={bw}, h={heads}, L={seq}, d={dim}): fwd+bwd {t * 1e3:7.1f} ms", flush=True)
            except Exception as e:                      # backend not available on this GPU / dtype / shape
                print(f"sdpa {name:9s} (B={bw}, h={heads}, L={seq}, d={dim}): unavailable ({type(e).__name__})", flush=True)


if __name__ == "__main__":
    main()
