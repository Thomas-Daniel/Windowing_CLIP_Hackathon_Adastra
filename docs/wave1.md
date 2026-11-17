# Wave 1: Atlas runs at 192³ (22 runs)

This branch adds to `main` only what these runs need. All runs use the Atlas backbone, a global batch of 128 and
4 MI300A nodes each.

## This branch: new environment, three conv implementations (`wave1-atlas-192-newenv`)

The code of the 2026-11-16 runs: `wave1-atlas-192-miopen`, plus the Triton convolution and the settings that run
torch 2.14.0+rocm10.1.0 on 4 MI300A nodes.

- **Venv**: `scripts/setup_env_rocm10_adastra.sh` (uv, Python 3.12, torch 2.14.0+rocm10.1.0, Triton 3.8, ROCm
  libraries inside the wheels). `scripts/launch_wave.sh` requires it as `VENV_DIR`.
- **`model.conv_impl`** (`configs/base.yaml`, default `native`) sets how every Conv3d is computed, Atlas's
  (`mcw/backbone.py`) and the windowing arm's (`mcw/model.py`):
  - `native`: MIOpen `F.conv3d`;
  - `gemm`: im2col + GEMM (`mcw/gemm_conv3d.py`, the code of `wave1-atlas-192`);
  - `triton`: implicit GEMM in Triton (`mcw/triton_conv3d.py`, prototype), without the column buffer.

  Per run: `--set model.conv_impl=triton`, or `CONV_IMPL=triton` with `scripts/launch_wave.sh`.
- **MIOpen Find**: `MCW_CUDNN_BENCHMARK=1` sets `torch.backends.cudnn.benchmark` (`train.py`), which makes MIOpen
  search for the fastest kernels. `launch_wave.sh` sets it. The first step takes ~4 min instead of ~1.
- **`scripts/env_adastra.sh`**, for the ROCm 7+ wheels (detected by `site-packages/_rocm_sdk_core/lib`):
  - RCCL plugin `aws-ofi-rccl/1.18.0_rocm_7`, with the wheel's libraries on `LD_LIBRARY_PATH`. The rocm6 plugin
    needs `libamdhip64.so.6`. Without the right plugin, RCCL fell back to TCP sockets (45 s/step).
  - Shim `$VENV_DIR/rocm_shim/libhsa-runtime64.so`, pointing to the wheel's `libhsa-runtime64.so.1`. Without it,
    libfabric loaded the system's ROCm 6 HSA runtime and GPU memory registration failed (`Bad address`,
    `cxil_map: write error`).
  - `TORCH_BLAS_PREFER_HIPBLASLT=0`. hipBLASLt spends ~25 ms of host CPU per matmul in this stack: Atlas fwd+bwd
    took 15.6 s instead of 0.54 s with rocBLAS.
  - `HSA_NO_SCRATCH_RECLAIM=1` (requested by this RCCL), `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3`,
    `NCCL_NET_GDR_LEVEL=3`.
- **`scripts/launch_wave.sh`**: `CONV_IMPL`, MIOpen Find for the training jobs, and `EXCLUDE` (default `a1014`,
  whose hsn0 was unreachable: "No route to host").
- **Network**: checked in the RCCL logs of the runs (`NCCL_DEBUG=INFO`):
  - all inter-node traffic went through Libfabric / CXI (Slingshot), 4 NICs per node, never TCP;
  - ~90 % of the network connections use GPUDirect RDMA;
  - the others, where a GPU uses another GPU's NIC, are staged through host memory.
- **Tests**: `TRITON_INTERPRET=1 python -m tests.test_triton_conv3d` (CPU, a few minutes). It compares
  `TritonConv3d` with `nn.Conv3d` (output, grad_x, grad_W) and checks the class of every Conv3d for each
  `conv_impl`.

### The three runs (2026-11-16)

`mix192r_fixed_reg4` with seeds 0, 1 and 2, one conv implementation each. Each run used 4 MI300A nodes, 8 epochs
(4 840 steps), commit `f9404c9`. Step times come from `s_per_step` in `metrics.jsonl`. Wall time and GENCI hours
cover the training jobs.

| `conv_impl` | run | mean s/step | median | p10 | p90 | volumes/s per APU | first step | wall | GENCI h |
|---|---|---|---|---|---|---|---|---|---|
| `gemm` | `mix192r_fixed_reg4` | 2.30 | 2.15 | 1.61 | 3.16 | 3.47 | 58 s | 3.31 h | 53.0 |
| `triton` | `mix192r_fixed_reg4_seed1` | **1.98** | 1.88 | 0.97 | 3.13 | **4.03** | 64 s | **2.85 h** | **45.6** |
| `native` + Find | `mix192r_fixed_reg4_seed2` | 2.07 | 2.02 | 1.19 | 3.07 | 3.86 | 264 s | 3.04 h | 48.6 |

- **Speed against GEMM**:
  - Triton: −14 % on the mean step, −40 % on the fastest steps (p10);
  - MIOpen with Find: −10 % on the mean, −26 % on p10.
- **Old environment** (GEMM, torch 2.5.1): the same config takes ~3.6 h and 57 GENCI h (reference timings below).
- **p90 ≈ 3.1 s with all three implementations.** The slow steps come from the multi-node collectives, not from
  the convolutions. A step loses `mean − p10` to them: 0.69 s with GEMM, 1.01 s with Triton, 0.88 s with MIOpen.

**Do not use the evaluations of these three runs.**

| `conv_impl` | probe mean | R@10 mean | zero-shot mean |
|---|---|---|---|
| `gemm` | 0.650 | 0.010 | 0.590 |
| `triton` | 0.702 | 0.072 | 0.730 |
| `native` + Find | 0.722 | 0.218 | 0.760 |
| reference `mix192r_clip`, old env, 3 seeds | 0.715 | 0.194 | 0.752 |
| reference `mix192r_pillar11`, old env, 3 seeds | 0.726 | 0.241 | 0.761 |

- **Unstable validation.** In all three runs, the validation R@10 jumps from one epoch to the next, whatever the
  conv implementation.
  - `gemm`, i2t R@10 per epoch: 0.10, 0.13, 0.34, 0.08, 0.49, 0.01, 0.57, 0.01.
  - In the old environment it rises steadily: `mix192r_clip`, 0.18 → 0.62.
- **The final evaluation follows the last epoch's state.** The last validation R@10 is 0.65 for `native` and 0.01
  for `gemm`.
- **BatchNorm statistics.** The checkpoints have negative BatchNorm `running_var`, mostly in the stem's
  `conv_down`. This happens in all three runs, and never in the old environment.
- **Cause unknown.** The BatchNorm alone is exact in this stack (MIOpen and native, centred inputs; job 6015590).

### MIOpen 3D convolutions (from `wave1-atlas-192-miopen`)

Measured on 1 exclusive MI300A (bf16, 192→192 3³ conv on 1 × 192 × 128³, TFLOP/s; jobs 5997815 / 6003917):

| torch / ROCm | MIOpen `F.conv3d`, fwd / fwd+bwd | `GemmConv3d`, fwd / fwd+bwd |
|---|---|---|
| 2.5.1 / 6.2 (runs so far) | fails (`miopenStatusInternalError`) | 154-163 / 125-127 |
| 2.14.1 / 7.2 | fails | unusable (hipBLASLt at 0.5 TFLOP/s on these shapes) |
| 2.15.0 / 7.14 | 103-120 / 164-169, equal to `GemmConv3d` to 5e-5 | 138-147 / 111-126 |

The first MIOpen call of each conv shape takes a few seconds to ~17 s (kernel search). The reference timings at the
end of this file were measured with the GEMM path in the old environment.

## The runs

| group | configs (`configs/<name>.yaml`) | runs | epochs |
|---|---|---|---|
| number of channels, single windows | `mix192r_fixed_reg2`, `mix192r_fixed_reg4`, `mix192r_hu_cdf`, `mix192r_abdo`, each `+ _seed1, _seed2` | 12 | 8 |
| lung window alone | `mix192r_lung_seed1`, `mix192r_lung_seed2` | 2 | 8 |
| long training | `mix192r_long24_clip_seed1/2`, `mix192r_long24_pillar11_seed1/2` | 4 | 24 |
| centred crop | `mix192c_clip_seed1/2`, `mix192c_pillar11_seed1/2` | 4 | 8 |

`mix192r_clip` and `mix192r_pillar11` (seeds 0-2), `mix192c_clip` and `mix192c_pillar11` are the reference runs.
They are already done, and their configs are included.

## What changed compared with `main`

- **`data.downsample: 2`** (`mcw/data.py` `block_mean_int16`, passed through by `train.py` and `eval/evaluate.py`):
  each 384³ volume becomes 192³ by averaging 2³ blocks of HU, rounded half up to int16. This is the full field of
  view at 3 mm. Every `mix192r_*` run uses it, and the evaluation applies the same resize.
- **`windowing.freeze_windows: true`** (`mcw/patch_windows.py`): `global_bank` with `window_init: uniform` keeps its
  K windows at init. This gives K fixed sigmoid windows evenly spaced over [-1000, 1000] HU (`fixed_reg2`,
  `fixed_reg4`).
- **Configs**: `mix192r_base` (downsample 2, Atlas laid out for 192³, micro-batch 4) and `mix192c_base` (centred
  192³ crop), plus the run configs above.
- **Data paths**: `configs/mix_base.yaml` reads the shared data in `/lus/scratch/CT3/cad17796/thdaniel/` and no
  longer reads each user's `${SCRATCHDIR}`. Outputs still go to `${SCRATCHDIR}/mix_clip_runs`.
- **`scripts/launch_wave.sh`**: for each run, it submits training, a resume job and the chained evaluation(s).
- **RCCL fix** (`scripts/env_adastra.sh`): `CXI_FORK_SAFE=1 CXI_FORK_SAFE_HP=1`.
  - The DataLoader re-forks its workers every epoch, after RCCL has registered its memory with the Slingshot NIC.
    Every collective then slowed down.
  - Crop test on 4 MI300 nodes, 1000 steps: the mean step after step 400 went from 2.51 s without the fix to
    1.91 s with it (median 2.47 → 1.52). Intermittent spikes remain: on about a third of the steps the all_reduce
    takes ~0.4-0.5 s instead of 0.03 s (see the reference timings below).
- **Tests**: `python -m tests.test_cpu fixed downsample`. The full suite (`python -m tests.test_cpu`) keeps the
  invariants of `docs/code.md`.

## Running

Prerequisites:
- read access to `/lus/scratch/CT3/cad17796/thdaniel/` (group `cad17796`);
- a Slurm account with MI300 hours;
- the venv of `scripts/setup_env_rocm10_adastra.sh`, passed as `VENV_DIR`.

`CONV_IMPL=gemm|triton|native` overrides `model.conv_impl` (default `native`) for the runs of one call. Example:
`CONV_IMPL=triton VENV_DIR=$WORKDIR/myenv ACCOUNT=<project> bash scripts/launch_wave.sh mix192r_abdo`.

From the repository root:

```bash
VENV_DIR=<venv> ACCOUNT=<project> bash scripts/launch_wave.sh \
  mix192r_fixed_reg2 mix192r_fixed_reg2_seed1 mix192r_fixed_reg2_seed2 \
  mix192r_fixed_reg4 mix192r_fixed_reg4_seed1 mix192r_fixed_reg4_seed2 \
  mix192r_hu_cdf mix192r_hu_cdf_seed1 mix192r_hu_cdf_seed2 \
  mix192r_abdo mix192r_abdo_seed1 mix192r_abdo_seed2
VENV_DIR=<venv> ACCOUNT=<project> bash scripts/launch_wave.sh mix192r_lung_seed1 mix192r_lung_seed2
LONG=1 VENV_DIR=<venv> ACCOUNT=<project> bash scripts/launch_wave.sh \
  mix192r_long24_clip_seed1 mix192r_long24_clip_seed2 mix192r_long24_pillar11_seed1 mix192r_long24_pillar11_seed2
VENV_DIR=<venv> ACCOUNT=<project> bash scripts/launch_wave.sh \
  mix192c_clip_seed1 mix192c_clip_seed2 mix192c_pillar11_seed1 mix192c_pillar11_seed2

squeue -u $USER -o "%.9i %.34j %.8T %.10M %.20S" | grep -v -E "resume|eval_"
```

## Reference timings (2026-11-16, `wave1-atlas-192`: `model.conv_impl: gemm`, torch 2.5.1+rocm6.2)

Measured with 1000-step runs (one end of epoch at step 605, i.e. one validation and one checkpoint) on exclusive
MI300A nodes. Use the **mean** of `s_per_step` (logged every 10 steps in `metrics.jsonl`): it sets the wall time,
and the median hides the spikes.

| config | nodes | mean s/step, steps 20-400 → 400-1000 | volumes/s per APU | 8 epochs, wall | 8 epochs, GENCI h (train) |
|---|---|---|---|---|---|
| `mix192r_clip` | 4 | 1.96 → 2.13 | 3.78 | 3.0 h | 47 |
| `mix192r_pillar11` | 4 | 2.13 → 2.48 | 3.26 | 3.4 h | 55 |
| `mix192r_fixed_reg4` | 4 | 2.10 → 2.57 | 3.16 | 3.6 h | 57 |
| `mix192r_clip` | 1 | 6.47 (300 steps) | 4.94 | 9.2 h | 37 |

- Evaluation of a resized run: ~22 min on 1 node (~1.5 GENCI h). A 24-epoch run is 3× the training.
- End of epoch (validation + checkpoint): ~1 min on 4 nodes. Start-up: ~40 s, mostly the first step.
- Peak memory 16.0-16.4 GiB per APU. Data wait 0.03-0.04 s per step on 4 nodes, the same in slow steps.
- **The multi-node inefficiency is the main target.** One node runs at 4.9 volumes/s per APU, 4 nodes at 3.2-3.8.
  Slow steps (~1/3, up to 3-5 s) come with collectives that take 0.4-0.5 s (all_reduce) instead of 0.03 s.
  The old runs without the fork-safety fix gave the same resized timings (`mix192r_clip`: 3.0 h, 48 h).

To reproduce a reference run (no evaluation, nothing kept but the metrics):

```bash
sbatch --account=<project> --nodes=4 --constraint=MI300 --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=48 \
  --time=01:30:00 --export=ALL,REPO_DIR=$PWD,VENV_DIR=<venv>,CONFIG=configs/mix192r_clip.yaml,EXTRA_ARGS="--output-dir \
  $SCRATCHDIR/bench192/clip_n4 --set training.stop_at_step=1000 training.ckpt_every=100000" scripts/submit_train.slurm
```
