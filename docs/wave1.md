# Wave 1: Atlas runs at 192³ (22 runs)

This branch adds to `main` only what these runs need. All runs use the Atlas backbone, a global batch of 128 and
4 MI300A nodes each.

## This branch: native MIOpen 3D convolutions (`wave1-atlas-192-miopen`)

`wave1-atlas-192` computes every Conv3d of Atlas as im2col + GEMM (`model.conv_impl: gemm`, `mcw/gemm_conv3d.py`),
because MIOpen's 3D convs were unusable with the software stack the runs used (torch 2.5.1+rocm6.2). This branch is
the same code with **`model.conv_impl: native`** (`configs/base.yaml`): the convs go through `F.conv3d`, i.e. MIOpen.
It is meant for a newer PyTorch / ROCm. The GEMM path is still there: set `model.conv_impl: gemm` to compare.

Differences from `wave1-atlas-192`:
- `configs/base.yaml`: `conv_impl: native`.
- `scripts/adastra_bind.sh`: MIOpen user db and kernel cache on the node-local `/tmp`, one per rank
  (`/tmp/miopen_<job>_<local rank>`). `scripts/env_adastra.sh` no longer points them to `$SCRATCHDIR/.miopen`.
- `scripts/launch_wave.sh`: `VENV_DIR` is required. The default venv of `wave1-atlas-192` (torch 2.5.1) cannot run
  these convs.

Measured on 1 exclusive MI300A (bf16, 192→192 3³ conv on 1 × 192 × 128³, TFLOP/s; jobs 5997815 / 6003917):

| torch / ROCm | MIOpen `F.conv3d`, fwd / fwd+bwd | `GemmConv3d`, fwd / fwd+bwd |
|---|---|---|
| 2.5.1 / 6.2 (runs so far) | fails (`miopenStatusInternalError`) | 154-163 / 125-127 |
| 2.14.1 / 7.2 | fails | unusable (hipBLASLt at 0.5 TFLOP/s on these shapes) |
| 2.15.0 / 7.14 | 103-120 / 164-169, equal to `GemmConv3d` to 5e-5 | 138-147 / 111-126 |

The first MIOpen call of each conv shape takes a few seconds to ~17 s (kernel search). The wave1 numbers in this file
(timings, costs) were measured with the GEMM path.

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
- a venv: by default the one the runs used, `/lus/scratch/CT3/cad17796/thdaniel/venvs/merlin_clip_1113`
  (torch 2.5.1+rocm6.2), or set `VENV_DIR`.

From the repository root:

```bash
ACCOUNT=<project> bash scripts/launch_wave.sh \
  mix192r_fixed_reg2 mix192r_fixed_reg2_seed1 mix192r_fixed_reg2_seed2 \
  mix192r_fixed_reg4 mix192r_fixed_reg4_seed1 mix192r_fixed_reg4_seed2 \
  mix192r_hu_cdf mix192r_hu_cdf_seed1 mix192r_hu_cdf_seed2 \
  mix192r_abdo mix192r_abdo_seed1 mix192r_abdo_seed2
ACCOUNT=<project> bash scripts/launch_wave.sh mix192r_lung_seed1 mix192r_lung_seed2
LONG=1 ACCOUNT=<project> bash scripts/launch_wave.sh \
  mix192r_long24_clip_seed1 mix192r_long24_clip_seed2 mix192r_long24_pillar11_seed1 mix192r_long24_pillar11_seed2
ACCOUNT=<project> bash scripts/launch_wave.sh \
  mix192c_clip_seed1 mix192c_clip_seed2 mix192c_pillar11_seed1 mix192c_pillar11_seed2

squeue -u $USER -o "%.9i %.34j %.8T %.10M %.20S" | grep -v -E "resume|eval_"
```

## Reference timings (2026-11-16, this branch, `model.conv_impl: gemm`, torch 2.5.1+rocm6.2)

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
