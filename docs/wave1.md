# Wave 1: Atlas runs at 192³ (22 runs)

This branch adds to `main` only what these runs need. All runs use the Atlas backbone, a global batch of 128 and
4 MI300A nodes each.

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
  - On 4 MI300 nodes, the step time went from 1.5 to 2.5 s after step ~500. With the fix, it stays at 1.5 s.
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

Before the RCCL fix, a run cost ~56 MI300 h (8 epochs) or ~180 h (24 epochs), training and evaluation included.
Expect roughly 35-40 % less with the fix.
