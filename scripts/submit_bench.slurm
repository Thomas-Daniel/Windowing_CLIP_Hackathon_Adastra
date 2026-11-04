#!/bin/bash
#==============================================================================
# submit_bench.slurm -- scripts/bench_step.py on one GCD (synthetic data, ~20 min).
#   REPO_DIR=$PWD VENV_DIR=$SCRATCHDIR/venvs/merlin_clip sbatch --account=cad17796 scripts/submit_bench.slurm
#==============================================================================
#SBATCH --job-name=mcw_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --constraint=MI250
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --export=ALL
set -eu
cd "${REPO_DIR:?}"
source scripts/env_adastra.sh
# shellcheck disable=SC2086
python -u scripts/bench_step.py ${BENCH_ARGS:-}
