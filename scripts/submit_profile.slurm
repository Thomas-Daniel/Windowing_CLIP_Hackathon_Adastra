#!/bin/bash
#SBATCH --job-name=mcw_profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --constraint=MI250
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --export=ALL
# scripts/profile_model.py on one GCD.
set -eu
cd "${REPO_DIR:?}"
source scripts/env_adastra.sh
# shellcheck disable=SC2086
python -u ${PROFILE_SCRIPT:-scripts/profile_model.py} ${PROFILE_ARGS:-}
