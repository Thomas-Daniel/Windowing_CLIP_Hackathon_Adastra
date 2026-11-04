#!/bin/bash
#SBATCH --job-name=mcw_benchconv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --constraint=MI250
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --export=ALL
# scripts/bench_conv3d.py under several MIOpen settings (diagnostic for the Atlas conv stem).
set -eu
cd "${REPO_DIR:?}"
source scripts/env_adastra.sh
echo "== defaults"; python -u scripts/bench_conv3d.py ${BENCH_ARGS:-}
echo "== PYTORCH_MIOPEN_SUGGEST_NHWC=1"; PYTORCH_MIOPEN_SUGGEST_NHWC=1 python -u scripts/bench_conv3d.py --formats ndhwc ${BENCH_ARGS:-}
echo "== MIOPEN_FIND_MODE=FAST"; MIOPEN_FIND_MODE=FAST python -u scripts/bench_conv3d.py --formats ncdhw ${BENCH_ARGS:-}
