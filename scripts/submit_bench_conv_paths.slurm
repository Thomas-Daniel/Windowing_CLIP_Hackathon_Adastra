#!/bin/bash
#SBATCH --job-name=mcw_convpaths
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --constraint=MI250
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --export=ALL
# The ways to compute the stem's 3D convs on one GCD (scripts/bench_conv3d_paths.py):
#   1. Triton kernels vs nn.Conv3d on the GPU (tests/test_triton_conv3d.py)
#   2. where the current gemm forward spends its time (pad / im2col copies / GEMMs)
#   3. gemm vs gemm with pipelined copies vs Triton implicit GEMM, on the 4 stem convs
#   4. MIOpen with a real Find (cudnn.benchmark=True, MIOPEN_FIND_MODE=NORMAL), printing the
#      solver it picks and the MIOpenDriver command to reproduce it outside PyTorch.
# MI300A: sbatch --constraint=MI300 ... ; another ROCm / PyTorch stack: VENV_DIR=<other venv>.
set -eu
cd "${REPO_DIR:?}"
source scripts/env_adastra.sh
python -c "import torch, triton; print('torch', torch.__version__, '| hip', torch.version.hip, '| triton', triton.__version__)"
echo "== 1. correctness on GPU";  python -u -m tests.test_triton_conv3d
echo "== 2-3. breakdown, gemm / gemm_pipe / triton"; python -u scripts/bench_conv3d_paths.py --breakdown
echo "== 4. MIOpen Find"
# fresh databases, so that the Find really benchmarks the solvers (and keeps its results for later runs)
export MIOPEN_USER_DB_PATH="$SCRATCHDIR/.miopen/db_find" MIOPEN_CUSTOM_CACHE_DIR="$SCRATCHDIR/.miopen/cache_find"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
MIOPEN_FIND_MODE=NORMAL MIOPEN_ENABLE_LOGGING_CMD=1 MIOPEN_LOG_LEVEL=5 \
    python -u scripts/bench_conv3d_paths.py --impl native --miopen-find --timeout 2400 \
    --convs c192to192_128 c64to192_128 down_192to384_s2
