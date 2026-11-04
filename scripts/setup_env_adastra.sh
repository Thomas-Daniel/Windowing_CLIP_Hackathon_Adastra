#!/bin/bash
# setup_env_adastra.sh -- build the ROCm PyTorch venv for merlin-clip-window on Adastra.
#
# Same toolchain as concerto-ct/scripts/setup_env_adastra.sh (torch 2.5.1+rocm6.2,
# validated on MI250X). Run ONCE from a login node (needs network through the CINES proxy):
#   VENV_DIR=$SCRATCHDIR/venvs/merlin_clip bash scripts/setup_env_adastra.sh
set -euo pipefail

VENV_DIR="${VENV_DIR:-$SCRATCHDIR/venvs/merlin_clip}"
ROCM_MODULE="${ROCM_MODULE:-amd-mixed/6.2.1}"
GFX_MODULE="${GFX_MODULE:-craype-accel-amd-gfx90a}"    # gfx90a = MI250X, gfx942 = MI300A
ROCM_WHEEL_TAG="${ROCM_WHEEL_TAG:-rocm6.2}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1 torchvision==0.20.1}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.11}"         # /usr/bin/python3 is 3.9

module purge
module load cpe/24.07
module load PrgEnv-gnu
module load "${ROCM_MODULE}"
module load "${GFX_MODULE}"

echo "==> Creating venv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

echo "==> Installing ROCm PyTorch (${ROCM_WHEEL_TAG})"
# shellcheck disable=SC2086
pip install ${TORCH_SPEC} --index-url "https://download.pytorch.org/whl/${ROCM_WHEEL_TAG}"

echo "==> Installing project dependencies"
pip install \
    "numpy>=1.24,<2.0" \
    "nibabel==5.3.2" \
    "pandas>=2.0" \
    "openpyxl" \
    "pyyaml>=6.0" \
    "scipy" \
    "scikit-learn" \
    "transformers>=4.51,<5" \
    "huggingface_hub" \
    "safetensors" \
    "timm>=1.0" \
    "einops" \
    matplotlib \
    tqdm

python -c "import torch, transformers, timm; print('torch', torch.__version__, 'transformers', transformers.__version__, 'timm', timm.__version__)"
