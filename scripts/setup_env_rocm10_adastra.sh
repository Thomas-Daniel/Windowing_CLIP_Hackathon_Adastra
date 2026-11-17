#!/bin/bash
# setup_env_rocm10_adastra.sh -- build the venv of the 2026-11-16 runs: torch 2.14.0+rocm10.1.0 (HIP 7.16, MIOpen 3.6,
# RCCL 2.30, Triton 3.8) from AMD's wheel index, Python 3.12, with uv. One venv for MI300A (gfx942) and MI250X (gfx90a).
# It runs the three conv implementations (model.conv_impl: native | gemm | triton); the ROCm libraries come with the
# wheels (site-packages/_rocm_sdk_core/lib), no ROCm module is needed. scripts/env_adastra.sh then sets what this stack
# needs on Adastra (rocm_7 RCCL plugin, libhsa-runtime64.so shim, TORCH_BLAS_PREFER_HIPBLASLT=0; see its comments).
# Run ONCE from a login node (needs network through the CINES proxy):
#   VENV_DIR=$WORKDIR/myenv bash scripts/setup_env_rocm10_adastra.sh
set -euo pipefail

VENV_DIR="${VENV_DIR:-$WORKDIR/myenv}"
TORCH_SPEC="${TORCH_SPEC:-torch[device-gfx942,device-gfx90a]==2.14.0+rocm10.1.0}"
INDEX="${INDEX:-https://stable.repo.amd.com/rocm/whl-next/}"

module load python                                      # Python 3.12.1 (/usr/bin/python3 is 3.9)
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh   # into ~/.local/bin, shell rc files untouched
fi
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
uv pip install --extra-index-url "$INDEX" "$TORCH_SPEC" torchvision torchaudio triton
uv pip install -r "$(dirname "$0")/../requirements.txt"

python - <<'PY'
import numpy, torch, transformers, triton
print("OK torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__,
      "numpy", numpy.__version__, "transformers", transformers.__version__)
PY
