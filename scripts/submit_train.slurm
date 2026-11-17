#!/bin/bash
#==============================================================================
# submit_train.slurm -- one CLIP run (one windowing arm). Default shape: 1 MI250X node (8 GCDs).
#
# Auto-resume: the run directory is fixed by run_name, train.py resumes ckpt_last.pt, so a
# run longer than 24 h is continued by resubmitting the same command (or chain with
# --dependency=afterany:<jobid>). A finished run (step = epochs x steps/epoch) exits at once.
#
#   REPO_DIR=$PWD VENV_DIR=$SCRATCHDIR/venvs/merlin_clip CONFIG=configs/win2_pillar11.yaml \
#       sbatch --account=cad17796 [--nodes=2] scripts/submit_train.slurm
# MI300A: add --constraint=MI300 --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=48.
# accum_steps adapts to the GPU count (global batch stays 128). Measured s/step (win4_local):
# 1 MI250X node 59.4, 2 nodes 29.8, 1 MI300A node 51.9, 2 nodes 23.6.
# Optional: EXTRA_ARGS="--set training.stop_at_step=200"
#==============================================================================
#SBATCH --job-name=mcw_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --constraint=MI300
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --export=ALL
# 15 min before the time limit SLURM sends SIGUSR1 to train.py: it saves ckpt_last.pt and exits.
#SBATCH --signal=USR1@900
set -eu
REPO_DIR="${REPO_DIR:?}"; CONFIG="${CONFIG:?set CONFIG to configs/win*.yaml}"
cd "$REPO_DIR"

source .venv/bin/activate

export MIOPEN_FIND_MODE=1
export OMP_NUM_THREADS=24

export MIOPEN_USER_DB_PATH="/tmp/miopen-${USER}-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3

echo "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST} started $(date -R): $CONFIG ${EXTRA_ARGS:-}"
MONITOR_LOG="logs/gpu_monitor_${SLURM_JOB_ID}.log"
srun --overlap --ntasks=1 --cpu-bind=none --mem-bind=none \
    bash -c 'while true; do echo "=== $(date -Is) ==="; rocm-smi --showuse --showmeminfo vram; sleep 60; done' \
    > "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
# shellcheck disable=SC2086
srun --cpu-bind=none --mem-bind=none -- bash scripts/adastra_bind.sh \
    python -u train.py --config "$CONFIG" ${EXTRA_ARGS:-} --set training.profile=true
echo "Job ${SLURM_JOB_ID} ended $(date -R)"
