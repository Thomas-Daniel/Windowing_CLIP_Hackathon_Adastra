#!/bin/bash
# adastra_bind.sh -- per-rank NUMA binding + DDP env wrapper for Adastra (MI250X or
# MI300A). Verbatim port of dinov3_axial_ct/scripts/mi250_bind.sh (validated on the
# DINOv3 runs): SLURM-native launch, 1 srun task = 1 GPU/GCD, RANK/LOCAL_RANK/
# WORLD_SIZE/MASTER_* exported for train.py's setup_ddp(), CPU set derived from the
# task's own Cpus_allowed_list so it works for any --cpus-per-task / GPUs-per-node
# shape (8 GCDs on MI250X, 4 APUs on MI300A) without editing.
#
# Usage (inside an sbatch script):
#   srun --cpu-bind=none --mem-bind=none -- bash scripts/adastra_bind.sh python -u train.py ...
set -eu

cd "${REPO_DIR:?REPO_DIR must be set (exported by the calling sbatch script)}"

export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29400}"
LOCAL_N="${SLURM_NTASKS_PER_NODE:?expected SLURM_NTASKS_PER_NODE to be set}"

if [ "${SKIP_NUMA_BIND:-0}" = "1" ]; then
    echo "[adastra_bind] rank=${RANK}/${WORLD_SIZE} local=${LOCAL_RANK} NUMA binding disabled (SKIP_NUMA_BIND=1) master=${MASTER_ADDR}:${MASTER_PORT}" >&2
    exec "$@"
fi

ALLOWED_LIST=$(grep -m1 Cpus_allowed_list /proc/self/status | awk '{print $2}')
CPUS=()
IFS=',' read -ra PARTS <<< "$ALLOWED_LIST"
for part in "${PARTS[@]}"; do
    if [[ "$part" == *-* ]]; then
        lo="${part%-*}"; hi="${part#*-}"
        for ((c = lo; c <= hi; c++)); do CPUS+=("$c"); done
    else
        CPUS+=("$part")
    fi
done
N=${#CPUS[@]}

if [ "$N" -lt "$LOCAL_N" ]; then
    echo "[adastra_bind] rank=${RANK}/${WORLD_SIZE} local=${LOCAL_RANK} only ${N} CPUs visible (Cpus_allowed_list=${ALLOWED_LIST}) for ${LOCAL_N} local ranks, skipping NUMA binding master=${MASTER_ADDR}:${MASTER_PORT}" >&2
    exec "$@"
fi

PER=$((N / LOCAL_N))
IDX=$((LOCAL_RANK % LOCAL_N))
START=$((IDX * PER))
SLICE=("${CPUS[@]:START:PER}")
CPU_SET=$(IFS=,; echo "${SLICE[*]}")
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$PER}"

echo "[adastra_bind] rank=${RANK}/${WORLD_SIZE} local=${LOCAL_RANK} node=$(hostname) cpu=${CPU_SET} (of ${N} allowed: ${ALLOWED_LIST}, ${LOCAL_N} local ranks) master=${MASTER_ADDR}:${MASTER_PORT} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}" >&2

exec numactl --localalloc --physcpubind="${CPU_SET}" -- "$@"
