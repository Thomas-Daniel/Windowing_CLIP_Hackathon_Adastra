#!/bin/bash
#SBATCH --job-name=CheckRcclNet
#SBATCH --constraint=MI300
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:10:00
#SBATCH --output=check_rccl_net_%j.log

# Which network does RCCL use between nodes? Bandwidth inside a node vs between nodes, RCCL INFO
# logs per process, summary at the end.
#   sbatch --account=<projet> tools/check_rccl_net.sh
#   RCCL_NET="<nom lu dans le résumé>" sbatch ... : force this network (init fails instead of
#   silently falling back to sockets)

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR:?}
VENV_PATH=${RADIO_VENV:-$PROJECT_ROOT/.venv}
if [ -f "${VENV_PATH}/bin/activate.poverlay" ]; then
    source "${VENV_PATH}/bin/activate.poverlay"
else
    source "${VENV_PATH}/bin/activate"
fi
cd "${PROJECT_ROOT}"

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29533
export OMP_NUM_THREADS=1

# Same settings as the training scripts (values from LUMI, not confirmed by the Adastra doc)
HSN_IFS=$(ls /sys/class/net | grep '^hsn' | paste -sd, || true)
if [ -n "${HSN_IFS}" ]; then
    export NCCL_SOCKET_IFNAME="${HSN_IFS}"
fi
export NCCL_NET_GDR_LEVEL=3
if [ -n "${RCCL_NET:-}" ]; then
    export NCCL_NET="${RCCL_NET}"
fi

LOG_DIR="${PROJECT_ROOT}/rccl_net_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG_FILE="${LOG_DIR}/%h.%p.log"

echo "nodes=${SLURM_JOB_NODELIST} hsn=${HSN_IFS:-none} NCCL_NET=${NCCL_NET:-auto}"
echo "==== RCCL network plugins found (LD_LIBRARY_PATH, venv)"
{ for d in ${LD_LIBRARY_PATH//:/ }; do ls "$d"/librccl-net*.so "$d"/libnccl-net*.so 2>/dev/null; done
  find "${VENV_PATH}/" -name 'librccl-net*.so' -o -name 'libnccl-net*.so' 2>/dev/null; } | sort -u || true

srun --cpu-bind=none python tools/check_rccl_net.py

echo "==== Summary of the RCCL logs (${LOG_DIR}), count = number of processes"
cat "${LOG_DIR}"/*.log \
    | grep -oE "Using network [A-Za-z ]+|NET/OFI Selected [Pp]rovider is [A-Za-z0-9_;]+|Bootstrap ?: Using [a-z0-9]+|NET/Socket : Using \[0\][a-z0-9]+|via NET/[A-Za-z /0-9]+" \
    | sed -E 's#via NET/([A-Za-z ]+)/[0-9]+#via NET/\1#' | sort | uniq -c