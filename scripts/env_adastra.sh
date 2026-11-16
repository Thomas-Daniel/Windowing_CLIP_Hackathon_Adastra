# Sourced by every SLURM script: venv + the Adastra MI250X runtime settings validated on the
# concerto-ct / dinov3 runs (see the comments for the job history behind each one).
# shellcheck disable=SC1091
source "${VENV_DIR:?set VENV_DIR to the venv built by scripts/setup_env_adastra.sh}/bin/activate"
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export HSA_FORCE_FINE_GRAIN_PCIE=1
# XNACK off: with HSA_XNACK=1 torch's AOTriton flash / mem-efficient SDPA kernels are rejected
# and attention silently falls back to the math backend (huge memory; concerto-ct job 5444076).
export HSA_XNACK=0
# numpy madvises every array >= 4 MB as hugepage; on these nodes (THP defrag=madvise) each
# 113 MB volume load then direct-compacts memory and loaders slow 3-4x after ~2.5 h.
export NUMPY_MADVISE_HUGEPAGE=0
# MIOpen (only used with model.conv_impl=native): its default user db was read-only on MI300A (miopenStatusInternalError)
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$SCRATCHDIR/.miopen/db}" MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$SCRATCHDIR/.miopen/cache}"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
# Multi-node: RCCL over Slingshot through the libfabric plugin (sets NCCL_NET_PLUGIN, FI_* and
# NCCL_NET_GDR_LEVEL); without it RCCL falls back to TCP sockets. RCCL_OFI=0 disables it.
# The module sets NCCL_NET_PLUGIN=librccl-net.so, which the RCCL bundled with torch 2.5.1 turns into
# "librccl-net-librccl-net.so.so"; unset, it loads librccl-net.so from LD_LIBRARY_PATH.
# Measured: 2 MI250X nodes scale x1.99 even over sockets (one ~370 MB all-reduce per ~30 s step).
if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ] && [ "${RCCL_OFI:-1}" = "1" ]; then
    module load aws-ofi-rccl/1.18.0_rocm6
    unset NCCL_NET_PLUGIN
    # CXI fork safety: the DataLoader re-forks its workers every epoch, after RCCL has registered its memory with
    # the NIC; without this, every collective then slows down (4 MI300 nodes: 1.5 -> 2.5 s/step from step ~400-600,
    # all_reduce 0.04 -> 0.30 s; 32 nodes: up to 8x). With it: flat 1.5 s/step (jobs 5997785 / 5997786, 2026-11-15).
    export CXI_FORK_SAFE=1 CXI_FORK_SAFE_HP=1
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
