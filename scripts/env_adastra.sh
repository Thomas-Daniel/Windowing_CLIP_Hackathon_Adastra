# Sourced by every SLURM script: venv + the Adastra MI250X runtime settings validated on the
# concerto-ct / dinov3 runs (see the comments for the job history behind each one).
# shellcheck disable=SC1091
source "${VENV_DIR:?set VENV_DIR to the venv built by scripts/setup_env_adastra.sh}/bin/activate"
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export HSA_FORCE_FINE_GRAIN_PCIE=1
# XNACK off: with HSA_XNACK=1 torch's AOTriton flash / mem-efficient SDPA kernels are rejected
# and attention silently falls back to the math backend (huge memory; concerto-ct job 5444076).
export HSA_XNACK=0
# numpy madvises every array >= 4 MB as hugepage; on these nodes (THP defrag=madvise) each
# 113 MB volume load then direct-compacts memory and loaders slow 3-4x after ~2.5 h.
export NUMPY_MADVISE_HUGEPAGE=0
# MIOpen user db / kernel cache: set per rank on the node-local /tmp by scripts/adastra_bind.sh
# Multi-node: RCCL over Slingshot through the libfabric plugin (sets NCCL_NET_PLUGIN, FI_* and
# NCCL_NET_GDR_LEVEL); without it RCCL falls back to TCP sockets. RCCL_OFI=0 disables it.
# The module sets NCCL_NET_PLUGIN=librccl-net.so, which the RCCL bundled with torch 2.5.1 turns into
# "librccl-net-librccl-net.so.so"; unset, it loads librccl-net.so from LD_LIBRARY_PATH.
# Measured: 2 MI250X nodes scale x1.99 even over sockets (one ~370 MB all-reduce per ~30 s step).
# ROCm 7+ wheels (torch 2.14.0+rocm10.1.0 from repo.amd.com: HIP 7.16, libraries in site-packages/_rocm_sdk_core/lib):
# the rocm6 plugin needs libamdhip64.so.6, fails to load, and RCCL falls back to TCP sockets (45 s/step on 4 MI300
# nodes, 2026-11-16). Use the rocm_7 plugin, let it find the wheel's libamdhip64.so.7, and set the variable the new
# RCCL asks for ("HSA_NO_SCRATCH_RECLAIM=1 must be set to avoid performance degradation").
ROCM_SDK_LIB=$(ls -d "$VENV_DIR"/lib/python3*/site-packages/_rocm_sdk_core/lib 2>/dev/null | head -n 1)
if [ -n "$ROCM_SDK_LIB" ]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi
if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ] && [ "${RCCL_OFI:-1}" = "1" ]; then
    if [ -n "$ROCM_SDK_LIB" ]; then
        module load aws-ofi-rccl/1.18.0_rocm_7
        export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$LD_LIBRARY_PATH"
        # libfabric's ROCr HMEM support dlopens "libhsa-runtime64.so" (unversioned). The wheel only has
        # libhsa-runtime64.so.1, so libfabric loaded the system's ROCm 6 HSA runtime next to the wheel's: GPU memory
        # registration then failed ("Bad address", cxil_map: write error). A shim directory with the unversioned name
        # pointing to the wheel's library makes dlopen return the runtime already loaded.
        if [ "${MCW_NO_HSA_SHIM:-0}" != "1" ]; then
            SHIM="$VENV_DIR/rocm_shim"
            [ -e "$SHIM/libhsa-runtime64.so" ] || { mkdir -p "$SHIM" && ln -sfn "$ROCM_SDK_LIB/libhsa-runtime64.so.1" "$SHIM/libhsa-runtime64.so"; }
            export LD_LIBRARY_PATH="$SHIM:$LD_LIBRARY_PATH"
        fi
    else
        module load aws-ofi-rccl/1.18.0_rocm6
    fi
    unset NCCL_NET_PLUGIN
    # override of the module's NCCL_NET_GDR_LEVEL=3 (e.g. 0: no GPUDirect, staged through host memory)
    [ -n "${MCW_NET_GDR_LEVEL:-}" ] && export NCCL_NET_GDR_LEVEL="$MCW_NET_GDR_LEVEL"
    # CXI fork safety: the DataLoader re-forks its workers every epoch, after RCCL has registered its memory with
    # the NIC; without this, every collective then slows down (4 MI300 nodes: 1.5 -> 2.5 s/step from step ~400-600,
    # all_reduce 0.04 -> 0.30 s; 32 nodes: up to 8x). With it, crop test: mean 1.91 instead of 2.51 s/step after step
    # 400 (median 1.52 vs 2.47); intermittent collective spikes remain (jobs 5997785 / 5997786, 2026-11-15).
    export CXI_FORK_SAFE=1 CXI_FORK_SAFE_HP=1
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
