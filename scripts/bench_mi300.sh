#!/bin/bash
#==============================================================================
# bench_mi300.sh -- MI300A vs MI250X benchmark campaign on the chest / abdomen mix (configs/mix_*.yaml), before
# choosing where the mix runs go. Submits short real-data trainings (train.py --set training.stop_at_step=N,
# log_every=1) to $SCRATCHDIR/mix_clip_runs/_bench/<name>, plus single-GPU synthetic micro-benchmarks
# (scripts/bench_step.py). Summary: python scripts/bench_summary.py
#
#   DEP=<jobid to wait for, e.g. the preprocessing> bash scripts/bench_mi300.sh
#==============================================================================
set -eu
cd "$(dirname "$0")/.."
OUT="$SCRATCHDIR/mix_clip_runs/_bench"
COMMON="ALL,REPO_DIR=$PWD,VENV_DIR=$SCRATCHDIR/venvs/merlin_clip"
DEPARG=${DEP:+--dependency=afterok:$DEP}
MI300="--constraint=MI300 --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=48"
MI250="--constraint=MI250 --gpus-per-node=8 --ntasks-per-node=8 --cpus-per-task=16"

train() {   # name nodes shape config steps "extra --set" "extra env"
    local name=$1 nodes=$2 shape=$3 cfg=$4 steps=$5 sets=${6:-} env=${7:-}
    local args="--output-dir $OUT/$name --set training.stop_at_step=$steps training.log_every=1 data.val_max_items=64 $sets"
    # shellcheck disable=SC2086
    id=$(sbatch --account=cad17796 --parsable --job-name="b_$name" --nodes="$nodes" --time=01:00:00 $shape $DEPARG \
        --export="$COMMON,CONFIG=configs/$cfg.yaml,EXTRA_ARGS=$args${env:+,$env}" scripts/submit_train.slurm)
    echo "$id $name"
}
# MI300A, 1 node (4 APUs, 128 GB each): micro-batch, conv implementation, BLAS backend, allocator, arm
train m300_1n_mb1        1 "$MI300" mix_patch_bank  20
train m300_1n_mb1_native 1 "$MI300" mix_patch_bank  20 "model.conv_impl=native"
train m300_1n_mb1_blaslt 1 "$MI300" mix_patch_bank  20 "" "TORCH_BLAS_PREFER_HIPBLASLT=1"
train m300_1n_mb2        1 "$MI300" mix_patch_bank  12 "training.micro_batch=2"
train m300_1n_mb2_exp    1 "$MI300" mix_patch_bank  12 "training.micro_batch=2" "PYTORCH_HIP_ALLOC_CONF=expandable_segments:True"
train m300_1n_mb2_ckpt   1 "$MI300" mix_patch_bank  12 "training.micro_batch=2 model.grad_checkpointing=true"
train m300_1n_mb1_global 1 "$MI300" mix_global_bank 20
# MI300A scaling
train m300_2n_mb1        2 "$MI300" mix_patch_bank  20
train m300_4n_mb1        4 "$MI300" mix_patch_bank  30
train m300_8n_mb1        8 "$MI300" mix_patch_bank  30
# MI250X references (4 nodes: job mix_bench4, 40 steps, in mix_clip_runs/_bench4)
train m250_2n_mb1        2 "$MI250" mix_patch_bank  20
train m250_8n_mb1        8 "$MI250" mix_patch_bank  30
# single-GPU synthetic micro-benchmarks: arms x micro-batch x activation checkpointing x conv, attention backends
# shellcheck disable=SC2086
sbatch --account=cad17796 --parsable --job-name=b_step_mi300 --constraint=MI300 --gpus=1 --cpus-per-task=48 --time=01:00:00 \
    --export="$COMMON,BENCH_ARGS=--windowing patch_bank global_bank --micro-batch 1 2 4 --ckpt 0 1 --conv-impl gemm native --sdpa" \
    scripts/submit_bench.slurm | sed 's/$/ step_mi300/'
sbatch --account=cad17796 --parsable --job-name=b_step_mi250 --time=01:00:00 \
    --export="$COMMON,BENCH_ARGS=--windowing patch_bank global_bank --micro-batch 1 2 --ckpt 0 1 --conv-impl gemm --sdpa" \
    scripts/submit_bench.slurm | sed 's/$/ step_mi250/'
