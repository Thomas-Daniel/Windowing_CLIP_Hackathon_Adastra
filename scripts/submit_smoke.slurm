#!/bin/bash
#==============================================================================
# submit_smoke.slurm -- 2 training steps of each windowing arm on real data (512-volume
# subset, 64 val volumes), then a small eval of the last checkpoint. ~40 min on one node for the 6 arms.
#   REPO_DIR=$PWD VENV_DIR=$SCRATCHDIR/venvs/merlin_clip sbatch --account=cad17796 scripts/submit_smoke.slurm
#==============================================================================
#SBATCH --job-name=mcw_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --constraint=MI250
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --export=ALL
set -eu
cd "${REPO_DIR:?}"
source scripts/env_adastra.sh
OUT="${SCRATCHDIR}/merlin_clip_runs/_smoke"
for c in ${SMOKE_CONFIGS:-win1_clip win2_pillar11 win3_learned win4_local win5_abdo win6_dense win7_local_shuf}; do
    rm -rf "$OUT/$c"
    srun --cpu-bind=none --mem-bind=none -- bash scripts/adastra_bind.sh python -u train.py \
        --config "configs/$c.yaml" --output-dir "$OUT/$c" \
        --set data.max_train_items=256 training.epochs=1 data.val_max_items=64 training.log_every=1
done
srun --cpu-bind=none --mem-bind=none -- bash scripts/adastra_bind.sh python -u -m eval.evaluate extract \
    --ckpt "$OUT/$c/ckpt_final.pt" --out-dir "$OUT/$c/eval" --splits test --max-items 256
python -u -m eval.evaluate report --ckpt "$OUT/$c/ckpt_final.pt" --out-dir "$OUT/$c/eval" --skip-probe
