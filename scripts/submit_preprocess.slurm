#!/bin/bash
#==============================================================================
# submit_preprocess.slurm -- NIfTI -> 384^3 @ 1.5 mm int16 HU [-1024, 3071] for all splits.
# CPU-only, on an MI250 node (cad17796 has no genoa hours). Resumable: resubmit after a timeout.
#
#   REPO_DIR=$PWD VENV_DIR=$SCRATCHDIR/venvs/merlin_clip DATA_DIR=$SCRATCHDIR/merlin_clip_data \
#       sbatch --account=cad17796 scripts/submit_preprocess.slurm
#==============================================================================
#SBATCH --job-name=mcw_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --constraint=MI250
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --export=ALL
set -eu
REPO_DIR="${REPO_DIR:?}"; DATA_DIR="${DATA_DIR:?}"
cd "$REPO_DIR"
source scripts/env_adastra.sh
export OMP_NUM_THREADS=2
echo "Job ${SLURM_JOB_ID} on $(hostname) started $(date -R)"
python -u scripts/preprocess_volumes.py --manifest "$DATA_DIR/manifest_nifti.csv" \
    --out-dir "$DATA_DIR/npy_384_1p5mm" --out-manifest "$DATA_DIR/manifest_384.csv" \
    --size 384 --spacing 1.5 --hu-min -1024 --hu-max 3071 --workers "${WORKERS:-64}"
echo "Job ${SLURM_JOB_ID} ended $(date -R)"
