#!/bin/bash
#==============================================================================
# submit_preprocess_multi.slurm -- chest/abdomen mix: CancerVerse + CT-RATE + INSPECT NIfTI -> 384^3 @ 1.5 mm int16
# HU [-1024, 3071] (Merlin volumes are reused). CPU-only on an MI250 node (cad17796 has no genoa hours). Resumable.
# Merlin's 25.5k volumes took 28 min with 64 workers; ~46k new ones are expected in ~1 h.
#
#   python scripts/build_manifest_multi.py --out-dir $DATA_DIR     # once, writes manifest_nifti.csv + reports.jsonl
#   REPO_DIR=$PWD VENV_DIR=$SCRATCHDIR/venvs/merlin_clip DATA_DIR=$SCRATCHDIR/chest_abdo_clip_data \
#       sbatch --account=cad17796 scripts/submit_preprocess_multi.slurm
# Variant D: add TAG=_D (inputs manifest_nifti_D.csv, outputs manifest_384_D.csv); volumes already in npy_384_1p5mm/ are skipped.
# Output: $DATA_DIR/npy_384_1p5mm/<source>/<study_id>.npy, $DATA_DIR/manifest_384.csv (+ .status.csv with the failures)
#==============================================================================
#SBATCH --job-name=mcw_prep_multi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
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
python -u scripts/preprocess_multisource.py --manifest "$DATA_DIR/manifest_nifti${TAG:-}.csv" \
    --out-dir "$DATA_DIR/npy_384_1p5mm" --out-manifest "$DATA_DIR/manifest_384${TAG:-}.csv" \
    --size 384 --spacing 1.5 --hu-min -1024 --hu-max 3071 --workers "${WORKERS:-64}"
echo "Job ${SLURM_JOB_ID} ended $(date -R)"
