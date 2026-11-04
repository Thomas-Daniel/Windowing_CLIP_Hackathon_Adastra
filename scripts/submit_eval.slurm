#!/bin/bash
#==============================================================================
# submit_eval.slurm -- test evaluation of one checkpoint: embeddings of all splits on 8 GCDs,
# then retrieval / zero-shot / linear probe (eval/evaluate.py), then the cross-run table
# (scripts/compare_evals.py -> <runs root>/eval_summary_<ckpt>.csv).
#
#   REPO_DIR=$PWD VENV_DIR=$SCRATCHDIR/venvs/merlin_clip \
#   CKPT=$SCRATCHDIR/merlin_clip_runs/win2_pillar11/ckpt_final.pt \
#       sbatch --account=cad17796 scripts/submit_eval.slurm
# Chest / abdomen mix runs: EXTRACT_ARGS="--splits train test --max-train-per-source 8000" (probe train subsets).
# Output: OUT_DIR (default: <ckpt dir>/eval_<ckpt name>)/summary.json, zeroshot_test.csv, linear_probe.csv
#
# Chained to the end of a training run (sbatch --dependency=afterany:<training job>), with
# RESUME_JOB=<the run's resume job id>, or a colon-separated chain of them (long runs: R1:R2:R3):
#   - CKPT exists (the run finished): the resume jobs still pending, now useless, are cancelled; the eval runs.
#   - CKPT missing (the run was interrupted): the eval resubmits itself after the first resume job of the
#     chain, with the rest of the chain as its RESUME_JOB, and exits.
#==============================================================================
#SBATCH --job-name=mcw_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --constraint=MI250
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --export=ALL
set -eu
REPO_DIR="${REPO_DIR:?}"; CKPT="${CKPT:?set CKPT to a checkpoint .pt}"
OUT_DIR="${OUT_DIR:-$(dirname "$CKPT")/eval_$(basename "${CKPT%.pt}")}"
cd "$REPO_DIR"
echo "Job ${SLURM_JOB_ID} started $(date -R): $CKPT -> $OUT_DIR"

FIRST=${RESUME_JOB:-}; FIRST=${FIRST%%:*}
REST=""; case "${RESUME_JOB:-}" in *:*) REST=${RESUME_JOB#*:} ;; esac
if [ ! -f "$CKPT" ]; then
    if [ -n "$FIRST" ] && squeue -h -j "$FIRST" 2>/dev/null | grep -q .; then
        # SUBMIT_SHAPE: the partition options of the original submission (e.g. the MI300 ones), reused here
        # shellcheck disable=SC2086
        NEXT=$(RESUME_JOB="$REST" sbatch --account="${SUBMIT_ACCOUNT:-cad17796}" --job-name="$SLURM_JOB_NAME" ${SUBMIT_SHAPE:-} \
               --dependency=afterany:"$FIRST" --parsable "$REPO_DIR/scripts/submit_eval.slurm")
        echo "$CKPT not there yet (run interrupted): eval resubmitted as job $NEXT, after resume job $FIRST${REST:+ (then $REST)}"
        exit 0
    fi
    echo "ERROR: $CKPT does not exist and no pending resume job: the run did not finish" >&2
    exit 1
fi
CHAIN=${RESUME_JOB:-}
for J in ${CHAIN//:/ }; do
    if squeue -h -j "$J" -t PENDING 2>/dev/null | grep -q .; then
        scancel "$J" && echo "run finished: cancelled its pending resume job $J"
    fi
done

source scripts/env_adastra.sh
srun --cpu-bind=none --mem-bind=none -- bash scripts/adastra_bind.sh \
    python -u -m eval.evaluate extract --ckpt "$CKPT" --out-dir "$OUT_DIR" ${EXTRACT_ARGS:-}
python -u -m eval.evaluate report --ckpt "$CKPT" --out-dir "$OUT_DIR"
python -u scripts/compare_evals.py --root "$(dirname "$(dirname "$CKPT")")" --ckpt "$(basename "${CKPT%.pt}")" || true
echo "Job ${SLURM_JOB_ID} ended $(date -R)"
