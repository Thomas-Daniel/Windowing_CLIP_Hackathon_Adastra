#!/bin/bash
#==============================================================================
# submit_embed.slurm -- Qwen3-Embedding report (+ zero-shot prompt) embeddings, 1 rank per GCD.
#
#   REPO_DIR=$PWD VENV_DIR=$SCRATCHDIR/venvs/merlin_clip DATA_DIR=$SCRATCHDIR/merlin_clip_data \
#   MODEL_DIR=$SCRATCHDIR/hf_models/Qwen3-Embedding-8B sbatch --account=cad17796 scripts/submit_embed.slurm
# REPORTS=<file> overrides $DATA_DIR/reports.jsonl (variant D: REPORTS=$DATA_DIR/reports_D.jsonl EMB_NAME=qwen3_emb_8b_D).
# Output: $DATA_DIR/text_emb/${EMB_NAME:-qwen3_emb_8b}/{last,mean}.npy, ids.txt, prompts.npz, meta.json
#==============================================================================
#SBATCH --job-name=mcw_embed
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
REPO_DIR="${REPO_DIR:?}"; DATA_DIR="${DATA_DIR:?}"; MODEL_DIR="${MODEL_DIR:?}"
OUT="$DATA_DIR/text_emb/${EMB_NAME:-qwen3_emb_8b}"
cd "$REPO_DIR"
source scripts/env_adastra.sh
echo "Job ${SLURM_JOB_ID} on $(hostname) started $(date -R): $MODEL_DIR -> $OUT"
rm -rf "$OUT/shards"
srun --cpu-bind=none --mem-bind=none -- bash scripts/adastra_bind.sh \
    python -u scripts/embed_reports.py embed --reports "${REPORTS:-$DATA_DIR/reports.jsonl}" --model "$MODEL_DIR" \
        --out "$OUT" --max-length "${MAX_LENGTH:-4096}" ${EXTRA_ARGS:-}
python -u scripts/embed_reports.py merge --out "$OUT"
echo "Job ${SLURM_JOB_ID} ended $(date -R)"
