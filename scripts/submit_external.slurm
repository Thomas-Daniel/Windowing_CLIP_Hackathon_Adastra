#!/bin/bash
#==============================================================================
# submit_external.slurm -- Jolia / Pillar-0 Abdomen CT on our Merlin test protocol (eval/external.py).
#   PHASES="text orient select"   : Qwen3-Embedding reports at 512 tokens (both models' text recipe),
#                                   orientation search on 256 val volumes (~30 min)
#   PHASES="extract report"       : train + test embeddings, then the metrics (~2-3 h)
#   DATA=multi                    : multi-source test sets (Merlin, CT-RATE, INSPECT, CancerVerse), orientation
#                                   and Merlin embeddings reused from the DATA=merlin run; e.g. on 2 nodes:
#       DATA=multi PHASES="text extract report" sbatch --account=cad17796 --nodes=2 --time=04:00:00 ...
#
#   REPO_DIR=$PWD VENV_DIR=$SCRATCHDIR/venvs/merlin_clip PHASES="text orient select" \
#       sbatch --account=cad17796 --time=00:59:00 scripts/submit_external.slurm
# Output: $SCRATCHDIR/merlin_clip_runs/_external/<model>/{orientation.json,summary.json,linear_probe.csv,...}
#==============================================================================
#SBATCH --job-name=mcw_external
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
REPO_DIR="${REPO_DIR:?}"; PHASES="${PHASES:?}"; DATA="${DATA:-merlin}"
if [ "$DATA" = multi ]; then DATA_DIR="$SCRATCHDIR/chest_abdo_clip_data"; else DATA_DIR="$SCRATCHDIR/merlin_clip_data"; fi
TEXT="$DATA_DIR/text_emb/qwen3_emb_8b_len512"
cd "$REPO_DIR"
source scripts/env_adastra.sh
echo "Job ${SLURM_JOB_ID} on $(hostname) started $(date -R): phases $PHASES"
run_ranks() { srun --cpu-bind=none --mem-bind=none -- bash scripts/adastra_bind.sh "$@"; }
for ph in $PHASES; do
    echo "=== $ph $(date -R)"
    case $ph in
        text)
            if [ ! -f "$TEXT/meta.json" ]; then
                rm -rf "$TEXT/shards"
                run_ranks python -u scripts/embed_reports.py embed --reports "$DATA_DIR/reports.jsonl" \
                    --model "$SCRATCHDIR/hf_models/Qwen3-Embedding-8B" --out "$TEXT" --max-length 512
                python -u scripts/embed_reports.py merge --out "$TEXT"
            fi ;;
        orient|extract) run_ranks python -u -m eval.external "$ph" --data "$DATA" --text-cache "$TEXT" ${EXTRA_ARGS:-} ;;
        select|report)  python -u -m eval.external "$ph" --data "$DATA" --text-cache "$TEXT" ${EXTRA_ARGS:-} ;;
        *) echo "unknown phase $ph"; exit 1 ;;
    esac
done
echo "Job ${SLURM_JOB_ID} ended $(date -R)"
