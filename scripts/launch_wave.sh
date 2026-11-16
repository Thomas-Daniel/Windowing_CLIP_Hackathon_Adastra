#!/bin/bash
# Launch Atlas runs on 4 MI300A nodes each: for every run, a training job, a resume job (afterany) and the eval
# job(s) chained after the training (submit_eval.slurm cancels the resume once the run is finished).
#
#   ACCOUNT=<project> bash scripts/launch_wave.sh <run> ...          8-epoch runs: eval of ckpt_final
#   LONG=1 ACCOUNT=<project> bash scripts/launch_wave.sh <run> ...   24-epoch runs: evals of ckpt_epoch008 / 016 / final
#
# <run> = a config name in configs/ (without .yaml). Run from the repository root.
# VENV_DIR (required): a venv whose PyTorch / ROCm runs MIOpen 3D convs on MI300A (this branch uses model.conv_impl:
# native); measured OK with torch 2.15.0+rocm7.14, failing with 2.5.1+rocm6.2 and 2.14.1+rocm7.2.
# RUNS_ROOT must be the configs' output_root (configs/mix_base.yaml: ${SCRATCHDIR}/mix_clip_runs).
set -eu
ACCOUNT="${ACCOUNT:?set ACCOUNT to the Slurm project to charge}"
VENV_DIR="${VENV_DIR:?set VENV_DIR to a venv with a ROCm whose MIOpen 3D convs work (see docs/wave1.md)}"
RUNS_ROOT="${RUNS_ROOT:-$SCRATCHDIR/mix_clip_runs}"
NODES="${NODES:-4}"
SHAPE="--constraint=MI300 --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=48"
EXP="ALL,REPO_DIR=$PWD,VENV_DIR=$VENV_DIR"
TL=$([ "${LONG:-0}" = 1 ] && echo 16:00:00 || echo 08:00:00)
CKS=$([ "${LONG:-0}" = 1 ] && echo "ckpt_epoch008 ckpt_epoch016 ckpt_final" || echo "ckpt_final")
[ -f train.py ] || { echo "run from the repository root"; exit 1; }
mkdir -p "$RUNS_ROOT"
for r in "$@"; do
  [ -f "configs/$r.yaml" ] || { echo "configs/$r.yaml not found, skipped"; continue; }
  R=$RUNS_ROOT/$r
  [ -e "$R" ] && { echo "$R exists, skipped"; continue; }
  t=$(sbatch --account=$ACCOUNT --parsable --nodes=$NODES $SHAPE --time=$TL --job-name=$r \
      --export=$EXP,CONFIG=configs/$r.yaml scripts/submit_train.slurm)
  rs=$(sbatch --account=$ACCOUNT --parsable --nodes=$NODES $SHAPE --time=$TL --job-name=${r}_resume \
      --dependency=afterany:$t --export=$EXP,CONFIG=configs/$r.yaml scripts/submit_train.slurm)
  ev=""
  for ck in $CKS; do
    n=eval_${r}$([ $ck = ckpt_final ] || echo "_e${ck: -2}")
    e=$(sbatch --account=$ACCOUNT --parsable --nodes=1 $SHAPE --job-name=$n --dependency=afterany:$t \
        "--export=$EXP,CKPT=$R/$ck.pt,RESUME_JOB=$rs,SUBMIT_ACCOUNT=$ACCOUNT,SUBMIT_SHAPE=$SHAPE,EXTRACT_ARGS=--splits train test --max-train-per-source 8000" \
        scripts/submit_eval.slurm)
    ev="$ev $ck=$e"
  done
  echo "$(date '+%F %H:%M') $r [$ACCOUNT, MI300 x$NODES, $TL] train=$t resume=$rs evals:$ev" | tee -a "$RUNS_ROOT/jobs.txt"
done
