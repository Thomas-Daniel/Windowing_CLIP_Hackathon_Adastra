#!/bin/bash
#==============================================================================
# archive_runs.sh -- copy finished runs from $SCRATCHDIR (purged) to $STOREDIR (kept), one tar per run
# (the store allows few files: ~10k per group), verified with `tar -d` against the run directory.
# The run directories on scratch are left untouched. Cross-run files (eval tables, bootstraps,
# progress plots) go to <dest>/runs_summary.tar.
#
#   bash scripts/archive_runs.sh win8_patch_bank win9_patch_cdf ...     # or no argument: every run
# Restore: tar -xf $STOREDIR/merlin_clip_window/<run>.tar -C $SCRATCHDIR/merlin_clip_runs
#==============================================================================
set -u
SRC="${SRC:-${SCRATCHDIR:?}/merlin_clip_runs}"          # mix runs: SRC=$SCRATCHDIR/mix_clip_runs
DEST="${DEST:-${STOREDIR:?}/merlin_clip_window}"
mkdir -p "$DEST"
cd "$SRC"
runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=($(ls -d */ | tr -d / | grep -v '^_'))   # every run dir (not _external, _smoke)
fail=0
for r in "${runs[@]}"; do
    if [ ! -f "$r/ckpt_final.pt" ] && [ -z "${FORCE:-}" ]; then echo "$r: no ckpt_final.pt (unfinished; FORCE=1 to archive anyway), skipped"; continue; fi
    tmp="$DEST/.$r.tar.partial"
    if tar --format=posix -cf "$tmp" "$r" && tar -df "$tmp" >/dev/null 2>&1; then
        mv -f "$tmp" "$DEST/$r.tar"
        echo "$r: $(du -h --apparent-size "$DEST/$r.tar" | cut -f1), $(tar -tf "$DEST/$r.tar" | wc -l) entries, verified"
    else
        rm -f "$tmp"; echo "$r: ARCHIVE FAILED"; fail=1
    fi
done
files=$(ls eval_summary_*.csv eval_bootstrap_*.txt progress.png progress.txt 2>/dev/null)
if [ -n "$files" ] && tar --format=posix -cf "$DEST/runs_summary.tar" $files && tar -df "$DEST/runs_summary.tar" >/dev/null 2>&1; then
    echo "runs_summary.tar: $(echo $files | wc -w) files, verified"
else
    echo "runs_summary.tar: FAILED or nothing to archive"; fail=1
fi
exit $fail
