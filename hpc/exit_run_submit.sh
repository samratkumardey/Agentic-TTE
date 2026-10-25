#!/bin/bash
# Phase 0 exit run as batch jobs: one sbatch job per (model, replicate), all on one GPU type.
# Run from the repository root after `source hpc/env.sh`:
#
#   bash hpc/exit_run_submit.sh gpu_requeue gpu:A100:1     # fast start, preemptible
#   bash hpc/exit_run_submit.sh gpu gpu:H100:1             # not preemptible, long queue
#
# Each job writes runs/exit-<GPU>/<model>/rep<k>/ (manifest.txt, protocols, logs) and uses its
# own fresh LLM cache under $ATTE_PROJECT_DIR/exit/<GPU>/<model>/rep<k>/. Job IDs are listed
# in runs/exit-<GPU>.jobs. Check progress with:  uv run python eval/check_exit_run.py runs/exit-<GPU>
set -euo pipefail
PARTITION="${1:?usage: bash hpc/exit_run_submit.sh <partition> <gres, e.g. gpu:A100:1>}"
GRES="${2:?usage: bash hpc/exit_run_submit.sh <partition> <gres, e.g. gpu:A100:1>}"
[[ -n "${ATTE_PROJECT_DIR:-}" ]] || { echo "[atte] source hpc/env.sh first"; exit 1; }
# shellcheck source=/dev/null
source hpc/exit_run_common.sh

GPU="$(echo "$GRES" | cut -d: -f2)"         # gpu:A100:1 -> A100 (GRES names are case-sensitive)
[[ "$GPU" =~ ^[A-Z][A-Za-z0-9]+$ ]] || { echo "[atte] GRES must name a GPU type, e.g. gpu:A100:1"; exit 1; }
LABEL="$(exit_label "$GPU")"                # A100, or A100-<tag> with ATTE_EXIT_TAG
exit_preflight

JOBS="runs/exit-$LABEL.jobs"
# Submitting from inside a (requeued) OnDemand job would pass its SLURM_RESTART_COUNT on through
# --export=ALL; each batch job must report only its own restarts (final run, 2026-10-25).
unset SLURM_RESTART_COUNT
mkdir -p "runs/exit-$LABEL" logs
for spec in $EXIT_MODELS; do
  MODEL="${spec%|*}"; THINK="${spec#*|}"; SLUG="$(exit_slug "$MODEL" "$THINK")"
  for REP in $(seq "$EXIT_REPS"); do
    CACHE="$ATTE_PROJECT_DIR/exit/$LABEL/$SLUG/rep$REP"
    RUN="runs/exit-$LABEL/$SLUG/rep$REP"
    exit_fresh_cache "$CACHE"
    # ATTE_ENV file: the job sources hpc/env.sh, then points the cache at this replicate.
    printf 'source hpc/env.sh\nexport ATTE_CACHE_DIR="%s"\n' "$CACHE" > "$CACHE/env.sh"
    if ! JID=$(sbatch --parsable -A "$SLURM_ACCOUNT" -p "$PARTITION" --gres="$GRES" \
      --job-name="atte-exit-$SLUG-r$REP" --output="logs/exit-$LABEL-$SLUG-rep$REP-%j.out" \
      --export=ALL,ATTE_ENV="$CACHE/env.sh",ATTE_MODEL="$MODEL",ATTE_THINK="$THINK",ATTE_REPLICATE="$REP",ATTE_RUN_DIR="$RUN",ATTE_TRIALS="$EXIT_TRIALS" \
      hpc/replay_phase0.sbatch); then
      rm -f "$CACHE/env.sh"   # not submitted: leave no "already used" marker behind
      echo "[atte] sbatch failed for $MODEL rep$REP; earlier submissions (if any) are in $JOBS"
      exit 1
    fi
    echo "$MODEL $THINK rep$REP $JID $RUN" | tee -a "$JOBS"
  done
done
echo "[atte] submitted; job list in $JOBS"
