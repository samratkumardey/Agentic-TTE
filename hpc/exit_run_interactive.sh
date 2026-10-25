#!/bin/bash
# Phase 0 benchmark run in an existing GPU session (Open OnDemand), one model and replicate after
# another, so every run uses the same GPU. Run from the repository root after
# `source hpc/env.sh`, ideally inside tmux so a closed browser tab does not stop it:
#
#   ATTE_EXIT_TAG=final bash hpc/exit_run_interactive.sh
#
# Reproducibility settings (repro test 2026-10-24, hpc/HELLBENDER.md): Ollama runs with
# OLLAMA_NUM_PARALLEL=1, is restarted before every replicate, and the model is unloaded before
# every trial. All three are recorded in each manifest.txt.
#
# Writes runs/exit-<GPU>[-<tag>]/<model arm>/rep<k>/ with manifest.txt, one protocol and log per
# trial, and a fresh LLM cache per replicate.
set -uo pipefail
[[ -n "${ATTE_PROJECT_DIR:-}" ]] || { echo "[atte] source hpc/env.sh first"; exit 1; }
# shellcheck source=/dev/null
source hpc/exit_run_common.sh
exit_preflight || exit 1

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
[[ -n "$GPU_NAME" ]] || { echo "[atte] no GPU in this session"; exit 1; }
GPU="$(echo "$GPU_NAME" | grep -oE 'H100|H200|A100|L40S' | head -n1)"
LABEL="$(exit_label "${GPU:-$(echo "$GPU_NAME" | tr -c 'A-Za-z0-9\n' '_')}")"
echo "[atte] benchmark run on $GPU_NAME -> runs/exit-$LABEL"

for spec in $EXIT_MODELS; do
  MODEL="${spec%|*}"; THINK="${spec#*|}"; SLUG="$(exit_slug "$MODEL" "$THINK")"
  export ATTE_MODEL="$MODEL" ATTE_THINK="$THINK"
  for REP in $(seq "$EXIT_REPS"); do
    CACHE="$ATTE_PROJECT_DIR/exit/$LABEL/$SLUG/rep$REP"
    RUN="runs/exit-$LABEL/$SLUG/rep$REP"
    exit_fresh_cache "$CACHE" || exit 1
    # A fresh server for every replicate, started with the recorded settings.
    # shellcheck source=/dev/null
    source hpc/interactive_ollama.sh stop > /dev/null
    # shellcheck source=/dev/null
    source hpc/interactive_ollama.sh || exit 1
    export ATTE_CACHE_DIR="$CACHE"          # after interactive_ollama.sh, which resets it
    mkdir -p "$RUN"
    exit_write_manifest "$RUN" "$MODEL" "$THINK" "$REP"
    exit_draft_trials "$RUN" "$MODEL" "$REP"
  done
done
# shellcheck source=/dev/null
source hpc/interactive_ollama.sh stop
echo "[atte] done. Check with:  uv run python eval/check_exit_run.py runs/exit-$LABEL"
