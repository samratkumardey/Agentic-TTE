#!/bin/bash
# Phase 0 benchmark run in an existing GPU session (Open OnDemand), one model and replicate after
# another, so every run uses the same GPU. Run from the repository root after
# `source hpc/env.sh`, ideally inside tmux so a closed browser tab does not stop it:
#
#   ATTE_EXIT_TAG=final bash hpc/exit_run_interactive.sh
#
# ATTE_EXIT_FIRST_REP=2 starts at replicate 2 (e.g. after rep1 finished and the run stopped).
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
  for REP in $(seq "${ATTE_EXIT_FIRST_REP:-1}" "$EXIT_REPS"); do  # FIRST_REP: resume a run
    CACHE="$ATTE_PROJECT_DIR/exit/$LABEL/$SLUG/rep$REP"
    RUN="runs/exit-$LABEL/$SLUG/rep$REP"
    # A fresh server for every replicate, started with the recorded settings.
    # shellcheck source=/dev/null
    source hpc/interactive_ollama.sh stop > /dev/null
    # With our server stopped, the GPU must be empty. Another process on it (2026-11-01: a
    # personal Ollama on port 11434 running qwen3:32b) slowed every call past the timeout.
    # Our stopped runner can take a while to release its memory: wait up to a minute.
    for _ in $(seq 12); do
      USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)"
      (( ${USED:-0} > 1024 )) || break
      sleep 5
    done
    if (( ${USED:-0} > 1024 )) && [[ "${ATTE_ALLOW_SHARED_GPU:-0}" != 1 ]]; then
      echo "[atte] the GPU already holds ${USED} MiB from another process (nvidia-smi lists it)."
      echo "[atte] Stop it, or use a batch job; a shared GPU makes calls time out. Not started."
      exit 1
    fi
    exit_fresh_cache "$CACHE" || exit 1
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
