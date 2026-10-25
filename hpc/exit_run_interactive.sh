#!/bin/bash
# Phase 0 exit run in an existing GPU session (Open OnDemand), one model and replicate after
# another, so every run uses the same GPU. Run from the repository root after
# `source hpc/env.sh`, ideally inside tmux so a closed browser tab does not stop it:
#
#   bash hpc/exit_run_interactive.sh
#
# A new arm on the same GPU needs a tag, e.g. gpt-oss with a fixed reasoning level:
#   ATTE_EXIT_TAG=gptoss-low ATTE_EXIT_MODELS="gpt-oss:20b|low" bash hpc/exit_run_interactive.sh
#
# Writes runs/exit-<GPU>[-<tag>]/<model>/rep<k>/ with manifest.txt, one protocol and log per
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
echo "[atte] exit run on $GPU_NAME -> runs/exit-$LABEL"

for spec in $EXIT_MODELS; do
  MODEL="${spec%|*}"; THINK="${spec#*|}"; SLUG="$(exit_slug "$MODEL")"
  export ATTE_MODEL="$MODEL" ATTE_THINK="$THINK"
  # Starts or reuses the project Ollama on $OLLAMA_PORT and checks the model is present.
  # shellcheck source=/dev/null
  source hpc/interactive_ollama.sh || exit 1
  for REP in $(seq "$EXIT_REPS"); do
    CACHE="$ATTE_PROJECT_DIR/exit/$LABEL/$SLUG/rep$REP"
    RUN="runs/exit-$LABEL/$SLUG/rep$REP"
    exit_fresh_cache "$CACHE" || exit 1
    export ATTE_CACHE_DIR="$CACHE"          # after interactive_ollama.sh, which resets it
    mkdir -p "$RUN"
    exit_write_manifest "$RUN" "$MODEL" "$THINK" "$REP"
    exit_draft_trials "$RUN" "$MODEL" "$REP"
  done
done
# shellcheck source=/dev/null
source hpc/interactive_ollama.sh stop
echo "[atte] done. Check with:  uv run python eval/check_exit_run.py runs/exit-$LABEL"
