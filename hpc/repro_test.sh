#!/bin/bash
# Reproducibility test: are replicates identical at temperature 0 when Ollama serves one
# request slot (OLLAMA_NUM_PARALLEL=1) instead of four? Run in an OnDemand GPU session from the
# repository root after `source hpc/env.sh`, inside tmux:
#
#   bash hpc/repro_test.sh
#
# For every configuration, model and replicate, Ollama is restarted (so each replicate starts
# from the same server state) and the trials are drafted with a fresh LLM cache. Output:
#   runs/repro-<GPU>/<config>/<model>/rep<k>/   (same layout as the exit run)
# Compare each configuration with:
#   for c in runs/repro-<GPU>/*/; do uv run python eval/score_phase0.py "$c" --out eval/results/repro; done
#
# Defaults: 3 trials whose replicates differed in the exit run, qwen3:32b and medgemma:27b,
# configurations np4_fa1 (exit-run setting) and np1_fa1. About 2 hours on an H100; under 10 MB.
# Override with ATTE_TRIALS, ATTE_EXIT_MODELS, ATTE_REPRO_CONFIGS ("name:num_parallel:flash").
set -uo pipefail
[[ -n "${ATTE_PROJECT_DIR:-}" ]] || { echo "[atte] source hpc/env.sh first"; exit 1; }
export ATTE_TRIALS="${ATTE_TRIALS:-NCT02856698 NCT06091982 NCT01223690}"
export ATTE_EXIT_MODELS="${ATTE_EXIT_MODELS:-qwen3:32b|false alibayram/medgemma:27b|none}"
CONFIGS="${ATTE_REPRO_CONFIGS:-np4_fa1:4:1 np1_fa1:1:1}"
# shellcheck source=/dev/null
source hpc/exit_run_common.sh
exit_preflight || exit 1

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
[[ -n "$GPU_NAME" ]] || { echo "[atte] no GPU in this session"; exit 1; }
GPU="$(echo "$GPU_NAME" | grep -oE 'H100|H200|A100|L40S' | head -n1)"
GPU="${GPU:-$(echo "$GPU_NAME" | tr -c 'A-Za-z0-9\n' '_')}"
echo "[atte] reproducibility test on $GPU_NAME -> runs/repro-$GPU"

# shellcheck source=/dev/null
source hpc/interactive_ollama.sh stop > /dev/null
for cfg in $CONFIGS; do
  NAME="${cfg%%:*}"; rest="${cfg#*:}"
  export ATTE_OLLAMA_NUM_PARALLEL="${rest%%:*}" ATTE_OLLAMA_FLASH_ATTENTION="${rest#*:}"
  for spec in $EXIT_MODELS; do
    MODEL="${spec%|*}"; THINK="${spec#*|}"; SLUG="$(exit_slug "$MODEL")"
    export ATTE_MODEL="$MODEL" ATTE_THINK="$THINK"
    for REP in $(seq "$EXIT_REPS"); do
      # shellcheck source=/dev/null
      source hpc/interactive_ollama.sh stop > /dev/null
      # shellcheck source=/dev/null
      source hpc/interactive_ollama.sh || exit 1
      CACHE="$ATTE_PROJECT_DIR/repro/$GPU/$NAME/$SLUG/rep$REP"
      RUN="runs/repro-$GPU/$NAME/$SLUG/rep$REP"
      exit_fresh_cache "$CACHE" || exit 1
      export ATTE_CACHE_DIR="$CACHE"
      mkdir -p "$RUN"
      exit_write_manifest "$RUN" "$MODEL" "$THINK" "$REP"
      exit_draft_trials "$RUN" "$MODEL" "$REP"
    done
  done
done
# shellcheck source=/dev/null
source hpc/interactive_ollama.sh stop
echo "[atte] done. Compare:  for c in runs/repro-$GPU/*/; do uv run python eval/score_phase0.py \"\$c\" --out eval/results/repro; done"
