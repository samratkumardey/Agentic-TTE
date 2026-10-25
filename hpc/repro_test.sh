#!/bin/bash
# Reproducibility tests. Run in an OnDemand GPU session from the repository root after
# `source hpc/env.sh`, inside tmux:
#
#   bash hpc/repro_test.sh
#
# Default (test 2, order independence): the same trials drafted in forward and in reverse order,
# OLLAMA_NUM_PARALLEL=1, model unloaded before every trial. If each trial's protocol is identical
# in both orders, outputs no longer depend on what the server ran before.
#   Test 1 (2026-10-24, done): ATTE_REPRO_CONFIGS="np4_fa1:4:1 np1_fa1:1:1" ATTE_EXIT_REPS=2
#   ATTE_UNLOAD_PER_TRIAL=0 showed replicates identical once Ollama restarts per replicate.
#
# For every configuration, model and replicate, Ollama is restarted and the trials are drafted
# with a fresh LLM cache. A configuration name ending in "_rev" drafts the trials in reverse.
# Output: runs/repro-<GPU>/<config>/<model>/rep<k>/  (same layout as the exit run). Compare:
#   uv run python eval/compare_runs.py runs/repro-<GPU>/np1_fwd runs/repro-<GPU>/np1_rev
#
# Defaults: 3 trials, qwen3:32b and medgemma:27b, 1 replicate per order. About 1.2 hours on an
# H100; under 10 MB. Override with ATTE_TRIALS, ATTE_EXIT_MODELS, ATTE_EXIT_REPS,
# ATTE_REPRO_CONFIGS ("name:num_parallel:flash_attention").
set -uo pipefail
[[ -n "${ATTE_PROJECT_DIR:-}" ]] || { echo "[atte] source hpc/env.sh first"; exit 1; }
export ATTE_TRIALS="${ATTE_TRIALS:-NCT02856698 NCT06091982 NCT01223690}"
export ATTE_EXIT_MODELS="${ATTE_EXIT_MODELS:-qwen3:32b|false alibayram/medgemma:27b|none}"
export ATTE_EXIT_REPS="${ATTE_EXIT_REPS:-1}"
CONFIGS="${ATTE_REPRO_CONFIGS:-np1_fwd:1:1 np1_rev:1:1}"
# shellcheck source=/dev/null
source hpc/exit_run_common.sh
exit_preflight || exit 1
FORWARD="$EXIT_TRIALS"
REVERSED="$(echo "$EXIT_TRIALS" | tr ' ' '\n' | tac | tr '\n' ' ' | sed 's/ $//')"

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
  if [[ "$NAME" == *_rev ]]; then EXIT_TRIALS="$REVERSED"; else EXIT_TRIALS="$FORWARD"; fi
  for spec in $EXIT_MODELS; do
    MODEL="${spec%|*}"; THINK="${spec#*|}"; SLUG="$(exit_slug "$MODEL" "$THINK")"
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
echo "[atte] done. Compare:  uv run python eval/compare_runs.py runs/repro-$GPU/np1_fwd runs/repro-$GPU/np1_rev"
