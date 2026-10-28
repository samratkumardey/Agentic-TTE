#!/bin/bash
# Trialist drafts of the 12 core trials (eval/SCORING.md, Phase 3: R1 and changelog items 11a,
# 13), with the Phase 0 scored settings: OLLAMA_NUM_PARALLEL=1, Ollama restarted per
# replicate, the model unloaded before every trial, a fresh LLM cache per replicate.
# Run from the repository root after `source hpc/env.sh`, on a node with internet (login or
# OnDemand) so the registry records can be fetched first:
#
#   bash hpc/core_draft_submit.sh gpu gpu:H100:1
#
# The trial list is read from eval/reference/replay_selection.json (the "core" flags), never
# typed by hand. Defaults: qwen3:32b (primary model, R1), 2 replicates (the second checks
# determinism only), tag "core" -> runs/exit-H100-core/<model>/rep<k>/. Override with
# ATTE_EXIT_MODELS, ATTE_EXIT_REPS or ATTE_EXIT_TAG. Space: well under 1 GB (protocols, logs,
# LLM caches under $ATTE_PROJECT_DIR/exit/).
set -euo pipefail
PARTITION="${1:?usage: bash hpc/core_draft_submit.sh <partition> <gres, e.g. gpu:H100:1>}"
GRES="${2:?usage: bash hpc/core_draft_submit.sh <partition> <gres, e.g. gpu:H100:1>}"
[[ -n "${ATTE_PROJECT_DIR:-}" ]] || { echo "[atte] source hpc/env.sh first"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"   # uv

CORE="$(uv run python -c "import json; d = json.load(open('eval/reference/replay_selection.json', encoding='utf-8')); print(' '.join(r['nct_id'] for r in d['candidates'] if r.get('core')))")"
N="$(wc -w <<< "$CORE")"
if [[ "$N" != 12 ]]; then
  echo "[atte] expected 12 core trials in eval/reference/replay_selection.json, found $N"
  exit 1
fi
echo "[atte] core trials: $CORE"

for t in $CORE; do
  if ! uv run agentic-tte fetch "$t" > /dev/null; then
    echo "[atte] could not fetch $t: run this script from a node with internet (login or OnDemand)"
    exit 1
  fi
done

export ATTE_TRIALS="$CORE"
export ATTE_EXIT_MODELS="${ATTE_EXIT_MODELS:-qwen3:32b|false}"
export ATTE_EXIT_REPS="${ATTE_EXIT_REPS:-2}"
export ATTE_EXIT_TAG="${ATTE_EXIT_TAG:-core}"
exec bash hpc/exit_run_submit.sh "$PARTITION" "$GRES"
