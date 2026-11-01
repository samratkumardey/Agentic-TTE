#!/bin/bash
# Clinician agent on the 12 core trials (eval/SCORING.md, Phase 3 changelog item 29): baseline
# confounders proposed from each trial's qwen3:32b draft and its result-blind literature, with
# the Phase 0 scored settings (OLLAMA_NUM_PARALLEL=1, Ollama restarted per replicate, the
# model unloaded before every trial, a fresh LLM cache per replicate).
# Run from the repository root after `source hpc/env.sh`, on a node with internet (login or
# OnDemand), so the literature can be built from PubMed first:
#
#   bash hpc/clinician_core_submit.sh gpu gpu:H100:1             # batch job, any partition/GPU
#   bash hpc/clinician_core_submit.sh interactive                # now, on this session's GPU
#
# "interactive" runs both replicates one after the other in the current GPU session (Open
# OnDemand), through exit_run_interactive.sh with the same settings, so nothing waits in the
# queue; run it inside tmux. Outputs then go to runs/exit-<this GPU>-core-clinician/.
#
# Inputs:
# - drafts: runs/exit-H100-core/qwen3_32b/rep1/<NCT>.protocol.json (the scored E3 drafts;
#   override with ATTE_DRAFTS);
# - literature: runs/literature/<NCT>.json (override with ATTE_LITERATURE). A missing file is
#   built here; an existing one is never rebuilt, so a literature, once used, stays fixed.
#   Delete a file to rebuild it. The job's manifest records one hash over all twelve.
# Outputs: runs/exit-H100-core-clinician/qwen3_32b/rep<k>/<NCT>.clinician.{json,md}, 2
# replicates (the second checks determinism only). Space: well under 100 MB (abstracts cached
# under $ATTE_PROJECT_DIR/cache/pubmed, sheets, logs, LLM caches).
set -euo pipefail
USAGE="usage: bash hpc/clinician_core_submit.sh <partition> <gres, e.g. gpu:H100:1> | interactive"
PARTITION="${1:?$USAGE}"
if [[ "$PARTITION" != interactive ]]; then
  GRES="${2:?$USAGE}"
fi
[[ -n "${ATTE_PROJECT_DIR:-}" ]] || { echo "[atte] source hpc/env.sh first"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"   # uv

CORE="$(uv run python -c "import json; d = json.load(open('eval/reference/replay_selection.json', encoding='utf-8')); print(' '.join(r['nct_id'] for r in d['candidates'] if r.get('core')))")"
N="$(wc -w <<< "$CORE")"
if [[ "$N" != 12 ]]; then
  echo "[atte] expected 12 core trials in eval/reference/replay_selection.json, found $N"
  exit 1
fi
DRAFTS="${ATTE_DRAFTS:-runs/exit-H100-core/qwen3_32b/rep1}"
LIT="${ATTE_LITERATURE:-runs/literature}"
mkdir -p "$LIT"
echo "[atte] core trials: $CORE"
echo "[atte] drafts: $DRAFTS   literature: $LIT"

for t in $CORE; do
  [[ -f "$DRAFTS/$t.protocol.json" ]] || { echo "[atte] missing draft $DRAFTS/$t.protocol.json"; exit 1; }
  if [[ -f "$LIT/$t.json" ]]; then
    echo "[atte] $t: literature exists, kept as it is"
  elif ! uv run agentic-tte literature "$t" --protocol "$DRAFTS/$t.protocol.json" -o "$LIT/$t.json"; then
    echo "[atte] could not build the literature for $t: run this script from a node with internet (login or OnDemand)"
    exit 1
  fi
done

export ATTE_TASK=clinician ATTE_DRAFTS="$DRAFTS" ATTE_LITERATURE="$LIT"
export ATTE_TRIALS="$CORE"
export ATTE_EXIT_MODELS="${ATTE_EXIT_MODELS:-qwen3:32b|false}"
export ATTE_EXIT_REPS="${ATTE_EXIT_REPS:-2}"
export ATTE_EXIT_TAG="${ATTE_EXIT_TAG:-core-clinician}"
if [[ "$PARTITION" == interactive ]]; then
  exec bash hpc/exit_run_interactive.sh
fi
exec bash hpc/exit_run_submit.sh "$PARTITION" "$GRES"
