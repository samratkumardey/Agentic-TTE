#!/bin/bash
# The Phase 3 Replay of the core trials on MIMIC-IV OMOP (PLAN.md §17.4 step 6), one review step
# at a time. Run from the repository root after `source hpc/env.sh`, inside tmux:
#
#   bash hpc/replay_core.sh status               # every core trial: waiting for which review
#   bash hpc/replay_core.sh start NCT00625209    # stops at design review (no GPU needed)
#   bash hpc/replay_core.sh resume NCT00625209   # after you filled in the review files
#
# Each run lives in runs/<label>/<NCT>/ (label: $ATTE_REPLAY_LABEL, default mimic-core): the
# review files to fill in, checkpoints.sqlite (aggregates only, no patient rows), report.md,
# result.json for the Concordance agent, and manifest.txt with one block per step.
#
# Settings (eval/SCORING.md Phase 3 changelog item 31; recorded in run.json and manifest.txt):
#   protocol:   the qwen3:32b core draft, $ATTE_DRAFTS/<NCT>.protocol.json
#   confounders: the scored Clinician v4 sheet, $ATTE_CLINICIAN_SHEETS/<NCT>.clinician.json,
#               shown at design review (no LLM call)
#   database:   $ATTE_CDM (default: the 2026-10-27 MIMIC-IV 3.1 CDM 5.4 build)
#   agents:     $ATTE_REPLAY_AGENTS, default "surrogates balance-advice"
#   bootstrap:  500 resamples (R4); S8 at the gate; data classification deidentified, approved
# A step that may call the model (surrogates, balance advice) starts a fresh Ollama with the
# scored settings (OLLAMA_NUM_PARALLEL=1) and needs a whole GPU (Open OnDemand session).
set -uo pipefail
[[ -n "${ATTE_PROJECT_DIR:-}" ]] || { echo "[atte] source hpc/env.sh first"; exit 1; }
# shellcheck source=/dev/null
source hpc/exit_run_common.sh

LABEL="${ATTE_REPLAY_LABEL:-mimic-core}"
DRAFTS="${ATTE_DRAFTS:-runs/exit-H100-core/qwen3_32b/rep1}"
SHEETS="${ATTE_CLINICIAN_SHEETS:-runs/exit-A100-core-clinician-v4/qwen3_32b/rep1}"
CDM="${ATTE_CDM:-${ATTE_OMOP_DIR:-}/mimiciv-3.1-cdm54-19365483.duckdb}"
AGENTS="${ATTE_REPLAY_AGENTS:-surrogates balance-advice}"
CORE="$(uv run python -c "import json; d = json.load(open('eval/reference/replay_selection.json', encoding='utf-8')); print(' '.join(r['nct_id'] for r in d['candidates'] if r.get('core')))")"

usage() { sed -n '2,8p' "$0"; exit 2; }
[[ $# -ge 1 ]] || usage
CMD="$1"; NCT="${2:-}"

if [[ "$CMD" == status ]]; then
  for t in $CORE; do
    d="runs/$LABEL/$t"
    if [[ -f "$d/pending.txt" ]]; then echo "$t  waiting for the $(cat "$d/pending.txt") review"
    elif [[ -f "$d/result.json" ]]; then
      echo "$t  $(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$d/result.json")"
    elif [[ -d "$d" ]]; then echo "$t  started, no review open (check $d/manifest.txt)"
    else echo "$t  not started"; fi
  done
  exit 0
fi

[[ "$CMD" == start || "$CMD" == resume ]] || usage
[[ " $CORE " == *" $NCT "* ]] || { echo "[atte] $NCT is not a core trial: $CORE"; exit 2; }
OUT="runs/$LABEL/$NCT"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[atte] uncommitted changes in the checkout; run 'git status' and 'git pull' first"; exit 1
fi
[[ -f "$CDM" ]] || { echo "[atte] no CDM at $CDM (set ATTE_CDM)"; exit 1; }

needs_llm=0
if [[ "$CMD" == resume ]]; then
  [[ -f "$OUT/pending.txt" ]] || { echo "[atte] $OUT has no open review"; exit 1; }
  grep -qE '"(surrogates|balance_advice)": true' "$OUT/run.json" && needs_llm=1
fi

step_manifest() {  # one block per step (CLAUDE.md rule 7)
  {
    echo "--- step $CMD $(date -Is)"
    echo "answering=$([[ -f "$OUT/pending.txt" ]] && cat "$OUT/pending.txt" || echo none)"
    echo "job_id=${SLURM_JOB_ID:-interactive}"
    echo "node=$(hostname -s)"
    echo "git_commit=$(git rev-parse --short HEAD)"
    echo "cdm=$CDM"
    echo "agents=$AGENTS"
    if [[ $needs_llm == 1 ]]; then
      echo "gpu=$(exit_gpu_query --query-gpu=name,memory.total --format=csv,noheader)"
      echo "ollama_version=$(curl -sf "$ATTE_OLLAMA_HOST/api/version")"
      echo "ollama_num_parallel=$OLLAMA_NUM_PARALLEL"
      echo "model=$ATTE_MODEL think=$ATTE_THINK"
      echo "llm_cache=$ATTE_CACHE_DIR"
    else
      echo "gpu=none (no LLM step)"
    fi
  } >> "$OUT/manifest.txt"
}

if [[ $needs_llm == 1 ]]; then
  exit_refuse_mig || exit 1
  [[ -n "$(exit_gpu_query --query-gpu=name --format=csv,noheader 2>/dev/null)" ]] || {
    echo "[atte] this step may call the model: run it in a GPU session"; exit 1; }
  # A fresh server per step, so no answer depends on the server's earlier requests.
  # shellcheck source=/dev/null
  source hpc/interactive_ollama.sh stop > /dev/null
  # shellcheck source=/dev/null
  source hpc/interactive_ollama.sh || exit 1
  export ATTE_CACHE_DIR="$ATTE_PROJECT_DIR/replay/$LABEL/$NCT"  # after interactive_ollama.sh
  mkdir -p "$ATTE_CACHE_DIR"
fi

if [[ "$CMD" == start ]]; then
  [[ -e "$OUT" ]] && { echo "[atte] $OUT exists: use resume, or move it aside to restart"; exit 1; }
  [[ -f "$DRAFTS/$NCT.protocol.json" ]] || { echo "[atte] no draft $DRAFTS/$NCT.protocol.json"; exit 1; }
  [[ -f "$SHEETS/$NCT.clinician.json" ]] || { echo "[atte] no sheet $SHEETS/$NCT.clinician.json"; exit 1; }
  flags=()
  for a in $AGENTS; do
    case "$a" in surrogates|balance-advice) flags+=("--$a") ;; *) echo "[atte] unknown agent $a"; exit 2 ;; esac
  done
  mkdir -p "$OUT"
  step_manifest
  uv run agentic-tte replay start --out "$OUT" --db "$CDM" --protocol "$DRAFTS/$NCT.protocol.json" \
    --clinician-sheet "$SHEETS/$NCT.clinician.json" --classification deidentified --approved \
    --bootstrap 500 --s8 "${flags[@]}" 2>&1 | tee -a "$OUT/steps.log"
else
  step_manifest
  uv run agentic-tte replay resume --out "$OUT" 2>&1 | tee -a "$OUT/steps.log"
fi
rc=${PIPESTATUS[0]}
echo "finished=$(date -Is) exit=$rc" >> "$OUT/manifest.txt"
if [[ $needs_llm == 1 ]]; then
  # shellcheck source=/dev/null
  source hpc/interactive_ollama.sh stop > /dev/null
fi
exit "$rc"
