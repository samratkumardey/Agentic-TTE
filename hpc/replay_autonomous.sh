#!/bin/bash
# The autonomous arm of the Phase 3 MIMIC Replay (eval/SCORING.md items 33-35): the 12 core
# trials, each run end to end with no person, under the frozen rules and policy. Run from the
# repository root after `source hpc/env.sh`, in a GPU session (Open OnDemand), inside tmux:
#
#   bash hpc/replay_autonomous.sh              # every core trial without a finished run
#   bash hpc/replay_autonomous.sh NCT02643381  # one trial
#   bash hpc/replay_autonomous.sh status       # each trial's status and stop reason
#
# Each trial: runs/<label>/<NCT>/ (label $ATTE_REPLAY_LABEL, default mimic-autonomous) with
# report.md, result.json (for the Concordance agent), run.json, checkpoints.sqlite (aggregates,
# no patient rows) and manifest.txt. A few MB per trial; under 200 MB for all 12.
#
# Settings (recorded in run.json and manifest.txt; none may change once a trial has run):
#   --repair --autonomy --balance-advice --auto eval/policies/autonomous-v1.json (item 34/35)
#   item 31: the qwen3:32b core draft, the Clinician v4 sheet, $ATTE_CDM, 500 bootstrap
#            resamples, S8 at the gate, data classification deidentified, approved
# The balance advice can call the model, so every trial starts a fresh Ollama with the scored
# settings (OLLAMA_NUM_PARALLEL=1) and its own LLM cache, and the session needs a whole GPU.
# A trial is never rerun once it has a result: a run that stopped or failed counts (item 33).
set -uo pipefail
[[ -n "${ATTE_PROJECT_DIR:-}" ]] || { echo "[atte] source hpc/env.sh first"; exit 1; }
# shellcheck source=/dev/null
source hpc/exit_run_common.sh

LABEL="${ATTE_REPLAY_LABEL:-mimic-autonomous}"
DRAFTS="${ATTE_DRAFTS:-runs/exit-H100-core/qwen3_32b/rep1}"
SHEETS="${ATTE_CLINICIAN_SHEETS:-runs/exit-A100-core-clinician-v4/qwen3_32b/rep1}"
CDM="${ATTE_CDM:-${ATTE_OMOP_DIR:-}/mimiciv-3.1-cdm54-19365483.duckdb}"
POLICY="eval/policies/autonomous-v1.json"
CORE="$(uv run python -c "import json; d = json.load(open('eval/reference/replay_selection.json', encoding='utf-8')); print(' '.join(r['nct_id'] for r in d['candidates'] if r.get('core')))")"

status_of() {  # status and stop reason of a finished run
  uv run python -c 'import json,sys; d = json.load(open(sys.argv[1])); print(d["status"], "|", d.get("stop_reason") or "-", "|", "analysed" if d.get("analysis") else "no analysis")' "$1"
}

if [[ "${1:-}" == status ]]; then
  for t in $CORE; do
    d="runs/$LABEL/$t"
    if [[ -f "$d/result.json" ]]; then echo "$t  $(status_of "$d/result.json")"
    elif [[ -d "$d" ]]; then echo "$t  started, no result (see $d/steps.log)"
    else echo "$t  not started"; fi
  done
  exit 0
fi

# The rules must be confirmed before any trial runs (items 34 and 35).
for item in 34 35; do
  grep -qE "^  $item\. .*confirmed by the user on" eval/SCORING.md || {
    echo "[atte] SCORING.md item $item is not confirmed: no autonomous run before it is"; exit 1; }
done
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[atte] uncommitted changes in the checkout; run 'git status' and 'git pull' first"; exit 1
fi
[[ -f "$CDM" ]] || { echo "[atte] no CDM at $CDM (set ATTE_CDM)"; exit 1; }
exit_refuse_mig || exit 1
[[ -n "$(exit_gpu_query --query-gpu=name --format=csv,noheader 2>/dev/null)" ]] || {
  echo "[atte] the balance advice may call the model: run this in a GPU session"; exit 1; }

TRIALS="${1:-$CORE}"
for t in $TRIALS; do
  [[ " $CORE " == *" $t "* ]] || { echo "[atte] $t is not a core trial: $CORE"; exit 2; }
done

for t in $TRIALS; do
  OUT="runs/$LABEL/$t"
  if [[ -e "$OUT" ]]; then
    echo "[atte] $t: $OUT exists, not run again ($( [[ -f "$OUT/result.json" ]] && status_of "$OUT/result.json" || echo 'no result: move it aside, with its reason, to restart'))"
    continue
  fi
  [[ -f "$DRAFTS/$t.protocol.json" ]] || { echo "[atte] $t: no draft $DRAFTS/$t.protocol.json"; continue; }
  [[ -f "$SHEETS/$t.clinician.json" ]] || { echo "[atte] $t: no sheet $SHEETS/$t.clinician.json"; continue; }
  mkdir -p "$OUT"
  # A fresh server and cache per trial, so no answer depends on another trial's requests.
  # shellcheck source=/dev/null
  source hpc/interactive_ollama.sh stop > /dev/null
  # shellcheck source=/dev/null
  source hpc/interactive_ollama.sh || exit 1
  export ATTE_CACHE_DIR="$ATTE_PROJECT_DIR/replay/$LABEL/$t"  # after interactive_ollama.sh
  mkdir -p "$ATTE_CACHE_DIR"
  {
    echo "--- autonomous run $(date -Is)"
    echo "job_id=${SLURM_JOB_ID:-interactive}"
    echo "node=$(hostname -s)"
    echo "git_commit=$(git rev-parse --short HEAD)"
    echo "cdm=$CDM"
    echo "draft=$DRAFTS/$t.protocol.json"
    echo "clinician_sheet=$SHEETS/$t.clinician.json"
    echo "policy=$POLICY sha256=$(sha256sum "$POLICY" | cut -c1-16)"
    echo "flags=--repair --autonomy --balance-advice --s8 --bootstrap 500"
    echo "gpu=$(exit_gpu_query --query-gpu=name,memory.total --format=csv,noheader)"
    echo "ollama_version=$(curl -sf "$ATTE_OLLAMA_HOST/api/version")"
    echo "ollama_num_parallel=$OLLAMA_NUM_PARALLEL"
    echo "model=$ATTE_MODEL think=$ATTE_THINK"
    echo "llm_cache=$ATTE_CACHE_DIR"
  } >> "$OUT/manifest.txt"
  echo "[atte] $t: autonomous run in $OUT"
  uv run agentic-tte replay start --out "$OUT" --db "$CDM" --protocol "$DRAFTS/$t.protocol.json" \
    --clinician-sheet "$SHEETS/$t.clinician.json" --classification deidentified --approved \
    --bootstrap 500 --s8 --repair --autonomy --balance-advice --auto "$POLICY" \
    2>&1 | tee -a "$OUT/steps.log"
  rc=${PIPESTATUS[0]}
  echo "finished=$(date -Is) exit=$rc" >> "$OUT/manifest.txt"
  if [[ -f "$OUT/result.json" ]]; then
    echo "[atte] $t: $(status_of "$OUT/result.json")"
  else
    echo "[atte] $t: no result (exit $rc): a failure of the autonomous arm; see $OUT/steps.log"
  fi
done
# shellcheck source=/dev/null
source hpc/interactive_ollama.sh stop > /dev/null
echo "[atte] done; summary: bash hpc/replay_autonomous.sh status"
