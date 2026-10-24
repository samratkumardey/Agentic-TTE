#!/bin/bash
# Shared settings for the Phase 0 exit run (eval/SCORING.md). Sourced by exit_run_submit.sh and
# exit_run_interactive.sh; not run on its own.

# The paper's 10 MIMIC-IV acute-care trials (eval/reference/replay_trials.json).
EXIT_TRIALS="${ATTE_TRIALS:-NCT00475852 NCT02856698 NCT03872011 NCT04134403 NCT06091982 NCT03258684 NCT03592693 NCT01560650 NCT01223690 NCT00953303}"
# "model|ATTE_THINK" pairs (eval/SCORING.md, Design).
EXIT_MODELS="${ATTE_EXIT_MODELS:-qwen3:32b|false gpt-oss:20b|none alibayram/medgemma:27b|none}"
EXIT_REPS="${ATTE_EXIT_REPS:-2}"

# Directory name for a model: qwen3:32b -> qwen3_32b
exit_slug() { echo "$1" | tr '/:' '__'; }

# Refuse to start if a trial record is missing from the shared CT.gov cache (compute nodes may
# have no internet) or if the working tree has uncommitted changes (the manifest would say
# -dirty and the run could not be reproduced from a commit).
exit_preflight() {
  local missing=0 t
  for t in $EXIT_TRIALS; do
    [[ -f "$ATTE_PROJECT_DIR/cache/ctgov/$t.json" ]] || { echo "[atte] missing CT.gov record: $t"; missing=1; }
  done
  if [[ $missing == 1 ]]; then
    echo "[atte] fetch first:  for t in $EXIT_TRIALS; do uv run agentic-tte fetch \"\$t\" > /dev/null; done"
    return 1
  fi
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[atte] uncommitted changes in the checkout; run 'git status' and 'git pull' first"
    return 1
  fi
}

# A fresh cache for one (GPU, model, replicate): empty LLM cache plus a copy of the CT.gov
# records. Refuses to reuse a cache that already holds LLM answers.
exit_fresh_cache() {
  local dir="$1"
  # env.sh marks a replicate already submitted as a batch job (it may still be queued).
  if [[ -f "$dir/llm_cache.sqlite" || -f "$dir/env.sh" ]]; then
    echo "[atte] $dir was already used (ran or submitted). Delete it to rerun that replicate."
    return 1
  fi
  mkdir -p "$dir/ctgov" && cp -n "$ATTE_PROJECT_DIR"/cache/ctgov/*.json "$dir/ctgov/"
}
