#!/bin/bash
# Shared settings for the Phase 0 exit run (eval/SCORING.md) and the reproducibility test.
# Sourced by exit_run_submit.sh, exit_run_interactive.sh and repro_test.sh; not run on its own.

# The paper's 10 MIMIC-IV acute-care trials (eval/reference/replay_trials.json).
EXIT_TRIALS="${ATTE_TRIALS:-NCT00475852 NCT02856698 NCT03872011 NCT04134403 NCT06091982 NCT03258684 NCT03592693 NCT01560650 NCT01223690 NCT00953303}"
# "model|ATTE_THINK" pairs (eval/SCORING.md, Design). ATTE_THINK: false/true, low/medium/high
# (gpt-oss reasoning level), or none (model default).
EXIT_MODELS="${ATTE_EXIT_MODELS:-qwen3:32b|false gpt-oss:20b|none alibayram/medgemma:27b|none}"
EXIT_REPS="${ATTE_EXIT_REPS:-2}"
# Optional tag for a new arm, so it never collides with an earlier run: runs/exit-H100-<tag>/
EXIT_TAG="${ATTE_EXIT_TAG:-}"

# Unload the model before every trial, so a trial's output never depends on which requests the
# server handled before it (repro test 2026-10-24: outputs depended on server state).
EXIT_UNLOAD_PER_TRIAL="${ATTE_UNLOAD_PER_TRIAL:-1}"

# Directory name for a model arm: qwen3:32b -> qwen3_32b; gpt-oss:20b with THINK=low ->
# gpt-oss_20b-low, so two reasoning levels of one model never share a directory.
exit_slug() {
  local slug
  slug="$(echo "$1" | tr '/:' '__')"
  case "${2:-}" in low|medium|high|true) slug="$slug-$2" ;; esac
  echo "$slug"
}

# Free the model's runner (and its prompt cache) without stopping the server.
exit_unload_model() {
  curl -sf "$ATTE_OLLAMA_HOST/api/generate" -d "{\"model\": \"$1\", \"keep_alive\": 0}" \
    > /dev/null || echo "[atte] warning: could not unload $1"
}

# "H100" -> "H100" or "H100-<tag>"
exit_label() { echo "$1${EXIT_TAG:+-$EXIT_TAG}"; }

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

# manifest.txt for an interactive run (CLAUDE.md rule 7). Batch jobs write theirs in
# replay_phase0.sbatch with the same keys.
exit_write_manifest() {
  local run="$1" model="$2" think="$3" rep="$4"
  {
    echo "job_id=${SLURM_JOB_ID:-interactive}"
    echo "partition=${SLURM_JOB_PARTITION:-}"
    echo "node=$(hostname -s)"
    echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
    echo "git_commit=$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
    echo "ollama_image=$OLLAMA_SIF"
    echo "ollama_version=$(curl -sf "$ATTE_OLLAMA_HOST/api/version")"
    echo "ollama_num_parallel=$OLLAMA_NUM_PARALLEL"
    echo "ollama_flash_attention=$OLLAMA_FLASH_ATTENTION"
    echo "unload_per_trial=$EXIT_UNLOAD_PER_TRIAL"
    echo "model=$model"
    echo "think=$think"
    echo "num_predict=${ATTE_NUM_PREDICT:-8192}"
    echo "llm_timeout_s=${ATTE_LLM_TIMEOUT_S:-600}"
    echo "replicate=$rep"
    echo "cache_dir=$ATTE_CACHE_DIR"
    echo "restart_count=0"
    echo "trials=$EXIT_TRIALS"
    echo "started=$(date -Is)"
  } | tee "$run/manifest.txt"
}

# Draft every trial into $run, then close the manifest with the finish time and failures.
exit_draft_trials() {
  local run="$1" model="$2" rep="$3" failed=0 t
  for t in $EXIT_TRIALS; do
    echo "[atte] $(date +%T) $model rep$rep $t"
    [[ "$EXIT_UNLOAD_PER_TRIAL" == "1" ]] && exit_unload_model "$model"
    if ! uv run agentic-tte draft "$t" -o "$run/$t.protocol.json" 2> "$run/$t.log"; then
      echo "[atte] $t failed, see $run/$t.log"
      failed=$((failed + 1))
    fi
    tail -n 3 "$run/$t.log"
  done
  echo "finished=$(date -Is) failed=$failed" >> "$run/manifest.txt"
}
