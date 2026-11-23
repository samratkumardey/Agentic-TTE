#!/bin/bash
# Check the autonomous arm's rules on MIMIC-IV OMOP before any run (eval/autonomy_check.py;
# eval/SCORING.md Phase 3 item 34). CPU only, no LLM. From the repository root after
# `source hpc/env.sh`:
#   sbatch -p "$SLURM_CPU_PARTITION" hpc/autonomy_check.sbatch
# Reads the OMOP CDM read-only and the qwen3:32b core drafts ($ATTE_DRAFTS); writes only pooled
# counts (no arms, no outcomes; small cells suppressed) to runs/autonomy-check-<job>/, a few KB.
#SBATCH --job-name=atte-autonomy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/autonomy-check-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PATH="$HOME/.local/bin:$PATH"   # uv
source "${ATTE_ENV:-hpc/env.sh}"

: "${ATTE_OMOP_DIR:?set ATTE_OMOP_DIR in hpc/env.sh}"
CDM="${ATTE_CDM:-$ATTE_OMOP_DIR/mimiciv-3.1-cdm54-19365483.duckdb}"
DRAFTS="${ATTE_DRAFTS:-runs/exit-H100-core/qwen3_32b/rep1}"
JOB="${SLURM_JOB_ID:-local}"
OUT="runs/autonomy-check-$JOB"
mkdir -p "$OUT" logs
[ -f "$CDM" ] || { echo "[atte] no CDM at $CDM (set ATTE_CDM)" >&2; exit 2; }
[ -d "$DRAFTS" ] || { echo "[atte] no drafts in $DRAFTS (set ATTE_DRAFTS)" >&2; exit 2; }

echo "[atte] $(date -u +%FT%TZ) autonomy check on $CDM"
{
  echo "job=$JOB"
  echo "node=$(hostname)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "cdm=$CDM"
  echo "drafts=$DRAFTS"
  echo "gpu=none (no LLM)"
  echo "started=$(date -u +%FT%TZ)"
} > "$OUT/manifest.txt"
rc=0
uv run python eval/autonomy_check.py --db "$CDM" --drafts "$DRAFTS" --out "$OUT" || rc=$?
echo "finished=$(date -u +%FT%TZ) exit=$rc" >> "$OUT/manifest.txt"
echo "[atte] done: $OUT/summary.md"
exit "$rc"
