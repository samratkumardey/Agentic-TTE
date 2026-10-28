#!/bin/bash
# E2 (a): compiled OMOP criteria vs the mimic-code derived concepts (eval/e2a_mimic_code.py;
# eval/SCORING.md, Phase 3 changelog item 17). Run after hpc/export_mimic_derived.sbatch, from
# the repository root after `source hpc/env.sh`:
#   sbatch -p "$SLURM_CPU_PARTITION" hpc/e2a_mimic_code.sbatch
# By-source breakdown of the disagreements (changelog item 19):
#   sbatch -p "$SLURM_CPU_PARTITION" --export=ALL,ATTE_E2A_BREAKDOWN=1 hpc/e2a_mimic_code.sbatch
# Reads the OMOP CDM read-only and the exports in $ATTE_MIMIC_DERIVED; writes only aggregate
# counts (small cells suppressed) to runs/e2a-<job>/.
#SBATCH --job-name=atte-e2a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e2a-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PATH="$HOME/.local/bin:$PATH"   # uv
source "${ATTE_ENV:-hpc/env.sh}"

: "${ATTE_OMOP_DIR:?set ATTE_OMOP_DIR in hpc/env.sh}"
: "${ATTE_MIMIC_DERIVED:?set ATTE_MIMIC_DERIVED in hpc/env.sh}"
CDM="${ATTE_CDM:-$ATTE_OMOP_DIR/mimiciv-3.1-cdm54-19365483.duckdb}"
JOB="${SLURM_JOB_ID:-local}"
OUT="runs/e2a-$JOB"
mkdir -p "$OUT" logs

for f in icustay_detail first_day_lab first_day_bg ventilation norepinephrine; do
  if [ ! -s "$ATTE_MIMIC_DERIVED/$f.csv.gz" ]; then
    echo "[atte] missing $ATTE_MIMIC_DERIVED/$f.csv.gz: run hpc/export_mimic_derived.sbatch first" >&2
    exit 2
  fi
done

echo "[atte] $(date -u +%FT%TZ) E2 (a) on $CDM"
EXTRA=()
if [ "${ATTE_E2A_BREAKDOWN:-0}" = "1" ]; then
  EXTRA=(--breakdown)
fi
uv run python eval/e2a_mimic_code.py --db "$CDM" --derived "$ATTE_MIMIC_DERIVED" -o "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
{
  echo "job=$JOB"
  echo "node=$(hostname)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "cdm=$CDM"
  echo "derived=$ATTE_MIMIC_DERIVED"
  echo "finished=$(date -u +%FT%TZ)"
} > "$OUT/manifest.txt"
echo "[atte] done: report in $OUT/"
