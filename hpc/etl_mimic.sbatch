#!/bin/bash
# MIMIC-IV 3.1 -> OMOP CDM 5.4 in DuckDB (PLAN.md §16) on a CPU node. Submit from the repository
# root after `source hpc/env.sh`:
#   sbatch -p "$SLURM_CPU_PARTITION" hpc/etl_mimic.sbatch                          # 1% pilot
#   sbatch -p "$SLURM_CPU_PARTITION" --export=ALL,ATTE_ETL_SAMPLE=full hpc/etl_mimic.sbatch
# Reviewed lab/input item maps: --export=ALL,ATTE_ITEM_MAPS="/path/labs.json /path/inputs.json"
# Labs use the OHDSI curated lab map when ATTE_OHDSI_MIMIC is set (hpc/env.example.sh).
#
# Space (all inside the ~/data quota): the 1% pilot writes about 1-2 GB, mostly the vocabulary.
# The full CDM is expected at 20-40 GB; the pilot's size gives the estimate. DuckDB spills go to
# node-local /local/scratch and are removed at the end.
# The CDM never leaves $ATTE_OMOP_DIR. The report and manifest (aggregates only, small cells
# suppressed) are copied to runs/etl-mimic-<job>/ so they can be brought back.
#SBATCH --job-name=atte-etl-mimic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/etl-mimic-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PATH="$HOME/.local/bin:$PATH"   # uv
# shellcheck source=/dev/null
source "${ATTE_ENV:-hpc/env.sh}"

: "${ATTE_MIMIC_SOURCE:?set ATTE_MIMIC_SOURCE in hpc/env.sh (see hpc/env.example.sh)}"
: "${ATTE_VOCAB:?set ATTE_VOCAB in hpc/env.sh}"
: "${ATTE_MIMIC_CONCEPT_MAP:?set ATTE_MIMIC_CONCEPT_MAP in hpc/env.sh}"
: "${ATTE_OMOP_DIR:?set ATTE_OMOP_DIR in hpc/env.sh}"

JOB="${SLURM_JOB_ID:-local}"
SAMPLE="${ATTE_ETL_SAMPLE:-1}"
if [ "$SAMPLE" = "full" ]; then
  NAME="mimiciv-3.1-cdm54-$JOB"
  SAMPLE_ARGS=()
else
  NAME="mimiciv-3.1-cdm54-pilot${SAMPLE}pct-$JOB"
  SAMPLE_ARGS=(--sample-percent "$SAMPLE")
fi
ITEM_ARGS=()
if [ -n "${ATTE_OHDSI_MIMIC:-}" ]; then
  ITEM_ARGS+=(--ohdsi-mimic "$ATTE_OHDSI_MIMIC")
fi
for m in ${ATTE_ITEM_MAPS:-}; do
  ITEM_ARGS+=(--item-map "$m")
done

OUT="$ATTE_OMOP_DIR/$NAME.duckdb"
TMP="${ATTE_ETL_TMP:-/local/scratch/${USER:-$(id -un)}/duckdb-$JOB}"
RUN_DIR="runs/etl-mimic-$JOB"
mkdir -p "$ATTE_OMOP_DIR" "$TMP" "$RUN_DIR" logs
trap 'rm -rf "$TMP"' EXIT

# DuckDB gets 80% of the job's memory; Slurm reports it in MB.
MEM_MB="${SLURM_MEM_PER_NODE:-98304}"
MEM_LIMIT="$((MEM_MB * 8 / 10))MB"
THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "[atte] $(date -u +%FT%TZ) ETL to $OUT (sample: $SAMPLE; threads $THREADS; memory $MEM_LIMIT)"
echo "[atte] current size of $ATTE_OMOP_DIR: $(du -sh "$ATTE_OMOP_DIR" 2>/dev/null | cut -f1)"

uv run agentic-tte etl mimic \
  --source "$ATTE_MIMIC_SOURCE" --vocab "$ATTE_VOCAB" --out "$OUT" \
  --concept-map "$ATTE_MIMIC_CONCEPT_MAP" \
  ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"} ${ITEM_ARGS[@]+"${ITEM_ARGS[@]}"} \
  --threads "$THREADS" --memory-limit "$MEM_LIMIT" --temp-directory "$TMP"

STEM="${OUT%.duckdb}"
{
  echo "partition=${SLURM_JOB_PARTITION:-}"
  echo "cpus=$THREADS"
  echo "mem_mb=$MEM_MB"
  echo "cdm_bytes=$(stat -c %s "$OUT")"
  echo "cdm_size=$(du -h "$OUT" | cut -f1)"
} >> "$STEM.manifest.txt"
cp "$STEM.report.md" "$STEM.report.json" "$STEM.manifest.txt" "$RUN_DIR/"
echo "[atte] $(date -u +%FT%TZ) done: $(du -h "$OUT" | cut -f1) at $OUT; report and manifest in $RUN_DIR/"
