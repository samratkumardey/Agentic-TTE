#!/bin/bash
# Export the MIT-LCP mimic-code derived concepts E2 (a) compares against, from the native
# Postgres build of MIMIC-IV (eval/SCORING.md, Phase 3 changelog item 17). Submit from the
# repository root after `source hpc/env.sh`:
#   sbatch -p "$SLURM_CPU_PARTITION" hpc/export_mimic_derived.sbatch
#
# Starts the Postgres server on this node only for the export, on a private socket in
# node-local scratch (no TCP), then stops it. Refuses to start if a postmaster.pid shows the
# data directory in use.
# Writes about 50-100 MB of gzipped CSVs to $ATTE_MIMIC_DERIVED (the lab folder, mode 700):
# patient-level data, which never leave it. Row counts and a manifest go to
# runs/export-derived-<job>/ so they can be brought back.
#SBATCH --job-name=atte-export-derived
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/export-derived-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source "${ATTE_ENV:-hpc/env.sh}"

: "${ATTE_MIMIC_PG:?set ATTE_MIMIC_PG in hpc/env.sh (see hpc/env.example.sh)}"
: "${ATTE_MIMIC_PG_ENV:?set ATTE_MIMIC_PG_ENV in hpc/env.sh}"
: "${ATTE_MIMIC_DERIVED:?set ATTE_MIMIC_DERIVED in hpc/env.sh}"

JOB="${SLURM_JOB_ID:-local}"
PGBIN="$ATTE_MIMIC_PG_ENV/bin"
PORT="${ATTE_MIMIC_PG_PORT:-5433}"
# Unix socket paths are limited to about 100 characters: keep it short.
SOCK="/local/scratch/${USER:-$(id -un)}/pg-$JOB"
RUN_DIR="runs/export-derived-$JOB"
mkdir -p "$SOCK" "$RUN_DIR" "$ATTE_MIMIC_DERIVED" logs

if [ -e "$ATTE_MIMIC_PG/postmaster.pid" ]; then
  echo "[atte] $ATTE_MIMIC_PG/postmaster.pid exists: a server may be running on it; not starting" >&2
  exit 2
fi

stop_server() {
  "$PGBIN/pg_ctl" -D "$ATTE_MIMIC_PG" -m fast -w stop >/dev/null 2>&1 || true
  rm -rf "$SOCK"
}
trap stop_server EXIT

echo "[atte] $(date -u +%FT%TZ) starting Postgres on socket $SOCK, port $PORT (no TCP)"
"$PGBIN/pg_ctl" -D "$ATTE_MIMIC_PG" -l "$RUN_DIR/postgres.log" -w \
  -o "-p $PORT -k $SOCK -c listen_addresses=''" start

PSQL=("$PGBIN/psql" -h "$SOCK" -p "$PORT" -v ON_ERROR_STOP=1 -X -q)
DB=""
for d in $("${PSQL[@]}" -d postgres -Atc "SELECT datname FROM pg_database WHERE NOT datistemplate"); do
  if [ "$("${PSQL[@]}" -d "$d" -Atc "SELECT count(*) FROM information_schema.schemata WHERE schema_name = 'mimiciv_derived'")" = "1" ]; then
    DB="$d"
    break
  fi
done
if [ -z "$DB" ]; then
  echo "[atte] no database has the mimiciv_derived schema" >&2
  exit 3
fi
echo "[atte] database $DB has mimiciv_derived"

export_table() {  # name, query
  local name="$1" query="$2"
  "${PSQL[@]}" -d "$DB" -c "\copy ($query) TO PROGRAM 'gzip > $ATTE_MIMIC_DERIVED/$name.csv.gz' WITH CSV HEADER"
  local n
  n=$("${PSQL[@]}" -d "$DB" -Atc "SELECT count(*) FROM ($query) q")
  echo "$name=$n" >> "$RUN_DIR/row_counts.txt"
  echo "[atte] $name: $n rows"
}

: > "$RUN_DIR/row_counts.txt"
export_table icustay_detail "SELECT subject_id, stay_id, icu_intime, admission_age, first_icu_stay FROM mimiciv_derived.icustay_detail"
export_table first_day_lab "SELECT stay_id, creatinine_max FROM mimiciv_derived.first_day_lab"
export_table first_day_bg "SELECT stay_id, lactate_max FROM mimiciv_derived.first_day_bg"
export_table ventilation "SELECT stay_id, starttime, endtime, ventilation_status FROM mimiciv_derived.ventilation"
export_table norepinephrine "SELECT stay_id, starttime, endtime FROM mimiciv_derived.norepinephrine"

{
  echo "job=$JOB"
  echo "node=$(hostname)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "postgres=$("$PGBIN/postgres" --version)"
  echo "database=$DB"
  echo "data_directory=$ATTE_MIMIC_PG"
  echo "output=$ATTE_MIMIC_DERIVED"
  echo "mimic_code=$(git -C "${ATTE_MIMIC_CODE:-/cluster/pixstor/data/$USER/mimic-code}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "finished=$(date -u +%FT%TZ)"
} > "$RUN_DIR/manifest.txt"
du -sh "$ATTE_MIMIC_DERIVED" | tee -a "$RUN_DIR/manifest.txt"
echo "[atte] $(date -u +%FT%TZ) done: row counts and manifest in $RUN_DIR/"
