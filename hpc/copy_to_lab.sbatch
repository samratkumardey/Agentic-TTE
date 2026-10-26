#!/bin/bash
# Copy one folder to another place (e.g. the lab storage) and verify the copy. The source is
# NEVER deleted: remove it yourself once the log says VERIFIED. Submit from the repository root:
#   sbatch -p "$SLURM_CPU_PARTITION" --export=ALL,SRC=/path/to/folder,DST=/path/to/new/folder \
#          hpc/copy_to_lab.sbatch
# rsync -a keeps owners, groups and permissions as they are (Postgres keeps its required 700
# data directory). Many files are world-readable, so privacy comes from the destination
# folder: create your lab folder with mode 700 first (see hpc/HELLBENDER.md).
# Do not copy a Postgres data directory while its server is running.
#SBATCH --job-name=atte-copy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/copy-%j.out

set -euo pipefail
: "${SRC:?set SRC to the folder to copy}"
: "${DST:?set DST to the new location}"
SRC="${SRC%/}"
DST="${DST%/}"
[ -d "$SRC" ] || { echo "[atte] source $SRC is not a folder"; exit 1; }
if [ -L "$SRC" ]; then
  echo "[atte] source $SRC is a link (already moved?): nothing to do"; exit 1
fi
PARENT=$(dirname "$DST")
if [ ! -d "$PARENT" ]; then
  echo "[atte] create $PARENT first, with mode 700 (only you can enter it)"; exit 1
fi
mkdir -p "$DST"

echo "[atte] $(date -u +%FT%TZ) copy $SRC -> $DST ($(du -sh "$SRC" | cut -f1))"
rsync -a --info=stats2 "$SRC/" "$DST/"

# Verify: a second dry run must find nothing left to transfer (same files, sizes and times).
CHANGES=$(rsync -a --dry-run --itemize-changes "$SRC/" "$DST/" | wc -l)
N_SRC=$(find "$SRC" -type f | wc -l)
N_DST=$(find "$DST" -type f | wc -l)
echo "[atte] files: source $N_SRC, copy $N_DST; differences on recheck: $CHANGES"
if [ "$CHANGES" -eq 0 ] && [ "$N_SRC" -eq "$N_DST" ]; then
  echo "[atte] $(date -u +%FT%TZ) VERIFIED: $DST ($(du -sh "$DST" | cut -f1)) matches $SRC"
  echo "[atte] the source is kept; remove it yourself and leave a link (see hpc/HELLBENDER.md)"
else
  echo "[atte] NOT VERIFIED: keep the source and look at the differences above"
  exit 1
fi
