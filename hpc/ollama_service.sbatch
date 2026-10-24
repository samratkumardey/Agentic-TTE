#!/bin/bash
# Long-running Ollama service on one H100. Submit with:  sbatch hpc/ollama_service.sbatch
# Partition, account and gres come from hpc/env.sh; pass them on the command line if your
# site does not allow #SBATCH lines to be overridden, e.g.
#   sbatch -p "$SLURM_GPU_PARTITION" --gres="$SLURM_GPU_GRES" hpc/ollama_service.sbatch
#SBATCH --job-name=atte-ollama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/ollama-%j.out

set -euo pipefail
# sbatch runs a spooled copy of this file, so locate env.sh from the submit directory (repo root).
# shellcheck source=/dev/null
source "${ATTE_ENV:-${SLURM_SUBMIT_DIR:-.}/hpc/env.sh}"

mkdir -p "$OLLAMA_MODELS" "$(dirname "$ATTE_ENDPOINT_FILE")" logs

export OLLAMA_HOST="$OLLAMA_BIND:$OLLAMA_PORT"
export OLLAMA_KEEP_ALIVE=-1          # keep the model in VRAM between agent calls
export OLLAMA_MAX_LOADED_MODELS=1    # one large model on the 80 GB card
export OLLAMA_NUM_PARALLEL=4         # concurrent agent requests
export OLLAMA_FLASH_ATTENTION=1

# Requeue before the wall-time limit so the service comes back by itself.
trap 'echo "[atte] wall time nearly over, requeueing"; scontrol requeue "$SLURM_JOB_ID"' USR1

if [[ "$OLLAMA_BIND" == "127.0.0.1" ]]; then
  ADVERTISED="http://127.0.0.1:$OLLAMA_PORT"
else
  ADVERTISED="http://$(hostname -f):$OLLAMA_PORT"
fi
echo "$ADVERTISED $SLURM_JOB_ID $(hostname -f)" > "$ATTE_ENDPOINT_FILE"
echo "[atte] Ollama at $ADVERTISED (job $SLURM_JOB_ID on $(hostname -f))"

apptainer exec --nv \
  --bind "$OLLAMA_MODELS:$OLLAMA_MODELS" \
  --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  "$OLLAMA_SIF" ollama serve &
SERVER_PID=$!
wait "$SERVER_PID"
