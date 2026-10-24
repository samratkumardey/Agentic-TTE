#!/bin/bash
# One self-contained job: start Ollama on the allocated H100, draft protocols for a list of
# trials, stop Ollama. Submit from the repository root:
#   sbatch -p "$SLURM_GPU_PARTITION" --gres="$SLURM_GPU_GRES" hpc/replay_phase0.sbatch
# Override the trial list or model with  --export=ALL,ATTE_TRIALS="NCT...",ATTE_MODEL=...
# Run `agentic-tte fetch <NCT>` on the login node first if compute nodes have no internet.
#SBATCH --job-name=atte-replay0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/replay0-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PATH="$HOME/.local/bin:$PATH"   # uv
# shellcheck source=/dev/null
source "${ATTE_ENV:-hpc/env.sh}"

# The paper's MIMIC-IV acute-care trials (PLAN.md §11).
TRIALS="${ATTE_TRIALS:-NCT00475852 NCT02856698 NCT03872011 NCT04134403 NCT06091982 NCT03258684 NCT03592693 NCT01560650 NCT01223690 NCT00953303}"
export ATTE_MODEL="${ATTE_MODEL:-qwen3:32b}"

# A per-job port, so two jobs that land on the same node don't collide.
PORT=$((20000 + SLURM_JOB_ID % 20000))
export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1
export ATTE_OLLAMA_HOST="http://$OLLAMA_HOST"

RUN_DIR="runs/replay0-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" logs "$OLLAMA_MODELS"

apptainer exec --nv --bind "$OLLAMA_MODELS" --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  "$OLLAMA_SIF" ollama serve > "$RUN_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

for _ in $(seq 60); do
  curl -sf "$ATTE_OLLAMA_HOST/api/version" > /dev/null && break
  sleep 2
done
curl -sf "$ATTE_OLLAMA_HOST/api/version" || { echo "[atte] Ollama did not start"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "${ATTE_PULL:-0}" == "1" ]]; then
  apptainer exec --env OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_SIF" ollama pull "$ATTE_MODEL"
fi
uv run agentic-tte llm-check || { echo "[atte] model $ATTE_MODEL missing; see hpc/README.md"; exit 1; }

failed=0
for t in $TRIALS; do
  echo "[atte] $(date +%T) drafting $t"
  if ! uv run agentic-tte draft "$t" -o "$RUN_DIR/$t.protocol.json" 2> "$RUN_DIR/$t.log"; then
    echo "[atte] $t failed, see $RUN_DIR/$t.log"
    failed=$((failed + 1))
  fi
  tail -n 2 "$RUN_DIR/$t.log"
done
echo "[atte] done: $RUN_DIR ($failed failed)"
