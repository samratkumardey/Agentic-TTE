#!/bin/bash
# Start the project's Ollama in an existing GPU session (Open OnDemand Jupyter/desktop, or
# `srun --pty`), for real-time testing without waiting in the batch queue. Source it from the
# repository root so the exports stay in your shell:
#
#   source hpc/interactive_ollama.sh          # start (or reuse) the server
#   uv run agentic-tte draft NCT04134403 -o runs/interactive/NCT04134403.protocol.json
#   source hpc/interactive_ollama.sh stop     # free the GPU memory when done
#
# Uses $OLLAMA_PORT from hpc/env.sh (11500), so it never collides with a personal Ollama on
# 11434. Results land in the same LLM cache as batch jobs.

# shellcheck source=/dev/null
source "${ATTE_ENV:-hpc/env.sh}"
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
export ATTE_OLLAMA_HOST="http://$OLLAMA_HOST"
export OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1
_atte_log="runs/interactive/ollama.log"
_atte_pidfile="runs/interactive/ollama.pid"
mkdir -p runs/interactive

if [[ "${1:-}" == "stop" ]]; then
  if [[ -f "$_atte_pidfile" ]] && kill "$(cat "$_atte_pidfile")" 2>/dev/null; then
    echo "[atte] stopped Ollama on $OLLAMA_HOST"
  else
    echo "[atte] no project Ollama running"
  fi
  rm -f "$_atte_pidfile"
  return 0 2>/dev/null || exit 0
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && ! nvidia-smi -L > /dev/null 2>&1; then
  echo "[atte] no GPU in this session; start one with a GPU (see hpc/HELLBENDER.md)"
  return 1 2>/dev/null || exit 1
fi

if curl -sf "$ATTE_OLLAMA_HOST/api/version" > /dev/null; then
  echo "[atte] reusing Ollama at $ATTE_OLLAMA_HOST"
else
  # --env for every OLLAMA_* setting: the image's own OLLAMA_HOST=0.0.0.0:11434 wins otherwise.
  nohup apptainer exec --nv --bind "$OLLAMA_MODELS" \
    --env OLLAMA_MODELS="$OLLAMA_MODELS" --env OLLAMA_HOST="$OLLAMA_HOST" \
    --env OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" --env OLLAMA_MAX_LOADED_MODELS="$OLLAMA_MAX_LOADED_MODELS" \
    --env OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL" --env OLLAMA_FLASH_ATTENTION="$OLLAMA_FLASH_ATTENTION" \
    "$OLLAMA_SIF" ollama serve > "$_atte_log" 2>&1 &
  echo $! > "$_atte_pidfile"
  for _ in $(seq 30); do
    curl -sf "$ATTE_OLLAMA_HOST/api/version" > /dev/null && break
    sleep 2
  done
  if ! curl -sf "$ATTE_OLLAMA_HOST/api/version" > /dev/null; then
    echo "[atte] Ollama did not start; last lines of $_atte_log:"
    tail -n 5 "$_atte_log"
    return 1 2>/dev/null || exit 1
  fi
  echo "[atte] Ollama started at $ATTE_OLLAMA_HOST (log: $_atte_log)"
fi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -n1
uv run agentic-tte llm-check
