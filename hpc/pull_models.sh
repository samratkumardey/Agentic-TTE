#!/bin/bash
# Pull the models in $ATTE_MODELS into $OLLAMA_MODELS. Run on the node where the service job is
# running (srun --jobid=<ollama job> --overlap --pty bash), or anywhere the endpoint is reachable.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ATTE_ENV:-$SCRIPT_DIR/env.sh}"

export OLLAMA_HOST="$(cut -d' ' -f1 "$ATTE_ENDPOINT_FILE")"
for m in $ATTE_MODELS; do
  echo "[atte] pulling $m"
  apptainer exec --env OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_SIF" ollama pull "$m"
done
apptainer exec --env OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_SIF" ollama list
