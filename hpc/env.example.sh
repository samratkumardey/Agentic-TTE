# Copy to hpc/env.sh (git-ignored) and fill in. Sourced by the Slurm scripts.
# Values marked CONFIRM need to be checked on Hellbender (see hpc/README.md, step 2).

export ATTE_PROJECT_DIR="/path/to/project/agentic-tte"          # CONFIRM: project storage, not $HOME
export OLLAMA_MODELS="$ATTE_PROJECT_DIR/ollama-models"            # model weights (tens of GB)
export OLLAMA_SIF="$ATTE_PROJECT_DIR/containers/ollama-0.40.3.sif"
export ATTE_ENDPOINT_FILE="$HOME/.agentic_tte/ollama_endpoint"    # where workers read host:port

# Shared between the login node (CT.gov fetches) and compute nodes (drafts, LLM cache).
export ATTE_CACHE_DIR="$ATTE_PROJECT_DIR/cache"
export UV_CACHE_DIR="$ATTE_PROJECT_DIR/uv-cache"                  # keeps $HOME quota free
export APPTAINER_CACHEDIR="$ATTE_PROJECT_DIR/apptainer-cache"

export SLURM_GPU_PARTITION="gpu"                                  # CONFIRM
export SLURM_ACCOUNT=""                                           # CONFIRM, if required
export SLURM_GPU_GRES="gpu:h100:1"                                # CONFIRM gres name

# Bind address for the long-running service (ollama_service.sbatch). 127.0.0.1 = only processes
# on the same node (safest). Ollama has no authentication, so binding 0.0.0.0 exposes it to
# every cluster user; only do that after agreeing network policy with RCSS.
export OLLAMA_BIND="127.0.0.1"
export OLLAMA_PORT="11434"

# Models to pull (see PLAN.md §5). One large model stays resident.
export ATTE_MODELS="qwen3:32b gpt-oss:120b medgemma:27b"
export ATTE_MODEL="${ATTE_MODEL:-qwen3:32b}"
# Qwen3 reasons by default, which multiplies tokens per call: "false" for Replay, "true" to
# compare. Use "none" (model default) for models without an on/off switch, e.g. gpt-oss.
export ATTE_THINK="${ATTE_THINK:-false}"
