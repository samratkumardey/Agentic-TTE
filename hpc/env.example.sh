# Copy to hpc/env.sh (git-ignored) and source it in every new shell:  source hpc/env.sh
# Values below are the confirmed Hellbender settings (2026-10-24); see hpc/HELLBENDER.md.

# --- Storage -------------------------------------------------------------------------------
# Repo checkout:      ~/data/Agentic-TTE        (~/data -> /mnt/pixstor/data/$USER, 500 GB quota)
# Shared model store: /cluster/pixstor/data/$USER/tte-project   (the SAME directory tree and
#                     quota as ~/data: /cluster/pixstor/data/$USER is ~/data; 2026-10-26)
# Never put large files in $HOME (50 GB quota). Nothing on Hellbender is backed up.
export ATTE_PROJECT_DIR="$HOME/data/Agentic-TTE/.hpc-data"      # caches; git-ignored
export ATTE_SHARED_DIR="/cluster/pixstor/data/$USER/tte-project" # models + container
export OLLAMA_MODELS="$ATTE_SHARED_DIR/ollama-models"
export OLLAMA_SIF="$ATTE_SHARED_DIR/containers/ollama-0.40.3.sif"
export ATTE_ENDPOINT_FILE="$HOME/.agentic_tte/ollama_endpoint"   # where workers read host:port

# Shared between the login/OnDemand node (CT.gov fetches) and compute nodes (LLM cache).
export ATTE_CACHE_DIR="$ATTE_PROJECT_DIR/cache"
export UV_CACHE_DIR="$ATTE_PROJECT_DIR/uv-cache"

# Apptainer build files go to node-local scratch: building into ~/data hit
# "Disk quota exceeded". Pull with:  apptainer pull --disable-cache "$OLLAMA_SIF" docker://...
export APPTAINER_TMPDIR="/local/scratch/$USER/apptainer-tmp"
export APPTAINER_CACHEDIR="/local/scratch/$USER/apptainer-cache"

# --- Slurm ---------------------------------------------------------------------------------
# GRES type names are case-sensitive: gpu:H100:1 works, gpu:h100:1 is rejected
# ("Requested node configuration is not available"). Never request MIG slices (1g.10gb, 2g.20gb).
export SLURM_ACCOUNT="general"
export SLURM_GPU_PARTITION="gpu"            # 3 nodes x 2 H100, not preemptible, long queue
export SLURM_GPU_GRES="gpu:H100:1"
export SLURM_CPU_PARTITION="general"        # CPU jobs (MIMIC ETL): 56 nodes, not preemptible
# Fallback with short waits (preemptible):  -p gpu_requeue --gres=gpu:A100:1  (or gpu:H100:1)

# --- Ollama --------------------------------------------------------------------------------
# 127.0.0.1 = only processes on the same node. Ollama has no authentication; do not bind
# 0.0.0.0 without agreeing network policy with RCSS.
export OLLAMA_BIND="127.0.0.1"
export OLLAMA_PORT="11500"   # not 11434: a personal Ollama 0.13.5 often already runs there

# Models already present in $OLLAMA_MODELS (pull nothing else without checking quota).
export ATTE_MODELS="qwen3:32b gpt-oss:20b alibayram/medgemma:27b"
export ATTE_MODEL="${ATTE_MODEL:-qwen3:32b}"
# Qwen3 reasons by default, which multiplies tokens per call: "false" for Replay, "true" to
# compare. Use "none" (model default) for models without an on/off switch: gpt-oss, medgemma.
export ATTE_THINK="${ATTE_THINK:-false}"

# --- MIMIC-IV and OMOP (Phase 3; hpc/HELLBENDER.md "Data already on Hellbender") ---------
# All inside the ~/data quota. The vocabulary folder name has a space: keep the quotes.
export ATTE_MIMIC_SOURCE="/cluster/pixstor/data/$USER/mimic-source/mimic-iv-3.1"
export ATTE_VOCAB="/cluster/pixstor/data/$USER/tte-project/athena/S07/Athena Verison"
export ATTE_MIMIC_CONCEPT_MAP="/cluster/pixstor/data/$USER/mimic-code/mimic-iv/concepts/concept_map"
export ATTE_OMOP_DIR="/cluster/pixstor/data/$USER/tte-project/omop"
export ATTE_ITEM_MAPS="${ATTE_ITEM_MAPS:-}"   # reviewed item maps (space-separated paths)
