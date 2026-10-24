# Running on Hellbender

Cluster facts, paths, quotas and known pitfalls are in [HELLBENDER.md](HELLBENDER.md). This page is the procedure.

There are three ways to run the Trialist on a GPU:

- **Interactive** (`interactive_ollama.sh`): in an existing GPU session (Open OnDemand). No queue; best for testing.
- **One batch job** (`replay_phase0.sbatch`): starts Ollama, drafts a list of trials, exits. Use for benchmark runs.
- **Long-running service** (`ollama_service.sbatch`): keeps Ollama up for interactive clients and, from Phase 4, the web app.

## 1. One-time setup

```bash
cd ~/data
git clone https://github.com/samratkumardey/Agentic-TTE.git && cd Agentic-TTE
cp hpc/env.example.sh hpc/env.sh       # values are the confirmed Hellbender settings
source hpc/env.sh
mkdir -p "$ATTE_PROJECT_DIR/cache" logs "$(dirname "$OLLAMA_SIF")" "$APPTAINER_TMPDIR"

curl -LsSf https://astral.sh/uv/install.sh | sh   # once; installs uv to ~/.local/bin
uv sync --extra dev
uv run pytest                                     # 47 tests, no GPU needed

# Only if the container is missing:
[ -f "$OLLAMA_SIF" ] || apptainer pull --disable-cache "$OLLAMA_SIF" docker://ollama/ollama:0.40.3
```

## 2. Every new shell

```bash
cd ~/data/Agentic-TTE
git pull
source hpc/env.sh
```

If `git pull` changed `hpc/env.example.sh`, copy the new lines into `hpc/env.sh`.

## 3. Pre-fetch trial records

```bash
for t in NCT00475852 NCT02856698 NCT03872011 NCT04134403 NCT06091982 \
         NCT03258684 NCT03592693 NCT01560650 NCT01223690 NCT00953303; do
  uv run agentic-tte fetch "$t" > /dev/null && echo "ok $t"
done
```

## 4. Interactive run (no queue)

In a session that has a GPU:

```bash
source hpc/interactive_ollama.sh          # starts Ollama on port 11500 and runs llm-check
uv run agentic-tte draft NCT04134403 -o runs/interactive/NCT04134403.protocol.json
source hpc/interactive_ollama.sh stop     # free the GPU when done
```

## 5. Batch run

```bash
# Final benchmark runs (one GPU type for every model):
sbatch -p "$SLURM_GPU_PARTITION" --gres="$SLURM_GPU_GRES" -A "$SLURM_ACCOUNT" hpc/replay_phase0.sbatch

# Faster start for development (preemptible):
sbatch -p gpu_requeue --gres=gpu:A100:1 -A "$SLURM_ACCOUNT" hpc/replay_phase0.sbatch

# Other model or trial list:
sbatch ... --export=ALL,ATTE_MODEL=gpt-oss:20b,ATTE_THINK=none,ATTE_TRIALS="NCT04134403" hpc/replay_phase0.sbatch

squeue -u $USER -o "%.10i %.9P %.8T %.20R %.20S"   # state, reason, estimated start
tail -f logs/replay0-<jobid>.out
```

Output goes to `runs/replay0-<jobid>/`: `manifest.txt` (job, node, GPU, git commit, Ollama version, model), one `<NCT>.protocol.json` and `<NCT>.log` per trial, and `ollama.log`. Each trial log ends with the LLM call and token counts and any incomplete TTE elements.

Re-running is cheap: identical requests come from the LLM cache in `$ATTE_CACHE_DIR`, keyed by model digest. Pass `--no-cache` to `agentic-tte draft` to force fresh calls.

## 6. Long-running service (needed from Phase 4)

```bash
sbatch -p "$SLURM_GPU_PARTITION" --gres="$SLURM_GPU_GRES" -A "$SLURM_ACCOUNT" hpc/ollama_service.sbatch
cat "$ATTE_ENDPOINT_FILE"                          # "<url> <jobid> <node>"
srun --jobid=<jobid> --overlap --pty bash          # a shell on the GPU node
export ATTE_OLLAMA_HOST="$(cut -d' ' -f1 "$ATTE_ENDPOINT_FILE")"
uv run agentic-tte llm-check
```

The service job requeues itself about 5 minutes before its wall-time limit. With `OLLAMA_BIND=127.0.0.1`, clients must run on the same node. Ollama has no authentication; binding `0.0.0.0` exposes the model to the whole cluster network, so agree on it with RCSS first.

## Still to confirm with RCSS

- Outbound internet from batch compute nodes (the OnDemand node has it)
- Whether the service tier (FastAPI, Postgres) runs as a Slurm job or a dedicated VM behind Open OnDemand
- Approved storage for MIMIC-IV / EHR data (data classification level)
