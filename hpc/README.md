# Running on Hellbender

Phase 0 runs the Trialist: it drafts target-trial protocols for registered trials with an open-weight model on an H100. There are two ways to run it:

- **One batch job** (`replay_phase0.sbatch`): starts Ollama, drafts a list of trials, then exits. Easiest; start here.
- **Long-running service** (`ollama_service.sbatch`): keeps Ollama up for interactive work and, from Phase 4, the web app.

## 1. Get the code onto Hellbender

From a Git remote (recommended):

```bash
ssh <pawprint>@<hellbender login host>
cd /path/to/project            # project storage, not $HOME
git clone <your repo url> agentic-tte && cd agentic-tte
```

Or copy it from your PC with `scp -r` / `rsync`, leaving out `.venv/` and `.cache/`.

## 2. Find the site-specific values

```bash
sinfo -o "%P %G %l %D" | grep -i -E "gpu|h100"        # partition, gres name, time limit
sacctmgr -n show assoc user=$USER format=account,qos  # account / QOS, if required
module avail apptainer 2>&1 | head                    # whether apptainer needs a module load
```

Then:

```bash
cp hpc/env.example.sh hpc/env.sh   # fill in the CONFIRM values from above
source hpc/env.sh
mkdir -p "$ATTE_PROJECT_DIR"/{containers,ollama-models,cache} logs
```

## 3. Python environment and container (login node)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv to ~/.local/bin, once
uv sync --extra dev
uv run pytest                                     # 47 tests, no GPU needed

apptainer pull "$OLLAMA_SIF" docker://ollama/ollama:0.40.3
```

`apptainer pull` needs no root. Use `hpc/ollama.def` only if you need to customise the image (`apptainer build --fakeroot`).

## 4. Pre-fetch trial records (login node)

Compute nodes may not have outbound internet. Fetching on the login node fills `$ATTE_CACHE_DIR`, and the job reads from it:

```bash
for t in NCT00475852 NCT02856698 NCT03872011 NCT04134403 NCT06091982 \
         NCT03258684 NCT03592693 NCT01560650 NCT01223690 NCT00953303; do
  uv run agentic-tte fetch "$t" > /dev/null && echo "ok $t"
done
```

## 5. Get the model weights

If compute nodes can reach the internet, let the first job pull the model (`ATTE_PULL=1`, below). Otherwise, pull once from an interactive GPU session on a node that can, or ask RCSS which nodes allow outbound HTTPS. The weights land in `$OLLAMA_MODELS` and are reused by every later job (`qwen3:32b` is ~20 GB).

## 6. Run the Replay batch

```bash
sbatch -p "$SLURM_GPU_PARTITION" --gres="$SLURM_GPU_GRES" ${SLURM_ACCOUNT:+-A $SLURM_ACCOUNT} \
       --export=ALL,ATTE_PULL=1 hpc/replay_phase0.sbatch
squeue -u $USER
tail -f logs/replay0-<jobid>.out
```

The output goes to `runs/replay0-<jobid>/`, with one `<NCT>.protocol.json` and one `<NCT>.log` per trial, plus `ollama.log`. Each log ends with the LLM call and token counts and any incomplete TTE elements. To use a different model or trial list:

```bash
sbatch ... --export=ALL,ATTE_MODEL=gpt-oss:120b,ATTE_THINK=none,ATTE_TRIALS="NCT04134403 NCT00475852" hpc/replay_phase0.sbatch
```

`ATTE_THINK=none` leaves reasoning at the model's default; gpt-oss has no on/off switch.

Re-running is cheap: identical requests come from the LLM cache in `$ATTE_CACHE_DIR`, which is keyed by the model digest. Pass `--no-cache` to `agentic-tte draft` to force fresh calls.

## 7. Long-running service (optional now, needed in Phase 4)

```bash
sbatch -p "$SLURM_GPU_PARTITION" --gres="$SLURM_GPU_GRES" hpc/ollama_service.sbatch
cat "$ATTE_ENDPOINT_FILE"                          # "<url> <jobid> <node>"
srun --jobid=<jobid> --overlap --pty bash          # a shell on the GPU node
bash hpc/pull_models.sh                            # pull $ATTE_MODELS
export ATTE_OLLAMA_HOST="$(cut -d' ' -f1 "$ATTE_ENDPOINT_FILE")"
uv run agentic-tte llm-check
uv run agentic-tte draft NCT04134403 -o runs/NCT04134403/protocol.json
```

The service job requeues itself about 5 minutes before its wall-time limit. With the default `OLLAMA_BIND=127.0.0.1`, clients must run on the same node. To let a separate service tier reach the GPU node, set `OLLAMA_BIND=0.0.0.0`. Ollama has no authentication, so agree on this with RCSS first; it exposes the model to the whole cluster network. By design the agents only send protocol text and schemas, never patient rows.

## Still to confirm with RCSS

- Outbound internet from compute nodes (CT.gov, model pulls)
- Maximum wall time for service jobs, and whether the service tier (FastAPI, Postgres) runs as a Slurm job or a dedicated VM behind Open OnDemand
- Project storage path and quota (~150 GB for the three default models)
