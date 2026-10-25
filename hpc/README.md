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
uv run pytest                                     # all must pass; no GPU needed

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

## 6. Phase 0 exit run

Design and scoring are fixed in [eval/SCORING.md](../eval/SCORING.md):
- 10 trials × 3 models (`qwen3:32b` with `ATTE_THINK=false`; `gpt-oss:20b` and `alibayram/medgemma:27b` with `none`) × 2 replicates;
- trialist-v2 at one commit; one GPU type for all six runs;
- a fresh LLM cache per model and replicate.

Disk use is under 50 MB in `~/data`: protocols, logs, manifests and six small SQLite caches. No new models are pulled.

Both options below write the same layout, which the checker and the scorer read:

```
runs/exit-<GPU>/<model>/rep<k>/   manifest.txt, <NCT>.protocol.json, <NCT>.log
.hpc-data/exit/<GPU>/<model>/rep<k>/   that replicate's fresh LLM cache and a CT.gov copy
```

Before either option, the checkout must be clean and on the latest commit, and the 10 records must be in the CT.gov cache (section 3). The scripts check both and refuse to start otherwise.

```bash
git pull && git status --short          # must print nothing
uv run pytest -q
```

**Option A: batch jobs (6 jobs, one per model and replicate)**

```bash
bash hpc/exit_run_submit.sh gpu_requeue gpu:A100:1    # usually starts quickly; preemptible
bash hpc/exit_run_submit.sh gpu gpu:H100:1            # not preemptible; the queue can take hours
squeue -u $USER -o "%.10i %.9P %.8T %.20R %.20S"
cat runs/exit-A100.jobs                                # model, think, replicate, job id, run dir
```

Run only one of the two lines: the label (`A100`, `H100`) comes from the GRES. A preempted job requeues under the same ID and resumes from its own cache. Its manifest then shows `restart_count` > 0, and cache hits are expected for that replicate only.

**Option B: in your OnDemand GPU session (sequential, no queue)**

```bash
tmux new -s atte                      # if available, so closing the browser tab doesn't stop the run
bash hpc/exit_run_interactive.sh      # all 3 models x 2 replicates on this session's GPU
```

The GPU label comes from `nvidia-smi`. g020 is an H100, so the output goes to `runs/exit-H100/`. The OnDemand session itself runs on `gpu_requeue` and can be preempted. If that happens, delete the unfinished replicate's run directory and its cache, then start the script again. It refuses to reuse a cache that already holds answers.

**After either option: check the run before scoring**

```bash
uv run python eval/check_exit_run.py runs/exit-A100    # or runs/exit-H100
```

It prints one row per run (protocols, calls, cache hits, tokens, failures). It ends with `OK: ready to score`, or with each problem it found: mixed GPU types, commits or Ollama versions; a dirty commit; missing trials; non-v2 protocols; cache hits without a restart.

To bring the results to the PC, archive the run directory (under 5 MB) and copy it the same way as the earlier protocol files:

```bash
tar czf runs/exit-A100.tgz runs/exit-A100 runs/exit-A100.jobs    # after Option A on A100
```
```bash
tar czf runs/exit-H100.tgz runs/exit-H100                        # after Option B on H100
```

### Reproducibility settings (apply to every scored run)

Test 1 (2026-10-24) found the cause of the exit run's replicate disagreement: server state.
- With Ollama restarted before each replicate, replicates were identical.
- The parallelism setting itself changes outputs.

The scripts therefore default to `OLLAMA_NUM_PARALLEL=1`, restart Ollama per replicate, and unload the model before every trial. Each `manifest.txt` records `ollama_num_parallel`, `ollama_flash_attention` and `unload_per_trial`. Results: `eval/results/repro/`.

### Test 2: order independence (run before the final benchmark)

The same 3 trials are drafted forward and in reverse, for qwen3:32b and medgemma:27b, one replicate each. It takes about 1.2 hours on the H100 and under 10 MB of disk.

```bash
tmux new -s atte
```
```bash
cd ~/data/Agentic-TTE && git pull && source hpc/env.sh && bash hpc/repro_test.sh
```
```bash
uv run python eval/compare_runs.py runs/repro-H100/np1_fwd runs/repro-H100/np1_rev
```

Send me the `compare_runs.py` output. "6 identical" means order doesn't matter, and the final run can start.

### Final benchmark run

Pre-registered in `eval/SCORING.md`, amendments 9 and 10:
- 10 trials × 4 arms × 2 replicates on H100: qwen3:32b, medgemma:27b and gpt-oss:20b as pre-registered, plus gpt-oss:20b `low` (exploratory).
- One commit, with the settings above.
- Disk use: under 50 MB. Time: about 10–12 GPU-hours, mostly gpt-oss.

**Recommended:** batch jobs on the non-preemptible `gpu` partition. That's 8 independent jobs, each within the 8-hour job limit:

```bash
ATTE_EXIT_TAG=final ATTE_EXIT_MODELS="qwen3:32b|false alibayram/medgemma:27b|none gpt-oss:20b|none gpt-oss:20b|low" bash hpc/exit_run_submit.sh gpu gpu:H100:1
```

**Alternative:** your OnDemand H100 session, in tmux, if it has at least 12 hours left:

```bash
ATTE_EXIT_TAG=final ATTE_EXIT_MODELS="qwen3:32b|false alibayram/medgemma:27b|none gpt-oss:20b|none gpt-oss:20b|low" bash hpc/exit_run_interactive.sh
```

Then check the run and archive it:

```bash
uv run python eval/check_exit_run.py runs/exit-H100-final
```
```bash
tar czf runs/exit-H100-final.tgz runs/exit-H100-final runs/exit-H100-final.jobs 2>/dev/null; ls -la runs/exit-H100-final.tgz
```

### New arms (tagged)

A new arm on a GPU type that already has a run needs `ATTE_EXIT_TAG`, so it never overwrites an earlier run. Arms of one model with different reasoning levels get separate directories (`gpt-oss_20b`, `gpt-oss_20b-low`).

## 7. Long-running service (needed from Phase 4)

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
