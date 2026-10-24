# Hellbender facts (confirmed 2026-10-24)

The source of truth for paths, quotas, Slurm settings and known pitfalls on MU's Hellbender cluster. Everything here was checked on the cluster, not assumed. Update this file whenever something on the cluster changes, and keep `hpc/env.example.sh` in line with it.

## Locations

| What | Path | Notes |
|---|---|---|
| Repo checkout | `~/data/Agentic-TTE` | Cloned from GitHub. Never edit code here; see "Sync rules" |
| Per-project caches | `~/data/Agentic-TTE/.hpc-data/` (`$ATTE_PROJECT_DIR`) | CT.gov cache, LLM cache, uv cache. Git-ignored |
| Ollama models | `/cluster/pixstor/data/$USER/tte-project/ollama-models` (`$OLLAMA_MODELS`) | Shared with older projects. Do not move |
| Ollama container | `/cluster/pixstor/data/$USER/tte-project/containers/ollama-0.40.3.sif` (`$OLLAMA_SIF`) | 3.2 GB |
| Run outputs | `~/data/Agentic-TTE/runs/` | Git-ignored. `runs/replay0-<jobid>/`, `runs/interactive/` |
| Slurm logs | `~/data/Agentic-TTE/logs/` | Git-ignored |
| Apptainer build scratch | `/local/scratch/$USER/apptainer-{tmp,cache}` | Node-local, temporary |

## Storage and quotas

| Area | Path | Quota | Use |
|---|---|---|---|
| Home | `/home/$USER` (on `/cluster/pixstor`) | 50 GB | Small config only |
| Personal Pixstor | `~/data` → `/mnt/pixstor/data/$USER` | 500 GB, **nearly full** | Repo, caches, outputs |
| Older project area | `/cluster/pixstor/data/$USER/tte-project` | Separate filesystem from `~/data` | Models, container |
| Local scratch | `/local/scratch` | 1.6–3.2 TB per node | Temporary build and job files; not shared between nodes |

- `df` reports whole-filesystem totals, not your quota, so it cannot tell you whether you are near your limit.
- Nothing is backed up. Code lives on GitHub; models can be re-pulled; results must be copied off the cluster if they matter.
- Cluster storage is approved only for lower data classification levels. DCL3/DCL4 data (identifiable EHR) needs an arrangement with RCSS before it comes to Hellbender. Phase 0 uses only public ClinicalTrials.gov text.

## Slurm

- Account `general`, QOS `normal`.
- GRES type names are **case-sensitive**: `gpu:H100:1` works; `gpu:h100:1` fails with "Requested node configuration is not available".
- Never request MIG slices (`1g.10gb`, `2g.20gb`); they are too small for the models.

| Partition | GPUs | Limits | When to use |
|---|---|---|---|
| `gpu` | 3 nodes × 2 H100 (64 CPU, ~490 GB RAM) | 2 days, not preemptible | Final benchmark runs. Queue waits can be many hours |
| `gpu_requeue` | H100 (×8, ×4, ×2 nodes), A100 ×4 on 15 nodes, H200, L40S | 2 days, **preemptible** | Development and fast turnaround; `--gres=gpu:A100:1` usually starts quickly |

Jobs request 8 CPUs and 64 GB RAM, which fits every GPU node. The model lives in GPU memory, not system RAM.

**GPU consistency:** results can differ slightly between GPU types (floating point). Every run writes the GPU name to `manifest.txt`. Run each final benchmark for all models on one GPU type, and report it in the methods.

## Models present in `$OLLAMA_MODELS`

`qwen3:32b` (20 GB, project default), `gpt-oss:20b` (13 GB), `alibayram/medgemma:27b` (16 GB), `qwen3:8b`, `mistral-small:24b`, `deepseek-r1:14b`, `phi4:14b`, `gemma3:12b`, `llama3.1:8b`, `qwen2.5:7b-instruct`, `mistral:7b`, `meditron:7b`, `cniongolo/biomistral`, `llama3.2:3b`, `nomic-embed-text`. All fit on a single A100, H100 or H200.

`ATTE_THINK=false` for qwen3; `ATTE_THINK=none` for models without a reasoning switch (gpt-oss, medgemma, the rest).

## Known pitfalls and their fixes

| Symptom | Cause | Fix (already in the scripts) |
|---|---|---|
| `listen tcp 0.0.0.0:11434: bind: address already in use` | The `ollama/ollama` image sets `OLLAMA_HOST=0.0.0.0:11434`, which overrides the host value inside Apptainer | Pass every `OLLAMA_*` setting with `--env` on `apptainer exec` |
| Port 11434 busy on OnDemand nodes | A personal Ollama 0.13.5 often runs there | Project uses port 11500 interactively and a per-job port in batch |
| `apptainer pull` → "Disk quota exceeded" | Build files and cache went to `~/data` | `APPTAINER_TMPDIR`/`APPTAINER_CACHEDIR` on `/local/scratch`, and `apptainer pull --disable-cache` |
| `WARNING: Error changing the container working directory` | `~/data` is a symlink to `/mnt/pixstor`, which is not bound into the container | Harmless. Never pass paths under `~/data` into the container |
| `Requested node configuration is not available` | Lowercase GRES (`h100`) | Use `gpu:H100:1` |
| Clone/push "Write access to repository not granted" (403) | Fine-grained token without access to this repo | Token with Agentic-TTE → Contents: Read and write |

## Environment notes

- `uv` is installed at `~/.local/bin`. The conda `(base)` environment is active by default and is ignored by `uv run`.
- `TMPDIR` is set to `~/data/tmp` in the user's shell profile.
- The OnDemand node g020 has outbound internet (Docker Hub and ClinicalTrials.gov both worked). Batch compute nodes are not yet confirmed, so trial records are pre-fetched into the cache.
- The OnDemand Jupyter session runs on `gpu_requeue` and can be preempted.

## Sync rules

1. Code changes are made on the PC, committed and pushed to GitHub. On Hellbender, only `git pull`.
2. `hpc/env.sh` is per-user and git-ignored. When `hpc/env.example.sh` changes, copy the new lines into `env.sh` by hand.
3. If something has to be fixed on Hellbender first, record it here and in the scripts on the PC, so the next `git pull` matches what works.
