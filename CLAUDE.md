# Agentic-TTE: notes for Claude Code

Read [PLAN.md](PLAN.md) for the architecture and roadmap, and [hpc/HELLBENDER.md](hpc/HELLBENDER.md) before changing anything that runs on the cluster.

## Where things run

- **This PC** is where code is written, tested (`uv run pytest`), committed and pushed.
- **Hellbender** (MU HPC) is where models run on GPUs. It only ever `git pull`s. The checkout is at `~/data/Agentic-TTE`.
- The user runs Hellbender commands themselves and pastes the output back. Give commands they can copy as they are, run from `~/data/Agentic-TTE` after `source hpc/env.sh`.

## Rules for cluster changes

1. `hpc/HELLBENDER.md` is the source of truth for paths, quotas, partitions, GRES names and pitfalls. Do not change a path, port, partition or resource request without checking it, and update that file in the same commit when a cluster fact changes.
2. `hpc/env.example.sh` must always hold working Hellbender values. `hpc/env.sh` is the user's git-ignored copy; when the example changes, tell the user which lines to copy over.
3. Every `apptainer exec ... ollama` call passes `OLLAMA_*` settings with `--env` (the image overrides `OLLAMA_HOST` otherwise).
4. Never default to port 11434 on Hellbender; a personal Ollama uses it.
5. GRES names are case-sensitive (`gpu:H100:1`, `gpu:A100:1`). Never request MIG slices.
6. Large files go under `~/data` or `/cluster/pixstor/data/$USER/tte-project`, never `$HOME` (50 GB). `~/data` is nearly full: say how much space a new step needs before it runs.
7. Every GPU run writes `manifest.txt` (job, node, GPU, git commit, Ollama version, model). Keep this for new run scripts too.
8. After changing anything under `hpc/`, run `bash -n` on the scripts and list the exact Hellbender commands the user should run next.

## Data rules

- No patient-level data in git (MIMIC-IV DUA, HIPAA). The pre-commit hook enforces this; do not bypass it.
- Before any phase brings MIMIC-IV or EHR data to Hellbender, remind the user to confirm the data classification level with RCSS.

## Current state (2026-10-24)

Phase 0 runs end to end on Hellbender: the Trialist drafts protocols with `qwen3:32b` in an OnDemand GPU session via `hpc/interactive_ollama.sh`. The code is now at trialist-v2 / protocol schema v2 (see PLAN.md §12); the only Hellbender output so far (NCT04134403) is trialist-v1. Next: the Phase 0 exit run (`hpc/exit_run_submit.sh` or `hpc/exit_run_interactive.sh`, see hpc/README.md §6). It covers 10 trials × `qwen3:32b`, `gpt-oss:20b` and `alibayram/medgemma:27b` × 2 replicates on one GPU type, each replicate with a fresh LLM cache. It is checked by `eval/check_exit_run.py` and scored as pre-registered in `eval/SCORING.md`. Reference facts for the 10 trials are in `eval/reference/`. Then Phase 1.
