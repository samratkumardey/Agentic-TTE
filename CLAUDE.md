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

The Phase 0 exit criterion is met. The exit run (commit 586e128, H100, interactive) is in `runs/exit-H100/` on the PC (not in git); results are in `eval/results/phase0_exit-H100.md`.
- qwen3:32b and medgemma:27b: 20/20 protocols each.
- gpt-oss:20b: 11/20; the rest timed out.
- Replicates differ at temperature 0.

Accuracy endpoints (`eval/SCORING.md` parts B and C) need the human gold annotation (`eval/gold/`), which does not exist yet. Never report an accuracy number without it.

Next on the cluster: `hpc/repro_test.sh` (OLLAMA_NUM_PARALLEL 4 vs 1), then the tagged gpt-oss `low` arm.

Phase 1 core is done (PLAN.md §13): the synthetic OMOP generator (`agentic_tte/synth/`) and the statistics port (`agentic_tte/stats/`), recovery-tested against potential-outcome truth.
- Truth for any estimator is the marginal effect in the population it estimates (`Cohort.truth_in`), never the planted conditional HR.
- Open: a Monte Carlo over seeds, bootstrap CIs, a doubly robust estimator.
- This PC's `.venv` sits in OneDrive, which can lock files during `uv add`. If it does, rerun once the lock clears; Hellbender is unaffected.
