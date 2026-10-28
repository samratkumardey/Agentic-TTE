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
- **MIMIC-IV 3.1 is already on Hellbender.** On 2026-10-26 the user confirmed that the data use agreement covers it and that RCSS has no issue with it (locations: hpc/HELLBENDER.md). No need to ask again for MIMIC-IV. Oracle or other EHR data still needs its own confirmation.
- MIMIC-IV, its Postgres build, the OMOP CDM and the vocabulary move to the user's mode-700 folder in the lab storage (`$ATTE_LAB_DIR`). The user confirmed the lab's PhysioNet licence on 2026-10-26. Keep that folder private, and never widen its permissions on the user's behalf.

## Current state (2026-10-24)

The Phase 0 exit criterion is met. The exit run (commit 586e128, H100, interactive) is in `runs/exit-H100/` on the PC (not in git); results are in `eval/results/phase0_exit-H100.md`.
- qwen3:32b and medgemma:27b: 20/20 protocols each.
- gpt-oss:20b: 11/20; the rest timed out.
- Replicates differ at temperature 0.

Accuracy endpoints (`eval/SCORING.md` parts B and C) need the human gold annotation (`eval/gold/`), which does not exist yet. Never report an accuracy number without it.

Reproducibility (eval/SCORING.md amendments 7-10, hpc/HELLBENDER.md):
- Replicates are identical once Ollama restarts per replicate. The exit run's disagreement came from server state.
- Every scored run uses OLLAMA_NUM_PARALLEL=1, restarts Ollama per replicate and unloads the model before each trial (the script defaults). Never score a run without these settings in its manifest.
- The 586e128 exit run is the pilot, not scored for accuracy.

Order independence was confirmed on 2026-10-25 (forward vs reverse, 6/6 identical). Hellbender writes outputs only under `runs/` (HELLBENDER.md sync rule 4).

**Final benchmark run done** (2026-10-25, commit 9e09bdf, `gpu` partition, H100 NVL; `runs/exit-H100-final/` on the PC, not in git; `eval/results/phase0_exit-H100-final.md`). Arms: qwen3:32b, medgemma:27b, gpt-oss:20b default, gpt-oss:20b `low`.
- 80/80 protocols, with 0 failures.
- Replicates identical in every arm.

This is the run that accuracy (parts B and C) will be scored on, once the gold annotation exists. No further GPU runs are needed for Phase 0.

Phase 2 (PLAN.md §15) has four parts built:
- the Data Support Gate (`agentic_tte/omop/gate.py`);
- OMOP concept mapping with clinician review (`agentic_tte/omop/vocab.py`, `concept_sets.py`);
- the criteria-to-SQL compiler (`agentic_tte/omop/compiler.py`);
- attrition and feasibility (`agentic_tte/omop/attrition.py`), the gate's last check.

Rules for that code:
- Concept IDs come only from vocabulary tables, and only reviewer-approved sets are used.
- The compiler never looks past time zero for eligibility.
- Criteria it cannot compile are reported, never dropped silently.
- Its frame must keep reproducing `synth.omop.ANALYSIS_SQL`.
- Counts that leave the data side go through `suppress` (`AttritionReport.public()`).

The end-to-end Replay is a LangGraph graph (`agentic_tte/graph/replay.py`). Its clinician reviews (design, concepts, amber gate, sign-off) are `interrupt()`s with a SQLite checkpointer.
- The Protocol Auditor (`agents/auditor.py`) runs between compile and the gate. A failure goes back to design review, where the reviewer fixes the design or overrides it with a reason.
- Never let a run reach the gate with an unexplained audit failure.

Rules for the graph:
- Keep the state and interrupt payloads plain JSON (`model_dump(mode="json")`).
- Never put the database connection or patient rows in the state.

The Phase 2 exit criterion is met on synthetic data (2026-10-26; `eval/results/phase2_replay.md`). Three trials were replayed end to end from the real qwen3:32b Phase 0 drafts, and every primary CI covers the planted marginal truth. The reviews were scripted, not clinical: never describe them as clinician review.

Reports now need sign-off (status `signed_off` or `returned`). Batch benchmarks use `AutoApprove`: their reports are labelled auto-approved, with status `auto_approved`. Never present an auto-approved run as reviewed or signed off.

Concept search expands abbreviations only from the protocol's own text and from reviewer-approved list entries. The shipped seed list (`omop/resources/abbreviations.json`) is unapproved: never mark its entries approved on a person's behalf.

**MIMIC-IV 3.1 is in OMOP CDM 5.4** (2026-10-27, job 19365483; `$ATTE_OMOP_DIR/mimiciv-3.1-cdm54-19365483.duckdb`, results in PLAN.md §16).
- Labs start from the OHDSI MIMIC-IV lab-to-LOINC map (`--ohdsi-mimic`, `ATTE_OHDSI_MIMIC`).
- All four item reviews were done by the user (reviewer Samrat Kumar Dey). The reviewed maps live in `$ATTE_OMOP_DIR/review/`.
- Confirm any hand-chosen concept ID by name against the vocabulary before a run: a pasted ID once lost its first digit.

Next (user, 2026-10-27): finish the Replay on MIMIC first, at parity with EmulatRx and then beyond it (PLAN.md §17).
- Pre-register the scoring and the trial-selection rule before any MIMIC estimate is seen.
- Then CREST-2 on the user's Oracle Health EHR (OMOP), building on their Protocol2TTE paper.
- Only 2 of EmulatRx's 10 MIMIC trials have a numeric published estimate (`eval/reference/replay_trials.json`). Never claim RCT agreement for a trial without a published result.

Phase 1 core is done (PLAN.md §13): the synthetic OMOP generator (`agentic_tte/synth/`) and the statistics port (`agentic_tte/stats/`), recovery-tested against potential-outcome truth.
- Truth for any estimator is the marginal effect in the population it estimates (`Cohort.truth_in`), never the planted conditional HR.
- Open: a Monte Carlo over seeds and a doubly robust estimator.
- Done: bootstrap CIs for RR, OR and RD (`stats/bootstrap.py`, R4). Scored runs use 500 resamples (the CLI default).
- This PC's `.venv` sits in OneDrive, which can lock files during `uv add`. If it does, rerun once the lock clears; Hellbender is unaffected.
