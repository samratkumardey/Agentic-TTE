# Agentic-TTE

An open-weight, multi-agent Target Trial Emulation platform. It reads a trial from ClinicalTrials.gov, drafts a computable target-trial protocol, and (in later phases) builds a cohort from EHR data in OMOP format and estimates causal effects. It has three modes: **Replay** (an existing trial), **Rehearse** (a new design) and **Redesign** (tuning a design).

It builds on and extends [EmulatRx](https://github.com/TrialLab/EmulatRx) (Li et al., *Nat. Commun.* 17:5501, 2026; MIT). See [PLAN.md](PLAN.md) for the architecture and roadmap, and [third_party/emulatrx/UPSTREAM.md](third_party/emulatrx/UPSTREAM.md) for what is vendored.

## Status: Phase 0 (foundations)

| Piece | Where |
|---|---|
| `TargetTrialProtocol`: the 7 TTE elements, versioning and diffs | `agentic_tte/protocol/` |
| ClinicalTrials.gov v2 client (cached) and a deterministic section extractor | `agentic_tte/sources/ctgov/` |
| `LLMClient` with Ollama and OpenAI-compatible (vLLM) backends, schema-constrained JSON, retries, SQLite cache | `agentic_tte/llm/` |
| Trialist agent: registry record → draft protocol | `agentic_tte/agents/trialist.py` |
| EmulatRx cache importer and regression fixture | `agentic_tte/protocol/emulatrx_import.py` |
| Hellbender Ollama service (Slurm + Apptainer) | `hpc/` |

## Status: Phase 1 (synthetic data and statistics)

| Piece | Where |
|---|---|
| Synthetic ICU cohorts with planted effects and potential-outcome truth | `agentic_tte/synth/dgp.py` |
| OMOP CDM v5.4 tables in DuckDB, with a SQL round trip | `agentic_tte/synth/omop.py` |
| Propensity scores, IPTW, matching, SMD | `agentic_tte/stats/balance.py` |
| Weighted Cox, KM risks, PH test, subgroups | `agentic_tte/stats/survival.py` |
| Clone-censor-weight for grace-period strategies | `agentic_tte/stats/ccw.py` |
| Schoenfeld sample size, E-values | `agentic_tte/stats/design.py` |
| Recovery table vs planted truth (cf. EmulatRx Table 3) | `eval/results/phase1_recovery.md` |

## Quick start (local)

```bash
uv sync --extra dev
uv run pytest                 # all tests (~5 min)
uv run pytest -m "not slow"   # skip the statistical recovery tests
```

Draft a protocol with a local Ollama model:

```bash
uv run agentic-tte llm-check
uv run agentic-tte draft NCT04134403 --model qwen3:8b -o runs/NCT04134403/protocol.json
```

Settings come from `ATTE_*` environment variables (see `agentic_tte/config.py`): for example `ATTE_OLLAMA_HOST`, `ATTE_MODEL` and `ATTE_CACHE_DIR`. For Hellbender, see [hpc/README.md](hpc/README.md) (procedure) and [hpc/HELLBENDER.md](hpc/HELLBENDER.md) (cluster facts and pitfalls).

Run the live test against a real model (deselected by default):

```bash
uv run pytest -m live
```

## Rules the code enforces

- The LLM only fills typed schemas. Numbers, splitting, durations and comparator arms are computed deterministically wherever possible.
- A failed parse is kept and flagged in `Criterion.notes`, never dropped. Concepts not found verbatim in the criterion are flagged. Values, time constraints, windows and grace periods the model wrote but the source never said are dropped, with a note.
- Every TTE element records where it came from (`registry`, `derived`, `agent`, `user`, `default`), and `completeness()` doesn't count defaults.
- No patient-level data in git: `.gitignore` plus a pre-commit hook (`scripts/check_no_patient_data.py`). Install the hook with `uv run pre-commit install`.
