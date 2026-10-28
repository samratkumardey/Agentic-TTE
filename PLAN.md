# Agentic-TTE — Architecture & Build Plan

An open-weight, multi-agent Target Trial Emulation (TTE) platform. It reads a trial from ClinicalTrials.gov, turns it into a computable TTE protocol, builds a cohort from EHR data, estimates causal effects, and supports three modes: **Replay**, **Rehearse** and **Redesign**.

Reference design: Li, Pan, Rajendran, Zang & Wang, *Empowering clinical trial design with agentic intelligence and real-world data* (EmulatRx), Nat. Commun. 17:5501 (2026), `s41467-026-74501-2.pdf`.

**We build on EmulatRx's code and extend it.** github.com/TrialLab/EmulatRx is MIT-licensed, so adopting it is allowed. Keep their `LICENSE` and copyright notice on any ported files, and cite the paper and the Zenodo DOI (10.5281/zenodo.20016649). The article text and figures are CC BY-NC-ND, so we don't copy them into our docs or UI.

---

## 0. Starting from the EmulatRx codebase

Upstream is pinned at commit `e353ee46e189` (2026-05-01). It is research code: notebook-driven (`v1-codebase.ipynb`), with flat modules, SQL written against the native MIMIC-IV `mimiciv_derived` schema, and per-trial artifacts stored in git-lfs.

**How we adopt it:** a pinned, read-only snapshot lives in `third_party/emulatrx` (see its `UPSTREAM.md`). Each module is **ported** into the `agentic_tte` package with tests, rather than importing the notebook. The snapshot stays as-is for diffing, with these exceptions:
- **Credentials:** upstream committed a live OpenAI key and org/project IDs in `trialistUtils/main.py` and `parse_info.py`. These are redacted to environment variables in our copy.
- **`trials/` left out:** upstream has filenames that are invalid on Windows (`>`), and its git-lfs server refused downloads. So the reference SQL is not available yet; re-fetch it on Hellbender (Linux) when LFS works.

| Upstream file | What it has | Where it goes / how we extend it |
|---|---|---|
| `trialist_utils.py` (`TrialInfo`, `TrialEligibilityCriterion`, `TrialTreatment`, `TrialOutcome`) | trial data classes | Becomes the seed of `protocol/`. Rewritten as Pydantic `TargetTrialProtocol`, adding time zero, grace period, causal contrast, versioning and diffs |
| `trialistUtils/` (`parse_info.py`, `prompts.py`, `nlp_utils.py`, `web_utils.py`) | CT.gov fetch, Criteria2Query-style parsing prompts, plus a cached example for NCT03872011 | `sources/ctgov/` and `agents/trialist`. Keep their prompts as the baseline; add JSON-schema-constrained output and the criteria IR |
| `dataframe_utils.py` (`DataframeBuilder`, `Informatician`) | LLM→SQL per criterion, cohort and covariate assembly, EC subsets | `agents/informatician` + `omop/`. Their SQL-generation path becomes **one backend**; we add the IR→SQL compiler and OMOP vocabulary grounding, and compare the two |
| `emulation_utils.py` | PSM/IPTW (causallib), Cox, KM, parametric, RSF, bootstrap survival curves, outlier/clean steps | `stats/`. Port these, add clone-censor-weight + IPCW, a doubly robust estimator, E-values and the Schoenfeld PH test, and put unit tests on the synthetic DGP before trusting them |
| `llm_zoo_rlhf.py` (`LLMZoo`, `OllamaModel`, `OpenaiClient`, `LLMwithReward`, BERT reward model) | model registry, Ollama/OpenAI clients, ipywidgets feedback, reward model | `llm/` (registry → `LLMClient`) and `feedback/` (reward model, later DPO). Replace the ipywidgets feedback with the web UI |
| `utils.py` | PDF/document chunking, CT.gov keyword download, Python executor | Chunking → `rag/`. **Drop `custom_python_executor`**: the agents call typed tools and never run arbitrary code |
| `trials/NCT00475852/` | per-criterion SQL, EC-optimization pickles, MIMIC-derived CSVs | Use the SQL as **golden reference queries** for MIMIC Replay. **Do not copy the CSVs** into our repo. They are MIMIC-derived patient-level data under the PhysioNet DUA; regenerate them locally on Hellbender |
| `v1-codebase.ipynb` | the LangGraph wiring and agent prompts | Port the graph and prompts into `graph/` and `agents/`, then extend them with the Auditor, Reporter, HITL interrupts and the persistent checkpointer |

**Where we go beyond EmulatRx:** open-weight-only serving with schema-constrained outputs; OMOP-native cohorts (their SQL targets the MIMIC schema); an explicit 7-element TTE protocol with versioning; Replay as a scored benchmark; Rehearse and Redesign as product modes; a web UI with human-review gates; an Auditor agent; guarded subgroup analyses; and a synthetic data generator with known ground truth.

---

## 1. What we keep from EmulatRx, and what we change

| EmulatRx | Keep | Change for this project |
|---|---|---|
| 5 agents: Supervisor, Trialist, Informatician, Clinician, Statistician | Yes | Add a **Protocol Auditor** (critic) that checks each of the 7 TTE elements, plus a **Reporter** |
| LangGraph with typed central state, cache, fixed seeds | Yes | Persistent checkpointer (Postgres/SQLite), so runs can be resumed, replayed and diffed |
| LLM writes OMOP SQL directly | **No** | The paper's own results show local models (Phi-4, Gemma 3, DeepSeek-R1) fail mostly at concept mapping and SQL syntax. Instead, the LLM emits a **JSON criteria IR** (validated against a schema), and a **deterministic compiler** turns the IR into SQL. Concept IDs come from a vocabulary lookup, not from LLM recall |
| LLM picks the method; deterministic libraries compute | Yes | This is why their HRs were identical across LLMs. We make it a hard rule: the LLM never produces a number |
| Neo4j trial knowledge graph | Later (Phase 3) | Start with the ClinicalTrials.gov v2 API, parsed into a Postgres + pgvector store. Add a graph only if cross-trial queries need it |
| PSM / IPTW / clone-censor-weight, Cox, KM, RSF, DR | Yes | Make the analysis plan follow Hernán's TTE specification explicitly (time zero, grace period, ITT vs per-protocol), not only "balance then Cox" |
| Subgroup analysis triggered when the primary result is null | **Guarded** | Exploratory only: pre-specified in the protocol, or labelled post hoc with a multiplicity note. Without this it is p-hacking |
| RLHF (PPO/DPO) | Phase 5 | Log expert ratings and pairwise preferences from day 1, so DPO data exists when it is needed |
| GPT-4o as best backbone | No | Open-weight only, served locally (PHI never leaves Hellbender) |

## 2. The TTE protocol is the central object

Everything revolves around one versioned, typed object, `TargetTrialProtocol` (Pydantic), that mirrors the standard target-trial table:

1. **Eligibility criteria**: a list of `Criterion` items (domain, concept set, value/operator/unit, temporal anchor + window, inclusion or exclusion, source text span, mapping confidence)
2. **Treatment strategies**: arms, drug/procedure concept sets, dose, timing, grace period
3. **Assignment procedure**: emulated randomization, i.e. the confounder set and the balancing method
4. **Time zero**: an explicit anchor event (e.g. ICU admission + 6 h). Eligibility, assignment and follow-up start must line up here
5. **Follow-up**: start, end, censoring events (death, loss to follow-up, administrative end)
6. **Outcomes**: primary and secondary, concept sets, ascertainment windows
7. **Causal contrast**: ITT analogue vs per-protocol; estimand (HR, RMST difference, risk difference at t)
8. **Analysis plan**: balancing method, outcome model, sensitivity analyses, pre-specified subgroups

Each run stores: protocol version → compiled SQL → cohort snapshot hash → analysis outputs → report. Rehearse and Redesign are just new protocol versions with a diff against their parent.

## 3. Agents (LangGraph nodes)

| Agent | Input → Output | Tools |
|---|---|---|
| **Supervisor** | user goal + state → next node / meeting / stop | routing policy, budget and iteration limits |
| **Trialist** | NCT ID or free-text design → draft `TargetTrialProtocol` | CT.gov v2 API, criteria parser (LLM → IR), similar-trial search (pgvector) |
| **Informatician** | protocol → concept sets → cohort + analysis frame | OMOP vocabulary lookup (Athena tables, plus embedding search over concept names), **IR→SQL compiler**, data-quality checks, attrition table |
| **Clinician** | questions from the others → structured answers (tagged concepts, surrogates, confounder lists) | RAG over PubMed abstracts / guidelines (FAISS or pgvector), cited passages |
| **Statistician** | analysis frame + plan → estimates, diagnostics | deterministic Python stats library (see §6); chooses methods from a fixed menu |
| **Protocol Auditor** | protocol + cohort → pass/fail per TTE element | rule checks: time-zero alignment, immortal-time risk, unmappable concepts, positivity, SMD < 0.1, E-value |
| **Reporter** | full state → report (Markdown/HTML/PDF) | templates, figures (KM, love plot, attrition flow) |

Feedback loops, as in the paper: Informatician ↔ Clinician (missing or unmappable concept → surrogate), Statistician ↔ Clinician (imbalance, implausible result), and Auditor → any agent. A loop exits on a pass, a max-iteration cap, or human interrupt.

**Human-in-the-loop**: LangGraph `interrupt()` at three gates. (a) protocol approval, (b) concept-set approval, (c) report sign-off. In Replay these can be set to auto-approve for batch benchmarking.

## 4. Three modes

### Replay: emulate an existing trial
NCT ID → fetch → parse → map → cohort → analyze → **compare with the published result** (CT.gov results section, or the paper's HR entered manually). This is also the **benchmark harness**. Metrics per trial:
- criteria parsing P/R/F1 against hand annotation (as in the paper's 266-concept gold set)
- % criteria mappable / dropped / surrogated
- cohort size and attrition
- estimate vs RCT: same direction, CI overlap, standardized difference
- wall-clock time and LLM tokens

### Rehearse: design a new trial
The trialist builds a protocol in the UI, starting from a blank form or from a cloned similar trial. Agents give: feasibility (eligible N per site and year), event rates, a Schoenfeld sample-size estimate, expected power, criteria that drive most of the attrition, and confounder suggestions.

### Redesign: tune the rehearsed design
Generate protocol variants and compare them:
- **Eligibility Shapley attribution** (Trial Pathfinder, Liu et al. Nature 2021), measuring each criterion's effect on N, HR and generalizability
- relax or tighten windows and thresholds; drop unmappable criteria with a sensitivity check
- side-by-side variant table, and a promote-to-new-baseline action
- every variant is a child protocol version, so the search is auditable

## 5. LLM serving on Hellbender (H100 80 GB)

- **Runtime**: Ollama inside a Slurm GPU job (Apptainer container or module), `OLLAMA_MODELS` on project storage, bound to the node port and reached by an SSH tunnel or from the app process on the same allocation. All agent code calls an **OpenAI-compatible client** behind one `LLMClient` interface, so **vLLM** can replace Ollama later without code changes. vLLM gives better concurrency for batch Replay and guided JSON decoding.
- **Structured output**: use Ollama's `format` = JSON schema for every agent output. Pydantic validates it, and the call retries with the validation error on failure.
- **Candidate models** (pick by benchmark, not by reputation; all fit in 80 GB):
  - Reasoning / Supervisor / Clinician: `gpt-oss:120b` (≈65 GB MXFP4) or `qwen3:32b`/`llama3.3:70b` (Q4)
  - Parsing / Trialist: `qwen3:32b`, `gemma3:27b`, or `medgemma:27b` (clinical text)
  - Embeddings: `nomic-embed-text` / `bge-m3`; biomedical: S-PubMedBERT via sentence-transformers
  - Only one large model is resident at a time. Use one model for all agents in v1, which also keeps the evaluation simple.
- **Reproducibility**: temperature 0, fixed seed, the model **digest** pinned in the run manifest, and a request/response cache keyed by (model digest, prompt hash).
- **Hosting (confirmed):** Hellbender allows long-running web services, and Open OnDemand is available. Two tiers:
  - **Service tier** (long-running): FastAPI + UI + Postgres + run workers, on a service allocation
  - **GPU tier**: Ollama on the H100 as a long-running service job. Workers reach it over the cluster network at a configured `OLLAMA_HOST`. Batch Replay can add short Slurm GPU jobs (vLLM) for parallel runs
  - **Open OnDemand** is the entry point for trialists (an OOD app that proxies to the UI), so users authenticate with MU credentials and need no SSH tunnel
- Still to confirm: GPU partition and QOS names, wall-time limits for service jobs (auto-restart/requeue if capped), and project storage paths for model weights and OMOP data.

## 6. Data layer

**Everything targets OMOP CDM v5.4.** It is the single adapter boundary: the agents and the SQL compiler see only OMOP.

| Source | Phase | Path to OMOP |
|---|---|---|
| **Synthetic** (ours) | 1 | Custom generator writing OMOP tables directly into DuckDB, with a **known data-generating process**: planted HR, confounding by indication, treatment-timing (immortal-time trap), missingness, effect modification. This is the only source where we can test that the estimator is *correct* |
| Synthea + ETL-Synthea | 1 (optional) | Realistic structure and vocabulary, but no controllable ground truth |
| MIMIC-IV | 3 | MIMIC-IV OMOP ETL (OHDSI); the MIMIC-IV demo works for CI. Needs PhysioNet credentialing and a DUA |
| Oracle Health (Cerner) EHR | 4 | Institutional OMOP extract if one exists, otherwise a FHIR R4 → OMOP ETL. IRB + HIPAA; on-prem only |

- Engine: DuckDB for synthetic/dev; Postgres for MIMIC/Oracle (SQLGlot transpiles compiler output across dialects)
- Vocabulary: OHDSI Athena download (SNOMED, RxNorm, LOINC, ICD9/10CM), loaded into the same database
- Privacy rule (as in the paper): LLMs see **schema, protocol and aggregate counts only**, never patient rows. Cell counts < 11 are suppressed in reports (as the MIMIC/CMS norm)

### Statistics engine (deterministic, tested against the synthetic DGP)
`lifelines` (Cox, KM, Schoenfeld PH test), `scikit-survival` (RSF), `scikit-learn` (propensity models), our own PSM/IPTW with stabilized and truncated weights, clone-censor-weight + IPCW, a doubly robust estimator, SMD/love plots, E-value, Schoenfeld sample size, and Monte-Carlo Shapley for eligibility criteria. Bootstrap CIs, seeded.

## 7. Application architecture

```
Browser (React UI) ──► FastAPI ──► Run service ──► LangGraph runtime ──► agents
                          │             │                 │
                          │             ▼                 ├─► LLMClient ─► Ollama / vLLM (GPU job)
                          │        Postgres: runs,        ├─► OMOP DB (DuckDB / Postgres)
                          │        protocol versions,     ├─► Vocabulary + RAG index
                          │        checkpoints, feedback  └─► Stats engine (Python)
                          └─► SSE stream of agent events (live run timeline)
```

- **UI pages**: Trial search/import · Protocol editor (the 7-element table, with source-text highlighting for each criterion) · Concept-set review · Run timeline (agent messages, tool calls, retries) · Results (attrition flow, love plot, KM, forest plot) · Variant comparison (Redesign) · Report export · Feedback (ratings and pairwise preference)
- **Long runs**: a persistent run service with a job queue (runs survive browser disconnects, resumable from checkpoints). Users reach the UI through an Open OnDemand app; batch Replay can fan out as Slurm jobs
- **Observability**: structured logs per agent step, plus Langfuse (self-hosted) or OpenTelemetry for traces, token counts and latency
- **Packaging**: `uv` + `pyproject.toml`; Apptainer image for Hellbender; Docker for local dev

## 8. Proposed repo layout

```
agentic_tte/
  protocol/        # Pydantic TargetTrialProtocol, Criterion IR, versioning/diff
  sources/ctgov/   # CT.gov v2 client, results parser
  llm/             # LLMClient (Ollama/vLLM), schema-constrained calls, cache
  agents/          # supervisor, trialist, informatician, clinician, statistician, auditor, reporter
  graph/           # LangGraph wiring, meetings, interrupts, checkpointer
  omop/            # vocabulary lookup, concept-set builder, IR→SQL compiler, DQ checks
  synth/           # synthetic OMOP generator with ground-truth DGP
  stats/           # balancing, survival, CCW, DR, Shapley EC, sample size
  rag/             # PubMed ingest, index, retrieval
  report/          # templates, figures
  api/             # FastAPI
ui/                # React app
hpc/               # Slurm scripts, Apptainer defs, Ollama launcher
eval/              # Replay benchmark set, gold annotations, metric scripts
tests/
```

## 9. Roadmap

| Phase | Scope | Exit criterion |
|---|---|---|
| **0. Foundations** (1–2 wk) | Repo; pinned EmulatRx subtree; port `LLMZoo` → `LLMClient` and `TrialInfo` → Protocol schema; CT.gov client; Ollama service job on Hellbender | Fetch an NCT ID and get a schema-valid draft protocol from a local model on the H100; upstream's cached NCT03872011 parse reproduced as a regression test |
| **1. Synthetic + stats core** (2–3 wk) | Synthetic OMOP generator with planted effects; port `emulation_utils` into `stats/`, then extend (CCW, DR, E-value); tests | Recover planted HRs 0.5/1/2/3 within CI; CCW removes the planted immortal-time bias; subgroup DGP recovered (the paper's Tables 3–4 as unit tests) |
| **2. Replay MVP** (3–4 wk) | Trialist + Informatician (IR→SQL compiler) + Statistician + Supervisor in LangGraph; CLI; Markdown report | End-to-end Replay of 3 trials from the paper's list (e.g. NCT04134403, NCT00475852, NCT06091982) on synthetic OMOP |
| **3. Clinician, Auditor, MIMIC** (3–4 wk) | RAG, feedback loops, meetings, MIMIC-IV OMOP; benchmark harness | Replay of the paper's 10 MIMIC trials; parsing F1 and SQL-error metrics comparable to the paper's local-model numbers or better |
| **4. Product UI + Rehearse** (4–6 wk) | FastAPI, React, protocol editor, HITL gates, run timeline, sample size/feasibility | A trialist designs, runs and exports a new protocol without using the CLI |
| **5. Redesign + learning** | Shapley EC, variant comparison, feedback logging → DPO fine-tune of the parser/mapper | Measured gain from fine-tuning on the held-out Replay set |
| **6. Oracle EHR** | ETL, IRB, deployment hardening, access control, audit log | Approved institutional deployment |

## 10. Key risks

- **Concept mapping is the bottleneck** (the paper's main failure mode for local models). Mitigations: vocabulary-grounded retrieval, human concept-set review gate, and later fine-tuning.
- **Free-text-only criteria** (biomarkers, notes) will be missing from structured data. Tag them `unmappable` and report them explicitly, never drop them silently. Clinical-notes NLP is a later add-on (MIMIC-IV-Note).
- **Statistical misuse by agents**: the method menu is fixed, assumptions are checked automatically, and post hoc analyses are labelled.
- **Inherited research code**: upstream has little test coverage and some duplicated functions (e.g. `perform_balancing_method` is defined twice in `emulation_utils.py`). Port behind tests; don't trust it untested.
- **Data-use compliance**: upstream ships MIMIC-derived patient-level CSVs. Our repo must never hold MIMIC or Oracle rows. Add a pre-commit check that blocks `*.csv`/`*.parquet` under `trials/` and `data/`.
- **Validity claims**: Replay agreement with an RCT is evidence, not proof. Reports need limitation sections generated from the Auditor's findings.

## 11. Decisions needed

1. UI stack: React + FastAPI (production) vs Streamlit (faster MVP, later rewrite)
2. Serving: Ollama only, or Ollama for dev and vLLM for batch Replay
3. First 3–5 Replay trials (proposed: the paper's MIMIC acute-care set, which allows direct comparison)
4. Graph DB: defer Neo4j (recommended) or adopt it now to match EmulatRx

Settled: long-running web services and Open OnDemand are available on Hellbender (§5); EmulatRx code is adopted and extended under its MIT license (§0).

Decided 2026-10-24: (1) React + FastAPI; (2) Ollama now, vLLM later behind the same `LLMClient` (`agentic_tte/llm/openai_compat.py` is ready); (3) the first Replay set is the paper's 10 MIMIC-IV acute-care trials (NCT00475852, NCT02856698, NCT03872011, NCT04134403, NCT06091982, NCT03258684, NCT03592693, NCT01560650, NCT01223690, NCT00953303); (4) Neo4j deferred.

## 12. Phase 0 status (2026-10-24)

Done: package scaffold; protocol schema with versioning and diffs; CT.gov client and deterministic extractor (reproduces upstream's GPT-4o criteria split for NCT03872011 exactly); `LLMClient` (Ollama + OpenAI-compatible, schema-constrained, retries, SQLite cache, digest pinning); Trialist draft path and CLI; EmulatRx cache importer as a regression test; Hellbender Slurm/Apptainer scripts; patient-data pre-commit guard.

Hellbender (2026-10-24): environment set up and documented in `hpc/HELLBENDER.md`; the Trialist drafts protocols with `qwen3:32b` on an H100 in an Open OnDemand session (`hpc/interactive_ollama.sh`). Open for the Phase 0 exit criterion: the 10-trial Replay batch (`hpc/replay_phase0.sbatch`) with `qwen3:32b`, `gpt-oss:20b` and `alibayram/medgemma:27b` on one GPU type.

**trialist-v2 / schema v2 (2026-10-24)**, from the review of qwen3:32b's NCT04134403 draft (`tests/test_nct04134403_regression.py`):
- Criteria are lists of atoms, each with its own value and time constraint (C1).
- Every element records its origin (`registry | derived | agent | user | default`); `completeness()` no longer counts defaults (C2, C3).
- There is no default censoring on death (C4). Time frames and treatment courses can end at an event ("28 days or discharge, whichever first") (C8, C10). Administrative criteria are labelled (C13).
- The extractor merges definition bullets, keeps AND/OR in joins, and records the design, status and enrollment type. It reads the detailed description, so ITT can come from the registry (C9, C11).
- Value and temporal text, windows, the time-zero evidence and the grace period must be grounded in the source text, or they are dropped with a note (C5, C14). Anchors such as "randomization" become `time_zero` (C6), "null" strings become null (C7), and a value with no number becomes `present` (C12).
- The few-shot example is no longer taken from a Replay trial. v1 protocol files still load (upgraded to one atom per criterion).

**Exit run (2026-10-24, H100, commit 586e128).** 10 trials × 3 models × 2 replicates, run interactively on g020. Results: [eval/results/phase0_exit-H100.md](eval/results/phase0_exit-H100.md).
- qwen3:32b and medgemma:27b drafted 20/20 schema-valid protocols each. gpt-oss:20b drafted 11/20; the other 9 ended in HTTP timeouts.
- Replicates are not identical at temperature 0: substantively, qwen3 6/10 and medgemma 4/10.
- The accuracy endpoints wait for the human gold annotation (`eval/gold/`).

Changes since the run, all after commit 586e128:
- Per-item timeout handling and an 8,192-token output cap.
- gpt-oss reasoning levels (`ATTE_THINK=low`).
- The unit-first time-frame fix ("Day 30").
- A checker that explains repeated requests within a run.
- Configurable Ollama parallelism, plus a reproducibility test (`hpc/repro_test.sh`).
- Scoring tools (`eval/phase0_scoring.py`, `eval/score_phase0.py`) and gold templates.

The Phase 0 exit criterion (§9) is met.

**Final benchmark run (2026-10-25, commit 9e09bdf, H100 NVL):** 4 arms × 2 replicates × 10 trials, 80/80 protocols, 0 failures, replicates identical in every arm ([eval/results/phase0_exit-H100-final.md](eval/results/phase0_exit-H100-final.md)). Reproducibility came from three settings: OLLAMA_NUM_PARALLEL=1, a fresh server per replicate, and the model unloaded per trial (`eval/SCORING.md` amendments 7–12). What remains for the Phase 0 paper results is the human gold annotation (`eval/gold/`), then scoring parts B and C.

## 13. Phase 1 status (2026-10-24)

**Synthetic data** (`agentic_tte/synth/`):
- A data-generating process with planted effects: confounding by SOFA, age and lactate; conditional HR 0.5/1/2/3; effect modification by SOFA; delayed start within a grace period.
- Ground truth from potential outcomes (marginal HR, risks, risk difference), computable in any analysis population (`Cohort.truth_in`).
- An OMOP CDM v5.4 writer in DuckDB, with a SQL round trip that matches the generator exactly.

**Statistics** (`agentic_tte/stats/`), ported from EmulatRx `emulation_utils.py`:
- Propensity scores, stabilized IPTW, caliper matching, SMD.
- Unpenalized marginal Cox with robust and clustered SEs; weighted KM risk difference; PH test; subgroups with an interaction test.
- Clone-censor-weight; Schoenfeld sample size; E-values.

The recovery tests pass (`tests/test_stats_recovery.py`); the table is in [eval/results/phase1_recovery.md](eval/results/phase1_recovery.md).

Findings for the paper:
1. The planted (conditional) HR is not the estimand of IPTW or PSM. The marginal HR is closer to 1 (2.45 for a planted 3.0), and EmulatRx Table 3 compared against the conditional value.
2. EmulatRx's Cox model used `penalizer=0.1`, which shrinks effects toward 1 (0.611 vs 0.573 on the same weighted data).
3. Caliper matching changes the estimand to the matched treated.
4. Clone-censor-weight targets the "start naturally, or at the deadline" grace-period strategy. That differs from "everyone starts at a random time within grace".

Exit criterion (§9), met:
- planted HRs recovered within 0.15 on the log scale, with the CI covering the truth;
- CCW removes the planted immortal-time bias (null, no confounding: naive 0.76, CCW 1.01);
- the subgroup DGP is recovered.

Open:
- a Monte Carlo study over many seeds, for bias and coverage claims (rows in the table share one seed);
- bootstrap CIs for CCW and KM risks;
- a doubly robust estimator (planned in §6, not yet ported);
- random survival forests (scikit-survival, not yet added).

## 14. Platform decisions (2026-10-24)

**Goal:** a production tool for clinicians and trialists (Replay, Rehearse, Redesign), not a replication of EmulatRx. The first manuscript uses a fully open-weight, automated pipeline.

### Models

- **Open-weight models are the primary pipeline.** The current three are `qwen3:32b`, `gpt-oss:20b` and `alibayram/medgemma:27b`. Adding a model is configuration (an Ollama tag in `ATTE_EXIT_MODELS`), not code.
- **Proprietary models are optional comparator arms, after the first manuscript.**
  - GPT and Grok: through the existing OpenAI-compatible backend (`agentic_tte/llm/openai_compat.py`); untested against those services.
  - Gemini: has an OpenAI-compatible endpoint.
  - Claude: needs one small backend for the Anthropic Messages API.
- **Rules for any external (hosted) model:**
  1. A per-backend `external` flag, enforced in code, refuses any request containing patient-derived content. External models see only registry text, literature, schemas and aggregate counts.
  2. Dated model versions are pinned and recorded in `manifest.txt`, because hosted models change without notice.
  3. They are scored with the same pre-registered method (`eval/SCORING.md`) as the open-weight models, and never pooled with them.

### Data sources and the Data Support Gate

- **One input format: OMOP CDM v5.4.** Every source is converted to OMOP first.
  - Synthetic: built.
  - MIMIC-IV: the OHDSI MIMIC-IV→OMOP conversion.
  - Oracle Health (Cerner): an OMOP extract or FHIR R4 → OMOP.
  - Other EHRs (Epic, PCORnet, i2b2): a conversion each, or FHIR R4 as the common route.
- **Engines:** DuckDB (synthetic, tests) and Postgres. Other dialects, including Oracle SQL, through SQLGlot.
- **Data Support Gate (first item of Phase 2).** A per-protocol report, green/amber/red, before any analysis. Red blocks the run; amber means a criterion needs a surrogate or is reported as unmappable.
  1. Valid OMOP: CDM version, required tables, vocabulary version.
  2. Protocol-to-data fit: patients with a record for each concept set; time resolution (datetime where criteria use hours); death completeness; observation-period coverage of the follow-up horizon.
  3. Data quality: completeness and plausibility, in the style of the OHDSI Data Quality Dashboard.
  4. Governance: data classification, de-identification status, no patient rows to any LLM.
  5. Feasibility: estimated eligible N and events (also feeds Rehearse).

### Cloud-resident data (Oracle EHR): bring the computation to the data

Oracle Health data cannot leave Oracle's cloud. The platform therefore has two parts:
- **Control plane** (Hellbender or elsewhere): agents, LLMs, interface, protocol versions. It produces the protocol and the compiled SQL, and never needs patient rows.
- **Data plane** (inside the data's environment): a small executor, the compiled SQL plus the statistics engine (`agentic_tte/stats/`, plain Python, packaged as a container). It returns **only aggregates**: counts, attrition, effect estimates, survival curves, with small cells suppressed.

Which form the data plane takes depends on MU's Oracle contract:
- (a) the executor container runs in MU's Oracle cloud tenancy;
- (b) Oracle-side analytics runs the generated SQL;
- (c) MU analysts run the scripts and return aggregates.

The same split covers federated multi-site emulation later.

To confirm with MU IT and Oracle: compute allowed in the tenancy, the data format available there (OMOP or FHIR), data-use terms, and the data classification (RCSS).

### Orchestration: free, open source

- **LangGraph**, the open-source library (MIT), as EmulatRx used. It provides the agent graph, persistent and resumable state (SQLite/Postgres checkpointer), and `interrupt()` for clinician approval gates.
- **Not used:**
  - LangGraph Platform and LangSmith (paid): our run manifests and logs replace them.
  - Most of LangChain: `agentic_tte/llm/LLMClient` stays the only model interface, so model swapping, caching and grounding stay ours.
- **Benchmark batches:** Slurm and the `hpc/` scripts, no Airflow or Prefect.
- **Testing:** the GitHub Actions free tier (not set up yet). Cost: $0.

### Order to the first manuscript (open-weight, automated)

1. Phase 2: Data Support Gate; OMOP concept mapping from the vocabulary tables; criteria-to-SQL compiler; attrition table.
2. Wire the agents with LangGraph.
3. End-to-end Replay on synthetic OMOP, where the truth is known.
4. Replay on MIMIC-IV, after PhysioNet credentialing and confirming the data classification with RCSS.
5. The three-model benchmark (`eval/SCORING.md`), with the gold annotation done in parallel.

Proprietary-model arms, the Oracle data plane, and the Rehearse and Redesign interfaces follow the first manuscript.

## 15. Phase 2 status (started 2026-10-25)

**1. Data Support Gate: built** (`agentic_tte/omop/gate.py`, tests in `tests/test_gate.py`).
- It checks an OMOP CDM database against a protocol and reports green/amber/red per check; red blocks the run.
- Checks: structure, CDM version (`cdm_source`), vocabulary, volume, time resolution against the protocol's finest window, death records, follow-up coverage, plausibility, patients per protocol concept (with descendants when `concept_ancestor` exists), and governance (declared data classification and RCSS approval; no patient rows to any LLM).
- Counts below 11 are suppressed. On the synthetic database with the NCT04134403 protocol it reports amber: local vocabulary only, and concepts not yet mapped.

**2. OMOP concept mapping with clinician review: built** (`agentic_tte/omop/vocab.py`, `agentic_tte/omop/concept_sets.py`; tests in `tests/test_concept_sets.py` and `tests/test_cli_concepts.py`).
- **IDs come only from the vocabulary tables.** A deterministic search runs within the concept's domain: exact name, then synonym, then all-words matches. Non-standard codes follow `Maps to` to their standard concept, and deprecated concepts are excluded. Bracketed abbreviations, generic words ("level") and drug salt words are tried as extra queries, and drugs prefer the ingredient.
- **A concept-set book holds one entry per eligibility concept and treatment component.** Each entry has its candidates and a proposed set. Administrative and demographic concepts are marked `not_needed`.
- **Clinician review:** `review.md` plus a `decisions.json` with one decision per entry: approve, replace, reject, unmappable, or surrogate (which needs a note). Every decision is recorded with the reviewer.
- **Only approved or surrogate sets reach the gate** (`book.gate_sets()`). `apply_to_protocol` writes the mappings into a new protocol version.
- **CLI:** `agentic-tte concepts propose|apply` and `agentic-tte gate`, which can use an Athena download (`load_athena`).
- **Test vocabulary:** `tests/fixtures/vocab/` is a made-up, Athena-format vocabulary. Only 196236 is a real concept ID.

Known limits, to address next:
- drug classes ("vasopressor") need class concepts (ATC or other drug classes) expanded to their ingredients;
- abbreviations ("MAP") rely on `concept_synonym`, or on a curated, reviewer-approved abbreviation list;
- a real Athena vocabulary still has to be downloaded by the user under their own account and licences.

The concept-set book also holds the confounders (`C:<name>`) and outcome concepts (`O:<name>`), so every concept the SQL uses is reviewed.

**3. Criteria-to-SQL compiler: built** (`agentic_tte/omop/compiler.py`, tests in `tests/test_compiler.py`). It is deterministic; no LLM writes SQL.
- **Inputs:** the protocol, the reviewed concept sets, and a `CompileSpec`. The spec holds what the protocol names but cannot ground itself: the time-zero event and named anchors ("hospital admission"), each as a concept set. `spec_template` lists what to fill in.
- **Outputs:**
  - `cohort_sql`: one row per index person, with `t0`, a boolean column per compiled criterion and `eligible`;
  - `frame_sql`: the analysis frame for eligible persons (confounders, `treated`, `treat_start`, `time`, `event`, `end_reason`);
  - a compile report, listing every criterion with how it was compiled, or why it was not.
- **No look-ahead:** eligibility reads only records at or before time zero. A criterion that needs later records is not compiled, and the report says to redefine time zero.
- **Time zero:** randomization, enrollment and baseline all mean time zero.
- **A criterion compiles fully or not at all.** A criterion that is not compiled is left out of `eligible` and listed, or raises in strict mode. Not compiled yet:
  - values on drugs or conditions (durations, counts);
  - qualitative results;
  - value constraints on atoms that mix domains;
  - cumulative durations;
  - undefined anchors;
  - follow-up end events ("discharge");
  - active comparators.
- **Recovery:** the compiled frame equals the hand-written `synth.omop.ANALYSIS_SQL` on the synthetic cohort (immediate and delayed treatment), except censoring times. Observation periods are dates, so censoring is resolved to one day; deaths are exact. Eligibility matches the same filter applied to the reference frame.
- **On NCT04134403 (qwen3 draft):** age compiles; the rest is reported. I1 needs look-ahead, because time zero is shock onset and ICU admission comes after it. I2 puts a lactate value on four concepts. E7 and E8 hold counts and durations. E5 needs a "primary admission" anchor.
- **CLI:** `agentic-tte compile PROTOCOL -o DIR --spec spec.json [--book reviewed.json] [--db DB]`. With `--db` it checks that the SQL runs and prints suppressed counts only.
- **Dialect:** DuckDB; Postgres and others go through SQLGlot later.

**4. Attrition and feasibility: built** (`agentic_tte/omop/attrition.py`, tests in `tests/test_attrition.py`). This closes the gate's last check.
- **`attrition(con, compiled)`** runs SQL aggregates on the data side.
  - The flow: persons → index event → in an observation period and alive at time zero → each compiled criterion in protocol order. Each step shows the number remaining and removed, plus the number meeting that criterion on its own, which shows which criterion is binding.
  - Criteria that were not applied are listed with their reasons.
  - For the frame: patients, events and person-days per arm, end-of-follow-up reasons, and the pooled Kaplan-Meier risk at the horizon.
- **`feasibility(report, protocol, design_hr=)`** is the gate's feasibility check.
  - Red: no eligible patients, an empty arm, or no events.
  - Amber:
    - fewer than 11 eligible patients, or fewer than 11 events in an arm;
    - fewer events than Schoenfeld requires for the design HR (at the observed treated share);
    - fewer than 10 events per confounder;
    - criteria not applied;
    - the frame not compiled.
  - The trial's registered enrollment is shown for comparison.
- **Small cells:** counts below 11 are suppressed in `public()`, the markdown and the gate check. Raw counts stay in the report object on the data side. Complementary suppression is not done yet.
- **CLI:** `agentic-tte compile ... --db` writes `attrition.md` and `attrition.json` (suppressed). `agentic-tte gate ... --compiled compile.json [--design-hr 0.7]` runs the feasibility check. It refuses a compile from another protocol version (content hash).

**5. End-to-end Replay in LangGraph: built** (`agentic_tte/graph/replay.py`, `agentic_tte/agents/statistician.py`; tests in `tests/test_replay_graph.py`).
- **Graph:** draft → design review → propose concepts → concept review → compile → gate → (red: stop; amber: gate review) → analyze → report.
- **Human reviews are LangGraph `interrupt()`s,** resumed with the reviewer's answer:
  - **design:** approve, or change elements and name the confounders. Any change makes a new protocol version (created by the user).
  - **concepts:** a decisions file, plus the compile spec (time zero and anchors).
  - **gate:** proceed or stop, asked only when the gate is amber.
- **Isolation and routing:** the database connection is closed over, not stored in the state, which holds the protocol versions, concept sets, SQL and suppressed aggregates only. The Supervisor is fixed routing for now.
- **Statistician** (deterministic menu, no LLM):
  - primary method: clone-censor-weight when the arm has a grace period; otherwise IPTW, or PSM if the plan says so;
  - always: the crude HR, IPTW, SMD before and after weighting, the PH test, IPTW-weighted KM risks at the horizon, and the E-value;
  - complete case on confounders.
- **Checkpoints:** in tests, `InMemorySaver`. In the CLI, `SqliteSaver` in the run directory, so the reviews can happen in separate sessions. State and interrupt payloads are plain JSON (no warnings with `LANGGRAPH_STRICT_MSGPACK=true`).
- **CLI:** `agentic-tte replay start -o RUN --db DB --classification synthetic (--nct NCT | --protocol p.json) [--design-hr]` writes the open review's files. `agentic-tte replay resume -o RUN` continues once they are filled in; every answer needs a reviewer.
- **Test result:** the Trialist (scripted LLM) ran on the real NCT03872011 record, and a scripted clinician did the reviews; the hydrocortisone arm became the synthetic study drug as a surrogate, with a note. Results on 3,000 synthetic patients:
  - **primary (clone-censor-weight):** HR 0.573 (0.515–0.636), against a true marginal HR of 0.561;
  - **IPTW:** HR 0.582;
  - **crude:** HR 0.916, showing the confounding the weighting removes;
  - **balance:** max SMD 0.72 before weighting, 0.035 after.
- **Bug found and fixed:** the CLI end-to-end run exposed a clone-censor-weight bug. Starts exactly at time zero were never counted as "at risk" in the first grace interval, so arm-0 weights were 1 and the estimate stayed confounded (0.73 against 0.53). Fixed in `stats/ccw.py`, with a regression test in `test_stats_recovery.py`.

**6. Phase 2 exit criterion met on synthetic data: Replay of three trials from real LLM drafts** (`eval/replay_phase2.py` → `eval/results/phase2_replay.md`; tests in `tests/test_replay_phase2.py`).
- **Inputs:** the drafts are the Phase 0 final benchmark outputs (qwen3:32b, H100, replicate 1; replicate 2 has the same content hash), copied to `tests/fixtures/protocols/*.final-H100.qwen3-32b.json`. Each trial has a synthetic OMOP profile (`synth.omop.Profile`): STASIS septic shock with three drugs; ASCEND-HF heart failure with nesiritide; post-cardiac-surgery AKI with RRT as a procedure and a prior surgery record. Each has a planted effect.
- **Reviews are scripted, not clinical:** a vocabulary match is approved, anything else is unmappable, and a few per-trial decisions are applied, each stated in the script. Every approved match was checked by hand and is correct.
- **Results:** every primary estimate's 95% CI covers the true marginal HR in the selected population.

  | Trial | Primary | HR | Truth |
  |---|---|---|---|
  | NCT04134403 | IPTW (no grace period) | 0.790 | 0.728 |
  | NCT00475852 | clone-censor-weight, 2-day grace; composite outcome as a reviewed definition | 0.913 | 0.846 |
  | NCT06091982 | clone-censor-weight, 3-day grace added in design review | 0.865 | 0.811 |

  As drafted, NCT06091982 had no grace period; the Statistician flags the immortal-time risk (all treated start after time zero).
- **Found in the real drafts, and fixed:**
  - A composite primary ("rehospitalization ... and all-cause mortality") was silently read as death. Now only pure mortality outcomes default to death. Others need a reviewed `CompileSpec.outcome`: events, plus whether death counts.
  - A placebo comparator component blocked the frame as an "active comparator". It now means no initiation.
  - Relative-change criteria ("increase > 0.3 mg/dL", "> 150%") were read as levels. They are now not compiled, with a reason.
  - Duplicate treatment components are counted once.
  - Several primary outcomes: the first is analysed and the rest are named.
  - A missing horizon or primary outcome no longer raises; it is reported.
  - The gate's time check failed on empty tables; empty tables are now skipped.
  - The four RRT modalities were drafted as a combination ("all"). A design-review fix (one RRT component) is shown next to the as-drafted run.
- **Synthetic CDM:** it now includes empty, typed `procedure_occurrence`, `observation` and `device_exposure` tables, as a real CDM does.
- **Not assessed:** agreement with the trials' published results, because the data are synthetic. That needs MIMIC-IV (Phase 3).

**7. Report sign-off and batch auto-approve: built** (§3 HITL; `agentic_tte/graph/replay.py`).
- **Sign-off:** after an analysed run's report, a fourth review signs it off. `{"approve": true, ...}` gives status `signed_off`; `false` gives `returned`, with the note. Blocked or stopped runs have nothing to sign off.
- **Audit trail:** every review is recorded in `reviews` (stage, reviewer, automatic or not, decision, note, UTC time) and shown as a table in the report.
- **`AutoApprove`:** a fixed answer policy for chosen stages, for batch benchmarking.
  - design: as given;
  - concepts: approve vocabulary proposals, mark the rest unmappable, apply `overrides`; a `spec` is required;
  - gate: proceed on amber;
  - sign-off: approve.
- **Labelling auto-approved runs:** the report opens with "AUTO-APPROVED BENCHMARK RUN. No person reviewed: <stages>. Not for clinical use." Its status is `auto_approved`, never `signed_off`. A policy may cover only some stages; the others still wait for a person. A run fully covered by the policy needs no checkpointer.
- **CLI:** `replay start --auto policy.json` saves the policy with the run, and `replay resume` uses it. The sign-off review writes `report_draft.md` and `signoff.json`.
- **Three-trial harness:** `eval/replay_phase2.py` now runs on `AutoApprove`, with the same estimates as before; its reports are labelled auto-approved.

**8. Abbreviations in concept search: built** (`agentic_tte/omop/abbreviations.py`; tests in `tests/test_abbreviations.py`).
- **Two sources, and only these:**
  - **The protocol's own registry text.** Definitions such as "acute decompensated heart failure (ADHF)", or the reverse, are found with the Schwartz–Hearst algorithm. Titles are skipped, because their acronym is the trial's name.
  - **A curated list:** only entries with a named approver are used.
- **The shipped seed list** (`agentic_tte/omop/resources/abbreviations.json`, 40 critical-care abbreviations, drafted by the developers) is entirely unapproved. `agentic-tte abbreviations init FILE` copies it, `show` lists it, and `approve FILE --reviewer NAME [--only A,B]` records approval. Pass the list to `concepts propose` or `replay start` with `--abbreviations FILE`.
- **Search rules:**
  - An abbreviation is expanded only where it is written in capitals as a word, so "map" is left alone.
  - A domain on an entry disambiguates: IHD is intermittent hemodialysis for procedures and ischemic heart disease for conditions.
  - The protocol's definition wins over the list's.
  - Every candidate found through an expansion records it ("ADHF = acute decompensated heart failure (protocol text: I1)"), and the review sheet shows it in bold, with the abbreviations used listed below it.
  - A direct match outranks an abbreviation match at the same match level, while an exact match through an expansion ("MAP" → Mean blood pressure) outranks a direct word match ("MAP kinase activity").
- **Three-trial Replay:** NCT00475852's "diagnosis of ADHF" now maps to the heart-failure concept through the protocol's own definition. The estimates are unchanged: I2 still has an unmappable dyspnea atom.
- **Packaging:** the seed list lives under `resources/` because the `.gitignore` patient-data rule ignores every `data/` folder. That rule stays.
- The compiler's follow-up end events ("discharge") and duration/count criteria.
- Complementary suppression.
- An LLM Supervisor and LLM method choice for the Statistician.

**Gold annotation (decision pending):** LLM pre-annotation is allowed only as a separate, disclosed arm, not as gold (see the discussion of 2026-10-25). The gold standard stays two blinded human annotators plus an adjudicator.

## 16. Phase 3 plan: MIMIC-IV in OMOP on Hellbender (draft, 2026-10-26)

**What is already there** (hpc/HELLBENDER.md, "Data already on Hellbender"):
- MIMIC-IV 3.1 raw (`hosp`, `icu`), and a validated Postgres build of it in the native schema (127 GB).
- Two Athena vocabularies; the frozen S07 package (2026-09-12, hashed) is the reproducible one.

No OMOP version of MIMIC exists yet. The user confirmed the data use agreement and RCSS on 2026-10-26.

**Route (recommended): our own ETL in DuckDB, from the raw 3.1 files, to OMOP CDM 5.4.**
- **Why DuckDB:**
  - It is the engine the compiler and gate already use.
  - It needs no database server on the cluster.
  - The input is the checksummed PhysioNet release, so a run is reproducible.
  - The output is one file.
- **The Postgres build stays** as the native-schema reference, for comparing against EmulatRx's `mimiciv_derived` SQL.
- **To check:** the OHDSI MIMIC-IV-to-OMOP ETL, as a cross-check of mapping decisions, not as the primary path.

**Steps:**
1. **Confirm locations, space and the CPU partition** (read-only commands; see the session of 2026-10-26).
2. **ETL** (`agentic_tte/etl/mimic_omop.py` plus an `hpc/` batch script on a CPU node).
   - **Mappings:**
     - `patients` / `admissions` → person, observation_period and visit_occurrence; `icustays` / `transfers` → visit_detail;
     - `diagnoses_icd` and `procedures_icd` → condition and procedure, through ICD9/ICD10 'Maps to';
     - `prescriptions` (NDC → RxNorm) and `inputevents` (vasopressors, fluids) → drug_exposure;
     - `labevents` (LOINC) and `chartevents` (vitals, scores) → measurement;
     - deaths from `patients.dod` and `admissions.deathtime`.
   - **Source codes go to standard concepts only through vocabulary 'Maps to'** or the MIT-LCP `concept_map` files, and every mapping table is reviewed.
   - **Unmapped records are kept**, with concept 0 and their source value, and reported, never dropped.
   - **MIMIC's date shifting:** calendar dates are not real, so `anchor_age` gives the age, and only within-patient times are meaningful.
   - **Output:** `$P/tte-project/omop/mimiciv-3.1-cdm54.duckdb`. It never goes in git or `~/data`. Its size will be measured on a 1% pilot first; 20–40 GB is expected.
   - **Manifest:** every run writes `manifest.txt` with the source SHA256SUMS hash, the vocabulary hash and the git commit.
3. **Vocabulary:** load the S07 package with `load_athena`.
4. **Validation** (aggregates only):
   - row reconciliation with the source counts (364,627 patients);
   - mapping coverage per domain;
   - the Data Support Gate (classification `deidentified`, approved).
5. **Pre-register the Replay-agreement endpoints** in `eval/SCORING.md` before any MIMIC result is seen: agreement of the HR with each trial's published result, with tolerances and the population.
6. **First real Replay:** NCT04134403 on MIMIC OMOP from the qwen3:32b draft, with real clinician reviews (not `AutoApprove`). The earlier run in `emulatrx-work/results/runs/NCT04134403_mimiciv_v1/` is the comparison.
7. **Phase 3 exit criterion:** Replay of the paper's 10 MIMIC trials, plus the Clinician and Auditor agents (§9).

**Tests on the PC:** a small made-up fixture shaped like the MIMIC tables. No MIMIC rows, demo or full, go into git.

**Route chosen by the user (2026-10-26): the DuckDB ETL from the raw files.**

**1% pilot (job 19333614, 2026-10-26, commit f516a8a, `general` c150).**
- **Resources:** 3.5 minutes, 3.1 GB of memory, a 520 MB CDM, mostly the vocabulary. Persons 2,337 (1,409 without an admission were left out), deaths 374, ICU stays 969. The pilot reads the full source files and samples after reading, so the full run mostly adds write time; the full CDM is expected at 15–30 GB with labs and drips.
- **Problems found:**
  - **Prescriptions: 0% mapped.** The S07 package appears to have no NDC vocabulary.
  - **Diagnoses: 51% mapped; ICD procedures: 46%.** Every top unmapped code is ICD-9 (4019, 25000, 4280; 3893, 9671), so ICD9CM and ICD9Proc appear missing too.
  - **Five curated procedure maps were rejected,** their targets non-standard with no 'Maps to'. They include Invasive Ventilation (225792), which needs a reviewed override. Reviewed maps for chart and procedure items are not supported yet.
- **Expected:** chartevents at 13% (only the 40 curated items; the rest are mostly nursing documentation).
- **Before the full run:** a vocabulary with ICD9CM, ICD9Proc and NDC; then rerun the pilot.

**Pilot A (job 19355552, 2026-10-26, commit 0915fae; Athena v5.0 29-AUG-26; data in the lab folder).**
- **Resources:** 3.4 minutes, 4.1 GB of memory, a 1.1 GB CDM (about 0.9 GB of it vocabulary).
- **Mapped:**

  | Source | 1% pilot | Pilot A |
  |---|---|---|
  | Prescriptions | 0% | **86.0%** |
  | Diagnoses | 51.3% | **92.7%** |
  | ICD procedures | 45.9% | **99.9%** |
  | Procedure events | 95.4% | 95.4% |

  The unmapped prescriptions are rows without an NDC (flushes, "Bag", "Vial", sterile water); the unmapped diagnoses are history, status and place codes (V15.82, Z95.5, Y92.230).
- **Procedure events:** 6 curated targets rejected (Chest Tube Placed is new in this release). The procedureevents review covers all 6.
- **Episode domain:** 41 records map to it, and are left out (no table).
- **Full CDM expected:** about 25–35 GB with labs and drips, in well under an hour.
- **Next:** Pilot B with the reviewed item maps, then the full run.

**Item reviews done (2026-10-27, reviewer Samrat Kumar Dey; reviewed maps in `$ATTE_OMOP_DIR/review/*.reviewed.json`, not in git).** Proposals pre-checked item by item; every hand-chosen concept confirmed by name against the vocabulary before the full run.
- **labevents:** OHDSI curated map (1,402 items) + 24 reviewed (small duplicate panels, eGFR CKD-EPI 2021, HIV screen, H. pylori stool Ag, Lyme) + 103 unmappable (tube holds, H/L/I indices, cancelled-specimen flags, coded comment and PCR-component items).
- **inputevents:** 193 mapped (RxNorm ingredients; clinical drug forms for pip/tazo, amp/sulbactam, SMX/TMP, insulin 70/30) + 114 unmappable (enteral formulas, TPN and lipids, blood components, carrier fluids, OR/PACU totals, LR: no LR concept in this release).
- **chartevents:** 23 reviewed (GCS ×3, FiO2, O2 flow, PEEP, tidal and minute volume, respiratory rates, heart rhythm, CVP, Braden, pupils, pain level, ventilator type) + 177 unmappable (nursing documentation, alarms; chart-charted labs left to labevents to avoid double counting). RASS and peak inspiratory pressure not found yet.
- **procedureevents:** Invasive Ventilation → 37158404, Chest Tube Placed → 40489839; ports and unplanned line removals unmappable.
- **Lesson:** a pasted lookup lost the first digit of a concept ID (45775571 → 5775571), which the vocabulary check caught. A clipped ID that exists would not be caught, so hand-chosen IDs are now confirmed by name before a run.

**Pilot B (job 19365411, 2026-10-27, 1%):** 6 minutes, 4.1 GB memory, 1.2 GB CDM; the four maps in use, 0 rejected targets after the glycopyrronium fix.

**Full CDM built (job 19365483, 2026-10-27, commit 9cad08c; `$ATTE_OMOP_DIR/mimiciv-3.1-cdm54-19365483.duckdb`, 4.7 GB).** 9 minutes on 16 CPUs, 83 GB peak memory (of 96 GB).
- **Persons:** 223,452 with an admission (141,175 without, left out; together the release's 364,627); 546,028 visits, 94,458 ICU stays, 36,882 deaths.
- **Rows:** measurement 229.4M, drug_exposure 28.5M, condition 6.1M, procedure 1.6M, observation 2.5M.
- **Mapped:** labevents 93.2%, procedureevents 99.6%, inputevents 74.6%, chartevents 21.0% (by design), diagnoses 92.6%, ICD procedures 99.9%, prescriptions 86.1%. 0 map targets rejected.
- **Left out:** 2,872 Episode-domain and 98 Specimen-domain records (no table); 21,890 prescriptions without a start time.

**Reviewed overrides for chartevents and procedureevents: built (2026-10-26;** `tests/test_item_overrides.py`).
- **Proposals:** `etl items propose --kind chartevents|procedureevents --concept-map <maps>` checks each curated target against the loaded vocabulary, as the ETL does.
  - Items whose curated map works are skipped.
  - Items whose curated target was rejected (e.g. Invasive Ventilation, 225792) are always listed, whatever their count, with the reason.
  - Frequent unmapped items are listed too.
  - Chart items are searched in Measurement and Observation; procedure items in Procedure, Observation and Device.
- **In the ETL**, a reviewed map of the same kind overrides the curated map item by item:
  - approve or replace: the reviewed concept;
  - reject or unmappable: the curated mapping is turned off;
  - undecided: the curated map stands.
  
  Reviewed targets are checked against the vocabulary like curated ones. The report counts overrides and names the reviewers. A reviewed map without `--concept-map` converts only its approved items.

**Search quality, from a pre-check of the first full proposals (2026-10-26).** The defects found:

| MIMIC label | Proposed | Should be |
|---|---|---|
| Invasive Ventilation | "Non-invasive ventilation" | Invasive mechanical ventilation |
| Potassium / Sodium / Chloride / Hematocrit / HCO3 "(serum)" | the specimen "Serum" | the lab test |
| Creatinine | "Serum creatinine above reference range" | a lab test, not a finding |
| Neutrophils and other differential counts | cord-blood tests | the blood test |
| H, L, I (lab indices) | single-letter synonyms | nothing |
| Potassium Chloride | no match (salt-only query dropped) | potassium chloride |
| LR, KCL | RxNorm Extension code names | nothing, without an approved expansion |

The fixes (commit 5d7675b):
- **Queries:** bracketed text is its own query only when it is an expansion or a drug brand. Single letters are dropped, salt-only queries are kept, and short abbreviations get exact matches only.
- **Ranking**, within a match level:
  - demote negations the query lacks, findings and cord-blood tests for measurements, and RxNorm Extension;
  - prefer LOINC for measurements, and the candidate found by the fullest query;
  - item proposals rank the specimen by category: serum or plasma for chemistry, blood for blood gas and haematology.
- **Tests:** the decoys are regression tests. Phase 2 results are unchanged. **Proposals made before this must be regenerated.**

**Second pre-check (2026-10-26).** Lab proposals were still wrong where it matters:

| MIMIC lab | Proposed |
|---|---|
| Creatinine | BUN/Creatinine ratio |
| Potassium | … in Red Blood Cells |
| Sodium | Calcium/Sodium ratio |
| Hemoglobin | Hemoglobin F1/total |
| Lactate | a gas panel |
| Hematocrit | an "adjusted by calculation" variant, via an exact synonym |

There were two causes. **Retrieval:** each query kept only the first ~30–45 rows the database returned, before ranking, so the right test among hundreds of LOINC matches was usually never seen. **Ranking:** it could not tell a different analyte from the one asked for.

The fixes:
- **Retrieval:** up to 500 matches, ordered so LOINC tests with a measured property and shorter names come first; all are then ranked.
- **Measurement penalties:**
  - a LOINC analyte (the text before `[`) with words the label lacks;
  - a LOINC name without a `[property]` (panels, narratives);
  - calculated, adjusted, ratio, goal or `[Measurement]` variants.
- **Ordering:** the penalty now ranks before the match level.

The decoys and a 600-decoy retrieval case are regression tests. Phase 2 results are unchanged.

**Step 2, part 1: built** (`agentic_tte/etl/mimic_omop.py`, `agentic-tte etl mimic`; tests in `tests/test_etl_mimic.py` on a made-up four-patient dataset written at test time).
- **Built:** person, observation_period, visit_occurrence (admissions; ER-and-inpatient when there was an ED stay), visit_detail (ICU stays), death, and the diagnoses, procedures and prescriptions coded through the vocabulary.
  - **Source codes:** ICD9CM, ICD10CM, ICD9Proc, ICD10PCS and NDC codes are matched with the dots removed, then followed through 'Maps to'. Each record goes to the table of its standard concept's domain.
  - **Unmapped records** are kept with concept 0 and their source value.
- **chartevents and procedureevents: built (step 2b)** through the MIT-LCP curated maps (`mimic-code/mimic-iv/concepts/concept_map/*_to_omop.csv`, SSSOM, with author and reviewer ORCIDs; `--concept-map`).
  - **Read where they are:** the maps are not vendored (the guard blocks CSVs in git), and the manifest records the `mimic-code` commit.
  - **Each curated target is checked** against the loaded vocabulary: accepted if standard and valid, followed through 'Maps to' if no longer standard, rejected if missing. The report lists every rejection, and match predicates are counted (`exactMatch` / `closeMatch`). The map rows used go into OMOP's `source_to_concept_map`.
  - **Items without a curated map are not converted.** They are counted by item, with labels, in the report.
- **labevents now start from a curated map too (2026-10-27):** the OHDSI MIMIC-IV ETL's lab-to-LOINC map (`github.com/OHDSI/MIMIC`, `custom_mapping_csv/gcpt_meas_lab_loinc_mod.csv`, Apache-2.0; 1,401 items), read from a checkout (`--ohdsi-mimic`, `ATTE_OHDSI_MIMIC`) and handled like the MIT-LCP maps: targets checked against the vocabulary, reviewed decisions override item by item, `etl items propose --kind labevents --ohdsi-mimic` lists only items it does not map.
  - **Why:** the third pre-check of the search-based lab proposals (916 items) still needed most items reviewed by hand. The OHDSI map covers 789 of them (93% of lab rows) and differed from our proposals in 415 of 594 compared items, mostly in method or specimen (e.g. Platelets by automated count, Neutrophils /100 leukocytes). The 127 items it leaves out are mostly specimen holds and serum indices (H, L, I), 7% of rows.
- **labevents and inputevents: built (step 2c), through reviewed item maps** (`agentic_tte/etl/item_maps.py`; tests in `tests/test_item_maps.py`).
  1. `agentic-tte etl items propose --kind labevents|inputevents` counts events per item (one scan) and proposes a standard concept from the vocabulary search on each item's label:
     - labs are searched in the Measurement domain, with the specimen ranking the LOINC candidates;
     - inputs in the Drug domain, preferring the ingredient.
     It writes `map.json`, `review.md` (most frequent first, with coverage) and `decisions.json`. Items with fewer than 11 events are left out by default.
  2. A person decides each item (approve, replace, reject, unmappable); a reviewer is required.
  3. `etl items apply` writes the reviewed map, and `etl mimic --item-map ...` uses it.
  - **Reviewed targets are checked against the vocabulary again,** like curated ones, and approved maps go into `source_to_concept_map` with the reviewers listed in the report.
  - **Labs join on the patient,** so outpatient labs are kept with no visit. Input amounts go to `quantity` and `dose_unit_source_value`.
  - **The review files stay on Hellbender:** they hold item labels and per-item counts from the credentialed data, with counts below 11 suppressed.
  - **The reviews are the user's or a clinician's to do;** nothing is approved automatically.
  - **Nothing is pre-filled** (user's decision, 2026-10-26). The decisions file shows each proposal for reference, but every `action` starts blank. Items left blank stay undecided and are not converted, so signing the generated file unread approves nothing. Concept-set decision files for protocols still pre-fill `approve`: `AutoApprove` relies on it, and the lists are short.
  - **First real proposals (inputevents, 2026-10-26)** exposed a search bug: all-words matching used substrings.
    - **The damage:**
      - "NaCl 0.9%" (1.6M rows) was proposed as linaclotide 0.29 MG;
      - 30 tube feeds labelled "(Full)" as buckminsterfullerene;
      - "LR" as elranatamab;
      - "KCL (Bolus)" as a *Cochliobolus* allergen.
      
      Substring matches made up 103 of 307 items and 42% of rows. Exact-name matches (149 items, 41% of rows) were sound.
    - **The fix** (`vocab.py`, shared with protocol concept sets):
      - whole-word matching;
      - administration words ("bolus", "full", "concentrate", …) and strengths are dropped from drug queries;
      - a query left with only such words is not run.
      
      The decoys are regression tests. Phase 2 results are unchanged. **Proposals made before this fix must be regenerated.**
- **Checks:** on the ETL output, the gate's structure, CDM-version and death checks pass, and a compiled cohort and frame run, with exact times.
- **Options,** all recorded in `cdm_source` and the manifest:
  - `diagnosis_time` (admission by default, a mild look-ahead; or discharge);
  - `death_followup_days` (365, for state death records);
  - `sample_percent` (deterministic `subject_id mod 10000`, for the 1% pilot).
- **Outputs:** the CDM file plus `<stem>.report.md` and `.report.json` (aggregates, suppressed) and `<stem>.manifest.txt` (job, node, git commit, source SHA256SUMS digest, vocabulary version, DuckDB version, options). Only the report and manifest may leave the cluster.
- **Known limits:**
  - Race and ethnicity are concept 0, with the source value kept.
  - Persons without any admission are left out and counted.
  - ~~The compiler's Visit domain reads only admissions.~~ **Fixed (2026-10-26):** the Visit domain is now admissions plus `visit_detail` stays, so ICU admission (32037) works as time zero, an anchor or a criterion. The gate requires `visit_detail` and counts Visit concepts in both tables. The synthetic CDM has an empty `visit_detail`, and its results are unchanged.
- **Batch script:** `hpc/etl_mimic.sbatch` (partition `general`, 16 CPUs, 96 GB, scratch on `/local/scratch`). The 1% pilot is the default, and `ATTE_ETL_SAMPLE=full` runs the whole release. It was dry-run on this PC against the made-up data. `~/data` and `/cluster/pixstor/data/$USER` are one directory and one 500 GB quota (2026-10-26), so the full CDM must fit there.

**Open decisions:**
- The ETL route above, versus loading from the Postgres build.
- Whether to delete the duplicate 9.9 GB MIMIC zip once the raw copy's checksums pass.

## 17. Replay on MIMIC: finish EmulatRx's pipeline, then go beyond it (plan, 2026-10-27)

The user's order (2026-10-27): first a successful Replay on MIMIC-IV covering what EmulatRx did, with clear contributions over it; the CREST-2 Replay on Oracle comes after.

### 17.1 What EmulatRx did (from the paper)

- **Agents (5):** Supervisor, Trialist, Informatician, Clinician, Statistician, in LangGraph.
- **Trials (20):**
  - 10 acute trials on MIMIC-IV (§11, item 3);
  - 10 chronic trials on INSIGHT (NYC);
  - all 20 used for criteria parsing (F1 against a 266-concept gold set) and SQL error types.
- **Models:** GPT-4o (best), plus Phi-4, DeepSeek-R1 14B and Gemma 3 12B, which were weaker at parsing and much weaker at SQL.
- **Statistician:** evaluated on synthetic data only (HR 0.5, 1, 2, 3). It was compared with the planted conditional HR, which is not what IPTW or PSM estimates (§13, findings 1–2).
- **Real effect estimates:** 3 MIMIC case studies only, with no pre-specified agreement test:
  - nesiritide (NCT00475852): HR 0.59, but ASCEND-HF was null (risk difference −0.7, 95% CI −2.1 to 0.7);
  - RRT in AKI (NCT06091982): HR 0.64, with no published result to compare against;
  - hydrocortisone in septic shock (NCT04134403): HR 1.30 (harm), against no published result for that trial.
- **Their stated limitations:**
  - no standard end-to-end benchmark;
  - structured data only (notes and imaging were future work);
  - two databases;
  - Clinician RAG without a knowledge graph.
- **Also in the paper:** RLHF (reward model, PPO and DPO), eligibility-criteria Shapley optimization, adaptive sample size, and Clinician answers rated by three experts on a Likert scale.

**What the reference file shows** (`eval/reference/replay_trials.json`): of the 10 MIMIC trials, only 2 have a numeric primary estimate with a CI (NCT00475852, NCT02856698), 3 more have a verified or probable result without an estimate we can score, and 4 have no published result (NCT04134403, NCT06091982, NCT03592693, NCT01560650). So agreement with the RCT cannot be scored for most of EmulatRx's own trials. A scored Replay needs a trial set chosen for that purpose.

### 17.2 Part A: finish the pipeline (parity with EmulatRx, on OMOP)

The MIMIC OMOP CDM exists (§16). What is missing for an end-to-end Replay of the 10 trials:
1. **Clinician agent:** RAG over PubMed abstracts for surrogate concepts and confounder lists, with cited passages. It answers the Informatician (unmappable concept → surrogate) and the Statistician (imbalance → drop or replace a covariate).
2. **Supervisor:** routing, loop caps and budgets, plus the Informatician ↔ Clinician and Statistician ↔ Clinician feedback loops (§3).
3. **Covariates on real data:** the confounder sets from the Clinician, compiled through reviewed concept sets (§15). EmulatRx used 87–89 covariates per case.
4. **Run the 10 trials** on the full CDM from the qwen3:32b Phase 0 drafts.
   - Reviews are done by the user, so they are labelled as such, not as clinician review.
   - The first trial is NCT04134403, compared with EmulatRx's three case-study numbers.
5. **Reports** for each trial, with attrition, balance, the estimate and the gate result.

**Exit:** 10/10 trials run end to end on MIMIC OMOP, with reports and manifests.

### 17.3 Part B: contributions over EmulatRx

**1. Agents (5 → 9), each with a job that can be measured.**

| Agent | In EmulatRx | Ours | Status |
|---|---|---|---|
| Supervisor | yes | yes | to build |
| Trialist | LLM parse | schema-constrained protocol with 7 TTE elements, versioned | built |
| Informatician | LLM writes SQL | LLM proposes concepts, vocabulary grounds them, a deterministic compiler writes the SQL | built |
| Clinician | RAG | RAG with citations | to build |
| Statistician | PSM/IPTW + Cox | adds CCW for grace periods, E-values, PH test and immortal-time flag; the LLM never produces a number | built |
| **Protocol Auditor** | no | checks time-zero alignment, immortal time, look-ahead, estimand vs the trial's | to build |
| **Data Support Gate** | no | feasibility, attrition, positivity, small-cell suppression | built |
| **Concordance agent** | no | pre-registered agreement metrics against the RCT | to build |
| **Reporter** | summary by GPT-4o | report with sign-off status (signed off / auto-approved) | built (basic) |

The count is not the contribution. Each new agent gets an ablation (run without it) so its effect on the results is shown.

**2. Trials: a scored benchmark, not case studies.**
- Keep EmulatRx's 10 for parity: parsing, compile success, cohorts.
- **Add MIMIC-emulable ICU trials with published results**, chosen by a rule written down before any is run:
  - adult ICU population;
  - intervention and comparator observable in MIMIC;
  - primary outcome observable in MIMIC (in-hospital or 28/90-day death; deaths are captured to 1 year);
  - a published primary estimate with a CI.
- **Target:** 20–30 trials in total.
- **Candidates to vet** (NCT IDs to be confirmed in the registry when selected):
  - ADRENAL and APROCCHSS (corticosteroids in septic shock);
  - SPICE III and MENDS2 (dexmedetomidine);
  - SUP-ICU (stress-ulcer prophylaxis);
  - AID-ICU (haloperidol);
  - VICTAS and LOVIT (vitamin C);
  - STARRT-AKI (timing of RRT);
  - SMART (balanced crystalloids);
  - TRISS (transfusion threshold);
  - NICE-SUGAR (glucose control).
- **Some candidates need inputs marked unmappable in the 2026-10-27 reviews:** LR for SMART, blood components for TRISS. Choosing them means revisiting those items first.

**3. Scoring, pre-registered** (`eval/SCORING.md`, before any MIMIC estimate is seen). This answers EmulatRx's first limitation.
- Agreement with the RCT: same direction, estimate inside the trial's CI, standardized difference, same significance conclusion, as in RCT-DUPLICATE.
- Parsing P/R/F1 against our gold annotation (`eval/gold/`, still to be made).
- Compile success and SQL failures per criterion, compared with EmulatRx's local-model error rates.
- Cohort size and attrition.
- Runtime and tokens.

**4. Open-weight only, reproducible.**
- Local models, with identical replicates at temperature 0 (§12). EmulatRx's best results needed GPT-4o.
- A manifest for every run: commit, model digest, vocabulary version, data hash.

**5. Portable: OMOP-native.**
- The same compiler and gate run on MIMIC and, later, on Oracle Health RWD (OMOP).
- EmulatRx's SQL targets MIMIC's native schema.

**6. Design-bias safeguards.**
- Clone-censor-weight for grace periods; no look-ahead in eligibility.
- Synthetic recovery scored against marginal truth (§13).
- Negative-control outcomes and E-values in every report.

**7. Human review that leaves a trail.**
- Interrupts for design, concepts, gate and sign-off, with reviewer names.
- Auto-approved benchmark runs are labelled as such.

### 17.4 Order of work

1. Pre-register the scoring (17.3 point 3) and the trial-selection rule (17.3 point 2), before any MIMIC estimate. **Done 2026-10-27:** `eval/SCORING.md`, section "Phase 3 Replay" (R1–R7).
   - The bootstrap CIs for risk measures (R4) must be built before any agreement-set trial is analysed.
   - **Screening done 2026-10-27 (AI, to be checked by the user):** `eval/reference/replay_selection.json`.
     - 1,193 candidates screened.
     - Tentative sets: 16 primary, 2 secondary, 8 exploratory; 8 uncertain on one named criterion.
     - **Wave 2** (wider conditions, PubMed result linkage, coverage rule; SCORING.md Phase 3 changelog items 5–9):
       - 2,309 candidates; 37 tentatively primary, 30 selected;
       - coverage met for F1–F4 and F7; F5 and F6 have 2 trials each.
     - Next: extract and verify S2 estimates (primary papers), then S8 counts on Hellbender.
2. **Design adopted 2026-10-27:** a core set plus stage-wise correctness and fault injection (SCORING.md Phase 3 changelog item 11).
   - The contribution is a pipeline shown correct stage by stage, not a trial count.
   - **Core set:** 12 trials with user-verified estimates. It is the primary agreement set.
   - **Breadth set:** the rest, reported on pipeline endpoints only.
   - **Endpoints:**
     - E1 fault injection: 10 fault types, ground truth by construction;
     - E2 cohort checks without hand-built cohorts: mimic-code validated concepts, metamorphic tests, differential testing;
     - E3 reference protocols written by the user, with a clinician audit;
     - E4 synthetic data;
     - E5 agreement with attribution by substitution;
     - E6 breadth.
3. **Done 2026-10-27:** the user verified the S2 estimates (SCORING.md Phase 3 changelog item 12).
   - **Core set (12), rebalanced (changelog item 13):**
     - 5 trials with significant published results: APROCCHSS, CAPE COD, LOVIT, EvK, ACURASYS;
     - 7 null trials: CHEST, ADRENAL, STARRT-AKI, CRISTAL, ECLS-SHOCK, BICAR-ICU, HYDRAREA.
   - **Breadth set (17):**
     - 10 with verified estimates also get agreement as a secondary, exploratory result;
     - 7 get pipeline endpoints only.
   - **8 trials failed S2** and are reported.
   - **Core coverage below target:** F3 (procedure), F4 (timing), F5 (composite), F6 (lab-defined).
4. **Build:**
   - **done 2026-10-27:** the bootstrap CIs (R4; `stats/bootstrap.py`; changelog item 15);
   - **done 2026-10-27:** the Protocol Auditor (`agents/auditor.py`, checks A-a to A-j) and the fault injector (`faults.py`, F-a to F-j).
     - On synthetic data, 10 of 10 faults are caught, with no alarm on the correct design (`eval/results/fault_injection_synthetic.md`; changelog item 14).
     - **Wired into the Replay graph (2026-10-27):** compile → audit → gate.
       - A failed audit goes back to design review with the findings. The reviewer changes the design, which is re-proposed and re-reviewed only if new concepts appear, or overrides it with a written reason, shown in the report.
       - 3 failed rounds stop the run.
       - Under `AutoApprove` the failure is overridden automatically (labelled) or stops the run (`audit="stop"`).
       - Sets approved as surrogates are exempt from the A-j name check.
     - Still to do: measure the Auditor's false alarms on real protocols. A-f fires on the qwen3:32b draft of NCT03872011, which has no new-user exclusion;
   - **done 2026-10-28:** the metamorphic and differential cohort tests (E2 (b), (c); `omop/cohort_checks.py`, `agentic-tte cohort check`; changelog item 16).
     - Still to do: the mimic-code comparison (E2 (a)), once the derived schema on the Postgres build is confirmed;
   - then the Clinician (RAG) and Supervisor.

   Tests on synthetic OMOP first.
5. **User:** write the 12 reference protocols from a template (E3), blind to model outputs; a clinician adjudicates the uncertain items and audits 20%.
6. Run the core set end to end on the full CDM (S8 counts first), with E1–E5 and the ablations; then the breadth set (E6).
7. Write up. Then CREST-2 on Oracle (the user's next goal), with the same pipeline and the user's clinician-adjudicated reference.
