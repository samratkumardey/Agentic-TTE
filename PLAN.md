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
