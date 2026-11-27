# Scoring methods (pre-registered)

This file holds two pre-registrations, each fixed by the commit that added it:
- **Phase 0 exit run** (protocol drafting; 2026-10-24): the sections from "Design" to the Phase 0 changelog.
- **Phase 3 Replay on MIMIC-IV** (trial selection and agreement with the trials; 2026-10-27): the section "Phase 3 Replay" at the end, with its own changelog.

# Phase 0 exit run: scoring method (pre-registered)

Fixed on 2026-10-24, before any exit-run output exists. The commit that adds this file is the pre-registration; cite its hash in the methods. Any change after outputs exist goes in the changelog at the end, with the reason.

## Design

| Factor | Levels |
|---|---|
| Trials | The 10 MIMIC-IV acute-care Replay trials ([reference/replay_trials.json](reference/replay_trials.json)) |
| Models | `qwen3:32b` (`ATTE_THINK=false`), `gpt-oss:20b` (`none`), `alibayram/medgemma:27b` (`none`) |
| Replicates | 2 per model. Each replicate uses a **fresh, empty LLM cache**, so no answer is reused |
| Code | trialist-v2 at one git commit (recorded in every `manifest.txt`) |
| Hardware | One GPU type for all 6 runs (recorded in `manifest.txt`) |
| Decoding | temperature 0, seed 42, `num_ctx` 16384 |

**Replicate 1 is the analysis set.** Replicate 2 is used only for stability (part D). Registry records come from the shared CT.gov cache (same SHA-256 for every run), so differences between runs come from the models.

## Gold standards

| Part | What is scored | Gold standard |
|---|---|---|
| A. Split | Registry text → criterion items | Two annotators' segmentation of the registry text. Secondary: EmulatRx Table 2 #EC (GPT-4o-derived, so a reference, not gold) |
| B. Concepts | Concepts, values, time constraints per criterion | Double annotation by two annotators, adjudicated by a third, blind to all model outputs (below) |
| C. TTE elements | The 8 elements of each protocol | A reference protocol per trial, written by the same annotators from the registry record (eligibility text, arms and interventions, outcomes, detailed description) |
| D. Stability | Replicate 1 vs replicate 2 | None: self-agreement |
| E. Operations | Calls, tokens, failures, time | None: logs and manifests |

### Gold annotation for B and C

- **Unit:** the criterion items produced by the deterministic extractor at the exit-run commit. These are the same for every model, so split errors are scored once, in part A, and not again in B.
- **Format:** the protocol schema v2 itself: `eval/gold/<NCT>.json`, holding criteria atoms plus a reference protocol. Annotators fill a template generated from the extractor output, with every model field empty.
- **Rules:** the trialist-v2 prompt rules (atoms, domains, value, temporal, `category`), which follow Criteria2Query 3.0 as adapted by EmulatRx, plus three clarifications:
  - a concept's `name` is the shortest clinically meaningful noun phrase;
  - abbreviations are expanded and listed in `eval/gold/synonyms.json`;
  - anchors are normalized, with randomization, enrollment and study entry → `time_zero`.
- **Agreement:** Cohen's κ between the two annotators on concept presence plus domain, before adjudication. Target ≥ 0.7, as in EmulatRx. Report it even if lower.
- **Registry errors** (for example NCT04134403 E2, "negative serum HCG"): gold follows the registry text and carries a flag. A model reproducing the registry is not penalized.

## A. Criteria split

- Exact item agreement between the extractor and the gold segmentation: precision and recall over items, matched by text after whitespace normalization.
- Secondary: the extractor's item count vs EmulatRx Table 2 #EC per trial (already in the reference JSON). Report counts, not a score.

## B. Concept-level precision, recall and F1 (primary endpoint)

Within each criterion, predicted and gold concepts are matched one-to-one (maximum bipartite matching).

- **Relaxed match:** the normalized name is equal (lowercase, punctuation removed, synonyms applied) and the domain is equal.
- **Strict match** (EmulatRx-comparable): relaxed, plus the concept's atom-level value (comparator, number, unit after normalization; qualitative text after normalization) and time constraint (relation, window amount and unit, normalized anchor) both equal the gold atom's. A concept inherits its atom's value and time constraint.

Rules:
- A failed parse (`parse_failed` in notes) counts every gold concept in that criterion as a false negative.
- Administrative criteria are included.
- Report micro precision, recall and F1 per model, pooled over all criteria of the 10 trials. Also report macro averages per trial.
- **Primary endpoint:** strict micro-F1 per model.
- **Uncertainty:** cluster bootstrap over trials (2,000 resamples, seed 20261024), giving 95% percentile intervals. Model comparisons use the paired bootstrap of the F1 difference. No p-values for model ranking.

Secondary measures, same matching:
- atoms per criterion, absolute difference from gold;
- value accuracy and time-constraint accuracy among relaxed-matched concepts;
- `category` accuracy (clinical vs administrative);
- `complexity` vs gold, as Cohen's κ;
- Spearman ρ between gold criterion complexity and errors per trial (EmulatRx reports ρ = 0.45).

## C. Agreement per TTE element

Each element of each (trial, model) protocol is scored **correct / partly correct / wrong / missing** against the reference protocol:

| Element | How it is scored |
|---|---|
| eligibility | The share of criteria whose concepts all strictly match and whose atom count is right. *Correct* ≥ 90%, *partly* 50–90%, *wrong* < 50% |
| treatment strategies | Automatic. Arms and comparator flag exact; components P/R on drug names; dose, frequency, course duration and end event exact after normalization. *Correct* = all exact; *partly* = arms and comparator right, some component field wrong |
| time zero | Two reviewers, blinded to the model (shuffled IDs), compare the anchor with the gold anchor and its list of acceptable alternatives on the 4-level scale. Report weighted κ; disagreements are adjudicated |
| follow-up | Automatic. Horizon, end event and end rule exact; censoring must not include death for all-cause mortality |
| outcomes | Automatic. The primary outcome is identified correctly, with an exact horizon and end event |
| causal contrast | Automatic. Value and origin (`registry` when stated, else `default`) both match gold |
| assignment, analysis plan | Phase 0 expects `origin: default`. Scored only on origin, to catch unsupported claims |

For each element, report the share correct per model. Also report the share of elements whose `origin` matches the gold origin. Secondary: Fleiss' κ across the three models for each element.

## D. Run-to-run stability

For each model, replicate 2 against replicate 1, with no gold:
1. **Identity:** the share of trials whose `content_hash()` is equal. This excludes version, status and provenance, so timestamps don't count.
2. **Concept F1 of replicate 2 against replicate 1:** relaxed and strict, matched as in B.
3. **Element agreement:** the share of TTE elements with the same part-C score in both replicates.
4. **Failures:** any `parse_failed` or retry that occurs in one replicate but not the other.

Exact repeatability isn't guaranteed even at temperature 0 with a fixed seed: Ollama batches up to 4 parallel requests, and GPU kernels aren't bit-exact. Report what is observed. Cross-GPU stability (for example H100 vs A100) is a separate, optional analysis and is never pooled with the main run.

## E. Operational measures

Per model and trial:
- LLM calls, cache hits (must be 0 in both replicates), and prompt and completion tokens, from the `[llm]` log line;
- the number of `parse_failed` and `dropped` events, and notes by type;
- wall time per run, from `manifest.txt` (`started`, `finished`).

Reasoning tokens (gpt-oss) are reported separately when the server exposes them, otherwise noted.

## Comparison with the EmulatRx paper

- **Concepts:** strict P/R/F1 (part B) are reported next to EmulatRx Fig. 2a (GPT-4o: P 96.7%, R 98.9%; Gemma 3: P 91.8%, R 88.0%). We state that the gold standards differ: theirs covers 20 trials and 266 concepts and isn't public.
- **Same-gold comparison (optional):** run the paper's local models (`phi4:14b`, `deepseek-r1:14b`, `gemma3:12b`) through our pipeline.
- **Split:** part A against Table 2.
- **Not compared in Phase 0:** effect estimates. They are listed in [reference/replay_trials.md](reference/replay_trials.md) for Phase 3.

## Outputs

`eval/score_phase0.py` (to be written against this document before any exit-run output is opened) writes one CSV per part plus a summary table with the bootstrap intervals. Annotators work from templates generated by `eval/make_gold_templates.py`.

## Changelog

- 2026-10-24: initial version (commit 586e128). The exit run ran at this commit.
- 2026-10-24, after the exit run and **before any accuracy scoring** (no gold exists yet). The amendments below were decided after seeing completion and replicate data, but not accuracy data.
  1. **Trial-level failures** (no protocol written; 9 of 20 gpt-oss:20b drafts, all HTTP read timeouts at 600 s). In parts B and C, every gold concept of that trial counts as a false negative and all 8 elements as *missing*. This is option (a). The trial-level failure rate per model is a reported result, next to the criterion-level `parse_failed` counts. A complete-case sensitivity analysis (trials every model completed in both replicates) is reported alongside, never instead.
  2. **No reruns of model-caused failures** in the pre-registered analysis: timeouts are the model's behaviour under the pre-registered settings. Infrastructure failures (preemption, node or server crash) may be rerun once, documented in this changelog. None occurred.
  3. **gpt-oss secondary arm** (option (c), exploratory). gpt-oss:20b with reasoning level `low`, at a later commit that fails per item instead of per trial on a timeout and caps output at 8,192 tokens. It is run under its own tag (`runs/exit-H100-gptoss-low/`), reported separately, and never pooled with the pre-registered run. Option (b), a longer timeout, is not used. gpt-oss averaged about 1,000 completion tokens per call, against about 170 for the other models, and stalled on the same criteria in both replicates (NCT00475852 I1, NCT04134403 E8). That points to runaway generation, which a longer timeout only prolongs.
  4. **Part D definitions made precise.** *Identical, all fields* compares `content_hash()`. *Substantive* identity ignores free text (time-zero rationale and evidence, notes, source spans, complexity labels), and ignores `atom_logic` over a single atom and `concept_logic` over a single concept, which select the same patients. Both are reported.
  5. **Deterministic parser defect found after the run.** Time frames with the unit first ("Randomization to Day 30", "Up to Day 14") gave no horizon, so follow-up is *missing* for NCT00475852 and NCT03258684 in every model. Fixed after the run. The primary analysis scores the outputs as produced; a re-derivation with the fixed parser (no LLM involved) is reported as a sensitivity analysis.
  6. **Reproducibility test** before any further benchmark run: OLLAMA_NUM_PARALLEL 4 vs 1, with Ollama restarted per replicate (`hpc/repro_test.sh`). If 1 makes replicates identical, every later run uses 1 and records it in `manifest.txt`.
- 2026-10-25, after the reproducibility test (commit df348a8, H100), **before any accuracy scoring**.
  7. **Finding.** With Ollama restarted before each replicate, replicates were identical on every field, for both qwen3:32b and medgemma:27b, at both OLLAMA_NUM_PARALLEL=4 and =1 (3 trials each; `eval/results/repro/`).
     - The exit run's replicate disagreement therefore came from **server state**: replicate 2 ran on a warm server after replicate 1.
     - The parallelism setting itself changes the outputs: np1 and np4 protocols differ substantively in 5 of 6 trial-model pairs (`eval/compare_runs.py`).
     - Outputs also appear to depend on which requests the server handled before a trial. That evidence is indirect: different commit and trial order.
  8. **Run settings for every scored run from now on:** OLLAMA_NUM_PARALLEL=1; Ollama restarted before every replicate; the model unloaded before every trial. All are recorded in `manifest.txt`.
     - Before the final run, an order-independence test (`hpc/repro_test.sh`: the same trials forward and reversed) must show identical protocols per trial.
     - If it does not, the final run uses one fixed trial order, and order dependence is reported as a limitation.
  9. **Final benchmark run.** The 586e128 exit run becomes the **pilot**. Its completion, operations and warm-server replicate disagreement are reported as findings; it is not scored for accuracy. Parts B and C are scored on the final run (replicate 1), at one commit that includes the post-pilot fixes (per-item failure handling, 8,192-token output cap, unit-first time frames).
     - Arms: qwen3:32b (`false`), alibayram/medgemma:27b (`none`), gpt-oss:20b (`none`), as pre-registered; plus gpt-oss:20b (`low`) as the exploratory secondary arm (amendment 3).
     - 10 trials × 2 replicates, one GPU type (H100), a fresh cache per replicate.
     - With identical replicates expected, part D becomes a check of determinism. Any remaining disagreement is reported, never averaged away.
  10. **gpt-oss timeouts under the final settings.** A call that runs away now ends at the output cap (or the 600 s timeout) as a criterion-level `parse_failed`, scored as missing (amendment 1). A trial-level failure, meaning no protocol at all, now only happens if the draft itself crashes, and is scored as all-missing. Both failure rates are reported per arm. No reruns of model-caused failures (amendment 2).
- 2026-10-25, order-independence test (commit 45dc961, g020 H100; `hpc/repro_test.sh`): qwen3:32b and medgemma:27b, 3 trials drafted forward and in reverse, with OLLAMA_NUM_PARALLEL=1 and the model unloaded before every trial.
  11. **All 6 trial-model pairs were identical on every field** in both orders. The gate in amendment 8 is passed. The final benchmark runs with these settings, in the standard trial order, with no fixed-order fallback needed.
- 2026-10-25, final benchmark run (commit 9e09bdf; `gpu` partition, nodes g032–g034, NVIDIA H100 NVL 95830 MiB; `runs/exit-H100-final/`; results `eval/results/phase0_exit-H100-final.md`). This is the run that parts B and C will be scored on, replicate 1.
  12. **Run facts recorded before any accuracy scoring.**
     - (a) All 80 drafts completed, with 0 criterion parse failures in every arm. gpt-oss:20b default used 105 calls per replicate (4 retries) against 101 for the other arms.
     - (b) Replicates were identical on every field in every arm (10/10 trials, 77/77 criteria), including replicate pairs that ran on different nodes.
     - (c) The GPU type differs from the pilot and the reproducibility tests (H100 80GB HBM3 on g020); the final run used one type throughout, as designed.
     - (d) Every manifest shows `restart_count=2`, inherited from the submitting OnDemand job's environment. The jobs ran once: their job IDs match `runs/exit-H100-final.jobs`, start and finish times are continuous, and cache hits equal the within-run repeats (2) exactly. Fixed for later runs in `hpc/exit_run_submit.sh`.

# Phase 3 Replay on MIMIC-IV: trial selection and agreement with the trials (pre-registered)

Fixed on 2026-10-27 (PLAN.md §17). The commit that adds this section is the pre-registration; cite its hash in the methods. Changes go in the changelog at the end of this section, with the date, the reason and what had been seen by then.

**What had been seen when this was fixed:**
- The MIMIC-IV 3.1 OMOP CDM (job 19365483) and its ETL report: aggregate counts only.
- The published results of the candidate trials, and EmulatRx's three case-study HRs.
- The Phase 2 synthetic Replay of three trials.
- **No effect estimate from MIMIC** had been computed by this project for any trial.

## R1. Data, pipeline and run settings

| Item | Fixed value |
|---|---|
| Data | MIMIC-IV 3.1 in OMOP CDM 5.4, `mimiciv-3.1-cdm54-19365483.duckdb` (built at commit 9cad08c, Athena v5.0 29-AUG-26, the reviewed item maps of 2026-10-27). A rebuild is a new data version and is recorded in the changelog |
| Primary model | `qwen3:32b`, think off. Its Phase 0 drafts are the start for the 10 parity trials; new trials are drafted under the Phase 0 run settings (amendment 8: OLLAMA_NUM_PARALLEL=1, a restart per replicate, the model unloaded per trial) |
| Pipeline | The LangGraph Replay (`agentic_tte/graph/replay.py`) at one commit, recorded in every manifest |
| Replicates | 1 for scoring. A second replicate checks determinism and is never averaged in |
| Reviews | Design, concepts and gate are reviewed **before** the analysis node runs. Each review is labelled with who did it: a clinician, the user, or `AutoApprove`. Only reviews by a named clinician are called clinician review |

## R2. Trial sets

- **Parity set:** EmulatRx's 10 MIMIC trials ([reference/replay_trials.json](reference/replay_trials.json)), all run end to end.
  - Scored on the pipeline endpoints (R5).
  - Scored on agreement (R4) only if a trial meets the selection rule (R3) on its own.
  - On the 2026-10-27 reference file, NCT00475852 and NCT02856698 have a verified numeric estimate. Four trials have no published result and are never scored for agreement.
  - NCT00475852's primary outcome includes heart-failure rehospitalization. Under S7 it can enter only the *secondary* agreement set.
- **Agreement set:** every trial, parity or new, that meets R3. This is the set the primary endpoint is computed on.

## R3. Trial-selection rule

**Candidates:**
1. The 10 parity trials.
2. The candidates named in PLAN.md §17.3 point 2.
3. A registry search, ClinicalTrials.gov API v2, run once and saved with its date:
   - interventional, randomized, status completed;
   - adults;
   - enrolment ≥ 100;
   - condition or keyword in: sepsis, septic shock, acute respiratory distress syndrome, acute respiratory failure, mechanical ventilation, acute kidney injury, delirium, critical illness, intensive care, acute heart failure.

Every candidate is screened against the same criteria. Each criterion is judged from the registry record and the primary publication only, never from MIMIC estimates.

| # | Criterion |
|---|---|
| S1 | Randomized and parallel-group (cluster and cluster-crossover allowed). The trial is superiority or non-inferiority |
| S2 | A published primary result: an effect estimate with a 95% CI, as HR, RR, OR or risk difference, for a binary or time-to-event primary outcome. Taken from the primary peer-reviewed paper or the CT.gov results, and checked against the source (`verification: verified`, with PMID or DOI and the table or figure) |
| S3 | Adult inpatients whose eligibility starts in a setting MIMIC records: an ICU stay, a hospital admission, or an ED visit leading to admission |
| S4 | The intervention and comparator are observable in structured data as a drug exposure or a procedure, or as their absence (placebo or usual care → not treated within the grace period) |
| S5 | **Dynamic strategies are excluded** from the agreement set: strategies defined by a physiological target (glucose, blood pressure, oxygen saturation, transfusion threshold). They may be run as exploratory |
| S6 | The primary outcome is **fully ascertainable** in MIMIC: all-cause death at a fixed horizon up to 365 days (deaths are recorded to one year after the last discharge), in-hospital death, or an event recorded in structured data during the stay (e.g. RRT, ventilation) |
| S7 | The outcome is not partly ascertainable. Outcomes needing other hospitals or later follow-up (rehospitalization, functional or patient-reported outcomes) put the trial in the *secondary* agreement set, never the primary |
| S8 | **Feasibility on MIMIC**, checked from counts only (the Data Support Gate): at least 50 eligible patients per arm after eligibility, and at least 20 outcome events pooled over arms. Counts are never split by arm and outcome before the analysis runs |

**Size:** all trials that pass. If more than 30 pass S1–S8, keep the 30 with the largest enrolment. Enrolment is fixed before any result, so this choice cannot depend on agreement.

**Screening log:** `eval/reference/replay_selection.json`. Every candidate goes in, with the criteria it passes or fails and the reason, and is reported as a flow diagram.

**Reference values:**
- The trial's primary estimate, in its own measure and orientation (intervention vs comparator as the trial defines them), with its 95% CI.
- Each value is checked by a second person before any emulation of that trial. Disagreements are settled against the source.

## R4. Agreement with the trial (agreement set)

**Measure:** the emulation reports the trial's own measure.
- HR from the weighted Cox model.
- RR, OR or risk difference from the weighted risks at the trial's horizon, with 95% CIs by nonparametric bootstrap: 500 resamples, seed 20261027, refitting the weights in every resample. This bootstrap must be implemented and tested before any agreement-set trial is analysed.
- Published estimates are never converted to another measure.

**Analysis:** the primary analysis is the one in the signed-off report, chosen by the pipeline's rules before the estimate exists:
- clone-censor-weight when the protocol has a grace period, IPTW otherwise;
- the ITT analogue.

Once an estimate exists, any change to the protocol, concepts or analysis makes a new protocol version. It is reported as post hoc and is never scored as the primary result.

**Metrics per trial** (as in RCT-DUPLICATE). Here *E* is the emulation and *R* the trial, on the log scale for ratios and the linear scale for differences, with standard errors taken from the 95% CIs.

| Metric | Definition |
|---|---|
| Standardized difference agreement | \|z\| < 1.96, where z = (θ_E − θ_R) / √(SE_E² + SE_R²) |
| Estimate agreement | θ_E inside the trial's 95% CI |
| Regulatory agreement | Same direction and same significance at 0.05 (both significant the same way, or both not significant) |
| Direction agreement | Same sign of θ, or both CIs containing the null |
| Ratio of ratios (ROR) | exp(θ_E − θ_R), with its 95% CI (for ratio measures) |

**Primary endpoints** (agreement set, primary outcome only):
1. The share of trials with standardized difference agreement, with an exact (Clopper–Pearson) 95% CI.
2. The pooled ROR over ratio-measure trials: random-effects meta-analysis (REML), with its 95% CI and τ².

Estimate, regulatory and direction agreement are secondary.

The emulation's own precision (CI width) is reported next to every agreement call, because a wide emulation CI makes standardized difference agreement easier.

**Failures count:** a trial in the agreement set that does not reach an estimate is scored as no agreement on every binary metric:
- the pipeline failed;
- the gate was red;
- feasibility fell below S8 at analysis;
- the analysis did not converge.

The reason is reported. A complete-case analysis is reported alongside, never instead.

**Secondary agreement set** (S7) and **dynamic-strategy trials** (S5): the same metrics, reported separately and never pooled with the primary.

## R5. Pipeline endpoints (all trials: parity and agreement sets)

- **End-to-end completion:** a signed-off report or an auto-approved one, labelled.
- **Criteria:**
  - share compiled;
  - share reported as not compilable, with reasons;
  - none dropped silently.
- **Concept sets:**
  - concepts proposed, approved, replaced and rejected at review;
  - the reviewer's edit count per interrupt (design, concepts, gate).
- **Gate:** green, amber or red, and the checks that failed.
- **Cohort:**
  - N per arm and attrition per criterion (counts below 11 suppressed);
  - maximum \|SMD\| after weighting (target < 0.1);
  - effective sample size per arm;
  - the immortal-time flag;
  - the proportional-hazards test;
  - the E-value.
- **Operations:** LLM calls, tokens and wall time per node, from the manifests.
- **Concepts:** parsing F1 against the gold annotation (Phase 0 parts B and C) once `eval/gold/` exists. It is not reported before that.

## R6. Comparison with EmulatRx

- **The three case-study trials:** our estimate next to EmulatRx's, with the RCT result where one exists. No agreement call is made for a trial without a published result:
  - NCT00475852: HR 0.59 (0.46–0.76);
  - NCT06091982: HR 0.64;
  - NCT04134403: HR 1.30 (1.07–1.58).
- **Parity set:** compile success and SQL failures per criterion against EmulatRx's error counts (their Table 2). The definitions differ (their error types vs our not-compilable reasons), and this is stated.

## R7. Planned secondary analyses

These are named now, and each is fully specified in this changelog before it runs:
- **Ablations:**
  - without the Protocol Auditor;
  - without the Clinician;
  - LLM-written SQL instead of the compiler.
- **Model comparison:** the other Phase 0 models.
- **Sensitivity of the primary estimate:** PSM instead of IPTW, the grace period ±50%, and the horizon as in the trial vs 28 days.
- **Negative-control outcomes:** one per trial, chosen before its analysis.

## Phase 3 changelog

- 2026-10-27: initial version.
- 2026-10-27, after the registry search and **before any MIMIC estimate**: how R3 is applied (`eval/replay_search.py`, `eval/replay_screen.py`; log in `eval/reference/replay_selection.json`).
  1. **Search run once** (2026-10-27, 1,184 hits, saved as `eval/reference/replay_search_2026-10-27.json`). The registry lists `NCT06091982`, an EmulatRx parity trial, without a randomized design.
  2. **Stage A, automatic, search hits only:**
     - S1 fails unless the design model is parallel, factorial or crossover;
     - S3 fails if the minimum age is under 18 or not given;
     - S4 fails if no intervention is a drug, biological, procedure, dietary supplement or combination product;
     - S2 fails if the record links no result publication and posts no primary analysis with a CI.

     The S2 screen uses only what the registry links, so trials with unlinked papers are missed. The named candidates and parity trials skip stage A for that reason.
  3. **Stage B, judged from the registry record:** S1–S7 for the 114 search hits left after stage A, the 10 parity trials and the 12 named candidates.
     - Investigational agents never used in US practice fail S4.
     - Items marked unmappable in the 2026-10-27 MIMIC item reviews (LR, blood components, enteral formulas, IV amino acids, delirium assessments, patient position) make the dependent trials fail S4 or S6. Revisiting those items is a data change, recorded here.
     - Status *uncertain* names the one criterion still to check.
  4. **Who screened:** Claude (AI), 2026-10-27. Every decision is to be checked by the user before any trial enters a set.
     - S2 values are extracted from the primary paper and checked by a second person.
     - S8 comes from Data Support Gate counts on Hellbender.
- 2026-10-27, **wave 2 and the coverage rule**, decided after the first screening (16 tentative primary trials, below the 20–30 target) and **before any MIMIC estimate**. The user approved them. R3's criteria S1–S8 are unchanged.
  5. **Wave-2 search, run once:** the same query as R3, with the condition and keyword list widened by:
     - shock, cardiogenic shock, cardiac arrest, critically ill;
     - pneumonia, gastrointestinal hemorrhage, acute pancreatitis;
     - cardiac surgery, pulmonary embolism;
     - subarachnoid hemorrhage, intracerebral hemorrhage, traumatic brain injury.

     Saved with its date. Wave 1 and wave 2 are screened together.
  6. **Result linkage through PubMed (S2 screen):** a trial passes the stage-A S2 screen if:
     - the registry links a result publication, or
     - it posts a primary analysis with a CI, or
     - PubMed lists an article carrying its NCT number (`<NCT>[si]`) with publication type "Randomized Controlled Trial".

     This applies to every hit of both waves that passes the other stage-A screens, so trials whose paper is not linked on CT.gov are no longer missed. The links are saved.
  7. **Items unmappable in the MIMIC reviews:** a trial that fails S4 or S6 only because of such an item has status *conditional* and names the item.
     - It joins a set only if the item is later mapped by a reviewed decision and the CDM is rebuilt. That is a data change, recorded here before any estimate.
     - The candidate items are LR and other balanced crystalloids, delirium assessments (CAM-ICU), patient position (prone) and IV amino acids.
  8. **Coverage rule:** each pipeline feature below should have at least 3 trials in the primary agreement set.

     | Feature | Meaning |
     |---|---|
     | F1 | Placebo or usual-care comparator, emulated as not treated |
     | F2 | Active comparator (drug vs drug) |
     | F3 | Procedure intervention |
     | F4 | Timing strategy with a grace period |
     | F5 | Composite outcome built from structured data |
     | F6 | Lab-defined outcome |
     | F7 | Mortality horizon other than 28 days |

     A trial can cover several features. A feature with fewer than 3 eligible trials is reported as a finding.

     **Selection when more than 30 pass:**
     1. For each feature, take its 3 largest eligible trials by enrolment.
     2. Fill up to 30 by enrolment.

     Enrolment and features are known before any result, so the choice cannot depend on agreement.
  9. **Wave-2 screening run** (2026-10-27, before any MIMIC estimate).
     - **Numbers:** 2,301 registry hits, 2,309 candidates in all. 622 trials were linked through PubMed.
     - **Outcome rule B1:** a search hit not judged in the first screening goes to stage B only if its first primary outcome is death or a structured in-hospital event. The rule is a fixed pattern (`outcome_rule_b1` in `eval/replay_screen.py`); it kept 149 of 720 such hits.
     - **Tentative result:**
       - 37 primary trials, 30 selected by the item-8 rule;
       - 2 secondary, 19 exploratory, 7 conditional, 34 uncertain.
     - **Coverage:** F5 (composite) and F6 (lab-defined) have 2 trials each, below the target of 3. The conditional trials SMART (F5) and PROTection (F6) would fill them if their items are mapped.
     - **Selection is recomputed, by the same outcome-blind rule, whenever a trial leaves the primary set:**
       - S2 not verified;
       - S8 below threshold;
       - or an uncertain or conditional trial resolved.

       Every recomputation is logged here.
  10. **S2 extraction** (2026-10-27, before any MIMIC estimate; `eval/reference/replay_reference_values.json`).
      - **What was done:** Claude extracted the primary estimate from PubMed abstracts for the 30 selected trials. Each value awaits the user's check against the full paper and a second person (R3).
      - **Result:**

        | Status | Trials | Which |
        |---|---|---|
        | 95% CI found | 21 | — |
        | CI at another level | 2 | PREVENT CLOT 96.2%, PRATO-ACS-2 90% |
        | No estimate in the abstract | 4 | VISEP, VANCS II, AKIKI, NCT03872011 |
        | No primary results paper | 3 | DISCO (pilot only), ALBIOSS-BALANCED, NCT01223690 |

      - **Proposed rule, for the user to confirm:** when a paper reports its primary CI at a level other than 95%, R4 takes the standard error from that interval with the matching normal quantile (2.075 for 96.2%, 1.645 for 90%). Nothing is converted to another measure.
      - **Correction:** the registry links HYDRAREA (NCT01636089) to an outpatient CT trial (PMID 32065601). The matching paper is PMID 28181941, pending the user's confirmation.
      - **What happens next:** trials confirmed as *not_found*, or still without an estimate after the full-text check, leave the primary set. The selection is then recomputed (item 9).
  11. **Core set, stage-wise correctness and fault injection** (2026-10-27, adopted by the user, **before any MIMIC estimate**).
      - **Why:** the contribution is a pipeline shown correct stage by stage, not a trial count.
      - **Constraint:** clinician time is scarce. Every endpoint below either needs no human reference, or needs only the user plus a small clinician audit.

      **11a. Core set (the primary agreement set from now on).** Taken from the 30 selected trials (item 9) by an outcome-blind rule:
      1. Keep only trials whose S2 estimate the user has verified against the full paper.
      2. Apply the item-8 coverage rule with a cap of **12**: up to 3 per feature, then by enrolment.

      The other selected trials form the **breadth set**. It is run end to end and reported on the R5 pipeline endpoints only, with no agreement claim. R4's primary endpoints are computed on the core set.

      **11b. Stage-wise endpoints.**

      | # | Stage | Reference (who makes it) | Metrics |
      |---|---|---|---|
      | E1 | Design checks: Protocol Auditor and review gates | **Fault injection** (by construction, no human) | Sensitivity per fault type; false alarms on the unmodified protocols; bias each fault causes in the estimate |
      | E2 | Cohort (compiler) | No hand-built cohorts: (a) independent validated definitions, (b) metamorphic tests, (c) differential testing (no clinician) | Concept-level agreement (κ, patient overlap counts, suppressed); share of metamorphic tests passed; patient-set Jaccard and adjudicated causes of difference |
      | E3 | Protocol (Trialist) | A reference protocol per core trial, written by the user from the trial paper with a fixed template, blind to model outputs. A clinician adjudicates only the items the user marks uncertain, plus a random 20% audit of fields | The 7 TTE elements scored as in Phase 0 part C. Reported as *investigator-written with clinician audit*, never as clinician gold |
      | E4 | Estimation | Synthetic data with known marginal truth (no human) | Bias, coverage, including clone-censor-weight and the E1 faults |
      | E5 | End to end | The published estimates (user-verified) | R4 metrics on the core set, plus attribution (11d) |
      | E6 | Breadth | None | R5 pipeline endpoints on the breadth set |

      **E2 references, in detail:**
      - (a) For criteria with a validated community definition, the compiled cohort is compared with MIT-LCP `mimic-code` derived concepts on the native-schema Postgres build: sepsis-3, KDIGO AKI, SOFA, ventilation, vasopressors.
      - (b) Metamorphic tests: dropping an inclusion criterion never shrinks the cohort; adding an exclusion never grows it; shifting all times leaves the cohort unchanged; criterion order does not matter.
      - (c) Differential testing against an independent implementation: LLM-written SQL as in EmulatRx, on the same OMOP data. The user adjudicates the causes of disagreement on aggregate counts only.

      **11c. Fault catalogue for E1, fixed now.** Each fault is applied to every core protocol where it is applicable:
      - F-a immortal time: treatment defined after time zero with no grace-period handling;
      - F-b look-ahead: an eligibility criterion uses data after time zero;
      - F-c time zero after treatment start;
      - F-d comparator swapped (active ↔ placebo or not treated);
      - F-e outcome horizon different from the trial's;
      - F-f prevalent users: prior treatment not excluded;
      - F-g death treated as censoring for an all-cause mortality outcome;
      - F-h a confounder measured after time zero;
      - F-i a timing strategy without a grace period;
      - F-j a concept set from the wrong drug class or condition.

      Ground truth is known by construction. Unmodified protocols measure false alarms.

      **11d. Attribution of disagreement by substitution.** For each core trial, the estimate is recomputed with the reference protocol (E3) substituted for the agent's. If agreement with the trial changes, the cause is placed in the protocol stage. Otherwise it lies in the cohort, confounding or the data.

      **Human budget:**
      - **User:** verify about 12 estimates (about 2 hours); write 12 reference protocols from a template (about 30–45 minutes each).
      - **Clinician:** about 2–3 hours of adjudication and audit.
      - **CREST-2,** with the user's existing clinician-adjudicated reference, stays the one fully clinician-validated case (PLAN.md §17, after MIMIC).
  12. **The user's S2 check and the final lists** (2026-10-27, before any MIMIC estimate; `eval/reference/replay_reference_values.json`, `replay_selection.json`).
      - **Verified by SKD:** 22 estimates. Corrections from the full texts:
        - HYDRAREA (NCT01636089) is PMID 28181941, which states the registration; its population is modified ITT (307 of 320).
        - NCT03872011's value is the ITT HR 1.08 (0.80–1.46). The paper does not name per-protocol as primary; the per-protocol HR is 1.07 (0.79–1.46).
      - **Fail S2,** leave the primary set, and are reported in the manuscript:
        - **No estimate with a CI:** VISEP (arm rates only), VANCS II and AKIKI (no estimate with a CI found).
        - **No primary results paper:** DISCO (pilot only), ALBIOSS-BALANCED, NCT01223690.
        - **Primary CI not at 95%:** PREVENT CLOT (96.2%) and PRATO-ACS-2 (90%). The user chose to drop them rather than rescale (the item-10 proposal is withdrawn).
      - **Selection recomputed (item 9):** 29 primary trials remain, all selected (cap 30). 22 have a verified estimate.
      - **Core set (item 11a, rule applied to the 22):** CHEST, ADRENAL, SUP-ICU, STARRT-AKI, CRISTAL, RSI, SOAP II, VIOLET, LOVIT, ECLS-SHOCK, BICAR-ICU, HYDRAREA.
      - **Core coverage:**

        | Met (3 or more) | Below target |
        |---|---|
        | F1 (6), F2 (5), F7 (6) | F3 (2), F4 (1), F5 (2), F6 (1) |

        The trials that would fill F3–F6 failed S2 (AKIKI, DISCO, PRATO-ACS-2) or are conditional on unmapped items (SMART, PROTection). This is reported as a finding.
      - **Breadth set:** the other 17 selected trials (10 with verified estimates).
      - **Still open:** the second-person check of R3.
  13. **Core rebalanced and breadth agreement added** (2026-10-27, by the user's decision, **before any MIMIC estimate**).
      - **Why:**
        - The item-12 core held one trial with a significant published result. Agreement between two null results is easy, so that core could barely show errors.
        - The trials' published results were public before any selection, so a rule based on them cannot depend on our emulations.
      - **Core rule (replaces 11a step 2):**
        1. Among the selected trials with a verified estimate, the trials whose published 95% CI excludes the null (1 for ratios, 0 for differences) enter first, largest first, up to 6.
        2. Then the item-8 coverage rule fills the core to 12, counting the trials already in towards each feature.
      - **Core (12):**

        | Group | Trials |
        |---|---|
        | Significant (5) | APROCCHSS, CAPE COD, LOVIT, EvK, ACURASYS |
        | Null (7) | CHEST, ADRENAL, STARRT-AKI, CRISTAL, ECLS-SHOCK, BICAR-ICU, HYDRAREA |

      - **Core coverage:**

        | Met (3 or more) | Below target |
        |---|---|
        | F1 (7), F2 (4), F7 (7) | F3 (2), F4 (1), F5 (2), F6 (1) |

      - **Breadth agreement:** the other selected trials with a verified estimate get the R4 metrics as a **secondary, exploratory** result.
        - They are reported separately, never pooled with the core, and never used for the primary endpoints.
        - The trials (10): SUP-ICU, RSI, SOAP II, VIOLET, ROSE, BICARICU-2, NCT03872011, CATS, DESIRE, NCT01284452.
      - **Breadth without agreement (7):** the remaining breadth trials have no verified estimate and get the R5 pipeline endpoints only.
  14. **E1 built and checked on synthetic data** (2026-10-27, before any MIMIC estimate).
      - **Code:**
        - `agentic_tte/faults.py`: the 10 faults;
        - `agentic_tte/agents/auditor.py`: the checks A-a to A-j, one per fault;
        - `eval/fault_injection.py` → `eval/results/fault_injection_synthetic.md`.
      - **F-h reinterpreted:**
        - The compiler reads every confounder at or before time zero by construction (tested in `tests/test_compiler.py`), so "a confounder measured after time zero" cannot be expressed in a compiled design.
        - F-h therefore plants the nearest expressible error: the treatment itself as a confounder.
        - The structural guarantee is reported as such.
      - **Result on synthetic data** (n = 3,000, treatment within a 3-day grace period, truth: marginal HR 0.590):
        - 10 of 10 faults caught by their own check, and no alarm on the correct design.
        - Immortal-time faults (F-a, F-i): HR 0.496 against 0.612 for the correct design, the expected bias toward benefit.
        - F-c (time zero after treatment start) and F-j (wrong concept set): no estimate (too few treated).
        - F-d (arms swapped): no estimate (the frame refuses an active comparator).
        - F-b, F-e, F-f, F-g and F-h leave the synthetic estimate unchanged. The synthetic design hides their effect: every patient meets I1, no one has the treatment before time zero, the data end at 28 days, death is never censored by the compiler, and the treatment confounder is constant. Their bias is to be measured on the MIMIC core trials.
      - **What this does and does not show:**
        - It shows that each guard works on a design where the truth is known.
        - It does not show the false-alarm rate on real protocols. That is measured on the core trials' reference protocols (E3) and the Trialist drafts before any claim. A-f in particular will flag every protocol without an explicit new-user exclusion.
  15. **R4 bootstrap built** (2026-10-27, before any MIMIC estimate; `agentic_tte/stats/bootstrap.py`).
      - **Risks:** the risks under each strategy at the trial's horizon come from the primary method's weighting:
        - clone-censor-weight: weighted KM on each arm's clones, in counting-process form;
        - otherwise IPTW: weighted KM per arm, stabilized ATE weights.
      - **Measures:** RR, OR of the cumulative risks, and RD.
      - **CIs:**
        - nonparametric bootstrap over patients, refitting the weights in every resample;
        - 500 resamples, seed 20261027 (the `replay start` default);
        - percentile intervals, with ratios taken on the log scale.
      - **Failed resamples:** a resample where a measure is undefined (no event in an arm) or a refit fails is dropped and counted. If more than 10% are dropped, that measure gets no CI rather than a biased one.
      - **Check on synthetic data** (n = 1,500, 100 resamples): the CIs cover the sample's potential-outcome truth for RR and RD, with IPTW (treatment at time zero) and with clone-censor-weight (3-day grace period).
      - **PSM:** a PSM primary analysis reports these measures with IPTW weighting, and says so, because PSM's estimand is the ATT.
  16. **E2 (b) and (c) built** (2026-10-28, before any MIMIC estimate; `agentic_tte/omop/cohort_checks.py`, `agentic-tte cohort check`).
      - **(b) Metamorphic relations, fixed now:**
        - dropping any compiled criterion gives a superset;
        - reversing the criteria gives the same cohort;
        - repeating a criterion gives the same cohort;
        - doubling a look-back window: superset for an inclusion, subset for an exclusion;
        - a measurement threshold 10% stricter: subset for an inclusion, superset for an exclusion;
        - moving every date by 400 years (a whole Gregorian cycle): the same cohort and the same frame totals.

        The metric is the number of patients who move the wrong way, which must be 0.
      - **(c) Differential testing, change of implementation:**
        - The independent implementation is a Python reference evaluator. It re-derives time zero, the index population and every compiled criterion from the raw OMOP tables, following the compiler's documented semantics, and is compared patient by patient.
        - Criteria it does not cover (anchors other than time zero, cumulative durations) are reported as not compared, never as agreeing.
        - LLM-written SQL as in EmulatRx stays planned as a second implementation. It needs the GPU (Hellbender).
      - **Checks of the checks:**
        - On synthetic data, 13 of 13 relations hold and 0 patients disagree on 6 criteria. On the compiler's hand-built edge cases (date-only records, the latest lab in the look-back, death before time zero, a minor), 0 disagree.
        - A compiler with one predicate negated is caught by both kinds of check (`tests/test_cohort_checks.py`).
      - **(a) The mimic-code comparison is not built yet.** It needs the derived-concepts schema of the native Postgres build. Its 2026-10-26 build was attempted but not checked (`hpc/HELLBENDER.md`), so the user first confirms that the schema exists.
      - **On MIMIC:** E2 runs per core trial once its protocol and concept sets are reviewed, on Hellbender. Only the suppressed counts come back.
  17. **E2 (a) built** (2026-10-28, before any MIMIC estimate; `eval/e2a_mimic_code.py`, `hpc/export_mimic_derived.sbatch`).
      - **The reference exists:** the user's mimic-code derived-concepts build finished with 0 errors.
      - **Five definitions, fixed now, for each patient's first ICU stay:**
        - age >= 65;
        - creatinine > 2.0 mg/dL on day 1;
        - lactate > 2.0 mmol/L on day 1;
        - invasive ventilation started on day 1;
        - norepinephrine started on day 1.

        Each is written as a compiled OMOP criterion (time zero at ICU admission, or ICU admission + 24 h, with mimic-code's first-day windows) and compared with `icustay_detail`, `first_day_lab`, `first_day_bg`, `ventilation` and `norepinephrine`.
      - **Metrics per definition:** both, ours only, mimic-code only, neither, and Cohen's κ, plus the patients with a time zero on one side only.
      - **Expected sources of disagreement, named before the run:**
        - anchor-based age at hospital admission vs our ICU-admission age;
        - procedureevents vs ventilator settings;
        - prescriptions counted in our norepinephrine;
        - deaths in the first 24 hours.
      - **Checked** on a made-up MIMIC-shaped database (`tests/test_e2a_mimic_code.py`).
  18. **E2 (a) result on MIMIC-IV** (2026-10-28, job 19373181; `eval/results/e2a_mimic_code.md`). This is a cohort check: no effect estimate was computed.
      - **creat2:** agreement is exact (κ = 1.0, 64,703 patients).
      - **The other disagreements match their named sources:**

        | Definition | κ | Disagreement |
        |---|---|---|
        | age65 | 0.976 | 773 positive by mimic-code only (anchor age) |
        | lact2 | 0.82 | 5,269 positive by ours only (more lactate sources) |
        | norepi24 | 0.95 | 732 positive by ours only (prescriptions) |
        | ivent24 | 0.715 | both directions (procedureevents vs ventilator settings) |

      - **No day-1 time zero:** 663 patients, who died or left before ICU admission + 24 h.
      - **Planned follow-up:** a by-source breakdown of the lactate and ventilation disagreements, as aggregate counts. It is reported as an adjudication; no definition is tuned to agree.
  19. **E2 (a) breakdown built** (2026-10-28; `eval/e2a_mimic_code.py --breakdown`). Within each disagreeing group it counts patients per source category, from each OMOP row's source value (the MIMIC item ID, or a drug name for prescriptions):
      - **lact2, ours only:** the items that gave our qualifying values.
      - **ivent24, ours only:** what mimic-code's ventilation table holds for the stay:
        - an invasive episode already running at ICU admission, or one starting after 24 h;
        - tracheostomy on day 1;
        - other support only;
        - no row.
      - **ivent24, mimic-code only:** whether our invasive-ventilation record starts before ICU admission, starts after 24 h, or is absent.
      - **norepi24, ours only:** the sources of the day-1 records.

      Counts only, small cells suppressed.
  20. **E2 (a) breakdown result** (job 19374062; `eval/results/e2a_mimic_code.md`).
      - **Lactate:** the source explanation named in item 18 is refuted. Every disagreeing patient's value is lab item 50813, mimic-code's own item, so the gap lies in mimic-code's blood-gas selection. A second breakdown checks it.
      - **Ventilation:** explained by timing (episodes that began before ICU admission) and by source (procedureevents vs ventilator settings). For trials that need ventilation, the concept review should consider the charted ventilator settings.
      - **Norepinephrine:** the first breakdown missed date-only records; fixed, together with the breakdown, which now places them at midnight like the compiler and labels them.
  21. **Second breakdown** (job 19374198).
      - **Lactate, explained:**
        - 3,817 of the 5,269 qualifying values have no hadm_id (emergency-department blood gases before admission);
        - 3,753 lie only in the 6 hours before ICU admission;
        - mimic-code's lactate_max is missing for 3,613 of the patients and <= 2 for 1,656.

        mimic-code counts only labs linked to the admission; our ETL keeps labs without one (PLAN.md §16). This is a definition choice, reported as such.
      - **Norepinephrine:** the breakdown matched the ingredient only, while the compiler also matches descendants (prescriptions map to clinical drugs). The breakdown now expands concept sets the same way.
  22. **E2 (a) closed** (job 19374310).
      - **Norepinephrine:** 714 of the 732 patients positive by ours only have a day-1 prescription; 18 have only an infusion that mimic-code does not list.
      - **Every E2 (a) disagreement is traced to a named data-source or definition difference; none to the compiler** (`eval/results/e2a_mimic_code.md`, summary).
      - **For concept review:** the ventilation, lactate and vasopressor definitions of the core trials are reviewed with these numbers.
  23. **Core drafting and E3 templates** (2026-10-28, before any MIMIC estimate).
      - **Drafts** (`hpc/core_draft_submit.sh`):
        - the 12 core trials, read from `replay_selection.json`;
        - qwen3:32b with think off (R1), 2 replicates;
        - the Phase 0 scored settings: OLLAMA_NUM_PARALLEL=1, a restart per replicate, the model unloaded per trial, a fresh cache per replicate;
        - tag `core`, written to `runs/exit-<GPU>-core/`.

        Replicate 2 checks determinism only.
      - **E3 templates** (`eval/make_reference_templates.py` → `eval/reference_protocols/templates/`): protocol skeletons in our schema, built from the registry record only.
      - **Blinding:** the investigator writes each reference from the trial paper before opening that trial's draft. The README states this, and the references record who wrote them. A reference must pass `--check` (valid and complete) before E3 or the attribution analysis uses it.
  24. **Second annotation of the E3 references** (2026-10-29, before any model draft or MIMIC estimate is seen, and before the second annotator reads the investigator's references).
      - **Why:** clinician time does not scale; a second, independent annotation makes the reference's reliability measurable instead of assumed.
      - **Annotators:**
        - investigator (SKD): `eval/reference_protocols/<NCT>.SKD.json`, written from the trial paper (item 23);
        - LLM second annotator (Claude Opus 5.5, the coding assistant that built the pipeline): `<NCT>.AI.json`, written from the same papers (`papers/`, git-ignored) and the same templates, without reading the investigator's references or any model draft.
      - **Limitation, stated in the paper:** the second annotator is not independent of the pipeline's design and may share its conventions, so agreement can be inflated. It is reported as an LLM annotation, never as a clinician's.
      - **Agreement:** element by element (the 7 TTE elements, as in Phase 0 part C, plus criterion atoms as in part B), reported before adjudication.
      - **Adjudication:** the investigator resolves each disagreement first; the clinician reviews the disputed items plus the random 20% audit of item 11b. The adjudicated protocol is the E3 reference.
      - **Reported as:** *investigator-written reference protocols with an independent LLM second annotation; disagreements adjudicated, with clinician audit*. Never as clinician gold.
  25. **Reference comparison measures** (2026-10-29, `eval/compare_references.py`, `eval/results/e3_reference_agreement.md`).
      - **Pre-registered** (Phase 0 parts B and C): strict and relaxed concept F1, category agreement and κ, arms, comparator, horizon, end event, contrast and estimand. Time zero is rated by hand at adjudication.
      - **Added after the first run, and reported as such:**
        - *strict as compiled*: no time constraint counts as "at time zero" and units are ignored, because the compiler reads them that way. The first run's strict F1 (2%) mostly counted that notation difference (one annotator left "at time zero" implicit, the other wrote it).
        - *lenient names*: one-to-one matching of concept names that differ only in wording.
        - A 0-hour grace period equals none, as the Statistician reads it.
      - Every disagreement, under any measure, is listed for adjudication; none is dropped.
  26. **Initiation before time zero** (2026-10-31; adjudication of HYDRAREA).
      - **Problem:** HYDRAREA's hydration starts 1 hour before contrast, the event that defines time zero. The compiler counted treatment only from time zero, so no patient would be treated; moving time zero to the start of hydration would put the contrast criterion after time zero (look-ahead).
      - **Change:** `TreatmentStrategy.initiation_before` (a duration, with evidence). Initiation in that window before time zero counts as initiation at time zero (`treat_start` 0). Assignment still uses only data up to time zero.
      - **Auditor A-c** ignores treatment records inside the stated window and still fails on earlier ones.
      - **Rule:** a prior-treatment (new-user) exclusion must end before the window, or it removes the patients the window counts as treated.
      - **Scope:** only arms that state the window. No other core trial uses it; the Trialist does not propose it yet.
  27. **Adjudication completed and adjudicated references built** (2026-10-31).
      - **Who decided:** the 7 grace periods by the investigator (SKD), accepting or correcting the second annotator's proposal; the other 94 items (12 time-zero ratings, 70 concept and category items, 12 confounder lists) by the LLM second annotator at the investigator's request. It wrote one side of every pair, so it is an interested party: those 94 decisions carry `audit: true` and all go to the clinician, not a 20% sample. The 7 investigator decisions enter the random 20% audit.
      - **Outcome:** 101 items: 71 keep the investigator's version, 17 take the second annotator's, 13 set a value (grace periods, HYDRAREA's initiation window, confounder additions and one removal). Five items are marked for clinician review in their notes (CRISTAL I3, CHEST E6, BICAR-ICU E5, CAPE COD I13, EvK's neuromuscular-blocker confounder).
      - **One correction to an investigator choice:** CAPE COD's confounder "time from severity criterion to treatment" is measured after time zero and was removed (post-baseline adjustment).
      - **Build:** `eval/build_adjudicated_references.py` writes `<NCT>.adjudicated.json` = the investigator's reference with each decision applied; the SKD and AI files stay as written. All 12 pass `--check`. These are the E3 references, reported as *investigator-written, with an LLM second annotation; disagreements adjudicated, with clinician audit*.
      - **Known non-compilable criteria,** reported and never dropped: ADRENAL I5 and CAPE COD E14 (cumulative duration), ADRENAL I2 (needs a derived SIRS count).
  28. **E3 scored for qwen3:32b** (2026-10-31; run `runs/exit-H100-core`, 12 core trials, 2 replicates; `eval/score_e3.py`, report `eval/results/e3_qwen3_32b.md`).
      - **Method:** replicate 1 against `<NCT>.adjudicated.json`, using the same comparison functions that compared the two annotators. Replicate 2 is checked for identity only (12/12 identical). Time zero goes to a rating sheet (`eval/results/e3_time_zero_ratings.json`), never scored automatically; it still needs a rater.
      - **Registry-structured criteria** (`S*`, e.g. minimumAge in APROCCHSS and LOVIT) have no counterpart in the references, whose templates hold free-text criteria only: listed, not scored.
      - **Two scoring fixes before the results were read as final:** (a) a draft that calls the treatment arm a comparator had its grace period counted as right and its components skipped; treatment arms are now the reference's, whatever the draft calls them (grace 8/12 became 6/12). (b) `parse_duration` read "Day 0 to Day 28" as 0 days; it now takes the last unit-first mention. The scored BICAR-ICU draft keeps its 0-day horizon as generated.
      - **Scope, stated with every E3 result:** the drafts have no confounders because the Trialist does not propose them (no Clinician agent yet), and the estimand is the Trialist's hazard-ratio default (no risk ratio in the schema). These measure missing parts of the pipeline, not wrong judgements, and must not be reported as the agent choosing badly.
      - **Time zero rated** (2026-10-31) by the LLM second annotator at the investigator's request: same 6/12, acceptable alternative 4/12, different 2/12 (CAPE COD anchors at study-drug start, which the comparator arm never has; ECLS-SHOCK at shock onset instead of angiography before revascularisation). It did not write the drafts but wrote one side of the references and adjudicated them, so all 12 ratings carry `audit: true` for the clinician. `score_e3.py` refuses a rated sheet whose anchors differ from the run's.
  29. **Clinician agent: confounder proposals** (2026-11-01; `agentic_tte/agents/clinician.py`, prompt `clinician-v1`). Fixed before any Clinician output on a core trial was seen.
      - **Literature, result-blind:** PubMed searches built from the protocol (prognostic factors of the main registry condition; observational studies of the treatment), human and, for adult trials, adult studies only, no case reports or opinion pieces; plus the registry's BACKGROUND references. Left out, each with its reason: the trial's own RESULT and DERIVED references, any article naming its NCT ID or (4+ letter) acronym, and anything published after its primary completion date. The literature is built on a machine with internet (`agentic-tte literature`) and kept under `runs/` (abstracts are publishers' text: never in git).
      - **Agent:** BM25 retrieval (15 passages, at most 3 per abstract; deterministic), one schema-constrained call. Deterministic checks: a quote must occur verbatim (case and whitespace ignored, at least 4 words) in the cited abstract or the citation is dropped; the treatment itself (F-h), the primary outcome and variables named as measured after time zero are dropped; duplicates are merged; age and sex are always included. Every dropped item is listed on the sheet with its reason.
      - **In the Replay:** with `--literature`, the Clinician proposes confounders after the draft (a new version, origin `agent`) unless the protocol already names some; the sheet is shown at design review, where approving keeps the list and `confounders` replaces it. Under `AutoApprove` the proposal stands unreviewed and the report says so.
      - **How it is scored:** the frozen E3 result (item 28) is not re-scored. Confounder recall and precision of the Clinician's lists against the adjudicated references are a separate, labelled result ("Trialist + Clinician").
      - **Disclosure:** the Clinician was built by the LLM that wrote the second annotation and adjudicated the references, after the E3 report showed confounders missing. Its prompt holds no trial-specific content, but its developer has seen the reference confounder lists; the Clinician's E3 confounder score is therefore not blind to development and must be reported with this note. The clinician audit of the references stands apart from it.
      - **Amendment (2026-11-01), prompt `clinician-v2`:** the first runs (`clinician-v1`, then v2) on the OnDemand H100 timed out at 600 s per trial. v2 bounds the answer schema (at most 20 confounders, 2 citations each, quotes and rationales under 300 characters) and says so in the prompt; that was a guess and did not fix it. A streamed probe (`agentic-tte clinician-probe`) then showed about one token per second with or without the schema, with the model fully on the GPU: the user's personal Ollama (port 11434, another program) was running qwen3:32b on the same H100 at 91% utilisation. Interactive runs now refuse to start on a GPU that already holds memory. The v2 bounds are kept (harmless, and a runaway guard). No Clinician answer was produced by either run, so no output was seen before the change; neither run is scored.
      - **Result, replicate 1** (2026-11-01; run `exit-H100-core-clinician-v2b`, qwen3:32b, OnDemand H100 with the GPU to itself; `eval/results/e3_clinician_qwen3_32b_v2.md`): over the 12 core trials, precision 26%, recall 46%, F1 33% (64 of 139 reference confounders matched; 246 proposed). Per trial, recall runs from 30% (EvK) to 54% (ECLS-SHOCK). Many citations failed the verbatim check (CAPE COD 0 of 21 items cited, CRISTAL 2 of 21); Reported with the disclosure above. Replicate 2 (same session, Ollama restarted) is byte-identical in 12/12 trials.
      - **Why citations failed** (read from the rep1 response cache, 2026-11-01): of 285 citations, 144 passed. Of the 138 that cite an article in the literature but fail: 33 differ only in punctuation or symbols; 41 join fragments with "..." (in 33 every fragment is verbatim); the rest are paraphrases or sentences not in the abstract (e.g. "The APACHE III prognostic system includes age as a predictive variable"). Some joined quotes are verbatim but weak evidence ("Exclusion criteria ... liver cirrhosis", from another study's exclusions). Citations do not change the confounder score (names are matched, not citations). Any change to the check is a new prompt or check version, scored as a new run.
      - **Amendment (2026-11-01), prompt `clinician-v3`:** made after the v2 rep1 output was seen, so v3 is a new run and v2 stays reported as it is. The quote check now matches whole words in order with case, punctuation and symbols ignored, and accepts "..." between fragments only if each is four or more words and they occur in order (`clinician.verbatim`). The prompt asks for one contiguous sentence or clause, no paraphrase or joined pieces, and says another study's inclusion or exclusion criteria are not evidence. Applied to the v2 rep1 answers as a check of the code only, 197 of 285 citations would pass (144 under v2); that is not a v3 result. Run tag `core-clinician-v3`. The confounder score is still on names; citations remain a separate, reported count.
      - **Result, `clinician-v3`** (2026-11-19; run `exit-A100-core-clinician-v3`, commit 6bdc009, batch jobs 19813099 and 19813100 on `gpu_requeue`, A100 80GB PCIe on g004 and g015; same drafts and literature as v2, `literature_sha256` 186cd205…; `eval/results/e3_clinician_qwen3_32b_v3.md`): precision 25%, recall 43%, F1 32% (60 of 139 reference confounders matched; 241 proposed). Replicate 2 is byte-identical in 12/12 trials. The confounder score is unchanged within noise from v2 (recall 46%, precision 26%; v2 ran on an H100, so prompt and GPU both differ); per trial, recall moved both ways (ECLS-SHOCK 54% to 23%, APROCCHSS 42% to 25%; STARRT-AKI 43% to 57%, ADRENAL 50% to 60%). Citations improved: 208 of 241 items keep a verified citation (v2: 120 of 246); 256 of 302 quotes pass. Of the 46 that fail, 20 are long sentences cut mid-word by the 300-character schema cap, 18 join a lead-in to a fragment under four words with "..." (in 10, every fragment is in the abstract), and 8 are paraphrases or not in the abstract. The citation fix does not move recall; retrieval is the next lever (off-topic abstracts persist, e.g. neutropenic cancer patients for CRISTAL). Reported with the disclosure above. The v2 report is renamed `e3_clinician_qwen3_32b_v2.md`; `score_clinician.py` now names each report by prompt version.
      - **Diagnosis of the v3 misses** (2026-11-19, no model run): of the 79 reference confounders v3 missed, 19 appear in the passages it was shown, 36 only elsewhere in its literature, and 24 nowhere in it (a word-match check, approximate). The misses repeat across trials: vasopressor use, serum lactate, mechanical ventilation, sepsis or infection source, surgical vs medical admission, acute kidney injury, and the severity score the trial used (often SAPS II). About 8 to 10 look like name-matching strictness rather than real misses ("Need for vasopressors" vs "Vasopressor use", "COPD" vs "chronic obstructive lung disease", "Septicemia" vs "Sepsis"); the pre-registered matching is not changed, and these go to the clinician audit.
      - **Amendment (2026-11-19), prompt `clinician-v4`: retrieval and a checklist.** Searches use every registry condition that is not a treatment component (up to 3), add a third search for adjusted observational studies of the condition (propensity, inverse probability, "adjusted for", confounders), and take 80 abstracts per search (was 40). The model is shown 40 passages, at most 2 per abstract (was 15 and 3). The prompt adds a generic checklist of confounder domains (demographics; admission type and diagnosis; severity scores; organ support at time zero; physiology and labs; source or aetiology; comorbidities; time from onset; treatments before time zero) and asks for every variable that matters whether or not a passage mentions it. The literature is rebuilt with the v4 searches (`runs/literature-v4/`, built on the PC on 2026-11-19: 35 to 215 abstracts per trial, was 32 to 82); the v3 literature is kept unchanged. Before any v4 model output, the passages shown mention 85 of the 139 reference confounders (v3 retrieval: 57), and the literature 122 (was 110). Prompts are 6,000 to 7,800 tokens, within the 16,384-token context with the 8,192-token output cap. **Disclosure:** the checklist and the retrieval changes were written after seeing which reference confounders v3 missed. They hold no trial-specific content, but the 12 core trials are from now on a development set for the Clinician: its scores on them are development results, not a test. An untouched test needs reference confounder lists for other trials (e.g. from the breadth set), written before any Clinician output for them is seen. Run tag `core-clinician-v4`.
      - **Result, `clinician-v4`** (2026-11-20; run `exit-A100-core-clinician-v4`, commit 4b3dbf1, batch jobs 19817888 and 19817889 on `gpu_requeue`, A100 80GB PCIe on g008; literature `runs/literature-v4`; `eval/results/e3_clinician_qwen3_32b_v4.md`), a development result on the 12 core trials: precision 28%, recall 48%, F1 35% (67 of 139 matched; 243 proposed). Replicate 2 is byte-identical in 12/12 trials. 211 of 243 items keep a verified citation (v3: 208 of 241). Recall rose from 43% (v3) and 46% (v2), well short of the hoped-for jump: per trial it moved both ways (BICAR-ICU 50% to 67%, CAPE COD 50% to 58%; STARRT-AKI 57% to 29%, whose acute kidney injury literature is mostly outside intensive care). Of the 72 references still missed, 32 are in the passages shown, 26 only elsewhere in the literature and 14 in none (was 19, 36 and 24 for v3), so retrieval now reaches most of them. Many of the remaining misses are names the lenient matching does not pair, not missing variables: "gender" for "sex" (the check adds sex unless gender is listed), "vasopressors or inotropes" for "Vasopressor use", "Invasive mechanical ventilation" for "Mechanical ventilation at baseline", "COPD" for "chronic obstructive lung disease", "use of nephrotoxic antibiotics" for "Nephrotoxic drugs". The pre-registered matching is not changed; a rated match review (each unmatched reference against the proposed list, with a rater and a note, like the time-zero ratings) would measure this. Also seen: the post-baseline check drops "time from ICU admission to randomization" (CRISTAL), a baseline variable the checklist asks for; and some proposals are measured after time zero without saying so ("units of packed red blood cells transfused in first 24 hours", STARRT-AKI).
      - **Match review, fixed before any rating (2026-11-20):** `eval/score_clinician.py --review` writes `eval/results/clinician_match_review_qwen3_32b_v4.json`, one entry for each of the 72 reference confounders the lenient matching left unmatched, listing that trial's unmatched proposals with their rationales. The rater sets `match` to the proposal that is the same baseline variable (a synonym, or a narrower or broader name a clinician would map to the same data) or to "none", with a note and their name; each proposal can match one reference only. The rating is done by the investigator (or a clinician), not by the LLM that built the Clinician. Once all 72 are rated, the report adds a reviewed precision, recall and F1 beside the automatic score, which stays the pre-registered result; a sheet made for a different run is refused. Example of what it measures: ACURASYS's "PEEP" against the proposal "positive end-expiratory pressure (PEEP)".
      - **Match review rated (2026-11-20, rater SKD, the investigator; `clinician_match_review_qwen3_32b_v4.json`):** 18 of the 72 unmatched reference confounders matched a proposal, 54 rated none, each with a note. **Reviewed v4 score: precision 35%, recall 61%, F1 45%** (85 of 139 matched), beside the automatic 28%, 48% and 35%, which stays the pre-registered result. Reviewed recall per trial runs from 43% (STARRT-AKI) to 75% (BICAR-ICU, CAPE COD). Matches include sex for "gender" (3 trials), admission type for surgical or operative admission (2), and vasopressor use for broader or timed variants (2); several notes flag a partial match (contrast volume without type, one infection source for the source taxonomy, ED location for intubation location, a vasopressor variable whose timing must be checked against time zero). Rated by the investigator, not a clinician: not a clinician review. Still a development result on the core trials (see the v4 disclosure). The most common real misses are the trial's own severity score (SAPS II, SOFA, APACHE II), sepsis status, serum lactate, acute kidney injury and treatment doses.
  30. **Concordance agent: how R4 is computed** (2026-11-20; `agentic_tte/agents/concordance.py`, `eval/score_concordance.py`). Fixed before any MIMIC estimate exists; tested on synthetic numbers only.
      - **Inputs:** each finished Replay writes `result.json` (status, protocol version and hash, the arms' labels and component concepts, the analysis aggregates). References are the user-verified values with S2 pass in `replay_reference_values.json`; sets come from `replay_selection.json`: the core set (12) is the primary agreement set, the breadth set (10) is secondary and exploratory. Sets are never pooled.
      - **Measure:** the trial's own. HR from the primary weighted Cox model; RR, OR or RD from the bootstrap risk measures (RD in percentage points). A risk measure from other than 500 resamples with seed 20261027 is flagged on its row. Adjusted published HRs (ACURASYS, NCT01284452) are compared with the pipeline's weighted HR, as published; never converted.
      - **Orientation:** the trial's contrast ("A vs B") is matched to the protocol's arms on words unique to each arm ("placebo", "control", "usual care" and "No X" mark a null arm). If the protocol's treatment arm is the trial's B, the emulation's contrast is reversed (1/ratio, −RD); the published estimate is never changed. If the match is unclear, or the protocol has no treatment arm, the trial is not scored until a person confirms the orientation by name (`--confirm`); it is listed outside the denominator with a count, and the endpoint is not final while any wait. On the qwen3:32b core drafts (before design review) 9 of 12 orient automatically (HYDRAREA reversed); ACURASYS, APROCCHSS and ADRENAL have no treatment arm (every arm drafted as a comparator; ADRENAL's comparator is hydrocortisone), which design review must fix.
      - **Survival:** a risk difference in survival (EvK) is compared with −RD of the emulation's event risk.
      - **Failures:** a trial with no run, a report not signed off or auto-approved (returned, blocked, stopped), no estimate in the trial's measure, or a non-positive ratio counts as no agreement on every binary metric, with the reason.
      - **Statistics:** standard errors from the 95% CIs on the log or linear scale; exact Clopper-Pearson CI for the share with standardized difference agreement; pooled ROR by random effects with τ² from the restricted likelihood (bounded optimisation, τ² = 0 when that fits at least as well).
      - **Written to:** `eval/results/e5_agreement_<run label>.md` and `.json`. Auto-approved trials are labelled on their rows.
  31. **The gate is blinded to outcomes by arm, S8 is applied at the gate, and the scored MIMIC run's settings** (2026-11-20, before any MIMIC estimate).
      - **Found and fixed:** the Data Support Gate's feasibility check and attrition table showed outcome events and person-time **per arm** before the analysis. That is a crude effect estimate, seen at the gate review and, since the balance loop (2026-11-20), before the design can still change; S8 says counts are never split by arm and outcome before the analysis runs. Now the gate and the attrition report show patients per arm only, and events, person-time, end-of-follow-up reasons and the Kaplan-Meier risk pooled. The amber rule "fewer than 11 events in an arm" became "fewer than 22 events pooled". No MIMIC gate had been run, so nothing was seen. Only the Statistician's analysis reports outcomes by arm.
      - **S8 at the gate** (`--s8`, on by default in `replay start`): at least 50 patients in each arm and at least 20 events pooled, else the gate is red. At the run's **first** gate this is the S8 screen: the run ends with status `excluded_s8`, the trial leaves the agreement set, and the set is recomputed by the outcome-blind rule (item 9); the Concordance agent lists it outside the denominator. Below S8 at a **later** gate (after a design change) the run is blocked, which counts as a failure (R4).
      - **Scored run settings, confirmed by the user (SKD) on 2026-11-20, before the first core trial started** (`hpc/replay_core.sh`): protocol = the qwen3:32b core draft (`runs/exit-H100-core`, rep1, commit of item 23); confounders = the scored Clinician v4 sheet for that draft (`runs/exit-A100-core-clinician-v4`, rep1), shown at design review, where the user approves, edits or replaces them; surrogate answers and the Statistician ↔ Clinician balance loop on (qwen3:32b, think off, `OLLAMA_NUM_PARALLEL=1`, a fresh server per step); the Supervisor's default budget; bootstrap 500 resamples, seed 20261027; S8 on; data classification deidentified, approved (CLAUDE.md, 2026-10-26); the MIMIC-IV 3.1 CDM 5.4 build of job 19365483. Every review is done by the user (SKD) and reported as the investigator's review, never as clinician review. The design review fixes drafts that cannot be emulated as written: ACURASYS, APROCCHSS and ADRENAL were drafted with every arm as a comparator (item 30).
  32. **The gate shows inclusion criteria that did not compile, and the review helpers** (2026-11-22, before any core trial reached a gate; **confirmed by the user (SKD) on 2026-11-23**, before the first core gate).
      - **Found on APROCCHSS** (the first core trial, at concept review): the compiler skips a whole criterion when any of its atoms cannot be compiled, and reports it (compile.md), but nothing stopped the run. On the qwen3:32b draft no septic-shock inclusion criterion would compile (a duration drafted as a value on a diagnosis, a qualitative value, a cumulative duration, doses on drugs), so the cohort would not have been restricted to septic shock; and E11 (platelets < 30000 /mm3) would have compiled against MIMIC values in K/uL (units are not converted) and excluded every patient with a platelet value.
      - **Gate check `compiled`**: an inclusion criterion that did not compile turns the gate **amber**, so the gate review sees it with the reasons; exclusion criteria that did not compile are listed (they remove no one). Amber never ends a run; S8 (item 31) is unchanged.
      - **Review helpers (no scored setting changes)**: `replay design-answer` writes the design answer from the reviewer's decisions (treatment arm; arms, components and criteria to drop; grace period; criterion edits: remove a value, set a window before time zero, set a threshold in the data's unit, drop an atom, all/any atoms) and warns about every criterion that will not compile while the design can still change (the compiler's own rule, `compiler.design_problem`); `concepts decide` writes the concept decisions and the compile spec (time zero, anchors) from the reviewer's decisions; `concepts counts` shows each kept set's vocabulary names, records, persons and, for measurements, the units and median value in the data (whole-database aggregates, small cells suppressed). Empty anchors are removed from the spec, so criteria that use them are reported as not compiled.
      - **Restarts before any gate**: APROCCHSS was restarted at design review (2026-11-22, `runs/mimic-core/NCT00625209.attempt1/` on Hellbender, with the reason in `RESTARTED.txt`) because the draft had turned amendment text into exclusions and had no grace period; it will be restarted once more to fix the criteria above. No gate, count by arm or estimate had been produced. A restart is allowed only before a trial's first gate, and every restart is kept and reported with its reason.
  33. **The study design: an autonomous primary arm, a rule-guided secondary arm and a clinician audit; the Repair agent (repair-v1); unit conversion** (2026-11-23, before any core trial reached a gate; nothing in it was chosen after an estimate, a gate or a count by arm). The three-arm design was adopted by the user in chat on 2026-11-23; the repair-v1 rules and the unit conversion were **confirmed by the user (SKD) on 2026-11-23**, before the first run.
      - **Why:** the user is not an expert in every trial's disease area and cannot fix or finalise each draft by hand. A Replay whose quality depends on one investigator's ad hoc edits would not be reproducible, and the edits would be the result. So the pipeline is scored as it runs on its own, and the human part becomes a fixed rule book plus an independent audit.
      - **Primary arm, fully autonomous:** the agents make every decision; every review stage is answered by `AutoApprove` and labelled auto-approved (status `auto_approved`, never `signed_off`); a run that stops, blocks or fails counts as a failure in R4 (item 30). Nothing is fixed by a person. This is the headline result (E5 agreement; E1/E2 stage-wise failures, with a failure taxonomy; ablations).
      - **Secondary arm, rule-guided review:** the same trials, with the design and concept reviews answered from a pre-registered rule book, coded where it can be. The user answers only the items the rules flag, and each such answer is recorded with the rule that flagged it. It is reported as investigator review following rules, never as clinician review. The rule book is fixed in its own item before the first secondary-arm run. Item 32's review helpers are its tools.
      - **Clinician audit:** a clinician audits a seeded random sample of about 20% of the trials, plus every item a rule flagged, on the design and concept decisions (not on estimates). The audit is reported as such; it does not change the scored runs.
      - **The Repair agent (`agentic_tte/agents/repair.py`, rules version `repair-v1`),** a node between the draft (and the Clinician) and design review, in both arms. It uses fixed rules, no LLM and no data, and records every change with its rule (a new protocol version created by the Repair agent, `repair.md` at design review and in the report). What the rules cannot decide is flagged and left as drafted. The rules: R-arm-null (registry PLACEBO/NO_INTERVENTION/SHAM, a label that names no treatment, or all-placebo components: the comparator, with its vehicle or usual-care components removed); R-arm-shared (components every arm has, or that the comparator also gets, are background); R-arm-one; R-arm-contrast (several real arms: the trial's published contrast picks the treatment and comparator, the other factor of a factorial is removed, other arms are dropped); R-amend (amendment text that removes criteria, and the list it introduces, is dropped; a single removal is flagged); R-value (an inclusion atom the compiler cannot compile loses the value it cannot read, and a cumulative duration becomes a window before time zero; exclusions are left as drafted). An active comparator is flagged: the compiler cannot frame it yet.
      - **The contrast is an input:** the "A vs B" of `eval/reference/replay_reference_values.json` (the comparison being emulated, like the NCT ID; never an estimate), used only to choose the arms, never their orientation in R4 (item 30 still needs the user's confirmation).
      - **Development and test sets:** repair-v1 was written while reading the 12 core-trial qwen3:32b drafts (`runs/exit-H100-core`, rep1), so the core set is its development set, disclosed as such. The breadth set is the held-out test of the rules. repair-v1 is frozen with this item; any change is a new rules version, in a new item, and every trial it affects is run again with it.
      - **Unit conversion in the compiler:** a measurement threshold in cells per mm3 (or per microlitre) is compared with values recorded in thousand per microlitre (or 10^3/uL, billion per litre) multiplied by 1000, matched on the unit concept's name. Only conversions exact by definition; a value in any other unit, or with no unit concept, is read as it is, and the criterion carries a warning.
      - **What this replaces:** item 31's "every review is done by the user (SKD)" now holds only for the secondary arm. Item 31's other settings and item 32 (the `compiled` gate check, confirmed on 2026-11-23) stand. The APROCCHSS attempts at design review (attempt1 to attempt3, `runs/mimic-core/NCT00625209*` on Hellbender) were made under the old plan; they are kept, are not scored, and never reached a gate.
      - **Still to fix before the first autonomous run (a new item):** the autonomous choice of time zero and of concepts with no records in the data, the `AutoApprove` settings (surrogates, balance advice), and the active-comparator frame. Until then no core trial is run in either arm.
  34. **The autonomous arm's remaining rules: active comparators, time zero, concept choice from the data, the policy; repair-v3** (2026-11-23, before any core trial reached a gate; nothing in it was chosen after an estimate, a gate or a count by arm; **confirmed by the user on 2026-11-24**, after the counts-only checks (jobs 19906650, 19910776, 19914511) and before any core trial ran; the rules are frozen from this date). It completes item 33's list of what had to be fixed first.
      - **Active-comparator frame (compiler).** When the comparator is an active treatment, eligible patients who start either strategy within the treatment arm's grace period are assigned to the one they start first and followed from that start (new-user, active-comparator design); patients who start neither, or both at the same time, are left out, and confounders are read at time zero. The Statistician then uses IPTW (or PSM when the plan says so), not clone-censor-weight, which is for initiation vs no initiation. Without a grace period the frame is not compiled.
      - **Fixed: initiation of a multi-component strategy.** "All components started" was computed with DuckDB's `greatest()`, which skips missing values, so a patient who started only one component of a combination (hydrocortisone without fludrocortisone, APROCCHSS) counted as treated. Now a missing component means not initiated. No MIMIC run has reached the analysis. The Phase 2 synthetic replays that had multi-component arms (STASIS; NCT06091982 as drafted) gave every treated patient all components at once, so their frames do not change; their tests are re-run with the fix. The Phase 0 runs compile no frame.
      - **repair-v2** (adds to item 33's repair-v1; repair-v1 never ran a trial): R-arm-standard (a comparator of standard fluids only, such as saline or crystalloids, is no initiation of the study fluid: CRISTAL, CHEST, BICAR-ICU 2018); R-arm-timing (the same intervention at another time, such as standard vs accelerated RRT, is no initiation within the grace period: STARRT-AKI); R-grace (a treatment arm with no grace period gets 24 hours, origin default, labelled); R-anchor (an anchor named after a temporal relation, a drafting slip, is time zero). After repair-v2 the only active comparator among the 12 core trials is EvK (etomidate vs ketamine).
      - **Time zero by rule (`tz-v1`, `agents/autonomy.py`).** Time zero starts at the first ICU admission (OMOP visit 32037; the first inpatient visit, 9201, for a trial whose protocol never mentions intensive care; both checked by name against the vocabulary before a run). It then waits for the time-stamped inclusion events: each positive drug, procedure or device atom, and each measurement atom with a numeric threshold, of an inclusion criterion whose atoms must all hold; for a criterion met by any of its atoms, the first of its drug, procedure or device alternatives. Each event counts from 24 hours before the ICU admission to 72 hours after it; a person without one of them has no time zero. Concepts of the treatment arm are never time-zero events. Named anchors map to a matching time-zero event, else to hospital or ICU admission, else they are left out (criteria that use them are reported as not compiled).
      - **Concepts chosen from the data (`data-v1`, replaced by data-v2 below).** A proposed concept with no records in the database is replaced by the vocabulary candidate with the most records (concept IDs still come only from the vocabulary search); with no candidate in the data, the set is unmappable and its criteria are reported as not compiled. A treatment component given intravenously (its route says IV, infusion, bolus or injection) keeps only the injectable forms of the drug that have records. The counts are whole-database aggregates, never split by arm or outcome, and small cells are suppressed in the notes.
      - **The autonomous policy** (`eval/policies/autonomous-v1.json`, with `replay start --repair --autonomy --balance-advice`): every stage auto-approved and labelled; time zero by rule; the Clinician's surrogate concepts **not** accepted (on APROCCHSS it proposed aspirin for "an indication for corticosteroids" and hand gel for "inappropriate antimicrobial treatment"; this choice was made on one development trial and is disclosed); the Clinician's result-blind balance advice applied, labelled; a failed Protocol Auditor **stops** the run, which counts as a failure (an ablation reruns with auto-override); an amber gate proceeds with its limitations listed; S8 and the other settings of item 31 are unchanged.
      - **Revised after the counts-only MIMIC check** (job 19906650, commit 0aa890d, 2026-11-23; `eval/autonomy_check.py`: pooled counts per criterion, nothing by arm, no outcome read, no frame run). The check showed rules that did not work on the data; they were revised inside this item, which is still unconfirmed and under which no trial has run. The check prompted every change below; this is disclosed, and the core set stays the development set.
          - **Compiler, death outcomes.** A primary outcome named "All cause mortality at 90 days after randomisation", "Day 28 all causes mortality" or "Reduction of the mortality rate of ARDS patients at d90" was read as a composite and left the frame uncompiled. Qualifiers of time and reference (after randomisation, causes, d90, reduction, the population "of ARDS patients") are now allowed; a death plus any other event is still a composite.
          - **Compiler, creatinine units.** A creatinine threshold in µmol/L is compared with values recorded in mg/dL × 88.4 (molar mass 113.12 g/mol). Other molar units are not converted. **All unit conversions read the recorded unit text when a record has no unit concept:** the MIMIC-IV OMOP build has none (job 19910776: platelets, creatinine, blood pressures, potassium and bicarbonate all without a unit concept), so the per-mm3 conversion of items 33 and 34 had never applied on MIMIC; MIMIC's own units (K/uL, mg/dL) are kept in `unit_source_value`.
          - **data-v2** (replaces data-v1, which never ran a trial): a replacement candidate must also match the protocol's term (`name_match`: the term's last word and at least two thirds of its content words, all of them for a term of up to two words; data-v1 chose "Chronic kidney disease" for "underlying disease"); with no matching candidate, the standard concepts of the domain in use in the data whose names match the term (a measurement takes the one with the most records, other domains all of the best-matching ones: the vocabulary search had missed MIMIC's platelet count and blood pressures); a visit named as intensive care is the ICU admission concept (32037); a drug, procedure, device or measurement of an inclusion criterion that fewer than 1% of the persons with the time-zero admission have is unmappable (the data barely record it: chest X-ray left hydrocortisone in pneumonia with no time zero at all).
          - **repair-v3** (replaces repair-v2, which never ran a trial): R-amend also drops criteria that say a criterion was changed or clarified (the criterion as listed is kept, flagged) and dated or "Amendments" headings; a criterion an amendment added is kept. R-subgroup drops a sub-group heading and the criteria right after it that name its condition (COVID-19 in NCT02517489). R-name removes timing and action words from a treatment component's name ("Accelerated RRT initiation", "ECLS insertion"), and expands a lone abbreviation from the registered title when the title spells it out (renal replacement therapy; extracorporeal life support).
          - Unchanged: tz-v1, the policy, and the composite outcomes (NCT02476253, NCT03680274) and non-death outcome without a concept (NCT01636089), which still compile no frame and count as failures of the autonomous arm.
      - **Development set:** like repair-v1, these rules were written while reading the 12 core drafts, and are disclosed as such; the breadth set is the held-out test. They are frozen with this item; any change is a new version in a new item, and every trial it affects is run again.
  35. **repair-v4 and an auditor correction, so the autonomous arm does not stop at the audit** (2026-11-24, before any core trial ran; nothing in it was chosen after an estimate, a gate or a count by arm; **confirmed by the user on 2026-11-24**, after the counts-only check with the auditor (job 19938541: all 12 pass the audit) and before any core trial ran; frozen from this date). Found by a dry run of the autonomous batch (`hpc/replay_autonomous.sh`) on a 30-patient synthetic database: the confirmed policy stops a run whose Protocol Audit fails (item 34), and after repair-v3 the audit would fail on 11 of the 12 core drafts (A-f, no exclusion of prior treatment use), and on others for look-ahead (A-b), the outcome horizon (A-e) and the arms (A-d). A person fixes these at design review; the autonomous arm needs rules for them.
      - **R-new-user:** when no exclusion names a treatment component before time zero, one exclusion per component (of the treatment and of an active comparator) is added: any record strictly before time zero, with all available lookback (the new-user design). "Strictly before", so a start at time zero itself (etomidate at the intubation that is EvK's time zero) is treatment, not prior use.
      - **R-lookahead:** an eligibility atom that needs records after time zero is removed, and a criterion left with none is dropped (no look-ahead can be emulated; the compiler would not compile it).
      - **R-horizon:** a primary outcome whose horizon is missing or differs from the trial's registered time frame takes that time frame (as the auditor's A-e parses it).
      - **R-arm-registry:** with an active comparator, when the registry marks the comparator EXPERIMENTAL and the treatment not, they swap. The published contrast's orientation stays the Concordance agent's, confirmed by the user (item 30).
      - **Auditor A-d corrected:** two EXPERIMENTAL arms (a head-to-head trial such as EvK) are not an arm error; one of them has to be the comparator. A-d still fails when the comparator is EXPERIMENTAL and the treatment is not.
      - **The counts-only check now runs the Protocol Auditor** as the run does, so every audit failure on MIMIC is seen before a trial runs; its data shares are pooled and no outcome is read.
      - **Development set:** like items 33 and 34, written on the 12 core drafts, disclosed; frozen with this item. The batch refuses to run until items 34 and 35 are both confirmed.
  36. **The gate's data-quality check: a measurement without a value is completeness, not a plausibility violation; the autonomous arm is run again once** (2026-11-26, before any estimate; **confirmed by the user on 2026-11-26**, before the second autonomous run).
      - **Found:** the first autonomous batch (2026-11-26, OnDemand g002, A100, commit 5f21642; `runs/mimic-autonomous/` on Hellbender) was the first time any MIMIC run reached the Data Support Gate. All 12 core trials were blocked by one check, Data quality: 29,060,684 measurement rows (12.67%) have no `value_as_number`, and the check, written in Phase 2 on synthetic data where every measurement had a value, turned red above 5%. Every other check was green or amber. The block is the same for every trial; it does not depend on a protocol, a rule or a count by arm.
      - **Why it is not a violation:** OMOP allows a measurement with no numeric value (a qualitative result in `value_as_concept_id`, or the record that a test was done), and a threshold criterion is simply not met by a missing value (the compiler). Plausibility violations (an end before a start, a death before birth, an age over 120) are impossible values; a missing value is not.
      - **Change:** the quality check counts measurements with neither `value_as_number` nor `value_as_concept_id` as missing values, reported in the check's detail and amber when above 5%, never red. Plausibility violations are unchanged (red above 5%).
      - **What had been seen:** the 12 blocked gate reports show the pooled counts the gate always shows (eligible persons, persons per arm, pooled events: S8, item 31). No estimate was computed and no outcome was split by arm. The change was chosen from the whole-database check alone.
      - **Rerun:** this is a pipeline defect shared by every trial, treated like an infrastructure failure (amendment 2: rerun once, documented here). The 12 blocked runs are kept in `runs/mimic-autonomous/` with this item as their reason and are reported. The autonomous arm runs again under the label `mimic-autonomous-2`, with the same rules (items 34 and 35), policy and item 31 settings. Disclosed: the first attempt reached each trial's first gate. A trial whose second run stops, blocks or fails counts as a failure (item 33); there is no third run.
      - **Also:** the batch now pins the session's first GPU (`CUDA_VISIBLE_DEVICES`), so a two-GPU session runs each model on one GPU, as every earlier run did, and records it in the manifest. The batch refuses to run until items 34, 35 and 36 are confirmed.
  37. **Pre-registered sensitivity analyses: missing confounders, weighting and positivity** (2026-11-27, before any estimate of any arm was opened; **proposed: the user confirms it here before any of it is coded or run**).
      - **Found, blind:** in the five trials the second autonomous run analysed (item 36), the balance check, which reads only treatment and confounders, and the gate's pooled counts showed the same pattern in the three trials whose confounders include labs or vital signs: only 7.7% to 42% of the eligible cohort has every confounder, and it stays imbalanced after weighting (largest SMD 0.19 to 0.36). The two trials whose confounders are diagnoses and demographics are complete and balanced. Treatment is rare in the three (2.9% to 4.9% of the eligible cohort), so the fixed 0.05 to 0.95 propensity share (69% to 82%) mostly reflects rarity, not a positivity violation, and the pooled 1st/99th-percentile truncation caps mostly treated weights. No estimate, outcome by arm or event count by arm was seen.
      - **The primary results do not change.** Each trial's primary analysis (the Statistician's method, complete cases, item 31 settings) stays the scored result, and agreement (item 30) is computed on it. The analyses below are secondary, pre-specified here, applied identically to every analysed trial in every arm (autonomous, secondary, breadth set) and never chosen, dropped or promoted per trial. Their agreement is reported beside the primary under the same rules.
      - **S-MI, multiple imputation of missing confounders:** chained equations, 20 imputations, 20 iterations, seed fixed and recorded. The imputation model holds every confounder, treatment, the event indicator and the Nelson-Aalen cumulative hazard at each person's event or censoring time (White and Royston 2009); continuous confounders by predictive mean matching, binary by logistic regression. Each imputed set runs the trial's primary method unchanged; the log hazard ratios and the risk measures are pooled with Rubin's rules, using 200 bootstrap resamples per imputation for the risk measures' variance. The fraction of missing information is reported. Imputation uses the outcome, so it is analysis-stage only: nothing from it reaches the design, the balance check or the Clinician's balance advice.
      - **S-W, weighting:** for the IPTW analysis of each trial (the primary for an active comparator, a secondary estimate otherwise): (a) untruncated stabilized weights; (b) stabilized weights truncated at the 1st and 99th percentiles within each arm; (c) overlap weights (Li, Morgan and Zaslavsky 2018), whose estimand is the average effect in the overlap population (ATO), reported with that label and never as the ATE. Clone-censor-weight analyses keep their own weights; S-W applies to their baseline IPTW estimate. Each arm's weight distribution (minimum, 1st, 50th and 99th percentiles, maximum) and Kish effective sample size are reported.
      - **S-P, positivity diagnostic:** each arm's propensity distribution (quantiles), the common support (from the larger of the arms' minima to the smaller of their maxima), each arm's share outside it, and the number of scores at exactly 0 or 1 (separation). This replaces the fixed 0.05 to 0.95 share as the reported positivity diagnostic; the old share stays in the balance check for continuity. S-P is outcome-blind.
      - **Validation before MIMIC:** recovery tests on the synthetic generator first. S-MI with missingness planted in a confounder (completely at random, and at random given treatment and the outcome) must cover the marginal truth where complete cases do not; overlap weights must recover the truth in the overlap population (the potential outcomes weighted by the true propensity times one minus it); S-W and S-P are checked against hand-computed values. None of it runs on MIMIC until these pass.
      - **How it runs:** a script reads each finished run's frame on Hellbender and writes `sensitivity/` beside the run (results, manifest, small cells suppressed); no Replay is rerun and no rule (items 34 and 35) changes. The CPU suffices.
      - **Order:** the arm orientations (item 30) and the secondary arm's rule book are confirmed before any estimate is opened, and this item is confirmed before it is coded.
      - **Reporting:** the manuscript reports this framework at the pipeline level (how often missing confounders, weighting and positivity change agreement, across trials); the per-trial diagnostics stay in each run's report and the supplement.
