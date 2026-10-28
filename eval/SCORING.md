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
