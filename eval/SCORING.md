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
