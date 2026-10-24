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

- 2026-10-24: initial version.
