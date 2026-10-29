# E3 reference agreement: SKD vs AI

Before adjudication (eval/SCORING.md, Phase 3 changelog item 24). SKD: investigator; AI: LLM second annotator. F1 is symmetric; precision and recall read AI against SKD. *Lenient* concept matching is exploratory, not pre-registered.

## Summary over the 12 core trials

| Element | Agreement |
|---|---|
| Eligibility: category (clinical vs administrative) | 183/188 criteria, kappa 0.92 |
| Eligibility: concepts, strict F1 | 2% |
| Eligibility: concepts, strict as compiled F1 (no time constraint = at time zero; units ignored) | 33% |
| Eligibility: concepts, relaxed F1 (name + domain) | 44% |
| Eligibility: concepts, lenient names F1 | 67% |
| Treatment: same arms kept | 12/12 trials |
| Treatment: same comparator | 12/12 trials |
| Treatment: components, lenient F1 | 82% |
| Treatment: same grace period | 5/12 trials |
| Time zero | rated at adjudication (free text) |
| Outcome: same horizon | 12/12 trials |
| Outcome: same end event | 11/12 trials |
| Causal contrast | 12/12 trials |
| Estimand | 12/12 trials |
| Confounders, lenient F1 | 71% |

## Per trial

| Trial | Criteria (category agree) | Concepts strict / as compiled / relaxed / lenient F1 | Comparator | Components F1 | Grace | Horizon | Contrast | Estimand | Confounders F1 | Disagreements |
|---|---|---|---|---|---|---|---|---|---|---|
| ACURASYS | 7 (7) | 9% / 18% / 45% / 64% | same | 100% | same | same | same | same | 78% | 7 |
| CRISTAL | 14 (14) | 0% / 22% / 40% / 58% | same | 67% | same | same | same | same | 76% | 10 |
| APROCCHSS | 36 (35) | 8% / 12% / 39% / 59% | same | 100% | differs | same | same | same | 86% | 14 |
| CHEST | 19 (18) | 0% / 22% / 30% / 74% | same | 100% | same | same | same | same | 76% | 8 |
| ADRENAL | 14 (14) | 0% / 51% / 56% / 65% | same | 100% | differs | same | same | same | 67% | 8 |
| HYDRAREA | 9 (9) | 10% / 48% / 57% / 76% | same | 100% | differs | same | same | same | 86% | 6 |
| BICAR-ICU | 9 (9) | 5% / 43% / 48% / 67% | same | 100% | differs | same | same | same | 82% | 9 |
| CAPE COD | 32 (29) | 0% / 24% / 36% / 55% | same | 100% | same | same | same | same | 50% | 13 |
| STARRT-AKI | 14 (14) | 5% / 28% / 33% / 70% | same | 67% | same | same | same | same | 81% | 8 |
| EvK | 7 (7) | 0% / 71% / 71% / 100% | same | 100% | differs | same | same | same | 53% | 3 |
| ECLS-SHOCK | 15 (15) | 0% / 49% / 57% / 83% | same | 0% | differs | same | same | same | 42% | 8 |
| LOVIT | 12 (12) | 0% / 41% / 55% / 62% | same | 100% | differs | same | same | same | 76% | 7 |

Disagreements to adjudicate: 101 (including one time-zero rating per trial), in `eval/reference_protocols/adjudication.md`.
