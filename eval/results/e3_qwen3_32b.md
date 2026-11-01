# E3: qwen3_32b protocols vs the adjudicated references

Run `runs/exit-H100-core`, replicate 1 (analysis set); replicate 2 identical in 12/12 trials. References: `eval/reference_protocols/<NCT>.adjudicated.json` (investigator-written, LLM second annotation, adjudicated; clinician audit pending). Precision: share of the draft's concepts in the reference; recall: share of the reference's concepts in the draft.

## Summary over the 12 core trials

| Element | Result |
|---|---|
| Eligibility concepts, strict (pre-registered) | P 7%, R 10%, F1 9% |
| Eligibility concepts, as compiled | P 13%, R 18%, F1 15% |
| Eligibility concepts, relaxed (name + domain) | P 18%, R 26%, F1 21% |
| Eligibility concepts, lenient names (exploratory) | P 39%, R 55%, F1 46% |
| Eligibility element correct (>= 90% of criteria exact), strict / as compiled | 0/12 / 0/12 trials |
| Criterion category (clinical vs administrative) | 160/188 criteria |
| Treatment: same arms | 11/12 trials |
| Treatment: right comparator | 8/12 trials |
| Treatment components (lenient names) | P 45%, R 56%, F1 50% |
| Treatment: right grace period | 6/12 trials |
| Time zero (rated, `e3_time_zero_ratings.json`) | same 6/12, acceptable alternative 4/12, different 2/12 |
| Outcome horizon | 10/12 trials |
| Outcome end event | 11/12 trials |
| Causal contrast | 12/12 trials |
| Estimand | 1/12 trials |
| Confounders (lenient names) | P n/a, R 0%, F1 n/a |

## Per trial

| Trial | Clinical criteria exact, strict / compiled | Concepts F1 strict / compiled / lenient | Comparator | Components F1 | Grace | Horizon | End event | Contrast | Estimand | Confounders (draft n, F1) | Registry-structured criteria (not scored) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ACURASYS | 0/6 / 1/6 | 0% / 9% / 55% | wrong | 67% | right | wrong | wrong | right | right | 0, n/a | - |
| CRISTAL | 2/12 / 2/12 | 9% / 9% / 51% | right | 0% | right | right | right | right | wrong | 0, n/a | - |
| APROCCHSS | 1/20 / 1/20 | 2% / 2% / 25% | wrong | 80% | wrong | right | right | right | wrong | 0, n/a | minimumAge: 18 Years |
| CHEST | 1/16 / 1/16 | 6% / 20% / 56% | right | 0% | right | right | right | right | wrong | 0, n/a | - |
| ADRENAL | 1/13 / 3/13 | 4% / 16% / 24% | wrong | 100% | wrong | right | right | right | wrong | 0, n/a | - |
| HYDRAREA | 2/9 / 3/9 | 19% / 29% / 57% | right | 100% | right | right | right | right | wrong | 0, n/a | - |
| BICAR-ICU | 1/9 / 2/9 | 22% / 30% / 52% | right | 100% | wrong | wrong | right | right | wrong | 0, n/a | - |
| CAPE COD | 3/22 / 4/22 | 13% / 18% / 37% | right | 100% | right | right | right | right | wrong | 0, n/a | - |
| STARRT-AKI | 0/14 / 1/14 | 0% / 4% / 61% | right | 0% | right | right | right | right | wrong | 0, n/a | - |
| EvK | 0/5 / 1/5 | 0% / 24% / 71% | wrong | 50% | wrong | right | right | right | wrong | 0, n/a | - |
| ECLS-SHOCK | 1/13 / 2/13 | 19% / 26% / 45% | right | 0% | wrong | right | right | right | wrong | 0, n/a | - |
| LOVIT | 1/9 / 1/9 | 6% / 6% / 71% | right | 100% | wrong | right | right | right | wrong | 0, n/a | minimumAge: 18 Years |

## Time zero (rated, not scored automatically)

Rater: Rated by the LLM second annotator (Claude Opus 5.5) at the investigator's request, 2026-10-31. It did not write the drafts but wrote one side of the references and adjudicated them, so every rating goes to the clinician audit.

| Trial | Rating | Reason |
|---|---|---|
| ACURASYS | acceptable alternative | Same event (first P/F < 150 on PEEP). The draft drops the 48 h ARDS-onset window and mechanical ventilation from the anchor and treats the 48 h as a grace period; it uses PEEP > 5 for the paper's >= 5. |
| CRISTAL | acceptable alternative | Same decision point (need for fluid resuscitation, randomised within 15 min), but stated as a day and a physician's judgement; it needs operationalising as the first fluid bolus to be found in the data. |
| APROCCHSS | same | Full septic-shock definition first met; same criteria. |
| CHEST | acceptable alternative | The first need for resuscitation with supporting clinical signs, where the reference uses the first resuscitation-fluid order: the same moment in practice, observed differently. |
| ADRENAL | same | First time all inclusion criteria are met, vasopressors for >= 4 h. The 24 h limit is an eligibility criterion (E1) in both. |
| HYDRAREA | same | Contrast injection. The pre-contrast hydration hour is the initiation window, not a different time zero. |
| BICAR-ICU | acceptable alternative | Same acidaemia criteria. The draft adds the registry's lactate >= 2 mmol/L to the anchor and drops the 48 h ICU window. |
| CAPE COD | different | Anchored at study-drug start, which exists only in the treated arm: the comparator has no time zero, and anchoring at treatment start selects on the future. The reference anchors at the first severity criterion with a 24 h grace period. |
| STARRT-AKI | same | First time the creatinine and severe-AKI criteria are met in the ICU; the no-urgent-RRT condition is an exclusion criterion in both. |
| EvK | same | Emergency intubation: the need and the induction are the same moment. |
| ECLS-SHOCK | different | Anchored at cardiogenic-shock onset; the trial randomised after diagnostic angiography, before revascularisation. Shock onset comes earlier and includes patients before the treatment decision could be made. |
| LOVIT | same | Vasopressors for infection in the ICU. The 24 h from ICU admission is in the eligibility criteria; the draft also calls it a grace period. |

## Element differences (draft vs reference)

- **ACURASYS:** treatment comparator: reference {"A": true, "B": false}, draft {"A": true, "B": true}; treatment arm 'B' components: reference ["Cisatracurium"], draft ["cisatracurium", "Cisatracurium besilate"]; outcome horizon_hours: reference 2160.0, draft 26298.0; outcome end_event: reference "hospital discharge", draft null; outcome end_rule: reference "first", draft null; confounders only one annotator listed: reference ["age", "sex", "saps ii", "pao2 fio2 ratio", "peep", "plateau pressure", "ards etiology pneumonia sepsis", "vasopressor use shock", "time from ards onset to time zero", "tidal volume"], draft []
- **CRISTAL:** treatment arm '2' components: reference ["Albumin human", "Gelatin (succinylated / polygeline)", "Hydroxyethyl starch", "Dextran"], draft ["Colloids"]; analysis_plan estimand: reference "risk_difference", draft "hazard_ratio"; confounders only one annotator listed: reference ["age", "sex", "saps ii", "sofa score", "admission diagnosis stratum sepsis trauma haemorrhage other", "mechanical ventilation", "vasopressor use", "mccabe chronic comorbidity class", "serum lactate", "fluids received before time zero"], draft []
- **APROCCHSS:** treatment arms kept: reference ["1", "2"], draft ["1", "2", "3", "4"]; treatment comparator: reference {"1": true, "2": false}, draft {"1": true, "2": true}; treatment arm '2' components: reference ["Hydrocortisone", "Fludrocortisone"], draft ["hydrocortisone", "fludrocortisone", "activated protein C"]; treatment arm '2' grace period (hours): reference 6.0, draft null; analysis_plan estimand: reference "risk_difference", draft "hazard_ratio"; confounders only one annotator listed: reference ["age", "sex", "saps ii", "sofa score", "source of infection", "norepinephrine dose", "serum lactate", "mechanical ventilation", "ards", "surgical vs medical admission", "chronic comorbidity mccabe", "community acquired infection"], draft []
- **CHEST:** treatment arm 'Hydroxy-ethyl starch' components: reference ["Hydroxyethyl starch 130/0.4 6%"], draft ["Hydroxy-ethyl starch"]; analysis_plan estimand: reference "risk_difference", draft "hazard_ratio"; confounders only one annotator listed: reference ["age", "sex", "apache ii", "operative vs non operative admission", "sepsis", "trauma traumatic brain injury", "serum creatinine rifle category", "mechanical ventilation", "vasopressor use", "serum lactate", "trauma", "acute kidney injury"], draft []
- **ADRENAL:** treatment comparator: reference {"Hydrocortisone": false, "Sterile air filled vial": true}, draft {"Hydrocortisone": true, "Sterile air filled vial": true}; treatment arm 'Hydrocortisone' grace period (hours): reference 6.0, draft null; analysis_plan estimand: reference "risk_difference", draft "hazard_ratio"; confounders only one annotator listed: reference ["age", "sex", "apache ii", "operative vs non operative admission", "catecholamine dose vs 15 ug min", "source of infection", "serum lactate", "renal replacement therapy", "sofa score", "time from shock onset to time zero"], draft []
- **HYDRAREA:** analysis_plan estimand: reference "risk_difference", draft "hazard_ratio"; confounders only one annotator listed: reference ["age", "sex", "saps ii", "baseline serum creatinine egfr", "diabetes mellitus", "contrast volume and type", "sepsis", "vasopressor use", "mechanical ventilation", "nephrotoxic drugs aminoglycosides vancomycin", "chronic kidney disease", "sofa score"], draft []
- **BICAR-ICU:** treatment arm 'Intervention' grace period (hours): reference 6.0, draft null; outcome horizon_hours: reference 672.0, draft 0.0; analysis_plan estimand: reference "risk_difference", draft "hazard_ratio"; confounders only one annotator listed: reference ["age", "sex", "sofa score", "saps ii", "sepsis at inclusion", "akin stage 2 3 at inclusion", "arterial ph", "serum lactate", "mechanical ventilation", "vasopressor use", "acute kidney injury", "serum bicarbonate"], draft []
- **CAPE COD:** analysis_plan estimand: reference "risk_difference", draft "hazard_ratio"; confounders only one annotator listed: reference ["age", "sex", "sofa score", "pneumonia severity index", "mechanical ventilation at baseline", "pao2 fio2 ratio", "immunosuppression", "c reactive protein", "saps ii", "noninvasive ventilation", "high flow nasal oxygen", "chronic obstructive lung disease"], draft []
- **STARRT-AKI:** treatment arm 'Accelerated RRT initiation' components: reference ["renal replacement therapy"], draft ["Accelerated RRT initiation"]; analysis_plan estimand: reference "risk_difference", draft "hazard_ratio"; confounders only one annotator listed: reference ["age", "sex", "sofa score", "apache ii", "baseline serum creatinine egfr", "urine output", "sepsis", "mechanical ventilation", "vasopressor use", "serum potassium", "serum bicarbonate", "surgical admission", "saps ii", "chronic kidney disease"], draft []
- **EvK:** treatment comparator: reference {"Etomidate": true, "Ketamine": false}, draft {"Etomidate": false, "Ketamine": false}; treatment arm 'Ketamine' components: reference ["Ketamine"], draft ["Ketamine", "Emergency Endotracheal Intubation", "Mechanical Ventilation"]; treatment arm 'Ketamine' grace period (hours): reference 1.0, draft null; analysis_plan estimand: reference "risk_difference", draft "hazard_ratio"; confounders only one annotator listed: reference ["age", "sex", "intubation location icu ward ed", "indication respiratory failure airway protection", "sepsis", "pre induction systolic blood pressure", "vasopressor use before intubation", "illness severity score", "neuromuscular blocker used", "trauma"], draft []
- **ECLS-SHOCK:** treatment arm 'ECLS' components: reference ["Extracorporeal life support (VA-ECMO)"], draft ["ECLS insertion", "PCI", "CABG", "Optimal medical treatment"]; treatment arm 'ECLS' grace period (hours): reference 6.0, draft null; analysis_plan estimand: reference "risk_difference", draft "hazard_ratio"; confounders only one annotator listed: reference ["age", "sex", "stemi vs nstemi", "resuscitation before time zero", "arterial lactate", "systolic blood pressure", "catecholamine use", "left ventricular ejection fraction", "multivessel coronary disease", "mechanical ventilation", "diabetes mellitus", "arterial ph", "serum creatinine"], draft []
- **LOVIT:** treatment arm 'Vitamin C' grace period (hours): reference 6.0, draft 24.0; analysis_plan estimand: reference "risk_difference", draft "hazard_ratio"; confounders only one annotator listed: reference ["age", "sex", "sofa score", "apache ii", "vasopressor dose", "serum lactate", "invasive mechanical ventilation", "source of infection", "covid 19", "charlson comorbidity index", "hydrocortisone", "thiamine"], draft []

## Reading these results

- These are the Trialist's drafts alone: no concept review, Clinician or Supervisor agent has touched them. The references are not gold: the clinician audit is still pending.
- **Confounders:** the Trialist does not propose confounders (the Clinician agent is not built), so recall is 0 by construction. This measures a missing agent, not a wrong one.
- **Estimand:** the Trialist defaults to a hazard ratio, and the schema has no risk ratio; the references chose risk differences for fixed-horizon mortality. A default, not a per-trial judgement.
- **Comparator:** wrong where the registry labels both arms ACTIVE_COMPARATOR (ACURASYS, APROCCHSS, ADRENAL) or neither (EvK), and the Trialist copied the label.
- **Horizon:** ACURASYS takes the registry's 36-month time frame (the paper's primary outcome is 90 days). BICAR-ICU's 0 days came from the time-frame parser reading 'Day 0 to Day 28' as day 0; the parser is fixed, the scored draft is kept as generated.
- **Grace period:** where the draft gives none and the reference gives one, the draft would be analysed by IPTW with an immortal-time flag rather than clone-censor-weight.
