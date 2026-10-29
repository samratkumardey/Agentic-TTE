# Adjudication: SKD vs AI

Record each decision in `adjudication.json` (`decision`: `SKD`, `AI` or your own value; `note`: why). Time zero also takes a `rating`: same, acceptable alternative, or different. Items you cannot settle go to the clinician.

## ACURASYS (NCT00299650)

**0. eligibility: I1 concepts**
- Registry text: ARDS with a PaO2 / FiO2 < 150 with a PEEP > 5 since less than 48 hrs
- SKD: ["PaO2/FiO2 ratio", "Positive end-expiratory pressure", "Acute respiratory distress syndrome"]
- AI: ["acute respiratory distress syndrome", "PaO2/FiO2 ratio", "positive end-expiratory pressure", "invasive mechanical ventilation"]

**1. eligibility: E2 concepts**
- Registry text: Continuous administration of NMBA for ARDS prior inclusion
- SKD: ["Neuromuscular blocking agent, continuous infusion"]
- AI: ["cisatracurium", "atracurium", "vecuronium", "rocuronium", "pancuronium"]

**2. eligibility: E5 concepts**
- Registry text: Persistent air leak
- SKD: ["Persistent air leak (bronchopleural fistula / pneumothorax with air leak)"]
- AI: ["persistent air leak", "pneumothorax"]

**3. time_zero: anchor event**
- SKD: "Randomization: first time ARDS with PaO2/FiO2 < 150 on PEEP >= 5 cmH2O is documented within 48 h of ARDS onset in a ventilated ICU patient; cisatracurium infusion starts at this point in the treated arm."
- AI: "severe ARDS (PaO2/FiO2 < 150 on PEEP >= 5 cm H2O) during invasive mechanical ventilation, within 48 hours of onset"

**4. outcome: end_event**
- SKD: null
- AI: "hospital discharge"

**5. outcome: end_rule**
- SKD: null
- AI: "first"

**6. confounders: only one annotator listed**
- SKD: ["ards etiology pneumonia sepsis", "time from ards onset to time zero"]
- AI: ["tidal volume", "sepsis"]

## CRISTAL (NCT00318942)

**7. eligibility: I2 concepts**
- Registry text: Hospitalized in intensive care unit
- SKD: ["Intensive care unit admission"]
- AI: ["intensive care unit stay"]

**8. eligibility: I3 concepts**
- Registry text: Need fluid resuscitation (according to the physician in charge of the patient)
- SKD: ["Hypovolemia", "Fluid resuscitation"]
- AI: ["hypotension", "systolic blood pressure", "serum lactate", "blood urea nitrogen"]

**9. eligibility: E3 concepts**
- Registry text: Coagulation disorders (haemophilia, Willebrand disease, etc.)
- SKD: ["Hemophilia", "Von Willebrand disease", "Coagulation factor deficiency"]
- AI: ["hemophilia", "von Willebrand disease", "coagulation defect"]

**10. eligibility: E5 concepts**
- Registry text: Severe hepatic failure
- SKD: ["Hepatic failure"]
- AI: ["hepatic failure", "cirrhosis of liver"]

**11. eligibility: E6 concepts**
- Registry text: Burns > 20 % of body surface
- SKD: ["Burn injury", "Total body surface area burned (%)"]
- AI: ["burn of more than 20% of body surface"]

**12. eligibility: E8 concepts**
- Registry text: Volume replacement: As a part of anaesthesia, for anaphylactic shock, or for dehydration
- SKD: ["Anaphylactic shock", "Dehydration", "Fluid replacement during anaesthesia"]
- AI: ["anesthesia", "anaphylactic shock", "dehydration"]

**13. eligibility: E10 concepts**
- Registry text: Advance directive of withholding or withdrawal of life-sustaining treatments
- SKD: ["Do not resuscitate / limitation of life-sustaining treatment"]
- AI: ["do not resuscitate status", "withdrawal of life-sustaining treatment"]

**14. treatment: arm '2' components**
- SKD: ["Albumin human", "Gelatin (succinylated / polygeline)", "Hydroxyethyl starch", "Dextran"]
- AI: ["colloid plasma substitute", "albumin human", "hydroxyethyl starch", "gelatin", "dextran"]

**15. time_zero: anchor event**
- SKD: "First fluid-resuscitation bolus ordered for hypovolaemia during the ICU stay (randomization was within ~15 min of the decision to resuscitate)."
- AI: "first need for fluid resuscitation for acute hypovolemia during the ICU stay (first resuscitation fluid bolus)"

**16. confounders: only one annotator listed**
- SKD: ["admission diagnosis stratum sepsis trauma haemorrhage other", "fluids received before time zero"]
- AI: ["sepsis", "multiple trauma", "hypovolemic shock"]

## APROCCHSS (NCT00625209)

**17. eligibility: I1 concepts**
- Registry text: hospitalized in intensive care unit for less than 7 days
- SKD: ["Intensive care unit admission"]
- AI: ["intensive care unit stay"]

**18. eligibility: I3 concepts**
- Registry text: at least one proven site of infection
- SKD: ["Infection, documented site"]
- AI: ["infection", "positive culture"]

**19. eligibility: I4 concepts**
- Registry text: at least 2 organ dysfunction as defined by a SOFA score =or> to 3 for at least 6 consecutive hours
- SKD: ["Number of organs with SOFA subscore >= 3"]
- AI: ["SOFA organ score"]

**20. eligibility: E1 concepts**
- Registry text: pregnancy or breath feeding
- SKD: ["Pregnancy", "Breastfeeding"]
- AI: ["pregnancy", "lactation"]

**21. eligibility: E2 concepts**
- Registry text: decision not to resuscitate
- SKD: ["Do not resuscitate / limitation of life-sustaining treatment"]
- AI: ["do not resuscitate status"]

**22. eligibility: E3 concepts**
- Registry text: underlying disease with an estimated life expectancy of less than 1 month
- SKD: ["Life expectancy < 1 month from underlying disease"]
- AI: ["terminal illness"]

**23. eligibility: E7 concepts**
- Registry text: chronic liver disease (Child C)
- SKD: ["Cirrhosis of liver", "Child-Pugh score"]
- AI: ["cirrhosis of liver, Child-Pugh class C"]

**24. eligibility: E8 concepts**
- Registry text: recent trauma (ie within the past 72 hours)
- SKD: ["Trauma"]
- AI: ["injury"]

**25. eligibility: E9 concepts**
- Registry text: intracranial process
- SKD: ["Intracranial mass / neoplasm"]
- AI: ["intracranial lesion"]

**26. eligibility: E12 concepts**
- Registry text: formal indication for curative anticoagulant; prophylactic use of heparin is allowed
- SKD: ["Therapeutic-dose anticoagulant"]
- AI: ["therapeutic anticoagulation"]

**27. eligibility: E23 category**
- Registry text: and a new exclusion criteria was added: "patients who had a previous episode of sepsis during the same hospital stay
- SKD: "clinical"
- AI: "administrative"

**28. treatment: arm '2' grace period (hours)**
- SKD: null
- AI: 6.0

**29. time_zero: anchor event**
- SKD: "Time septic shock criteria are first all met (documented infection, >= 2 organ SOFA >= 3, vasopressor at threshold dose for >= 6 h), within 24 h of shock onset."
- AI: "septic shock criteria met (infection, vasopressor >= 0.25 ug/kg/min for >= 6 h, SOFA >= 3 in >= 2 organs for >= 6 h), within 24 hours of shock onset"

**30. confounders: only one annotator listed**
- SKD: ["ards", "chronic comorbidity mccabe"]
- AI: ["community acquired infection"]

## CHEST (NCT00935168)

**31. eligibility: I3 category**
- Registry text: The ICU clinician considers that both 6% hydroxyethyl starch (130/0.4) and saline are equally appropriate for the patient and that no specific indication or contraindication for either exists.
- SKD: "administrative"
- AI: "clinical"

**32. eligibility: I4 concepts**
- Registry text: The requirement for fluid resuscitation must be supported by AT LEAST ONE of the following clinical signs: Heart rate > 90 beats per minute; Systolic blood pressure (SBP) < 100mmHg or mean arterial pressure (MAP) < 75mmHg or at least 40mmHg decrease in SBP or MAP from the baseline recording; Central
- SKD: ["Heart rate", "Systolic blood pressure", "Mean arterial pressure", "Central venous pressure", "Pulmonary artery wedge pressure", "Respiratory variation in arterial pressure (mmHg)", "Capillary refill time (s)", "Urine output (mL/kg/h)"]
- AI: ["heart rate", "systolic blood pressure", "mean arterial pressure", "central venous pressure", "urine output", "serum lactate"]

**33. eligibility: E2 concepts**
- Registry text: Primary non-traumatic intracranial haemorrhage or severe traumatic intracranial haemorrhage (mass lesion > 25 ml).
- SKD: ["Non-traumatic intracranial hemorrhage", "Traumatic intracranial hemorrhage, mass lesion > 25 mL"]
- AI: ["intracranial hemorrhage"]

**34. eligibility: E6 concepts**
- Registry text: Women of child bearing age (18-49 years old), unless evidence of documented menopause, hysterectomy or surgical sterilisation or negative pregnancy test before randomisation
- SKD: ["Sex", "Age", "Age"]
- AI: ["pregnancy"]

**35. eligibility: E10 concepts**
- Registry text: Patients admitted to the ICU for the treatment of burns or following liver transplantation surgery.
- SKD: ["Burn injury", "Liver transplantation"]
- AI: ["burn", "liver transplantation"]

**36. eligibility: E11 concepts**
- Registry text: Death is deemed imminent and inevitable or the patient has an underlying disease process with a life expectancy of < 90 days.
- SKD: ["Death imminent / life expectancy < 90 days"]
- AI: ["terminal illness"]

**37. time_zero: anchor event**
- SKD: "First fluid-resuscitation order in the study ICU during the current ICU admission (randomization occurred at the first resuscitation decision)."
- AI: "first fluid resuscitation bolus in the ICU judged necessary by the clinician (first ICU resuscitation fluid order)"

**38. confounders: only one annotator listed**
- SKD: ["operative vs non operative admission", "serum lactate"]
- AI: ["trauma", "acute kidney injury", "surgical admission"]

## ADRENAL (NCT01448109)

**39. eligibility: I2 concepts**
- Registry text: Documented site of infection, or strong suspicion of infection, with 2 of the 4 clinical signs of inflammation: Core temperature > 38°C or < 35°C; Heart rate > 90 beats per minute; White cell count > 12 x 109/L or < 4 x 109/L or > 10% immature neutrophils; Respiratory rate > 20 breaths per minute, o
- SKD: ["Infection, documented or suspected", "SIRS criteria met (count of temperature, HR, WBC, RR/PaCO2/MV)"]
- AI: ["infection", "body temperature", "body temperature", "heart rate", "white blood cell count", "respiratory rate"]

**40. eligibility: I4 concepts**
- Registry text: Being treated with vasopressors or inotropes to maintain a systolic blood pressure > 90mmHg, or mean arterial blood pressure > 60mmHg, or a MAP target set by the treating clinician for maintaining perfusion
- SKD: ["Norepinephrine", "Epinephrine", "Vasopressin", "Dopamine", "Phenylephrine", "Dobutamine"]
- AI: ["norepinephrine", "epinephrine", "vasopressin", "dopamine", "phenylephrine", "dobutamine", "milrinone"]

**41. eligibility: I5 concepts**
- Registry text: Administration of vasopressors or inotropes for = 4 hours and present at time of randomisation.
- SKD: ["Vasopressor/inotrope continuous infusion duration (h)"]
- AI: ["vasopressor"]

**42. eligibility: E1 concepts**
- Registry text: Met all inclusion criteria more than 24 hours ago
- SKD: ["Time since all inclusion criteria met (h)"]
- AI: ["septic shock"]

**43. eligibility: E8 concepts**
- Registry text: Death from underlying disease is likely within 90 days
- SKD: ["Life expectancy < 90 days from underlying disease"]
- AI: ["terminal illness"]

**44. treatment: arm 'Hydrocortisone' grace period (hours)**
- SKD: null
- AI: 6.0

**45. time_zero: anchor event**
- SKD: "First time all inclusion criteria are met: suspected/documented infection with >= 2 SIRS signs, on mechanical ventilation, and on vasopressor/inotrope for >= 4 h; randomization within 24 h of that."
- AI: "septic shock while mechanically ventilated: infection with >= 2 SIRS criteria and vasopressors or inotropes for >= 4 hours, within 24 hours of all criteria being met"

**46. confounders: only one annotator listed**
- SKD: ["operative vs non operative admission", "catecholamine dose vs 15 ug min", "sofa score"]
- AI: ["surgical admission", "time from shock onset", "norepinephrine dose"]

## HYDRAREA (NCT01636089)

**47. eligibility: I2 concepts**
- Registry text: in intensive care unit since 24h and for up to 48 hours
- SKD: ["Intensive care unit admission"]
- AI: ["intensive care unit stay"]

**48. eligibility: I3 concepts**
- Registry text: with contrast medium injection
- SKD: ["Iodinated contrast media, intravascular"]
- AI: ["contrast media", "computed tomography with contrast"]

**49. eligibility: E2 concepts**
- Registry text: rise in creatinine > 26µmol/L within 48hours
- SKD: ["Serum creatinine rise (umol/L)"]
- AI: ["serum creatinine increase"]

**50. treatment: arm 'bicarbonates' grace period (hours)**
- SKD: null
- AI: 1.0

**51. time_zero: anchor event**
- SKD: "Intravascular iodinated contrast administration in an ICU patient; study hydration starts 1 h before contrast (3 mL/kg) and runs 6 h after (1 mL/kg/h)."
- AI: "intravascular contrast medium administration for imaging in an ICU patient who has been in the ICU for at least 24 hours"

**52. confounders: only one annotator listed**
- SKD: ["nephrotoxic drugs aminoglycosides vancomycin"]
- AI: ["sofa score", "chronic kidney disease"]

## BICAR-ICU (NCT02476253)

**53. eligibility: I3 concepts**
- Registry text: Lactatemia of 2mmol/l or above, with pH of 7.20 or below and PaCO2 of 45mmHg or below and bicarbonatemia of 20mmol/l or below
- SKD: ["Arterial pH", "PaCO2 (mmHg)", "Serum bicarbonate (mmol/L)", "Serum lactate (mmol/L)"]
- AI: ["serum lactate", "arterial pH", "arterial PaCO2", "serum bicarbonate"]

**54. eligibility: E2 concepts**
- Registry text: Single respiratory disorder (PaCO2 > 50 mmHg, Bicarbonatemia equal or higher than (PaCO2-40)/10 + 24
- SKD: ["PaCO2 (mmHg)"]
- AI: ["respiratory acidosis", "arterial PaCO2"]

**55. eligibility: E3 concepts**
- Registry text: Acute diarrhea, ileostomy or biliary drainage
- SKD: ["Acute diarrhea", "Ileostomy", "Biliary drainage"]
- AI: ["diarrhea", "ileostomy", "biliary drainage"]

**56. eligibility: E4 concepts**
- Registry text: Stage IV kidney failure or chronic dialysis
- SKD: ["Chronic kidney disease stage 4", "Chronic maintenance dialysis"]
- AI: ["chronic kidney disease stage 4", "chronic kidney disease stage 5", "chronic dialysis"]

**57. eligibility: E5 concepts**
- Registry text: Tubular acidosis, ketoacidosis, high anion gap acids poisoning (PEG, aspirin, methanol)
- SKD: ["Renal tubular acidosis", "Ketoacidosis", "Poisoning by high anion gap acid (polyethylene glycol, salicylate, methanol)"]
- AI: ["renal tubular acidosis", "ketoacidosis", "poisoning by salicylate", "poisoning by methanol", "poisoning by ethylene glycol"]

**58. eligibility: E6 concepts**
- Registry text: PaCO2 equal to 45mmHg or above and spontaneous breathing, pregnancy, protected patients, moribund patient (life expectancy of 48h or below)
- SKD: ["PaCO2 (mmHg) with spontaneous breathing", "Pregnancy", "Life expectancy <= 48 h (moribund)"]
- AI: ["arterial PaCO2", "pregnancy", "terminal illness"]

**59. treatment: arm 'Intervention' grace period (hours)**
- SKD: null
- AI: 24.0

**60. time_zero: anchor event**
- SKD: "First arterial blood gas meeting severe metabolic acidaemia (pH <= 7.20, PaCO2 <= 45 mmHg, HCO3 <= 20 mmol/L) within 48 h of ICU admission."
- AI: "severe metabolic acidaemia (pH <= 7.20, PaCO2 <= 45 mm Hg, bicarbonate <= 20 mmol/L) within 48 hours of ICU admission"

**61. confounders: only one annotator listed**
- SKD: ["sepsis at inclusion"]
- AI: ["sepsis", "acute kidney injury", "serum bicarbonate"]

## CAPE COD (NCT02517489)

**62. eligibility: I3 concepts**
- Registry text: Admission to an Intensive Care Unit (ICU) or intermediate care unit participating to the trial
- SKD: ["Intensive care unit admission", "Intermediate care unit admission"]
- AI: ["intensive care unit stay"]

**63. eligibility: I7 category**
- Registry text: Study drug infusion initiated no longer than 24 hours post first severity criterion
- SKD: "administrative"
- AI: "clinical"

**64. eligibility: I8 concepts**
- Registry text: Severity defined by at least one of the following: Pneumonia Severity Index (PSI) > 130 (Fine class V); Patient placed on mechanical ventilation (invasive or not) for acute respiratory failure, with a PEEP level of 5 cm of water or more; Patient treated by high-flow oxygen therapy with a FiO2 of 50%
- SKD: ["Pneumonia Severity Index", "PEEP on invasive or non-invasive ventilation (cmH2O)", "PaO2/FiO2 on high-flow oxygen with FiO2 >= 0.5", "PaO2 on reservoir-bag mask (flow-dependent threshold, mmHg)"]
- AI: ["invasive mechanical ventilation", "noninvasive ventilation", "high-flow nasal oxygen", "Pneumonia Severity Index"]

**65. eligibility: I9 category**
- Registry text: Oxygen flow (L/min) 6 7 8 9 10 or more PaO2 (mmHg) less than 180 210 240 270 300
- SKD: "administrative"
- AI: "clinical"

**66. eligibility: I13 category**
- Registry text: Diagnosis of COVID19 either as certain (PCR) or probable (evocative clinical and radiological features AND epidemic context AND absence of other microbiological documentation).
- SKD: "administrative"
- AI: "clinical"

**67. eligibility: E1 concepts**
- Registry text: Patient treated by vasopressors for septic shock at the time of inclusion
- SKD: ["Septic shock", "Norepinephrine", "Epinephrine", "Vasopressin", "Dopamine", "Phenylephrine"]
- AI: ["vasopressor"]

**68. eligibility: E7 concepts**
- Registry text: Patients in which rapid PCR-test is positive for flu
- SKD: ["Influenza, PCR-confirmed"]
- AI: ["influenza"]

**69. eligibility: E8 concepts**
- Registry text: Active tuberculosis or fungal infection
- SKD: ["Active tuberculosis", "Fungal infection"]
- AI: ["tuberculosis", "mycosis"]

**70. eligibility: E10 concepts**
- Registry text: Myelosuppression
- SKD: ["Bone marrow suppression"]
- AI: ["bone marrow suppression", "neutropenia"]

**71. eligibility: E11 concepts**
- Registry text: Decision of withholding mechanical ventilation or endotracheal intubation
- SKD: ["Decision to withhold intubation / mechanical ventilation"]
- AI: ["do not intubate status"]

**72. eligibility: E14 concepts**
- Registry text: Patients under treatment by more than 15 mg/d of prednisone (or equivalent) for more than 30 days
- SKD: ["Prednisone-equivalent dose (mg/day)"]
- AI: ["prednisone"]

**73. time_zero: anchor event**
- SKD: "First time a severity criterion (I8) is met in a patient hospitalised with community-acquired pneumonia."
- AI: "first severity criterion of community-acquired pneumonia in the ICU (mechanical ventilation with PEEP >= 5, high-flow oxygen with FiO2 >= 50% and PaO2/FiO2 < 300, or PSI > 130)"

**74. confounders: only one annotator listed**
- SKD: ["pneumonia severity index", "mechanical ventilation at baseline", "c reactive protein", "time from severity criterion to treatment"]
- AI: ["saps ii", "psi score", "invasive mechanical ventilation", "noninvasive ventilation", "high flow nasal oxygen", "chronic obstructive lung disease"]

## STARRT-AKI (NCT02568722)

**75. eligibility: I2 concepts**
- Registry text: Admission to an intensive care unit (ICU)
- SKD: ["Intensive care unit admission"]
- AI: ["intensive care unit stay"]

**76. eligibility: I4 concepts**
- Registry text: Evidence of severe AKI defined by at least 1 of the following 3 criteria: ≥ 2-fold increase in serum creatinine from a known pre-morbid baseline or during the current hospitalization; OR ii) Achievement of a serum creatinine ≥ 354 µmol/L with evidence of a minimum increase of 27 µmol/L from pre-morb
- SKD: ["Serum creatinine / baseline ratio", "Serum creatinine (umol/L)", "Urine output over preceding 12 h (mL/kg)"]
- AI: ["acute kidney injury KDIGO stage 2", "acute kidney injury KDIGO stage 3", "serum creatinine", "urine output"]

**77. eligibility: E3 concepts**
- Registry text: Presence of a drug overdose that necessitates initiation of RRT
- SKD: ["Drug overdose requiring dialysis"]
- AI: ["poisoning"]

**78. eligibility: E7 concepts**
- Registry text: Known pre-hospitalization advanced chronic kidney disease, defined by an estimated glomerular filtration rate < 20 mL/min/1.73 m2
- SKD: ["eGFR before hospitalisation (mL/min/1.73m2)"]
- AI: ["chronic kidney disease stage 4", "chronic kidney disease stage 5", "estimated glomerular filtration rate"]

**79. eligibility: E8 concepts**
- Registry text: Presence or clinical suspicion of renal obstruction, rapidly progressive glomerulonephritis, vasculitis, thrombotic microangiopathy or acute interstitial nephritis
- SKD: ["Urinary tract obstruction", "Rapidly progressive glomerulonephritis", "Vasculitis", "Thrombotic microangiopathy", "Acute interstitial nephritis"]
- AI: ["obstructive uropathy", "rapidly progressive glomerulonephritis", "vasculitis", "thrombotic microangiopathy", "acute interstitial nephritis"]

**80. treatment: arm 'Accelerated RRT initiation' components**
- SKD: ["Renal replacement therapy", "Dialysis catheter insertion"]
- AI: ["renal replacement therapy"]

**81. time_zero: anchor event**
- SKD: "First time all eligibility criteria are met (ICU, creatinine threshold, severe AKI criterion) with no urgent RRT indication."
- AI: "full eligibility for severe AKI (KDIGO stage 2-3 with creatinine >= 100/130 umol/L, no emergency indication for RRT) in an ICU patient"

**82. confounders: only one annotator listed**
- SKD: ["apache ii"]
- AI: ["saps ii", "chronic kidney disease", "fluid balance"]

## EvK (NCT02643381)

**83. treatment: arm 'Ketamine' grace period (hours)**
- SKD: null
- AI: 1.0

**84. time_zero: anchor event**
- SKD: "Emergency endotracheal intubation (induction drug administration)."
- AI: "emergency endotracheal intubation (induction of anesthesia) in an adult hospital patient"

**85. confounders: only one annotator listed**
- SKD: ["indication respiratory failure airway protection", "pre induction systolic blood pressure", "illness severity score", "neuromuscular blocker used"]
- AI: ["sofa score", "trauma", "hypotension", "admission diagnosis", "neuromuscular blocking agent"]

## ECLS-SHOCK (NCT03637205)

**86. eligibility: I3 concepts**
- Registry text: Systolic blood pressure <90 mmHg >30 min or catecholamines required to maintain pressure >90 mmHg during Systole
- SKD: ["Systolic blood pressure < 90 mmHg for > 30 min", "Norepinephrine", "Epinephrine", "Vasopressin", "Dopamine", "Phenylephrine"]
- AI: ["systolic blood pressure", "norepinephrine", "epinephrine", "dopamine", "dobutamine"]

**87. eligibility: E2 concepts**
- Registry text: Mechanical cause of cardiogenic shock
- SKD: ["Ventricular septal rupture", "Papillary muscle rupture", "Left ventricular free wall rupture"]
- AI: ["ventricular septal rupture", "papillary muscle rupture", "rupture of cardiac free wall"]

**88. eligibility: E4 concepts**
- Registry text: Severe peripheral artery disease with impossibility to insert ECLS cannulae
- SKD: ["Severe peripheral artery disease"]
- AI: ["peripheral arterial disease"]

**89. eligibility: E7 concepts**
- Registry text: Other severe concomitant disease with limited life expectancy <6 months
- SKD: ["Life expectancy < 6 months from other disease"]
- AI: ["terminal illness"]

**90. treatment: arm 'ECLS' components**
- SKD: ["Extracorporeal life support (VA-ECMO)"]
- AI: ["extracorporeal membrane oxygenation"]

**91. treatment: arm 'ECLS' grace period (hours)**
- SKD: null
- AI: 6.0

**92. time_zero: anchor event**
- SKD: "Diagnostic coronary angiography in AMI with cardiogenic shock, before revascularization."
- AI: "coronary angiography in acute myocardial infarction with cardiogenic shock (SBP < 90 mm Hg > 30 min or catecholamines, lactate > 3 mmol/L, hypoperfusion) with planned revascularization"

**93. confounders: only one annotator listed**
- SKD: ["stemi vs nstemi", "resuscitation before time zero", "arterial lactate", "systolic blood pressure", "catecholamine use", "multivessel coronary disease"]
- AI: ["serum lactate", "arterial ph", "cardiac arrest", "st elevation myocardial infarction", "scai shock stage", "serum creatinine", "number of diseased coronary vessels", "vasopressor use"]

## LOVIT (NCT03680274)

**94. eligibility: I1 concepts**
- Registry text: Admitted to the intensive care unit with proven or suspected infection as the main diagnosis;
- SKD: ["Intensive care unit admission", "Infection, proven or suspected"]
- AI: ["intensive care unit stay", "infection", "sepsis"]

**95. eligibility: E1 concepts**
- Registry text: > 24 hours of intensive care unit admission;
- SKD: ["Time since ICU admission (h)"]
- AI: ["intensive care unit stay"]

**96. eligibility: E4 concepts**
- Registry text: Known allergy to vitamin C;
- SKD: ["Allergy to vitamin C"]
- AI: ["allergy to ascorbic acid"]

**97. eligibility: E7 concepts**
- Registry text: Expected death or withdrawal of life-sustaining treatments within 48 hours;
- SKD: ["Expected death / withdrawal of life support within 48 h"]
- AI: ["limitation of treatment"]

**98. treatment: arm 'Vitamin C' grace period (hours)**
- SKD: null
- AI: 6.0

**99. time_zero: anchor event**
- SKD: "First time an ICU patient with proven/suspected infection is on a continuous vasopressor infusion, within 24 h of ICU admission."
- AI: "vasopressor infusion for proven or suspected infection within the first 24 hours of ICU admission"

**100. confounders: only one annotator listed**
- SKD: ["vasopressor dose", "charlson comorbidity index"]
- AI: ["norepinephrine dose", "hydrocortisone", "thiamine"]
