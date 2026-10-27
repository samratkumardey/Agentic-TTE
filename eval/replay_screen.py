"""Phase 3 Replay: screening of candidate trials against S1-S8 (eval/SCORING.md, R3).

Builds eval/reference/replay_selection.json from the saved registry search
(eval/replay_search.py) plus the parity trials and the PLAN.md §17.3 named candidates.

Stage A is automatic, from registry fields. Stage B is the judgment recorded in JUDGED below:
made by Claude (AI) on 2026-10-27 from the registry record, and to be checked by the user.
S2 values still need extraction from the primary paper and a second-person check; S8 needs
counts from the Data Support Gate on Hellbender.

    uv run python eval/replay_screen.py
"""

import json
import re

# ruff: noqa: E501  (one screening decision per line)
import sys
from pathlib import Path

REF = Path(__file__).parent / "reference"
SEARCH = REF / "replay_search_2026-10-27.json"  # wave 1 (R3)
SEARCH2 = REF / "replay_search_wave2_2026-10-27.json"  # wave 2 (changelog item 5)
PUBMED = REF / "replay_pubmed_links_2026-10-27.json"  # changelog item 6
NAMED = REF / "replay_candidates_named_2026-10-27.json"  # the parity and named trials, fetched

OBSERVABLE_TYPES = {"DRUG", "BIOLOGICAL", "PROCEDURE", "DIETARY_SUPPLEMENT", "COMBINATION_PRODUCT"}
MODELS = {"PARALLEL", "FACTORIAL", "CROSSOVER"}

# Stage B. status: primary | secondary (S7) | exploratory (S5) | excluded | uncertain |
# conditional (passes once an unmapped MIMIC item is mapped; changelog item 7).
# fail: the first criterion not met; note: what the decision rests on.
JUDGED: dict[str, tuple[str, str | None, str]] = {
    # --- passes S1-S7 on the registry record (S2 extraction and S8 pending) ---
    "NCT00148278": ("primary", None, "CATS: norepinephrine + dobutamine vs epinephrine; 28-day death"),
    "NCT00318942": ("primary", None, "CRISTAL: colloids vs crystalloids; 28-day death. LR (a crystalloid) is unmapped in our inputevents map; saline and colloids are mapped"),
    "NCT00625209": ("primary", None, "APROCCHSS: hydrocortisone + fludrocortisone vs placebo; 90-day death (the activated protein C arm was stopped)"),
    "NCT02509078": ("primary", None, "ROSE: early cisatracurium in ARDS; in-hospital death to day 90"),
    "NCT03345992": ("primary", None, "INCLASS: clarithromycin vs placebo in sepsis; 28-day death"),
    "NCT03654001": ("primary", None, "ALBIOSS-BALANCED: albumin factorial; 90-day death. The balanced-solution factor needs LR, unmapped"),
    "NCT04010630": ("primary", None, "BICARICU-2: sodium bicarbonate in severe metabolic acidosis with AKI; 90-day death"),
    "NCT01636089": ("primary", None, "HYDRAREA: bicarbonate vs saline before contrast in ICU patients; contrast AKI at 72 h (creatinine, structured)"),
    "NCT01448109": ("primary", None, "ADRENAL: hydrocortisone vs placebo in septic shock on ventilation; 90-day death"),
    "NCT02467621": ("primary", None, "SUP-ICU: pantoprazole vs placebo; 90-day death"),
    "NCT02856698": ("primary", None, "MIMO (parity): midazolam vs morphine in acute pulmonary oedema (ED); in-hospital death"),
    "NCT03872011": ("primary", None, "Parity: vitamin C + thiamine + hydrocortisone vs placebo; 90-day death"),
    "NCT03258684": ("primary", None, "Parity: hydrocortisone + vitamin C + thiamine vs saline; hospital survival to day 14"),
    "NCT01223690": ("primary", None, "Parity: clarithromycin vs placebo; 28-day death"),
    "NCT02568722": ("primary", None, "STARRT-AKI: accelerated vs standard RRT initiation; 90-day death. A timing strategy with a grace period (clone-censor-weight), not a physiological target"),
    "NCT03680274": ("primary", None, "LOVIT: vitamin C vs placebo; death or persistent organ dysfunction (vasopressors, ventilation, RRT) at day 28, all in structured data while in hospital"),
    # --- passes, but one criterion needs a closer look ---
    "NCT00147004": ("uncertain", "S3", "CORTICUS: hydrocortisone vs placebo; the primary population is ACTH-test non-responders, which MIMIC may not record"),
    "NCT00604214": ("uncertain", "S8", "PROWESS-SHOCK: drotrecogin alfa vs placebo; 28-day death. Drotrecogin is in MIMIC (Xigris) but rare: S8 likely fails"),
    "NCT00832039": ("uncertain", "S4", "SISPCT: sodium selenite factorial; 28-day death. Whether IV selenite is given in MIMIC is unknown (S8)"),
    "NCT01915719": ("uncertain", "S4", "IVNICTUS: early non-invasive ventilation vs oxygen; 28-day death. NIV must be identifiable in procedureevents"),
    "NCT02962102": ("uncertain", "S2", "ACTIVATE-AKI: calcifediol/calcitriol vs placebo; death within 7 days, a phase 2 trial; the published estimate must be checked"),
    "NCT06390748": ("uncertain", "S6", "Esmolol in sepsis: the primary outcome's horizon is not stated in the registry"),
    "NCT03653832": ("uncertain", "S6", "A2B: dexmedetomidine or clonidine vs propofol; time to successful extubation (procedureevents). Extubation as an event needs checking"),
    "NCT01728558": ("uncertain", "S4", "SPICE III: early goal-directed sedation (dexmedetomidine-based) vs usual care; 90-day death. Emulable only as early dexmedetomidine vs not"),
    # --- secondary agreement set (S7: partly ascertainable outcome) ---
    "NCT00962156": ("secondary", "S7", "6S: HES vs Ringer's acetate; death or dialysis dependence at 90 days (dialysis after discharge is not seen)"),
    "NCT00475852": ("secondary", "S7", "ASCEND-HF (parity): nesiritide vs placebo; HF rehospitalization or death at 30 days (rehospitalization elsewhere is not seen)"),
    "NCT02444988": ("conditional", "S4", "SMART: balanced crystalloids vs saline; MAKE30 (structured). Needs LR and Plasma-Lyte mapped (changelog item 7)"),
    # --- exploratory (S5: dynamic or protocolized strategies) ---
    "NCT00115479": ("exploratory", "S5", "Leuven medical ICU: intensive insulin to a glucose target"),
    "NCT00320099": ("exploratory", "S5", "COIITSS: the primary comparison is intensive insulin to a glucose target"),
    "NCT01485315": ("exploratory", "S5", "TRISS: haemoglobin transfusion threshold"),
    "NCT00220987": ("exploratory", "S5", "NICE-SUGAR: glucose target"),
    "NCT03229941": ("exploratory", "S5", "TOP: transfusion threshold; composite with myocardial infarction"),
    "NCT03434028": ("exploratory", "S5", "CLOVERS: protocolized restrictive vs liberal fluid resuscitation"),
    "NCT05629780": ("exploratory", "S5", "CLASSIC sub-study: restrictive fluid strategy; time to lactate resolution"),
    "NCT04425031": ("exploratory", "S5", "HOT-COVID: oxygenation target; days alive without organ support"),
    # --- excluded: intervention not observable as given in MIMIC (S4) ---
    "NCT00089986": ("excluded", "S4", "Investigational phospholipid emulsion, never marketed"),
    "NCT00143611": ("excluded", "S4", "Resatorvid, investigational"),
    "NCT02330432": ("excluded", "S4", "Mycobacterium w, not used in US practice"),
    "NCT02867267": ("excluded", "S4", "Thymosin alpha 1, not used in US practice"),
    "NCT01598831": ("excluded", "S4", "ART-123 (thrombomodulin), investigational"),
    "NCT01870778": ("excluded", "S4", "Serelaxin, investigational; cardiovascular death is not recorded"),
    "NCT02182440": ("excluded", "S4", "Recombinant alkaline phosphatase, investigational"),
    "NCT01777165": ("excluded", "S4", "ABT-719, investigational"),
    "NCT03941483": ("excluded", "S4", "ASP1128, investigational"),
    "NCT02338843": ("excluded", "S6", "ATHOS-3: the outcome is a MAP response at 3 hours"),
    "NCT04402866": ("excluded", "S4", "TD-0903, investigational"),
    "NCT04478071": ("excluded", "S4", "Vadadustat for COVID ARDS, investigational use"),
    "NCT04426695": ("excluded", "S4", "SARS-CoV-2 antibodies, outpatients"),
    "NCT00076219": ("excluded", "S4", "ATN: RRT intensity (dose), not identifiable in structured data"),
    "NCT00527813": ("conditional", "S4", "PROSEVA: prone positioning; 28-day death. Needs the patient-position items mapped (changelog item 7)"),
    "NCT01799993": ("excluded", "S4", "INHALE: inhaled amikacin; route is not kept in drug_exposure"),
    "NCT03452839": ("excluded", "S4", "MERCY: continuous vs bolus meropenem; administration mode not kept; composite with resistance"),
    "NCT03098524": ("excluded", "S4", "MECANO: ventilation during cardiopulmonary bypass (intraoperative)"),
    "NCT03709264": ("conditional", "S4", "PROTection: IV amino acids after cardiac surgery; AKI (creatinine). Needs IV amino acids mapped (changelog item 7)"),
    "NCT04325906": ("excluded", "S4", "HFNC with prone positioning, not observable"),
    "NCT01572337": ("excluded", "S4", "NAVIGATE: NIV outside the ICU; the outcome is ICU admission"),
    "NCT07249151": ("excluded", "S4", "Nursing assessment bundle in the ED"),
    "NCT03573739": ("excluded", "S4", "NUTRIREA3: low-calorie enteral feeding; enteral formulas are unmapped"),
    "NCT03884543": ("excluded", "S4", "DESIGNATION: intraoperative PEEP strategy"),
    "NCT03201263": ("excluded", "S4", "PROTECTION: sigh ventilation; feasibility outcome"),
    "NCT02424773": ("excluded", "S4", "RESPIR-OH: oxygen devices"),
    "NCT00213499": ("excluded", "S4", "VENISE: NIV for weaning; weaning failure"),
    "NCT05465291": ("excluded", "S4", "Neuromuscular electrical stimulation"),
    "NCT05932134": ("excluded", "S4", "Diet and rehabilitation"),
    "NCT04714736": ("excluded", "S4", "DyeVert contrast-reduction device"),
    "NCT01560650": ("excluded", "S4", "Parity: CRRT intensity (dose), not identifiable; no published result"),
    # --- excluded: setting (S3) ---
    "NCT02473263": ("excluded", "S3", "Prehospital (SAMU) treatment"),
    "NCT03871491": ("excluded", "S3", "A-PLUS: women in labour; outcome over 6 weeks after delivery"),
    "NCT04308668": ("excluded", "S3", "COVID post-exposure prophylaxis, outpatients"),
    "NCT04311177": ("excluded", "S3", "COVID outpatients"),
    "NCT04328467": ("excluded", "S3", "COVID pre-exposure prophylaxis, outpatients"),
    "NCT00835003": ("excluded", "S3", "Timing of elective caesarean; neonatal outcome"),
    "NCT00337051": ("excluded", "S3", "Renal transplantation anaesthesia; graft outcome"),
    "NCT03098706": ("excluded", "S3", "Brain-dead organ donors; graft outcome"),
    "NCT00318188": ("excluded", "S3", "Scleroderma rehabilitation, outpatients"),
    "NCT00402311": ("excluded", "S3", "Outpatient multidisciplinary care"),
    "NCT00175227": ("excluded", "S3", "Hydration before elective angiography, largely outpatients"),
    "NCT02980003": ("excluded", "S3", "TEATE: urine alkalinization before angiography"),
    "NCT01467466": ("excluded", "S3", "PRESERVE: angiography patients; 90-day outcome after discharge (S7)"),
    "NCT02554890": ("excluded", "S6", "PIONEER-HF: NT-proBNP outcome"),
    "NCT02970669": ("excluded", "S3", "AWAKE-HF: outpatients; activity outcome"),
    "NCT07651150": ("excluded", "S6", "Preoperative dexamethasone; delirium (unmapped)"),
    # --- excluded: no binary or time-to-event primary outcome (S2) ---
    "NCT00097630": ("excluded", "S2", "ABC: ventilator-free days"),
    "NCT00464204": ("excluded", "S2", "CRYSTMAS: volume to stabilization"),
    "NCT00441792": ("excluded", "S2", "Length of stay"),
    "NCT00466492": ("excluded", "S2", "Duration of ventilation and stay"),
    "NCT00472160": ("excluded", "S2", "PREOXY: maximum SOFA"),
    "NCT00534287": ("excluded", "S2", "MaxSep: mean SOFA"),
    "NCT01132846": ("excluded", "S2", "ROSE-AHF: cystatin C change"),
    "NCT01335932": ("excluded", "S2", "GRAIL: IL-6 level"),
    "NCT01420302": ("excluded", "S2", "LOGIC-1: glycaemic penalty index"),
    "NCT01626612": ("excluded", "S2", "Length of stay"),
    "NCT01993836": ("excluded", "S2", "CSF biomarker correlation"),
    "NCT02105415": ("excluded", "S2", "KEEP PACE: mean arterial pressure"),
    "NCT02881957": ("excluded", "S2", "Length of stay"),
    "NCT03389555": ("excluded", "S2", "ACTS: SOFA change"),
    "NCT04235608": ("excluded", "S2", "SESAR: ventilator-free days"),
    "NCT04291508": ("excluded", "S2", "ASTER: days alive and free of organ support"),
    "NCT04578886": ("excluded", "S2", "Days without delirium"),
    "NCT04640194": ("excluded", "S2", "TRISTARDS: time to improvement, small"),
    "NCT05167734": ("excluded", "S2", "Haemoglobin concentration"),
    "NCT05312385": ("excluded", "S2", "INSPiRE-ICU1: time in sedation range"),
    "NCT05327296": ("excluded", "S2", "INSPiRE-ICU2: time in sedation range"),
    "NCT05411991": ("excluded", "S2", "DECONGEST: hierarchical composite"),
    "NCT05682716": ("excluded", "S2", "Creatinine level"),
    "NCT05916586": ("excluded", "S2", "CRP change"),
    "NCT04315298": ("excluded", "S2", "Sarilumab: CRP change"),
    "NCT01739933": ("excluded", "S2", "MENDS2: delirium/coma-free days"),
    "NCT03392376": ("excluded", "S2", "AID-ICU: days alive out of hospital"),
    "NCT03509350": ("excluded", "S2", "VICTAS: vasopressor- and ventilator-free days"),
    # --- excluded: outcome not ascertainable in MIMIC (S6) ---
    "NCT00135291": ("excluded", "S6", "Infection within 30 days"),
    "NCT00182364": ("excluded", "S6", "PROTECT: proximal DVT on screening ultrasound"),
    "NCT00277888": ("excluded", "S6", "CATHEDIA: catheter insertion complications"),
    "NCT00893763": ("excluded", "S6", "SToPP2: VAP by CPIS"),
    "NCT01137487": ("excluded", "S6", "NUTRIREA1: VAP"),
    "NCT01504867": ("excluded", "S6", "LIPS-A: ARDS development"),
    "NCT01690988": ("excluded", "S6", "PODCAST: delirium (CAM items unmapped)"),
    "NCT01945983": ("excluded", "S6", "CENSER: shock-control target at 6 hours"),
    "NCT01962116": ("excluded", "S6", "VERROU-REA: catheter dysfunction-free survival"),
    "NCT02267538": ("conditional", "S6", "Dexmedetomidine after cardiac surgery; delirium. Needs the delirium assessments (CAM-ICU) mapped (changelog item 7)"),
    "NCT02361528": ("excluded", "S6", "GRID: ICU-acquired infection"),
    "NCT02723383": ("excluded", "S6", "BACLOREA: agitation-related adverse events"),
    "NCT02856594": ("conditional", "S6", "MINDDS: dexmedetomidine after cardiac surgery; delirium. Needs CAM-ICU mapped (changelog item 7)"),
    "NCT02937454": ("excluded", "S6", "AFFIRM-AHF: HF hospitalization and cardiovascular death"),
    "NCT03138733": ("excluded", "S6", "Clinical success at day 70"),
    "NCT03207100": ("excluded", "S6", "ASSICHH: blood-pressure control at 1 hour"),
    "NCT03317067": ("excluded", "S6", "4D: duration of agitation"),
    "NCT04341844": ("excluded", "S6", "Postoperative delirium (unmapped)"),
    "NCT06346522": ("excluded", "S6", "Catheter-related bladder discomfort"),
    "NCT06443957": ("excluded", "S6", "Postoperative delirium; intranasal insulin"),
    "NCT00953303": ("excluded", "S6", "Parity: cardiovascular death (cause of death is not recorded)"),
    # --- excluded: no published result (S2) or not randomized (S1) ---
    "NCT04134403": ("excluded", "S2", "STASIS (parity): no published result found"),
    "NCT03592693": ("excluded", "S2", "CORVICTES (parity): no published result found; n = 24"),
    "NCT06091982": ("excluded", "S1", "Parity: the registry record has no randomized design (observational?); no published result"),
}

# Wave 2 (changelog items 5-9): stage-B decisions for the hits kept by the outcome rule B1.
JUDGED_W2: dict[str, tuple[str, str | None, str]] = {
    # --- primary ---
    "NCT02984384": ("primary", None, "PREVENT CLOT: aspirin vs LMWH after orthopaedic trauma (inpatients); 90-day death"),
    "NCT00935168": ("primary", None, "CHEST: hydroxyethyl starch vs saline in ICU; 90-day death"),
    "NCT05277896": ("primary", None, "RSI: ketamine vs etomidate for emergency intubation; 28-day in-hospital death"),
    "NCT00314704": ("primary", None, "SOAP II: dopamine vs norepinephrine in shock; 28-day death"),
    "NCT03096314": ("primary", None, "VIOLET: enteral vitamin D3 vs placebo; 90-day death"),
    "NCT02309151": ("primary", None, "DISCO: immediate vs deferred coronary angiography after cardiac arrest; 30-day survival"),
    "NCT02517489": ("primary", None, "CAPE COD: hydrocortisone vs placebo in severe community-acquired pneumonia; 28-day death"),
    "NCT02643381": ("primary", None, "EvK: etomidate vs ketamine for emergency intubation; day-7 survival"),
    "NCT01870804": ("primary", None, "PRATO-ACS-2: rosuvastatin vs atorvastatin in ACS before angiography; contrast AKI at 72 h"),
    "NCT01932190": ("primary", None, "AKIKI: early vs delayed RRT; 60-day survival"),
    "NCT00135473": ("primary", None, "VISEP: hydroxyethyl starch vs crystalloid (the volume comparison); 28-day death. The insulin comparison is a glucose target (S5)"),
    "NCT03637205": ("primary", None, "ECLS-SHOCK: ECLS vs usual care in infarct-related cardiogenic shock; 30-day death"),
    "NCT02476253": ("primary", None, "BICAR-ICU: sodium bicarbonate vs none in severe acidaemia; day-28 death or organ failure at day 7"),
    "NCT00299650": ("primary", None, "ACURASYS: cisatracurium vs placebo in ARDS; 90-day death"),
    "NCT01718613": ("primary", None, "VANCS II: vasopressin vs norepinephrine in cancer patients with septic shock; 28-day death"),
    "NCT01760967": ("primary", None, "DESIRE: dexmedetomidine vs none in ventilated sepsis; 28-day death"),
    "NCT01284452": ("primary", None, "Hydrocortisone vs placebo in severe sepsis; 28-day death"),
    "NCT03338569": ("primary", None, "Vitamin C vs placebo in septic shock; ICU death to day 28"),
    "NCT02580240": ("primary", None, "Hydrocortisone vs saline in septic shock; 28-day death"),
    "NCT01455688": ("primary", None, "Early vs late antiretroviral therapy in critically ill HIV patients; hospital death"),
    "NCT05778838": ("primary", None, "Midodrine vs none in septic shock; death (horizon to confirm in the paper)"),
    # --- uncertain ---
    "NCT03680963": ("uncertain", "S4", "EVERDAC: early arterial catheter vs a non-invasive strategy; 28-day death. Arterial lines are in procedureevents; the strategy needs a grace period"),
    "NCT00639743": ("uncertain", "S6", "PEITHO: tenecteplase in intermediate-risk PE; death or haemodynamic collapse at day 7. Collapse needs a structured definition"),
    "NCT02019420": ("uncertain", "S8", "Tedizolid vs linezolid in ventilated pneumonia; 28-day death. Tedizolid is likely rare in MIMIC"),
    "NCT02070757": ("uncertain", "S8", "ASPECT-NP: ceftolozane/tazobactam vs meropenem; 28-day death. Ceftolozane is likely rare"),
    "NCT01927549": ("uncertain", "S4", "CULPRIT-SHOCK: culprit-only vs multivessel PCI; 30-day death or RRT. Vessel count only in ICD-10-PCS"),
    "NCT04636671": ("uncertain", "S8", "MEDEAS: methylprednisolone vs dexamethasone in COVID pneumonia; survival. COVID admissions in MIMIC-IV 3.1 to check"),
    "NCT00621790": ("uncertain", "S8", "Fenoldopam vs placebo after cardiac surgery; RRT. Fenoldopam use to check"),
    "NCT00791648": ("uncertain", "S4", "Atorvastatin vs placebo before cardiac surgery; AKI. Statin start may be before admission"),
    "NCT02493764": ("uncertain", "S8", "RESTORE-IMI 2: imipenem/relebactam vs piperacillin/tazobactam; 28-day death. Relebactam is likely rare"),
    "NCT03869437": ("uncertain", "S8", "GAMECHANGER: cefiderocol vs best available therapy; 14-day death. Cefiderocol is likely rare"),
    "NCT04362813": ("uncertain", "S8", "CAN-COVID: canakinumab; survival without ventilation. COVID admissions and use to check"),
    "NCT04343729": ("uncertain", "S8", "MetCOVID: methylprednisolone vs placebo in COVID; 28-day death. COVID admissions to check"),
    "NCT04372186": ("uncertain", "S8", "EMPACTA: tocilizumab in COVID; death or ventilation by day 28. COVID admissions to check"),
    "NCT03032380": ("uncertain", "S8", "APEKS-NP: cefiderocol vs meropenem; 14-day death. Cefiderocol is likely rare"),
    "NCT03152474": ("uncertain", "S8", "Methylprednisolone or vitamin A vs placebo in sepsis; 28-day death. IV vitamin A use to check"),
    "NCT03583333": ("uncertain", "S8", "Imipenem/relebactam vs piperacillin/tazobactam; 28-day death. Relebactam is likely rare"),
    "NCT02836899": ("uncertain", "S4", "Nitric oxide during cardiac surgery; AKI. Delivery during bypass is not recorded"),
    "NCT01791868": ("uncertain", "S6", "VALSE: IV valproate in status epilepticus; discharged alive at day 15 (definition to check)"),
    "NCT00322530": ("uncertain", "S4", "Dialysis strategies in critically ill patients; 90-day death. Which modalities are compared needs checking"),
    "NCT04598594": ("uncertain", "S8", "NICOVID-REA: nicotine patches in ventilated COVID; 28-day death. COVID admissions to check"),
    "NCT01577862": ("uncertain", "S8", "CoRAb: colistin with vs without rifampicin for MDR Acinetobacter; 30-day death. The population needs microbiology, not in our CDM"),
    "NCT00297674": ("uncertain", "S6", "IV clarithromycin in septic syndrome; mortality, horizon not stated in the registry"),
    "NCT03549923": ("uncertain", "S4", "ELITE: early CRRT on ECMO; 30-day death. Timing strategy to check"),
    "NCT04443881": ("uncertain", "S8", "Anakinra in COVID cytokine storm; no ventilation by day 15. COVID admissions to check"),
    "NCT01840241": ("uncertain", "S3", "Sodium bicarbonate vs saline to prevent AKI; population to check"),
    "NCT01394419": ("uncertain", "S3", "N-acetylcysteine vs saline to prevent AKI after surgery; population to check"),
    # --- conditional (changelog item 7) ---
    "NCT02875873": ("conditional", "S4", "BaSICS: Plasma-Lyte vs saline; 90-day death. Needs balanced crystalloids mapped"),
    "NCT02721654": ("conditional", "S4", "PLUS: Plasma-Lyte vs saline; 90-day death. Needs balanced crystalloids mapped"),
    # --- exploratory (S5) ---
    "NCT00107601": ("exploratory", "S5", "Glucontrol: glucose target"),
    "NCT03174002": ("exploratory", "S5", "HOT-ICU: oxygenation target"),
    "NCT02908308": ("exploratory", "S5", "TTM-2: temperature target after cardiac arrest"),
    "NCT03668236": ("exploratory", "S5", "CLASSIC: restrictive fluid strategy"),
    "NCT00510835": ("exploratory", "S5", "ProCESS: protocolized resuscitation"),
    "NCT01020916": ("exploratory", "S5", "TTM: 33 vs 36 degrees"),
    "NCT01149278": ("exploratory", "S5", "SEPSISPAM: blood-pressure target"),
    "NCT00182195": ("exploratory", "S5", "LOVS: open-lung ventilation strategy"),
    "NCT00282698": ("exploratory", "S5", "Tight glucose control in cardiac surgery"),
    "NCT00270673": ("exploratory", "S5", "Lactate-directed therapy"),
    "NCT00372502": ("exploratory", "S5", "LactATES: resuscitation end point"),
    # --- excluded: setting (S3) ---
    "NCT01401647": ("excluded", "S3", "ALPS: out-of-hospital cardiac arrest drugs"),
    "NCT01988428": ("excluded", "S3", "PHANTASi: prehospital antibiotics"),
    "NCT00127907": ("excluded", "S3", "Out-of-hospital cardiac arrest drugs"),
    "NCT00365950": ("excluded", "S3", "Anticoagulation duration for DVT, outpatients"),
    "NCT02327026": ("excluded", "S3", "CAAM: prehospital airway"),
    "NCT03452917": ("excluded", "S3", "SNOCAT: prehospital cardiac arrest"),
    "NCT02000674": ("excluded", "S3", "CURASMUR: prehospital intubation"),
    "NCT04684719": ("excluded", "S3", "TOWAR: prehospital whole blood"),
    "NCT00121524": ("excluded", "S3", "Out-of-hospital cardiac arrest drugs"),
    "NCT03872960": ("excluded", "S3", "ARREST: prehospital transfer"),
    "NCT01818427": ("excluded", "S3", "PAMPer: prehospital plasma"),
    "NCT04080986": ("excluded", "S3", "DOSEVF: prehospital defibrillation"),
    "NCT03080883": ("excluded", "S3", "Apixaban in cancer outpatients"),
    "NCT01612013": ("excluded", "S3", "Contrast AKI prevention in angiography patients"),
    "NCT02799368": ("excluded", "S3", "Contrast AKI prevention in angiography patients"),
    "NCT02225431": ("excluded", "S3", "Contrast AKI prevention in angiography patients"),
    "NCT02070679": ("excluded", "S3", "Contrast AKI prevention in angiography patients"),
    "NCT00131274": ("excluded", "S3", "Idiopathic pulmonary fibrosis outpatients"),
    # --- excluded: intervention not observable (S4) ---
    "NCT02105610": ("excluded", "S4", "MYRIAD: intraoperative volatile anaesthesia"),
    "NCT00293592": ("excluded", "S4", "DECS: intraoperative dexamethasone"),
    "NCT02306746": ("excluded", "S4", "TARGET: enteral energy (formulas unmapped)"),
    "NCT00049777": ("excluded", "S4", "Heparin during drotrecogin alfa"),
    "NCT00334828": ("excluded", "S4", "ACCESS: eritoran, investigational"),
    "NCT00221013": ("excluded", "S4", "RENAL: CRRT intensity (dose)"),
    "NCT00348504": ("excluded", "S4", "SURVIVE: levosimendan, not used in the US"),
    "NCT03218553": ("excluded", "S4", "PACMAN: perioperative dexamethasone"),
    "NCT00271752": ("excluded", "S4", "PASS: procalcitonin-guided strategy"),
    "NCT00997217": ("excluded", "S4", "Remote ischaemic preconditioning"),
    "NCT00558311": ("excluded", "S4", "CONSCIOUS-2: clazosentan, investigational"),
    "NCT02195726": ("excluded", "S4", "Remote ischaemic preconditioning"),
    "NCT00994825": ("excluded", "S4", "CHEETAH: levosimendan, not used in the US"),
    "NCT04447469": ("excluded", "S4", "Mavrilimumab, investigational"),
    "NCT02739451": ("excluded", "S4", "HIGH: high-flow nasal oxygen device"),
    "NCT02238639": ("excluded", "S4", "SLICE: diagnostic strategy"),
    "NCT00525707": ("excluded", "S4", "VERITAS 1: tezosentan, investigational"),
    "NCT00524433": ("excluded", "S4", "VERITAS 2: tezosentan, investigational"),
    "NCT01967680": ("excluded", "S4", "NONSEDA: no-sedation strategy"),
    "NCT04716556": ("excluded", "S4", "Convalescent plasma (blood products unmapped)"),
    "NCT01915563": ("excluded", "S4", "Rest after a breathing trial"),
    "NCT01477151": ("excluded", "S4", "RISCCS: intraoperative volatile anaesthesia"),
    "NCT05204563": ("excluded", "S4", "XNW4107, investigational"),
    "NCT00711620": ("excluded", "S4", "ETASS: thymosin alpha 1"),
    "NCT01526382": ("excluded", "S4", "PiCCO monitoring device"),
    "NCT02610283": ("excluded", "S4", "QPI-1002, investigational"),
    "NCT04688320": ("excluded", "S4", "Recombinant prourokinase, investigational"),
    "NCT03085758": ("excluded", "S4", "AdrenOSS-2: adrecizumab, investigational"),
    "NCT04878055": ("excluded", "S4", "Reparixin, investigational"),
    "NCT04323527": ("excluded", "S4", "CloroCOVID19: chloroquine dose comparison"),
    "NCT00226096": ("excluded", "S6", "Intensive BP lowering in ICH; death or dependency (mRS)"),
    "NCT00630656": ("excluded", "S4", "Talactoferrin, investigational"),
    "NCT01464567": ("excluded", "S2", "Breathing-trial mode; days on ventilation"),
    "NCT02634411": ("excluded", "S6", "iDIAPASON: antibiotic duration; composite with relapse"),
    "NCT00791505": ("excluded", "S6", "COPD exacerbation antibiotics; clinical failure"),
    "NCT00248638": ("excluded", "S4", "GLND: parenteral glutamine (PN unmapped)"),
    "NCT00224770": ("excluded", "S4", "MISTIE: minimally invasive surgery with rtPA"),
    "NCT01499485": ("excluded", "S2", "TRAMA: duration of ventilation"),
    "NCT03101787": ("excluded", "S6", "INCEPTION: ECPR; survival with good neurological status"),
    "NCT01931228": ("excluded", "S4", "Mechanical insufflation-exsufflation"),
    "NCT01828788": ("excluded", "S6", "Postoperative rehabilitation criteria"),
    "NCT06841731": ("excluded", "S4", "HRS-8427, investigational"),
    "NCT04775472": ("excluded", "S4", "Left atrial septostomy on ECMO"),
    "NCT00539708": ("excluded", "S6", "NIV after extubation; respiratory failure"),
    "NCT00411879": ("excluded", "S6", "In-hospital cardiac arrest drugs; ROSC and neurological survival"),
    "NCT02329444": ("excluded", "S4", "Remote ischaemic preconditioning"),
    "NCT04311697": ("excluded", "S4", "Aviptadil, investigational"),
    "NCT03894046": ("excluded", "S4", "ATTACK: durlobactam, investigational"),
    "NCT03667690": ("excluded", "S4", "ReSTORE: rezafungin, investigational"),
    "NCT01710891": ("excluded", "S4", "Videolaryngoscope"),
    "NCT00248443": ("excluded", "S4", "Bronchoscopy strategy"),
    "NCT01581229": ("excluded", "S4", "Early NIV; intubation criteria"),
    "NCT00821262": ("excluded", "S4", "SEVO AIFA: intraoperative sevoflurane"),
    "NCT01077349": ("excluded", "S4", "HEROICS: high-volume haemofiltration (dose)"),
    # --- excluded: outcome (S6, S2) ---
    "NCT00571649": ("excluded", "S6", "MAGELLAN: VTE by screening ultrasound"),
    "NCT00457002": ("excluded", "S6", "ADOPT: VTE by screening ultrasound"),
    "NCT00423319": ("excluded", "S6", "Apixaban: VTE composite"),
    "NCT00371683": ("excluded", "S6", "Apixaban: VTE composite"),
    "NCT00452530": ("excluded", "S6", "ADVANCE-2: VTE composite"),
    "NCT00071032": ("excluded", "S6", "FOCUS: walking ability at 60 days"),
    "NCT00894387": ("excluded", "S6", "ASTRONAUT: CV death or HF rehospitalization after discharge"),
    "NCT04372589": ("excluded", "S2", "ATTACC: organ support-free days"),
    "NCT00512759": ("excluded", "S7", "GALACTIC: death or HF rehospitalization at 180 days"),
    "NCT03382548": ("excluded", "S6", "REGARD-VAP: death or pneumonia recurrence"),
    "NCT00427076": ("excluded", "S6", "Clinical cure of MRSA infection"),
    "NCT03333278": ("excluded", "S2", "VITAMINS: time alive and free of vasopressors"),
    "NCT05335954": ("excluded", "S6", "Hypotension within 20 minutes of intubation"),
    "NCT00729794": ("excluded", "S6", "VSE-2: ROSC and neurological survival after in-hospital arrest"),
    "NCT02265406": ("excluded", "S6", "PROPHY-VAP: early VAP"),
    "NCT04111926": ("excluded", "S6", "Survival over 3 years"),
    "NCT03289325": ("excluded", "S6", "Survival over 6 years"),
}
JUDGED.update(JUDGED_W2)

# Outcome rule B1 (changelog item 9): a wave-2 hit not decided above is kept for judgment
# only if its first primary outcome is death or a structured in-hospital event.
_B1_KEEP = re.compile(r"mortalit|death|died|dead|surviv|alive|acute kidney injury|\bAKI\b|"
                      r"renal replacement|dialysis|MAKE|intubat|extubat|mechanical ventilation|"
                      r"reintubation|kidney", re.I)  # fmt: skip
_B1_CONT = re.compile(r"free days|days alive|length of stay|duration|score|level|concentration|"
                      r"change in|mean |time to (?!death)", re.I)  # fmt: skip
_DEATH = re.compile(r"mortalit|death|surviv", re.I)


def outcome_rule_b1(s: dict) -> bool:
    """True if the first primary outcome passes rule B1."""
    po = (s["primary_outcomes"][0]["measure"] or "") if s["primary_outcomes"] else ""
    return bool(_B1_KEEP.search(po)) and not (_B1_CONT.search(po) and not _DEATH.search(po))


def stage_a_without_s2(s: dict) -> str | None:
    """The first of S1, S3, S4 a search hit fails on registry fields, or None."""
    if s["model"] not in MODELS:
        return "S1"
    if (s["min_age_years"] or 0) < 18:
        return "S3"
    if not any(i["type"] in OBSERVABLE_TYPES for i in s["interventions"]):
        return "S4"
    return None


def stage_a(s: dict, pubmed: dict[str, list[str]]) -> str | None:
    """The first S-criterion a search hit fails on registry fields (with the PubMed links of
    changelog item 6 for S2), or None."""
    failed = stage_a_without_s2(s)
    if failed:
        return failed
    if not (s["result_refs"] or any(a["ci_lower"] for a in s["primary_analyses"])
            or pubmed.get(s["nct_id"])):  # fmt: skip
        return "S2"
    return None


def registry_estimate(s: dict) -> dict | None:
    """The first primary-outcome analysis with a CI posted on CT.gov (to be checked)."""
    for a in s["primary_analyses"]:
        if a["ci_lower"] and a["param_type"]:
            return {k: a[k] for k in ("outcome", "param_type", "value", "ci_lower", "ci_upper",
                                      "ci_pct", "p", "groups_desc")}  # fmt: skip
    return None


# Pipeline features (changelog item 8) of each trial in the primary agreement set.
FEATURES: dict[str, list[str]] = {
    "NCT00148278": ["F2"],  # CATS
    "NCT00318942": ["F2"],  # CRISTAL
    "NCT00625209": ["F1", "F7"],  # APROCCHSS
    "NCT02509078": ["F1", "F7"],  # ROSE
    "NCT03345992": ["F1"],  # INCLASS
    "NCT03654001": ["F1", "F7"],  # ALBIOSS-BALANCED
    "NCT04010630": ["F1", "F7"],  # BICARICU-2
    "NCT01636089": ["F2", "F6"],  # HYDRAREA
    "NCT01448109": ["F1", "F7"],  # ADRENAL
    "NCT02467621": ["F1", "F7"],  # SUP-ICU
    "NCT02856698": ["F2", "F7"],  # MIMO
    "NCT03872011": ["F1", "F7"],
    "NCT03258684": ["F1", "F7"],
    "NCT01223690": ["F1"],
    "NCT02568722": ["F3", "F4", "F7"],  # STARRT-AKI
    "NCT03680274": ["F1", "F5"],  # LOVIT
    "NCT02984384": ["F2", "F7"],  # PREVENT CLOT
    "NCT00935168": ["F2", "F7"],  # CHEST
    "NCT05277896": ["F2"],  # RSI
    "NCT00314704": ["F2"],  # SOAP II
    "NCT03096314": ["F1", "F7"],  # VIOLET
    "NCT02309151": ["F3", "F4", "F7"],  # DISCO
    "NCT02517489": ["F1"],  # CAPE COD
    "NCT02643381": ["F2", "F7"],  # EvK
    "NCT01870804": ["F2", "F6"],  # PRATO-ACS-2
    "NCT01932190": ["F3", "F4", "F7"],  # AKIKI
    "NCT00135473": ["F1"],  # VISEP
    "NCT03637205": ["F1", "F3", "F7"],  # ECLS-SHOCK
    "NCT02476253": ["F1", "F5"],  # BICAR-ICU
    "NCT00299650": ["F1", "F7"],  # ACURASYS
    "NCT01718613": ["F2"],  # VANCS II
    "NCT01760967": ["F1"],  # DESIRE
    "NCT01284452": ["F1"],
    "NCT03338569": ["F1"],
    "NCT02580240": ["F1"],
    "NCT01455688": ["F4", "F7"],
    "NCT05778838": ["F1"],
}

FEATURE_NAMES = {
    "F1": "placebo or usual-care comparator (not treated)",
    "F2": "active comparator (drug vs drug)",
    "F3": "procedure intervention",
    "F4": "timing strategy with a grace period",
    "F5": "composite outcome from structured data",
    "F6": "lab-defined outcome",
    "F7": "mortality horizon other than 28 days",
}


def coverage_select(primary: list[dict], cap: int = 30) -> tuple[list[str], dict]:
    """Changelog item 8: up to 3 largest per feature first, then fill by enrolment."""
    by_size = sorted(primary, key=lambda r: (-(r["enrollment"] or 0), r["nct_id"]))
    chosen: list[str] = []
    if len(primary) <= cap:
        chosen = [r["nct_id"] for r in by_size]
    else:
        for f in FEATURE_NAMES:
            for r in [r for r in by_size if f in r.get("features", [])][:3]:
                if r["nct_id"] not in chosen:
                    chosen.append(r["nct_id"])
        for r in by_size:
            if len(chosen) >= cap:
                break
            if r["nct_id"] not in chosen:
                chosen.append(r["nct_id"])
    got = {f: sum(f in r.get("features", []) for r in primary if r["nct_id"] in chosen)
           for f in FEATURE_NAMES}  # fmt: skip
    return chosen, {f: {"name": FEATURE_NAMES[f], "trials": n, "met": n >= 3}
                    for f, n in got.items()}  # fmt: skip


def main() -> None:
    named = json.loads(NAMED.read_text(encoding="utf-8"))
    pubmed = json.loads(PUBMED.read_text(encoding="utf-8"))["links"]
    values = {v["nct_id"]: v for v in json.loads(
        (REF / "replay_reference_values.json").read_text(encoding="utf-8"))["values"]}  # fmt: skip
    parity = {t["nct_id"]: t for t in json.loads(
        (REF / "replay_trials.json").read_text(encoding="utf-8"))["trials"]}  # fmt: skip
    wave1 = {s["nct_id"]: s for s in json.loads(SEARCH.read_text("utf-8"))["studies"]}
    wave2 = {s["nct_id"]: s for s in json.loads(SEARCH2.read_text("utf-8"))["studies"]}
    sources: dict[str, list[str]] = {}
    for i in wave1:
        sources.setdefault(i, []).append("search")
    for i in set(wave2) - set(wave1):
        sources.setdefault(i, []).append("search-wave2")
    for i in named["named"]:
        sources.setdefault(i, []).append("named")
    for i in parity:
        sources.setdefault(i, []).append("parity")
    records = {**wave2, **wave1, **named["studies"]}
    only_search = {"search", "search-wave2"}

    out, missing = [], []
    for i in sorted(sources):
        s = records[i]
        rec = {"nct_id": i, "acronym": s.get("acronym"), "title": s.get("title"),
               "source": sources[i], "enrollment": s.get("enrollment")}  # fmt: skip
        failed_a = stage_a(s, pubmed) if set(sources[i]) <= only_search else None
        if failed_a:
            rec.update(status="excluded", failed=failed_a, stage="A",
                       note="registry fields (see operational_rules)")
        elif i in JUDGED:
            status, failed, note = JUDGED[i]
            rec.update(status=status, failed=failed, stage="B", note=note)
            if values.get(i, {}).get("s2") == "fail":  # the user's full-text check (item 12)
                rec.update(status="excluded", failed="S2", stage="S2 check",
                           note=f"{note}. S2: {values[i]['status']}: {values[i].get('note', '')}")
        elif set(sources[i]) <= only_search and not outcome_rule_b1(s):
            rec.update(status="excluded", failed="S6", stage="B1",
                       note="outcome rule B1: the primary outcome is not death or a structured "
                            "in-hospital event")  # fmt: skip
        else:
            missing.append(i)
            continue
        if rec["status"] != "excluded":
            pub = parity.get(i, {}).get("published_primary_result") or {}
            ok = pub.get("verification") == "verified" and bool(pub.get("ci95"))
            value = values.get(i)
            rec["reference"] = {
                "verified": pub if ok else None,
                "extracted": value,
                "registry_estimate": registry_estimate(s),
                "registry_result_pmids": s.get("result_refs", []),
                "pubmed_rct_pmids": pubmed.get(i, []),
                "s2_status": "verified" if ok or (value and value.get("verified_by"))
                else value["status"] if value else "to extract and check",
            }  # fmt: skip
            rec["s8"] = "pending (Data Support Gate counts on Hellbender)"
            if rec["status"] == "primary":
                rec["features"] = FEATURES.get(i, [])
        out.append(rec)
    if missing:
        sys.exit(f"stage A passed but not judged ({len(missing)}): {missing}")
    unused = set(JUDGED) - set(sources)
    if unused:
        sys.exit(f"judged but not a candidate: {sorted(unused)}")
    no_features = [r["nct_id"] for r in out if r["status"] == "primary" and not r["features"]]
    if no_features:
        sys.exit(f"primary trials without features: {no_features}")

    primary = [r for r in out if r["status"] == "primary"]
    chosen, coverage = coverage_select(primary)
    for r in primary:
        r["selected"] = r["nct_id"] in chosen
    # Core set (changelog item 11a): selected trials with a user-verified S2 estimate, cap 12.
    verified = [r for r in primary if r["selected"] and values.get(r["nct_id"], {}).get("s2") == "pass"]
    core, core_coverage = coverage_select(verified, cap=12)
    for r in primary:
        r["core"] = r["nct_id"] in core
    flow: dict[str, int] = {"candidates": len(out)}
    for r in out:
        key = f"excluded_{r['failed']}" if r["status"] == "excluded" else r["status"]
        flow[key] = flow.get(key, 0) + 1
    flow["primary_selected"] = len(chosen)
    flow["selected_s2_verified"] = len(verified)
    flow["core"] = len(core)
    doc = {
        "_about": "Screening of Replay candidates against eval/SCORING.md R3 (S1-S8).",
        "pre_registration": "eval/SCORING.md, section Phase 3 Replay (commit 77bbc85), "
                            "with changelog items 1-8 (wave 2 and coverage: commit a37fc10)",
        "searches": [SEARCH.name, SEARCH2.name],
        "pubmed_links": PUBMED.name,
        "screened": "2026-10-27 by Claude (AI) from registry records; every decision is to be "
                    "checked by the user. S2 values need extraction from the primary paper and "
                    "a second-person check; S8 needs Data Support Gate counts.",
        "operational_rules": [
            "Stage A (search hits only, registry fields): S1 fails if the design model is not "
            "parallel, factorial or crossover; S3 fails if the minimum age is under 18 or not "
            "given; S4 fails if no intervention is a drug, biological, procedure, dietary "
            "supplement or combination product; S2 fails if there is no linked result "
            "publication, no posted primary analysis with a CI and no PubMed RCT article "
            "carrying the NCT number.",
            "Stage B: the remaining search hits, the parity trials and the named candidates, "
            "judged on S1-S7 from the registry record. Interventions never used in US practice "
            "(investigational agents) fail S4.",
            "conditional: passes once the named unmapped MIMIC item is mapped (changelog "
            "item 7). uncertain: one criterion needs a check (named in 'failed').",
            "B1 (search hits not judged in the first screening, changelog item 9): kept for stage B only if the first "
            "primary outcome is death or a structured in-hospital event (eval/replay_screen.py, "
            "outcome_rule_b1).",
            "features and selection: changelog item 8.",
        ],
        "flow": flow,
        "coverage": coverage,
        "core_coverage": core_coverage,
        "candidates": out,
    }
    (REF / "replay_selection.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False),
                                               encoding="utf-8")  # fmt: skip
    print(json.dumps({"flow": flow, "coverage": coverage}, indent=1))


if __name__ == "__main__":
    main()
