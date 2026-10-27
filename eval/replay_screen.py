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

# ruff: noqa: E501  (one screening decision per line)
import sys
from pathlib import Path

REF = Path(__file__).parent / "reference"
SEARCH = REF / "replay_search_2026-10-27.json"
NAMED = REF / "replay_candidates_named_2026-10-27.json"  # the parity and named trials, fetched

OBSERVABLE_TYPES = {"DRUG", "BIOLOGICAL", "PROCEDURE", "DIETARY_SUPPLEMENT", "COMBINATION_PRODUCT"}
MODELS = {"PARALLEL", "FACTORIAL", "CROSSOVER"}

# Stage B. status: primary | secondary (S7) | exploratory (S5) | excluded | uncertain.
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
    "NCT02444988": ("excluded", "S4", "SMART: balanced crystalloids vs saline; MAKE30. Balanced crystalloids (LR, Plasma-Lyte) are unmapped in our inputevents map; revisit if chosen"),
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
    "NCT00527813": ("excluded", "S4", "PROSEVA: prone positioning; the position items are unmapped nursing documentation"),
    "NCT01799993": ("excluded", "S4", "INHALE: inhaled amikacin; route is not kept in drug_exposure"),
    "NCT03452839": ("excluded", "S4", "MERCY: continuous vs bolus meropenem; administration mode not kept; composite with resistance"),
    "NCT03098524": ("excluded", "S4", "MECANO: ventilation during cardiopulmonary bypass (intraoperative)"),
    "NCT03709264": ("excluded", "S4", "PROTection: IV amino acids, unmapped in our inputevents map"),
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
    "NCT02267538": ("excluded", "S6", "Postoperative delirium (unmapped)"),
    "NCT02361528": ("excluded", "S6", "GRID: ICU-acquired infection"),
    "NCT02723383": ("excluded", "S6", "BACLOREA: agitation-related adverse events"),
    "NCT02856594": ("excluded", "S6", "MINDDS: delirium (unmapped)"),
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


def stage_a(s: dict) -> str | None:
    """The first S-criterion a search hit fails on registry fields, or None."""
    if s["model"] not in MODELS:
        return "S1"
    if (s["min_age_years"] or 0) < 18:
        return "S3"
    if not any(i["type"] in OBSERVABLE_TYPES for i in s["interventions"]):
        return "S4"
    if not (s["result_refs"] or any(a["ci_lower"] for a in s["primary_analyses"])):
        return "S2"
    return None


def registry_estimate(s: dict) -> dict | None:
    """The first primary-outcome analysis with a CI posted on CT.gov (to be checked)."""
    for a in s["primary_analyses"]:
        if a["ci_lower"] and a["param_type"]:
            return {k: a[k] for k in ("outcome", "param_type", "value", "ci_lower", "ci_upper",
                                      "ci_pct", "p", "groups_desc")}  # fmt: skip
    return None


def main() -> None:
    search = json.loads(SEARCH.read_text(encoding="utf-8"))
    named = json.loads(NAMED.read_text(encoding="utf-8"))
    parity = {t["nct_id"]: t for t in json.loads(
        (REF / "replay_trials.json").read_text(encoding="utf-8"))["trials"]}  # fmt: skip
    studies = {s["nct_id"]: s for s in search["studies"]}
    sources: dict[str, list[str]] = {}
    for i in studies:
        sources.setdefault(i, []).append("search")
    for i in named["named"]:
        sources.setdefault(i, []).append("named")
    for i in parity:
        sources.setdefault(i, []).append("parity")
    records = {**studies, **named["studies"]}

    out, missing = [], []
    for i in sorted(sources):
        s = records[i]
        rec = {"nct_id": i, "acronym": s.get("acronym"), "title": s.get("title"),
               "source": sources[i], "enrollment": s.get("enrollment")}  # fmt: skip
        failed_a = stage_a(s) if sources[i] == ["search"] else None
        if failed_a:
            rec.update(status="excluded", failed=failed_a, stage="A",
                       note="registry fields (see operational_rules)")
        elif i in JUDGED:
            status, failed, note = JUDGED[i]
            rec.update(status=status, failed=failed, stage="B", note=note)
        else:
            missing.append(i)
            continue
        if rec["status"] not in ("excluded",):
            pub = parity.get(i, {}).get("published_primary_result") or {}
            rec["reference"] = {
                "verified": pub if pub.get("verification") == "verified" and pub.get("ci95") else None,
                "registry_estimate": registry_estimate(s),
                "result_pmids": s.get("result_refs", []),
                "s2_status": "verified" if pub.get("verification") == "verified" and pub.get("ci95")
                else "to extract and check",
            }  # fmt: skip
            rec["s8"] = "pending (Data Support Gate counts on Hellbender)"
        out.append(rec)
    if missing:
        sys.exit(f"stage A passed but not judged: {missing}")
    unused = set(JUDGED) - set(sources)
    if unused:
        sys.exit(f"judged but not a candidate: {sorted(unused)}")

    flow: dict[str, int] = {"candidates": len(out)}
    for r in out:
        key = f"excluded_{r['failed']}" if r["status"] == "excluded" else r["status"]
        flow[key] = flow.get(key, 0) + 1
    doc = {
        "_about": "Screening of Replay candidates against eval/SCORING.md R3 (S1-S8).",
        "pre_registration": "eval/SCORING.md, section Phase 3 Replay (commit 77bbc85)",
        "search": SEARCH.name,
        "screened": "2026-10-27 by Claude (AI) from registry records; every decision is to be "
                    "checked by the user. S2 values need extraction from the primary paper and "
                    "a second-person check; S8 needs Data Support Gate counts.",
        "operational_rules": [
            "Stage A (search hits only, registry fields): S1 fails if the design model is not "
            "parallel, factorial or crossover; S3 fails if the minimum age is under 18 or not "
            "given; S4 fails if no intervention is a drug, biological, procedure, dietary "
            "supplement or combination product; S2 fails if the record links no result "
            "publication and posts no primary analysis with a CI.",
            "Stage B: the remaining search hits, the parity trials and the named candidates, "
            "judged on S1-S7 from the registry record. Interventions never used in US practice "
            "(investigational agents) fail S4. Items marked unmappable in the 2026-10-27 MIMIC "
            "item reviews (LR, blood components, enteral formulas, delirium assessments, "
            "position) make the dependent trials fail S4 or S6 until revisited.",
            "uncertain: one criterion needs a check (named in 'failed') before the trial is "
            "included or excluded.",
        ],
        "flow": flow,
        "candidates": out,
    }
    (REF / "replay_selection.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False),
                                               encoding="utf-8")  # fmt: skip
    print(json.dumps(flow, indent=1))


if __name__ == "__main__":
    main()
