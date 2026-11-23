"""The autonomous arm's rules (agentic_tte/agents/autonomy.py, SCORING.md Phase 3 item 34):
concepts chosen from record counts, and time zero by rule; and the compiler's time-zero events
and all-components initiation that they rely on."""

import pandas as pd
import pytest

from agentic_tte.agents.autonomy import (
    ICU_VISIT,
    INPATIENT_VISIT,
    check_visits,
    choose_in_data,
    name_match,
    time_zero_spec,
)
from agentic_tte.omop.compiler import CompileSpec, EventDef, compile_protocol
from agentic_tte.omop.concept_sets import ConceptSet, ConceptSetBook
from agentic_tte.omop.vocab import Candidate
from agentic_tte.protocol import (
    AssignmentProcedure,
    ConceptRef,
    Criterion,
    CriterionAtom,
    Duration,
    Outcome,
    TargetTrialProtocol,
    TemporalConstraint,
    TreatmentComponent,
    TreatmentStrategy,
    ValueConstraint,
)
from agentic_tte.synth import omop as so
from tests.test_compiler import SPEC, T0, D, H, _db, atom, crit, ref


def cand(cid, name, domain="Drug"):
    return Candidate(concept_id=cid, concept_name=name, vocabulary_id="RxNorm", domain_id=domain,
                     standard=True, match="exact_name", matched_text=name, score=1.0)


def _vocab_db():
    t = pd.Timestamp("2020-01-01 08:00")
    concept = pd.DataFrame({"concept_id": [1, 11, 12, 2, 3, 4, ICU_VISIT, INPATIENT_VISIT],
                            "concept_name": ["hydrocortisone", "hydrocortisone 100 MG Injection",
                                             "hydrocortisone 10 MG Oral Tablet", "SOFA score",
                                             "Lactate [Moles/volume] in Blood", "procalcitonin",
                                             "Intensive Care", "Inpatient Visit"],
                            "domain_id": ["Drug", "Drug", "Drug", "Measurement", "Measurement",
                                          "Measurement", "Visit", "Visit"],
                            "standard_concept": ["S"] * 8})
    ancestor = pd.DataFrame({"ancestor_concept_id": [1, 1], "descendant_concept_id": [11, 12]})
    drug = pd.DataFrame({"person_id": [1, 2, 3], "drug_concept_id": [11, 12, 12],
                         "drug_exposure_start_datetime": [t] * 3,
                         "drug_exposure_start_date": [t.date()] * 3})
    meas = pd.DataFrame({"person_id": [1, 2], "measurement_concept_id": [3, 3],
                         "measurement_datetime": [t] * 2, "measurement_date": [t.date()] * 2,
                         "value_as_number": [2.5, 1.0]})
    return so.write_duckdb({"concept": concept, "concept_ancestor": ancestor,
                            "drug_exposure": drug, "measurement": meas})  # fmt: skip


def test_concepts_move_to_candidates_with_records():
    con = _vocab_db()
    check_visits(con)
    p = TargetTrialProtocol(protocol_id="T", treatment_strategies=[
        TreatmentStrategy(label="HC", components=[TreatmentComponent(
            concept=ConceptRef(name="hydrocortisone", domain="Drug"), route="iv bolus")]),
        TreatmentStrategy(label="Placebo", is_comparator=True)])  # fmt: skip
    book = ConceptSetBook(protocol_id="T", protocol_version=1, sets=[
        ConceptSet(key="T:HC:hydrocortisone", scope="treatment", criterion_text="", domain="Drug",
                   concept_name="hydrocortisone", candidates=[cand(1, "hydrocortisone")],
                   concept_ids=[1], status="proposed"),
        ConceptSet(key="I1:lactate", scope="eligibility", criterion_text="", domain="Measurement",
                   concept_name="lactate", candidates=[cand(2, "SOFA score", "Measurement"),
                                                       cand(3, "Lactate", "Measurement")],
                   concept_ids=[2], status="proposed"),
        ConceptSet(key="I2:procalcitonin", scope="eligibility", criterion_text="",
                   domain="Measurement", concept_name="procalcitonin",
                   candidates=[cand(4, "procalcitonin", "Measurement")], concept_ids=[4],
                   status="proposed"),
        # a candidate with records whose name does not match the term is not taken
        ConceptSet(key="I3:severity", scope="eligibility", criterion_text="", domain="Measurement",
                   concept_name="severity", candidates=[cand(2, "SOFA score", "Measurement"),
                                                        cand(3, "Lactate", "Measurement")],
                   concept_ids=[2], status="proposed"),
        # no vocabulary candidate: the concepts in use whose name matches
        ConceptSet(key="I4:blood lactate", scope="eligibility", criterion_text="",
                   domain="Measurement", concept_name="blood lactate", status="no_match"),
        ConceptSet(key="I5:intensive care unit", scope="eligibility", criterion_text="",
                   domain="Visit", concept_name="admission to an intensive care unit",
                   status="no_match")])  # fmt: skip
    p = p.model_copy(update={"eligibility": [Criterion(id="I6", kind="inclusion", text="icu")]})
    out, notes = choose_in_data(con, book, p)
    sets = {s.key: s for s in out.sets}
    assert sets["T:HC:hydrocortisone"].concept_ids == [11]  # the injectable form only
    assert sets["I1:lactate"].concept_ids == [3] and "has no records" in sets["I1:lactate"].note
    pct = sets["I2:procalcitonin"]
    assert pct.status == "no_match" and not pct.concept_ids
    assert sets["I3:severity"].status == "no_match"  # not 'Lactate', though it has records
    assert sets["I4:blood lactate"].concept_ids == [3] and "in use" in sets["I4:blood lactate"].note
    assert sets["I5:intensive care unit"].concept_ids == [ICU_VISIT]
    assert len(notes.notes) == 6 and notes.rules == "data-v2"
    assert book.sets[1].concept_ids == [2]  # the input is unchanged


def test_name_match():
    assert name_match("platelet count", "Platelets [#/volume] in Blood by Automated count") == 1
    assert name_match("mean arterial pressure", "Mean blood pressure") == pytest.approx(2 / 3)
    assert name_match("underlying disease", "Chronic kidney disease") == 0  # two words: both
    assert name_match("mean arterial pressure", "Arterial blood gas") == 0  # the head is missing
    assert name_match("Hydroxy-ethyl starch", "hydroxyethyl starch 130/0.4") == 1


def test_check_visits_refuses_a_wrong_concept():
    con = _vocab_db()
    con.execute(f"UPDATE concept SET concept_name = 'ER' WHERE concept_id = {ICU_VISIT}")
    with pytest.raises(ValueError, match="not 'intensive care'"):
        check_visits(con)


def test_time_zero_waits_for_the_inclusion_events():
    vaso = CriterionAtom(concepts=[ConceptRef(name="norepinephrine", domain="Drug"),
                                   ConceptRef(name="epinephrine", domain="Drug")])
    map_ = CriterionAtom(concepts=[ConceptRef(name="MAP", domain="Measurement")],
                         value=ValueConstraint(comparator=">=", value=65, unit="mmHg"))
    p = TargetTrialProtocol(protocol_id="T", eligibility=[
        Criterion(id="I1", kind="inclusion", text="in the ICU, vasopressor or MAP",
                  atoms=[vaso, map_],
                  atom_logic="any"),
        Criterion(id="I2", kind="inclusion", text="lactate > 2", atoms=[CriterionAtom(
            concepts=[ConceptRef(name="lactate", domain="Measurement")],
            value=ValueConstraint(comparator=">", value=2, unit="mmol/L"))]),
        Criterion(id="I3", kind="inclusion", text="study drug", atoms=[CriterionAtom(
            concepts=[ConceptRef(name="hydrocortisone", domain="Drug")])]),
        Criterion(id="E1", kind="exclusion", text="burn in this admission", atoms=[CriterionAtom(
            concepts=[ConceptRef(name="burn", domain="Condition")],
            temporal=TemporalConstraint(relation="x_after_y", anchor="this admission"))]),
        Criterion(id="E2", kind="exclusion", text="since norepinephrine", atoms=[CriterionAtom(
            concepts=[ConceptRef(name="steroids", domain="Drug")],
            temporal=TemporalConstraint(relation="x_after_y", anchor="norepinephrine start"))]),
    ], treatment_strategies=[TreatmentStrategy(label="HC", components=[TreatmentComponent(
        concept=ConceptRef(name="hydrocortisone", domain="Drug"))])])  # fmt: skip
    sets = {"I1:norepinephrine": [10], "I1:epinephrine": [11], "I1:MAP": [20],
            "I2:lactate": [30], "I3:hydrocortisone": [40], "T:HC:hydrocortisone": [40]}
    spec, notes = time_zero_spec(p, sets)
    CompileSpec.model_validate(spec)
    assert spec["time_zero"]["concept_ids"] == [ICU_VISIT]
    after = [(e["domain"], e["concept_ids"], (e["value"] or {}).get("value"))
             for e in spec["time_zero_after"]]  # fmt: skip
    assert after == [("Drug", [10, 11], None), ("Measurement", [30], 2)]
    assert any("a treatment drug, not used" in n for n in notes.notes)
    assert spec["anchors"]["this admission"]["concept_ids"] == [INPATIENT_VISIT]
    assert spec["anchors"]["norepinephrine start"]["concept_ids"] == [10, 11]
    spec, _ = time_zero_spec(TargetTrialProtocol(protocol_id="T"), {})
    assert spec["time_zero"]["concept_ids"] == [INPATIENT_VISIT]  # no word of intensive care


def _frame_protocol(*components):
    return TargetTrialProtocol(
        protocol_id="T", eligibility=[crit("A", "inclusion", atom(ref("x", "Condition", 100)))],
        treatment_strategies=[
            TreatmentStrategy(label="Rx", components=[TreatmentComponent(concept=c)
                                                      for c in components]),
            TreatmentStrategy(label="None", is_comparator=True)],
        assignment=AssignmentProcedure(confounders=[ref("age", "Demographic")]),
        outcomes=[Outcome(name="organ failure", primary=True, concept=ref("of", "Condition", 500),
                          horizon=Duration(amount=28, unit="day"))])  # fmt: skip


def test_all_components_must_start_for_initiation():
    one = compile_protocol(_frame_protocol(ref("drug", "Drug", 300)), SPEC)
    both = compile_protocol(_frame_protocol(ref("drug", "Drug", 300), ref("other", "Drug", 301)),
                            SPEC)  # fmt: skip
    con = _db()
    assert con.execute(one.frame_sql).df().set_index("person_id").loc[1, "treated"] == 1
    # p1 never gets drug 301: starting one of two components is not initiation.
    assert con.execute(both.frame_sql).df().set_index("person_id").loc[1, "treated"] == 0


def test_time_zero_events_in_the_compiler():
    p = TargetTrialProtocol(protocol_id="T")
    drug = SPEC.model_copy(update={"time_zero_after": [EventDef(domain="Drug",
                                                                concept_ids=[300])]})
    df = _db().execute(compile_protocol(p, drug).cohort_sql).df().set_index("person_id")
    assert list(df.index) == [1] and df.loc[1, "t0"] == T0 + D  # p1's drug at +1 day
    lac = SPEC.model_copy(update={"time_zero_after": [EventDef(
        domain="Measurement", concept_ids=[200],
        value=ValueConstraint(comparator=">", value=2))]})  # fmt: skip
    cc = compile_protocol(p, lac)
    df = _db().execute(cc.cohort_sql).df().set_index("person_id")
    # p1 3.0 at -2 h and p2 4.0 at -3 h are within the 24-hour lookback: time zero stays T0.
    assert sorted(df.index) == [1, 2] and (df.t0 == T0).all()
    assert any("time zero waits for the first Measurement" in a for a in cc.assumptions)
    late = SPEC.model_copy(update={"time_zero_after": drug.time_zero_after,
                                   "time_zero_window": Duration(amount=12, unit="hour")})
    assert _db().execute(compile_protocol(p, late).cohort_sql).df().empty  # +1 day is too late
    assert H  # (the hand-built database's hour, kept for symmetry with test_compiler)


def test_the_mimic_check_on_a_small_database(tmp_path):
    """eval/autonomy_check.py end to end on the fixture vocabulary and 30 hand-made patients."""
    import importlib.util
    from pathlib import Path

    from agentic_tte.omop.vocab import load_athena

    spec_ = importlib.util.spec_from_file_location(
        "autonomy_check", Path(__file__).parents[1] / "eval" / "autonomy_check.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    t = pd.Timestamp("2020-01-01 08:00")
    ids = list(range(1, 31))
    tables = {
        "person": pd.DataFrame({"person_id": ids, "gender_concept_id": 8507, "year_of_birth": 1960,
                                "month_of_birth": 1, "day_of_birth": 1,
                                "birth_datetime": pd.Timestamp("1960-01-01")}),
        "observation_period": pd.DataFrame({"person_id": ids,
                                            "observation_period_start_date": (t - D * 30).date(),
                                            "observation_period_end_date": (t + D * 90).date()}),
        "visit_occurrence": pd.DataFrame({"person_id": ids, "visit_concept_id": INPATIENT_VISIT,
                                          "visit_start_datetime": t - H,
                                          "visit_start_date": t.date()}),
        "visit_detail": pd.DataFrame({"person_id": ids, "visit_detail_concept_id": ICU_VISIT,
                                      "visit_detail_start_datetime": t,
                                      "visit_detail_start_date": t.date()}),
        "condition_occurrence": pd.DataFrame({"person_id": ids, "condition_concept_id": 196236,
                                              "condition_start_datetime": t,
                                              "condition_start_date": t.date()}),
        # norepinephrine 2 h after ICU admission for 20 patients; hydrocortisone injections
        "drug_exposure": pd.DataFrame(
            [{"person_id": i, "drug_concept_id": 9000060, "drug_exposure_start_datetime": t + 2 * H,
              "drug_exposure_start_date": t.date()} for i in ids[:20]]
            + [{"person_id": i, "drug_concept_id": 9000021,
                "drug_exposure_start_datetime": t + 5 * H, "drug_exposure_start_date": t.date()}
               for i in ids[:12]]),
        "measurement": pd.DataFrame({"person_id": [1], "measurement_concept_id": [9000030],
                                     "measurement_datetime": [t], "measurement_date": [t.date()],
                                     "value_as_number": [3.0], "unit_concept_id": [None]}),
        "death": pd.DataFrame({"person_id": [1], "death_date": [(t + D * 5).date()],
                               "death_datetime": [t + D * 5]}),
    }  # fmt: skip
    con = so.write_duckdb(tables)
    load_athena(con, Path(__file__).parent / "fixtures" / "vocab")
    p = TargetTrialProtocol(
        protocol_id="T", eligibility=[
            Criterion(id="I1", kind="inclusion", text="septic shock in the ICU", atoms=[
                CriterionAtom(concepts=[ConceptRef(name="septic shock", domain="Condition")])]),
            Criterion(id="I2", kind="inclusion", text="norepinephrine", atoms=[CriterionAtom(
                concepts=[ConceptRef(name="norepinephrine", domain="Drug")])]),
            Criterion(id="E1", kind="exclusion", text="pregnancy", atoms=[CriterionAtom(
                concepts=[ConceptRef(name="pregnancy", domain="Condition")])])],
        treatment_strategies=[
            TreatmentStrategy(label="Hydrocortisone", components=[TreatmentComponent(
                concept=ConceptRef(name="hydrocortisone", domain="Drug"), route="iv bolus")]),
            TreatmentStrategy(label="Placebo", is_comparator=True, components=[
                TreatmentComponent(concept=ConceptRef(name="placebo", domain="Drug"))])],
        outcomes=[Outcome(name="28-day mortality", primary=True,
                          horizon=Duration(amount=28, unit="day"))])  # fmt: skip
    res, md = mod.check(con, p, "hydrocortisone vs placebo")
    assert res["persons_with_admission"] == 30 and res["persons_with_time_zero"] == 20
    assert res["eligible"] == 20 and res["frame_design"] == "no_initiation"
    crit = {c["id"]: c for c in res["criteria"]}
    assert crit["E1"]["status"] == "not_compiled"  # pregnancy has no records: unmappable
    assert crit["I2"]["persons_meeting"] == 20
    assert any("injectable" in n for n in res["data_rules"]["notes"])
    assert any("time zero waits for I2" in n for n in res["time_zero"]["notes"])
    assert "R-grace" in md and "Pooled counts only" in md
    assert not {"arms", "events", "risk_at_horizon"} & set(res)  # nothing by arm or outcome
    # An inclusion event too few persons have is not recorded in the data: no time-zero event.
    import agentic_tte.agents.autonomy as au

    old, au.SPARSE = au.SPARSE, 0.9  # 20 of 30 is under 90%
    try:
        res, _ = mod.check(con, p, "hydrocortisone vs placebo")
    finally:
        au.SPARSE = old
    assert res["persons_with_time_zero"] == 30
    assert any("barely record" in n for n in res["data_rules"]["notes"])
    assert {c["id"]: c for c in res["criteria"]}["I2"]["status"] == "not_compiled"
