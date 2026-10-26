import pytest

from agentic_tte.omop import DataContext, run_gate
from agentic_tte.omop.gate import concept_key, suppress
from agentic_tte.protocol import (
    ConceptRef,
    Criterion,
    CriterionAtom,
    Duration,
    FollowUp,
    TargetTrialProtocol,
    TemporalConstraint,
)
from agentic_tte.synth.dgp import Scenario, generate
from agentic_tte.synth.omop import SEPTIC_SHOCK, STUDY_DRUG, to_omop, write_duckdb

SYNTH = DataContext(data_classification="synthetic")


def db(n=300, **tweaks):
    tables = to_omop(generate(Scenario(n=n), with_truth=False))
    for fn in tweaks.values():
        fn(tables)
    return write_duckdb(tables)


def crit(cid, kind, *concepts, window=None, category="clinical"):
    temporal = None
    if window:
        temporal = TemporalConstraint(relation="x_within_t_before_y",
                                      window=Duration(amount=window[0], unit=window[1]))
    return Criterion(id=cid, kind=kind, text=cid, category=category,
                     atoms=[CriterionAtom(concepts=list(concepts), temporal=temporal)])


def protocol(*criteria, horizon=28):
    return TargetTrialProtocol(protocol_id="T", eligibility=list(criteria),
                               follow_up=FollowUp(horizon=Duration(amount=horizon, unit="day")))


SHOCK = ConceptRef(name="septic shock", domain="Condition")
DRUG = ConceptRef(name="study drug", domain="Drug")
AGE = ConceptRef(name="age", domain="Demographic")


def test_synthetic_database_passes_with_vocabulary_limitation():
    r = run_gate(db(), protocol(crit("I1", "inclusion", SHOCK)),
                 concept_sets={"I1:septic shock": [SEPTIC_SHOCK]}, context=SYNTH)
    assert not r.blocks_run
    assert r.overall == "amber"  # only local concepts, no concept_ancestor
    assert r.check("vocabulary").status == "amber"
    for cid in ("structure", "cdm_metadata", "governance", "volume", "time", "death",
                "follow_up", "quality", "concepts"):
        assert r.check(cid).status == "green", (cid, r.check(cid).detail)
    assert r.check("feasibility").status == "not_run"
    assert r.cdm_version == "5.4"
    assert "Data Support Gate: AMBER" in r.to_markdown()


def test_undeclared_classification_blocks():
    r = run_gate(db(), protocol(crit("I1", "inclusion", SHOCK)))
    assert r.check("governance").status == "red" and r.blocks_run
    approved = run_gate(db(), None, context=DataContext(data_classification="identified",
                                                        approved_by_rcss=True))
    assert approved.check("governance").status == "green"
    deid = run_gate(db(), None, context=DataContext(data_classification="deidentified"))
    assert deid.check("governance").status == "amber"


def test_missing_table_blocks_and_skips_data_checks():
    r = run_gate(db(death=lambda t: t.pop("death")), None, context=SYNTH)
    assert r.check("structure").status == "red"
    assert "death" in r.check("structure").values["missing_tables"]
    assert r.check("death").status == "not_run" and r.blocks_run


def test_missing_cdm_source_is_a_limitation():
    r = run_gate(db(src=lambda t: t.pop("cdm_source")), None, context=SYNTH)
    assert r.check("cdm_metadata").status == "amber" and not r.blocks_run


def test_concept_fit():
    p = protocol(
        crit("I1", "inclusion", SHOCK, AGE),
        crit("I2", "inclusion", ConceptRef(name="unknown thing", domain="Condition")),
        crit("E1", "exclusion", ConceptRef(name="absent drug", domain="Drug")),
        crit("E2", "exclusion", DRUG),
        crit("E3", "exclusion", ConceptRef(name="prisoner", domain="Observation"),
             category="administrative"),
        crit("E4", "exclusion", ConceptRef(name="never mapped", domain="Condition")),
    )
    sets = {concept_key(p.eligibility[0], "septic shock"): [SEPTIC_SHOCK],
            concept_key(p.eligibility[1], "unknown thing"): [999_999_999],
            concept_key(p.eligibility[2], "absent drug"): [888_888_888],
            concept_key(p.eligibility[3], "study drug"): [STUDY_DRUG]}  # fmt: skip
    r = run_gate(db(), p, concept_sets=sets, context=SYNTH)
    found = r.check("concepts").values["concepts"]
    rows = {(x["criterion"], x["concept"]): x["result"] for x in found}
    assert rows[("I1", "septic shock")] == "300 patients"
    assert rows[("I1", "age")] == "available in person"
    assert rows[("I2", "unknown thing")] == "0 patients"  # inclusion nobody meets: blocks
    assert rows[("E1", "absent drug")] == "0 patients"  # exclusion nobody meets: limitation
    assert rows[("E3", None)].startswith("administrative")
    assert rows[("E4", "never mapped")] == "unmapped"
    assert r.check("concepts").status == "red" and r.blocks_run
    assert "I2 unknown thing" in r.check("concepts").detail


def test_concept_in_missing_domain_table():
    p = protocol(crit("I1", "inclusion", ConceptRef(name="ventilation", domain="Procedure")))
    con = db(drop=lambda t: t.pop("procedure_occurrence"))
    r = run_gate(con, p, concept_sets={"I1:ventilation": [123]}, context=SYNTH)
    rows = r.check("concepts").values["concepts"]
    assert rows[0]["result"] == "no procedure_occurrence table"
    assert r.check("concepts").status == "red"


def _midnight(tables):
    for name, col in (("visit_occurrence", "visit_start_datetime"),
                      ("condition_occurrence", "condition_start_datetime"),
                      ("drug_exposure", "drug_exposure_start_datetime"),
                      ("measurement", "measurement_datetime")):  # fmt: skip
        tables[name][col] = tables[name][col].dt.normalize()


def test_hour_windows_need_timestamps():
    p = protocol(crit("I1", "inclusion", SHOCK, window=(6, "hour")))
    sets = {"I1:septic shock": [SEPTIC_SHOCK]}
    ok = run_gate(db(), p, concept_sets=sets, context=SYNTH)
    assert ok.check("time").status == "green"
    dates_only = run_gate(db(t=_midnight), p, concept_sets=sets, context=SYNTH)
    assert dates_only.check("time").status == "red"
    assert "6-hour windows" in dates_only.check("time").detail
    day_window = protocol(crit("I1", "inclusion", SHOCK, window=(3, "day")))
    assert run_gate(db(t=_midnight), day_window, concept_sets=sets,
                    context=SYNTH).check("time").status == "green"


def test_quality_violations():
    def bad_visits(tables):
        v = tables["visit_occurrence"]
        v.loc[v.index[:30], "visit_end_date"] = (
            v.loc[v.index[:30], "visit_start_date"].map(lambda d: d.replace(year=d.year - 1)))

    r = run_gate(db(q=bad_visits), None, context=SYNTH)
    q = r.check("quality")
    assert q.status == "red"  # 10% of persons
    assert q.values["violations"]["visit ends before it starts"] == 30


def test_small_cells_are_suppressed():
    assert suppress(0) == 0 and suppress(5) == "<11" and suppress(11) == 11
    r = run_gate(db(n=6), protocol(crit("I1", "inclusion", SHOCK)),
                 concept_sets={"I1:septic shock": [SEPTIC_SHOCK]}, context=SYNTH)
    assert r.check("volume").detail.startswith("<11 persons")
    rows = r.check("concepts").values["concepts"]
    assert rows[0]["result"] == "<11 patients" and rows[0]["patients"] == "<11"
    assert r.check("concepts").status == "amber"
    assert "6" not in r.check("volume").detail


@pytest.mark.parametrize("horizon,expected", [(28, "green"), (400, "amber")])
def test_follow_up_coverage(horizon, expected):
    r = run_gate(db(), protocol(crit("I1", "inclusion", SHOCK), horizon=horizon), context=SYNTH)
    assert r.check("follow_up").status == expected
