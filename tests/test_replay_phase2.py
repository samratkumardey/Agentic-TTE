"""Phase 2 exit check: the three-trial Replay from the real Phase 0 drafts
(eval/replay_phase2.py), plus the compiler and gate rules those drafts exposed."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from agentic_tte.omop import DataContext, run_gate
from agentic_tte.omop.compiler import (
    CompileSpec,
    EventDef,
    OutcomeSpec,
    _death_only,
    compile_protocol,
)
from agentic_tte.protocol import Duration, TargetTrialProtocol
from agentic_tte.synth import omop as so
from agentic_tte.synth.dgp import Scenario, generate
from tests.test_compiler import (
    SYNTH_SPEC,
    _synth_protocol,
    atom,
    crit,
    gt,
    ref,
)

EVAL = Path(__file__).resolve().parents[1] / "eval"
_spec = importlib.util.spec_from_file_location("replay_phase2", EVAL / "replay_phase2.py")
rp = importlib.util.module_from_spec(_spec)
sys.modules["replay_phase2"] = rp
_spec.loader.exec_module(rp)


# --- rules found in the real drafts ----------------------------------------------------------


@pytest.mark.parametrize("name,death", [
    ("In-hospital or 28-day mortality rate", True), ("90-day mortality", True),
    ("Mortality", True), ("28-day all-cause mortality", True),
    ("Composite of Rehospitalization Due to Heart Failure and All-Cause Mortality", False),
    ("Death or disability", False), ("Ventilator-free days", False),
])  # fmt: skip
def test_only_pure_mortality_outcomes_default_to_death(name, death):
    assert _death_only(name) is death


def _with(p, **updates):
    return p.model_copy(update=updates)


def test_composite_outcome_needs_a_reviewed_definition():
    from agentic_tte.protocol import Outcome

    con = so.write_duckdb(so.to_omop(generate(Scenario(n=300), with_truth=False)))
    p = _with(_synth_protocol([crit("I1", "inclusion", atom(ref("septic shock", "Condition")))]),
              outcomes=[Outcome(name="Rehospitalization for heart failure and all-cause "
                                     "mortality", primary=True)])  # fmt: skip
    cc = compile_protocol(p, SYNTH_SPEC)  # no horizon: reported, not raised
    assert cc.frame_sql is None and any("horizon" in n for n in cc.frame_notes)
    p = _with(p, outcomes=[p.outcomes[0].model_copy(update={"horizon": Duration(amount=28,
                                                                                unit="day")})])
    cc = compile_protocol(p, SYNTH_SPEC)
    assert cc.frame_sql is None
    assert any("composite" in n and "compile spec" in n for n in cc.frame_notes)
    spec = SYNTH_SPEC.model_copy(update={"outcome": OutcomeSpec(
        events=[EventDef(domain="Visit", concept_ids=[so.ICU_VISIT])], death=True)})
    cc = compile_protocol(p, spec)
    f = con.execute(cc.frame_sql).df()
    ref_frame = so.analysis_frame(con)  # no visits after time zero: the composite is death
    assert (f.event.to_numpy() == ref_frame.event.to_numpy()).all()


def test_placebo_comparator_and_duplicate_components():
    from agentic_tte.protocol import TreatmentComponent, TreatmentStrategy

    p = _synth_protocol([crit("I1", "inclusion", atom(ref("septic shock", "Condition")))])
    drug = TreatmentComponent(concept=ref("study drug", "Drug"))
    p = _with(p, treatment_strategies=[
        TreatmentStrategy(label="Study drug", components=[drug, drug]),
        TreatmentStrategy(label="Placebo arm", is_comparator=True, components=[
            TreatmentComponent(concept=ref("Placebo", "Drug"))])])  # fmt: skip
    cc = compile_protocol(p, SYNTH_SPEC)
    assert cc.frame_sql is not None
    assert any("placebo" in n for n in cc.frame_notes)
    assert any("same component more than once" in n for n in cc.frame_notes)
    active = _with(p, treatment_strategies=[p.treatment_strategies[0], TreatmentStrategy(
        label="Other drug", is_comparator=True,
        components=[TreatmentComponent(concept=ref("dobutamine", "Drug"))])])  # fmt: skip
    assert compile_protocol(active, SYNTH_SPEC).frame_sql is None  # truly active: not yet


def test_relative_change_is_not_read_as_a_level():
    from agentic_tte.protocol import Criterion

    rise = Criterion(id="I9", kind="inclusion", text="an increase in serum creatinine > 0.3",
                     atoms=[atom(ref("creatinine", "Measurement", 1), value=gt(0.3))])
    pct = Criterion(id="I8", kind="inclusion", text="creatinine 150%", atoms=[atom(
        ref("creatinine", "Measurement", 1), value=gt(150, "%"))])  # fmt: skip
    level = Criterion(id="I7", kind="inclusion", text="lactate > 2", atoms=[atom(
        ref("lactate", "Measurement", 1), value=gt(2))])  # fmt: skip
    cc = compile_protocol(TargetTrialProtocol(protocol_id="X", eligibility=[rise, pct, level]),
                          CompileSpec(time_zero=EventDef(domain="Visit", concept_ids=[1])))
    st = {c.id: c for c in cc.criteria}
    assert st["I9"].status == st["I8"].status == "not_compiled"
    assert "change from a baseline" in st["I9"].reasons[0]
    assert st["I7"].status == "compiled"


def test_empty_tables_do_not_fail_the_time_check():
    prof = rp.TRIALS[2].profile  # RRT is a procedure: drug_exposure is empty
    con = so.write_duckdb(so.to_omop(generate(Scenario(n=200), with_truth=False), prof))
    p = TargetTrialProtocol.model_validate_json(rp.TRIALS[2].draft.read_text(encoding="utf-8"))
    r = run_gate(con, p, context=DataContext(data_classification="synthetic"))
    t = r.check("time")
    assert t.status == "green" and "drug_exposure" not in t.values["coverage"]


def test_profiles_write_trial_concepts():
    prof = rp.TRIALS[2].profile
    t = so.to_omop(generate(Scenario(n=50, timing="delayed"), with_truth=False), prof)
    assert len(t["drug_exposure"]) == 0 and {"procedure_occurrence", "observation",
                                             "device_exposure"} <= set(t)  # fmt: skip
    procs = t["procedure_occurrence"].procedure_concept_id.value_counts()
    assert procs[so.PRIOR_PROCEDURE] == 50 and procs[so.STUDY_DRUG] > 0
    names = set(t["concept"].concept_name)
    assert {"Renal replacement therapy (synthetic)", "Cardiac surgery (synthetic)"} <= names


# --- the three-trial Replay ------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("i", [0, 1, 2], ids=[t.nct for t in rp.TRIALS])
def test_three_trial_replay(i):
    t = rp.TRIALS[i]
    t = replace(t, scenario=replace(t.scenario, n=2500))
    res = rp.replay(t)
    st = res["state"]
    assert st["status"] == "done"
    a = st["analysis"]
    p = next(e for e in a["estimates"] if e["method"] == a["primary_method"])
    assert abs(np.log(p["hr"]) - np.log(res["truth"])) < 0.2
    assert p["lower"] <= res["truth"] <= p["upper"]
    assert max(a["smd_after"].values()) < 0.1
    expected = {0: "iptw", 1: "clone-censor-weight", 2: "clone-censor-weight"}[i]
    assert a["primary_method"] == expected
    if t.nct == "NCT06091982":
        drafted = rp.replay(t, design={"reviewer": rp.REVIEWER,
                                       "confounders": t.design["confounders"]},
                            overrides=rp.AS_DRAFTED_RRT)  # fmt: skip
        assert drafted["state"]["analysis"]["immortal_time_risk"]
        assert not a["immortal_time_risk"]
