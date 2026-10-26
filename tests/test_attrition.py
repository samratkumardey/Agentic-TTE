"""Attrition table and feasibility from the compiled cohort."""

import pytest

from agentic_tte.omop import DataContext, run_gate
from agentic_tte.omop.attrition import attrition, feasibility
from agentic_tte.omop.compiler import compile_protocol
from agentic_tte.protocol import TargetTrialProtocol, ValueConstraint
from agentic_tte.synth import omop as so
from agentic_tte.synth.dgp import Scenario, generate
from tests.test_compiler import (
    SPEC,
    SYNTH_SPEC,
    _db,
    _synth_protocol,
    atom,
    crit,
    gt,
    ref,
    within,
)


def _age(op, v):
    return atom(ref("age", "Demographic"), value=ValueConstraint(comparator=op, value=v,
                                                                 unit="years"))


def test_flow_on_hand_built_db():
    p = TargetTrialProtocol(protocol_id="T", eligibility=[
        crit("I1", "inclusion", _age(">=", 18)),
        crit("E1", "exclusion", atom(ref("shock", "Condition", 100), negated=True)),
        crit("E2", "exclusion", atom(ref("unmapped", "Condition"))),
        crit("K", "exclusion", atom(ref("prisoner", "Observation")), category="administrative"),
    ])  # fmt: skip
    rep = attrition(_db(), compile_protocol(p, SPEC))
    flow = [(s.criterion, s.remaining, s.removed, s.met_alone) for s in rep.steps]
    assert flow == [
        (None, 6, 0, None),  # persons
        (None, 6, 0, None),  # with the index event
        (None, 5, 1, None),  # p5 died before time zero
        ("I1", 4, 1, 4),  # p3 is 17
        ("E1", 3, 1, 2),  # p2 has no shock record up to time zero (p3 neither)
    ]
    assert rep.eligible == 3
    assert {c["id"] for c in rep.not_applied} == {"E2", "K"}
    assert rep.arms is None  # no treatment strategies: frame not compiled
    pub = rep.public()
    assert pub["eligible"] == "<11" and pub["steps"][3]["remaining"] == "<11"
    md = rep.to_markdown()
    assert "Not applied" in md and "E2 (not_compiled)" in md and " 3 " not in md
    check = feasibility(rep, p)
    assert check.status == "amber" and "events unknown" in check.detail
    assert check.values["eligible"] == "<11"


def _synth(n=1500, **kw):
    c = generate(Scenario(n=n, **kw), with_truth=False)
    return c, so.write_duckdb(so.to_omop(c))


def test_synthetic_arms_match_the_reference_frame():
    _, con = _synth()
    p = _synth_protocol([crit("I1", "inclusion", atom(ref("septic shock", "Condition"))),
                         crit("I2", "inclusion", _age(">=", 50))])
    rep = attrition(con, compile_protocol(p, SYNTH_SPEC))
    ref_frame = so.analysis_frame(con)
    expected = ref_frame[ref_frame.age >= 50]
    assert rep.eligible == len(expected)
    arms = {a.arm: a for a in rep.arms}
    assert arms["treatment"].patients == int(expected.treated.sum())
    assert arms["treatment"].events == int(expected[expected.treated == 1].event.sum())
    assert rep.events == int(expected.event.sum())
    assert sum(rep.end_reasons.values()) == rep.eligible
    assert 0 < rep.risk_at_horizon < 1 and rep.confounders == 4
    assert rep.risk_at_horizon >= expected.event.mean()  # KM accounts for censoring

    ok = feasibility(rep, p, design_hr=0.5)
    assert ok.status == "green", ok.detail
    assert ok.values["events_required"] < rep.events and ok.values["n_required"] > 0
    weak = feasibility(rep, p, design_hr=0.95)
    assert weak.status == "amber" and "not enough events" in weak.detail
    assert "power not assessed" in feasibility(rep, p).detail


def test_red_when_nobody_or_no_events():
    _, con = _synth(n=300)
    nobody = _synth_protocol([crit("I1", "inclusion", _age(">=", 200))])
    rep = attrition(con, compile_protocol(nobody, SYNTH_SPEC))
    assert rep.eligible == 0 and feasibility(rep, nobody).status == "red"
    _, con = _synth(n=300, baseline_hazard=1e-9)
    p = _synth_protocol([crit("I1", "inclusion", atom(ref("septic shock", "Condition")))])
    rep = attrition(con, compile_protocol(p, SYNTH_SPEC))
    check = feasibility(rep, p)
    assert rep.events == 0 and check.status == "red" and "no outcome events" in check.detail


def test_few_events_per_confounder_is_amber():
    _, con = _synth(n=300, baseline_hazard=0.001)
    p = _synth_protocol([crit("I1", "inclusion", atom(ref("septic shock", "Condition")))])
    rep = attrition(con, compile_protocol(p, SYNTH_SPEC))
    assert 0 < rep.events < 40
    assert "events per confounder" in feasibility(rep, p).detail


def test_gate_runs_feasibility_from_attrition():
    _, con = _synth(n=400)
    p = _synth_protocol([crit("I1", "inclusion", atom(ref("septic shock", "Condition"))),
                         crit("I2", "inclusion", atom(ref("lactate", "Measurement"),
                                                      value=gt(2), temporal=within(
                                                          "x_at_time_of_y")))])
    p = p.model_copy(update={"source": p.source.model_copy(update={"enrollment": 200})})
    sets = {**SYNTH_SPEC.concept_sets, "I2:lactate": [so.LACTATE]}
    rep = attrition(con, compile_protocol(p, SYNTH_SPEC.model_copy(
        update={"concept_sets": sets})))  # fmt: skip
    ctx = DataContext(data_classification="synthetic")
    before = run_gate(con, p, context=ctx)
    assert before.check("feasibility").status == "not_run"
    after = run_gate(con, p, context=ctx, attrition=rep, design_hr=0.6)
    f = after.check("feasibility")
    assert f.status in ("green", "amber") and "the trial enrolled 200" in f.detail
    assert f.values["eligible"] == rep.eligible


@pytest.mark.parametrize("n", [5])
def test_small_arms_are_suppressed_everywhere(n):
    _, con = _synth(n=n)
    p = _synth_protocol([crit("I1", "inclusion", atom(ref("septic shock", "Condition")))])
    rep = attrition(con, compile_protocol(p, SYNTH_SPEC))
    pub = rep.public()
    assert all(a["patients"] in ("<11", 0) for a in pub["arms"])
    assert all(a["person_days"] is None for a in pub["arms"] if a["patients"] == "<11")
    check = feasibility(rep, p)
    assert check.status in ("amber", "red")
    assert all(str(v["patients"]) in ("<11", "0") for v in check.values["arms"].values())


def test_cli_compile_then_gate(tmp_path):
    import json

    from typer.testing import CliRunner

    from agentic_tte.cli import app

    db = tmp_path / "synth.duckdb"
    so.write_duckdb(so.to_omop(generate(Scenario(n=400), with_truth=False)), db).close()
    proto = _synth_protocol([crit("I1", "inclusion", atom(ref("septic shock", "Condition")))])
    p, spec, out = tmp_path / "p.json", tmp_path / "spec.json", tmp_path / "out"
    p.write_text(proto.model_dump_json(), encoding="utf-8")
    spec.write_text(SYNTH_SPEC.model_dump_json(), encoding="utf-8")
    run = CliRunner().invoke
    r = run(app, ["compile", str(p), "-o", str(out), "--spec", str(spec), "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert json.loads((out / "attrition.json").read_text())["eligible"] == 400
    r = run(app, ["gate", str(p), "--db", str(db), "--classification", "synthetic",
                  "--compiled", str(out / "compile.json"), "--design-hr", "0.5"])
    assert "Feasibility" in r.output and "400 eligible" in r.output
    # A compile from another protocol version is refused.
    p.write_text(proto.model_copy(update={"protocol_id": "OTHER"}).model_dump_json(), "utf-8")
    r = run(app, ["gate", str(p), "--db", str(db), "--classification", "synthetic",
                  "--compiled", str(out / "compile.json")])
    assert r.exit_code == 2
