"""E2 cohort checks (eval/SCORING.md Phase 3 changelog item 16): metamorphic relations and the
differential test against the Python reference evaluator hold on correct compiles, and catch a
deliberately broken one."""

import importlib.util
import sys
from pathlib import Path

import duckdb
import pytest

from agentic_tte.omop import cohort_checks as ch
from agentic_tte.omop.compiler import compile_protocol
from agentic_tte.protocol import (
    ConceptRef,
    Criterion,
    CriterionAtom,
    Duration,
    TargetTrialProtocol,
    TemporalConstraint,
    ValueConstraint,
)
from agentic_tte.synth import omop as so
from agentic_tte.synth.dgp import Scenario, generate
from tests.test_compiler import SPEC, _db, atom, crit, gt, ref, within

EVAL = Path(__file__).parents[1] / "eval"
_spec = importlib.util.spec_from_file_location("fault_injection", EVAL / "fault_injection.py")
fi = importlib.util.module_from_spec(_spec)
sys.modules["fault_injection"] = fi
_spec.loader.exec_module(fi)


def _design():
    p, spec = fi.synthetic_design()
    p.eligibility += [
        Criterion(id="I2", kind="inclusion", text="age >= 40", atoms=[CriterionAtom(
            concepts=[ConceptRef(name="age", domain="Demographic")],
            value=ValueConstraint(comparator=">=", value=40, unit="years"))]),
        Criterion(id="I3", kind="inclusion", text="lactate > 2", atoms=[CriterionAtom(
            concepts=[ConceptRef(name="lactate", domain="Measurement")],
            value=ValueConstraint(comparator=">", value=2, unit="mmol/L"))]),
        Criterion(id="E2", kind="exclusion", text="female", atoms=[CriterionAtom(
            concepts=[ConceptRef(name="female", domain="Demographic")])]),
        Criterion(id="I4", kind="inclusion", text="SOFA within 24 h", atoms=[CriterionAtom(
            concepts=[ConceptRef(name="sofa", domain="Measurement")],
            temporal=TemporalConstraint(relation="x_within_t_before_y",
                                        window=Duration(amount=24, unit="hour")))]),
    ]  # fmt: skip
    spec.concept_sets.update({"I3:lactate": [so.LACTATE], "I4:sofa": [so.SOFA]})
    return p, spec


@pytest.fixture(scope="module")
def synth_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("e2") / "synth.duckdb"
    so.write_duckdb(so.to_omop(generate(Scenario(n=600, seed=3, timing="delayed"),
                                        with_truth=False)), path).close()  # fmt: skip
    con = duckdb.connect(str(path), read_only=True)
    yield con, path
    con.close()


def test_relations_and_differential_hold_on_a_correct_compile(synth_db):
    con, path = synth_db
    p, spec = _design()
    rep = ch.check_cohort(con, p, spec, db_path=path)
    kinds = {r.relation for r in rep.relations}
    assert kinds == {"drop", "order", "duplicate", "threshold", "window", "time_shift"}
    assert all(r.held for r in rep.relations), rep.to_markdown()
    assert rep.index_disagree == 0
    assert [d.disagree for d in rep.differential] == [0] * 6
    assert all(d.compared == 600 for d in rep.differential)
    md = rep.to_markdown()
    assert "13 of 13 held" in md and "time zero: 0 patients differ" in md


def test_the_checks_catch_a_broken_compiler(synth_db, monkeypatch):
    """Power: negate one compiled predicate (a plausible SQL bug) and both kinds of check
    report it."""
    con, path = synth_db
    p, spec = _design()
    real = compile_protocol

    def broken(protocol, spec_):
        cc = real(protocol, spec_)
        for c in cc.criteria:
            if c.id == "I3" and c.predicate:
                cc.ctes = cc.ctes.replace(f"{c.predicate} AS {c.column}",
                                          f"(NOT {c.predicate}) AS {c.column}")
        return cc

    monkeypatch.setattr(ch, "compile_protocol", broken)
    diffs = {d.id: d.disagree for d in ch.differential(con, p, spec)[0]}
    assert diffs["I3"] > 0 and diffs["I2"] == 0
    # Stricter threshold should shrink the cohort; with the predicate negated it grows it.
    rel = {r.detail: r for r in ch.metamorphic(con, p, spec)}
    assert not rel["I3 threshold 10% stricter"].held


def test_differential_on_the_hand_built_edge_cases():
    """Date-only records, the latest lab within the look-back, a death before time zero and a
    minor: the reference evaluator agrees with the SQL patient by patient; a criterion with an
    anchor other than time zero is reported as not compared."""
    con = _db()
    p = TargetTrialProtocol(protocol_id="EDGE", eligibility=[
        crit("I1", "inclusion", atom(ref("cond", "Condition", 100))),
        crit("I2", "inclusion", atom(ref("lactate", "Measurement", 200), value=gt(2, "mmol/L"),
                                     temporal=within("x_at_time_of_y"))),
        crit("I3", "inclusion", atom(ref("age", "Demographic"), value=ValueConstraint(
            comparator=">=", value=18, unit="years"))),
        crit("E1", "exclusion", atom(ref("cond", "Condition", 100),
                                     temporal=within("x_within_t_before_y", 1, "day"))),
        crit("E2", "exclusion", atom(ref("cond", "Condition", 100),
                                     temporal=within("x_before_y", anchor="hospital admission"))),
    ])  # fmt: skip
    diffs, index_disagree = ch.differential(con, p, SPEC)
    assert index_disagree == 0
    by = {d.id: d for d in diffs}
    assert all(by[i].disagree == 0 and by[i].compared > 0 for i in ("I1", "I2", "I3", "E1"))
    assert by["E2"].compared == 0 and "anchor" in by["E2"].note


def test_public_counts_are_suppressed(synth_db):
    con, _ = synth_db
    p, spec = _design()
    rep = ch.check_cohort(con, p, spec)
    pub = rep.public()
    assert all(isinstance(r["violations"], int) for r in pub["relations"])
    assert "time-shift relation not run" in rep.to_markdown()
    small = ch.Relation(relation="drop", detail="x", expect="superset", n_before=5, n_after=7,
                        violations=2)  # fmt: skip
    rep.relations = [small]
    assert rep.public()["relations"][0]["n_before"] == "<11"


def test_cli_cohort_check(synth_db, tmp_path):
    from typer.testing import CliRunner

    from agentic_tte.cli import app

    _, path = synth_db
    p, spec = _design()
    (tmp_path / "protocol.json").write_text(p.model_dump_json(), encoding="utf-8")
    (tmp_path / "spec.json").write_text(spec.model_dump_json(), encoding="utf-8")
    out = tmp_path / "e2"
    r = CliRunner().invoke(app, ["cohort", "check", "--db", str(path), "--protocol",
                                 str(tmp_path / "protocol.json"), "--spec",
                                 str(tmp_path / "spec.json"), "-o", str(out)])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "13 of 13 held" in (out / "cohort_checks.md").read_text(encoding="utf-8")
    assert (out / "cohort_checks.json").exists()
