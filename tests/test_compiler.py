"""Criteria-to-SQL compiler: temporal semantics on a hand-built OMOP database, and recovery of
the hand-written reference SQL on the synthetic cohort."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_tte.omop.compiler import (
    CompileError,
    CompileSpec,
    EventDef,
    compile_protocol,
    spec_template,
)
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
from agentic_tte.synth.dgp import Scenario, generate

FIXTURE = Path(__file__).parent / "fixtures" / "protocols" / "NCT04134403.trialist-v1.qwen3-32b.json"  # noqa: E501

# Hand-built database: index visit (concept 10) at T0 for everyone; 11 = hospital admission;
# condition 100 has descendant 101; 200 = lactate; 300 = a drug; 500 = an outcome condition.
T0 = pd.Timestamp("2020-01-10 12:00")
H, D = pd.Timedelta(hours=1), pd.Timedelta(days=1)


def _db():
    persons = [  # person_id, female, birth
        (1, 0, T0 - pd.Timedelta(days=365.25 * 70)),
        (2, 1, T0 - pd.Timedelta(days=365.25 * 50)),
        (3, 1, T0 - pd.Timedelta(days=365.25 * 17)),
        (4, 0, T0 - pd.Timedelta(days=365.25 * 40)),
        (5, 0, T0 - pd.Timedelta(days=365.25 * 60)),  # dies before time zero
        (6, 1, T0 - pd.Timedelta(days=365.25 * 30)),
    ]
    person = pd.DataFrame([{"person_id": i, "gender_concept_id": 8532 if f else 8507,
                            "year_of_birth": b.year, "month_of_birth": b.month,
                            "day_of_birth": b.day, "birth_datetime": b} for i, f, b in persons])
    ids = [p[0] for p in persons]
    op = pd.DataFrame({"person_id": ids,
                       "observation_period_start_date": pd.Timestamp("2019-01-01").date(),
                       "observation_period_end_date": pd.Timestamp("2020-12-31").date()})

    def visits(rows):
        return pd.DataFrame([{"person_id": p, "visit_concept_id": c, "visit_start_datetime": t,
                              "visit_start_date": t.date()} for p, c, t in rows])

    def cond(rows):
        return pd.DataFrame([{"person_id": p, "condition_concept_id": c,
                              "condition_start_datetime": t, "condition_start_date": d.date()}
                             for p, c, t, d in rows])

    visit = visits([(i, 10, T0) for i in ids] + [(1, 11, T0 - 3 * D), (4, 11, T0 - 3 * D)])
    condition = cond([(1, 100, T0 - 2 * D, T0 - 2 * D), (2, 100, T0 + D, T0 + D),
                      (4, 101, T0 - H, T0 - H), (6, 100, pd.NaT, T0),  # date only
                      (1, 500, T0 + 4 * D, T0 + 4 * D)])
    measurement = pd.DataFrame([{"person_id": p, "measurement_concept_id": 200,
                                 "measurement_datetime": t, "measurement_date": t.date(),
                                 "value_as_number": v} for p, t, v in [
        (1, T0 - 2 * H, 3.0), (1, T0 - 30 * H, 1.0), (2, T0 - H, 1.5), (2, T0 - 3 * H, 4.0)]])
    drug = pd.DataFrame([{"person_id": 1, "drug_concept_id": 300,
                          "drug_exposure_start_datetime": T0 + D,
                          "drug_exposure_start_date": (T0 + D).date()}])
    death = pd.DataFrame([{"person_id": 5, "death_date": (T0 - 2 * D).date(),
                           "death_datetime": T0 - 2 * D},
                          {"person_id": 2, "death_date": (T0 + 5 * D).date(),
                           "death_datetime": T0 + 5 * D}])
    ancestor = pd.DataFrame({"ancestor_concept_id": [100, 100],
                             "descendant_concept_id": [100, 101]})
    return so.write_duckdb({"person": person, "observation_period": op, "visit_occurrence": visit,
                            "condition_occurrence": condition, "measurement": measurement,
                            "drug_exposure": drug, "death": death, "concept_ancestor": ancestor})


SPEC = CompileSpec(time_zero=EventDef(domain="Visit", concept_ids=[10]),
                   anchors={"hospital admission": EventDef(domain="Visit", concept_ids=[11],
                                                           occurrence="last")})


def ref(name, domain, cid=None, **kw):
    return ConceptRef(name=name, domain=domain, omop_concept_id=cid, **kw)


def crit(cid, kind="inclusion", *atoms, **kw):
    return Criterion(id=cid, kind=kind, text=f"{cid} text", atoms=list(atoms), **kw)


def atom(*concepts, value=None, temporal=None, **kw):
    return CriterionAtom(concepts=list(concepts), value=value, temporal=temporal, **kw)


def gt(v, unit=None):
    return ValueConstraint(comparator=">", value=v, unit=unit)


def within(relation, amount=None, unit="hour", anchor="time_zero"):
    w = Duration(amount=amount, unit=unit) if amount is not None else None
    return TemporalConstraint(relation=relation, window=w, anchor=anchor)


def flags(criteria, spec=SPEC, con=None):
    p = TargetTrialProtocol(protocol_id="T", eligibility=criteria)
    cc = compile_protocol(p, spec)
    df = (con or _db()).execute(cc.cohort_sql).df().set_index("person_id")
    return cc, df


def met(df, col):
    return sorted(df.index[df[col]].tolist())


def test_base_and_history_with_descendants():
    cc, df = flags([crit("A", "inclusion", atom(ref("shock", "Condition", 100)))])
    assert 5 not in df.index  # died before time zero
    # p2's record is after time zero (no look-ahead); p4 has the descendant 101; p6 has a
    # date-only record on the index day, placed at midnight.
    assert met(df, "c_a") == [1, 4, 6]
    assert cc.criteria[0].status == "compiled"
    _, df = flags([crit("A", "inclusion", atom(ref("shock", "Condition", 100)))],
                  spec=SPEC.model_copy(update={"descendants": False}))
    assert met(df, "c_a") == [1, 6]


def test_measurement_latest_value_versus_window():
    at = atom(ref("lactate", "Measurement", 200), value=gt(2, "mmol/L"),
              temporal=within("x_at_time_of_y"))
    win = atom(ref("lactate", "Measurement", 200), value=gt(2, "mmol/L"),
               temporal=within("x_within_t_before_y", 6))
    cc, df = flags([crit("L1", "inclusion", at), crit("L2", "inclusion", win)])
    assert met(df, "c_l1") == [1]  # p2's latest value is 1.5
    assert met(df, "c_l2") == [1, 2]  # p2 had 4.0 three hours before
    assert any("units are not converted" in w for w in cc.criteria[0].warnings)
    # Outside the 24-hour lookback: p1's 1.0 at -30 h is never the latest value used.
    lb = SPEC.model_copy(update={"measurement_lookback": Duration(amount=1, unit="hour")})
    _, df = flags([crit("L1", "inclusion", at)], spec=lb)
    assert met(df, "c_l1") == []


def test_negation_demographics_and_eligible():
    cc, df = flags([
        crit("I1", "inclusion", atom(ref("age", "Demographic"), value=ValueConstraint(
            comparator=">=", value=18, unit="years"))),
        crit("E1", "exclusion", atom(ref("shock", "Condition", 100), negated=True)),
        crit("E2", "exclusion", atom(ref("sex", "Demographic"), value=ValueConstraint(
            comparator="present", text="female"))),
    ])
    assert met(df, "c_i1") == [1, 2, 4, 6]  # p3 is 17
    assert met(df, "c_e1") == [2, 3]
    assert met(df, "c_e2") == [2, 3, 6]
    assert met(df, "eligible") == [1, 4]


def test_named_anchor_and_look_ahead():
    after_admission = atom(ref("shock", "Condition", 100),
                           temporal=within("x_within_t_after_y", 2, "day", "hospital admission"))
    cc, df = flags([crit("A", "inclusion", after_admission)])
    assert met(df, "c_a") == [1]  # p4's record is 3 days minus 1 hour after admission
    assert any("no look-ahead" in w for w in cc.criteria[0].warnings)
    # Anchored at time zero (here via "randomization"), "after" needs the future.
    after_t0 = within("x_after_y", anchor="randomization")
    cc, _ = flags([crit("B", "inclusion", atom(ref("shock", "Condition", 100), temporal=after_t0))])
    assert cc.criteria[0].status == "not_compiled"
    assert "look-ahead" in cc.criteria[0].reasons[0]


def test_what_is_not_compiled_is_reported():
    cc, df = flags([
        crit("U", "inclusion", atom(ref("unmapped thing", "Condition"))),
        crit("V", "exclusion", atom(ref("vasopressor", "Drug", 300), value=gt(24, "hour"))),
        crit("M", "inclusion", atom(ref("sepsis", "Condition", 100),
                                    ref("lactate", "Measurement", 200), value=gt(2))),
        crit("N", "inclusion", atom(ref("shock", "Condition", 100),
                                    temporal=within("x_before_y", anchor="ICU discharge"))),
        crit("K", "exclusion", atom(ref("prisoner", "Observation")), category="administrative"),
        crit("Y", "exclusion", atom(ref("shock", "Condition", 100),
                                    ref("rare", "Condition", status="unmappable"),
                                    concept_logic="any")),
    ])
    st = {c.id: (c.status, c.reasons) for c in cc.criteria}
    assert st["U"][0] == "not_compiled" and "no approved concept set" in st["U"][1][0]
    assert st["V"][0] == "not_compiled" and "Drug" in st["V"][1][0]
    assert st["M"][0] == "not_compiled" and "split the atom" in st["M"][1][0]
    assert st["N"][0] == "not_compiled" and "ICU discharge" in st["N"][1][0]
    assert st["K"][0] == "not_emulable"
    assert st["Y"][0] == "compiled"
    assert any("unmappable" in w for w in cc.criteria[-1].warnings)
    # Only Y counts towards eligibility; everything else is listed, never silently used.
    assert set(c for c in df.columns if c.startswith("c_")) == {"c_y"}
    assert met(df, "eligible") == [2, 3]
    md = cc.to_markdown()
    assert "4 not compiled" in md and "no look-ahead" in md
    with pytest.raises(CompileError, match="U:"):
        compile_protocol(TargetTrialProtocol(protocol_id="T", eligibility=[cc_crit("U")]),
                         SPEC.model_copy(update={"strict": True}))


def cc_crit(cid):
    return crit(cid, "inclusion", atom(ref("unmapped thing", "Condition")))


def test_concept_sets_from_the_book_take_precedence():
    spec = SPEC.model_copy(update={"concept_sets": {"A:shock": [101]}})
    _, df = flags([crit("A", "inclusion", atom(ref("shock", "Condition", 100)))], spec=spec)
    assert met(df, "c_a") == [4]


def test_frame_with_concept_outcome_on_hand_built_db():
    p = TargetTrialProtocol(
        protocol_id="T", eligibility=[crit("A", "inclusion", atom(ref("x", "Condition", 100)))],
        treatment_strategies=[
            TreatmentStrategy(label="Rx", components=[TreatmentComponent(
                concept=ref("drug", "Drug", 300))]),
            TreatmentStrategy(label="None", is_comparator=True)],
        assignment=AssignmentProcedure(confounders=[ref("lactate", "Measurement", 200),
                                                    ref("age", "Demographic")]),
        outcomes=[Outcome(name="organ failure", primary=True, concept=ref("of", "Condition", 500),
                          horizon=Duration(amount=28, unit="day"))])  # fmt: skip
    cc = compile_protocol(p, SPEC)
    f = _db().execute(cc.frame_sql).df().set_index("person_id")
    assert list(f.columns) == cc.frame_columns[1:]
    assert sorted(f.index) == [1, 4, 6]
    assert f.loc[1, "treated"] == 1 and f.loc[1, "treat_start"] == pytest.approx(1.0)
    assert f.loc[1, "event"] == 1 and f.loc[1, "time"] == pytest.approx(4.0)
    assert f.loc[1, "lactate"] == 3.0 and f.loc[1, "age"] == pytest.approx(70, abs=0.01)
    assert f.loc[4, "end_reason"] == "end_of_follow_up" and f.loc[4, "time"] == pytest.approx(28)
    assert np.isnan(f.loc[4, "lactate"])


def _synth_protocol(eligibility):
    return TargetTrialProtocol(
        protocol_id="SYNTH", eligibility=eligibility,
        treatment_strategies=[
            TreatmentStrategy(label="Study drug", components=[TreatmentComponent(
                concept=ref("study drug", "Drug"))]),
            TreatmentStrategy(label="No study drug", is_comparator=True)],
        assignment=AssignmentProcedure(confounders=[
            ref("age", "Demographic"), ref("sex", "Demographic"),
            ref("sofa", "Measurement"), ref("lactate", "Measurement")]),
        outcomes=[Outcome(name="28-day all-cause mortality", primary=True,
                          horizon=Duration(amount=28, unit="day"))])  # fmt: skip


SYNTH_SETS = {"T:Study drug:study drug": [so.STUDY_DRUG], "C:sofa": [so.SOFA],
              "C:lactate": [so.LACTATE], "I1:septic shock": [so.SEPTIC_SHOCK],
              "I3:lactate": [so.LACTATE]}  # fmt: skip
SYNTH_SPEC = CompileSpec(time_zero=EventDef(domain="Visit", concept_ids=[so.ICU_VISIT]),
                         concept_sets=SYNTH_SETS, descendants=False)


@pytest.mark.parametrize("timing", ["at_time_zero", "delayed"])
def test_frame_reproduces_reference_sql(timing):
    """The compiled analysis frame equals synth.omop.ANALYSIS_SQL, the hand-written
    reference, except censoring times, which observation-period dates resolve to a day."""
    con = so.write_duckdb(so.to_omop(generate(Scenario(n=600, timing=timing,
                                                       missing_lactate=0.1), with_truth=False)))
    p = _synth_protocol([crit("I1", "inclusion", atom(ref("septic shock", "Condition")))])
    cc = compile_protocol(p, SYNTH_SPEC)
    ref_frame, f = so.analysis_frame(con), con.execute(cc.frame_sql).df()
    assert cc.frame_columns == ["person_id", "age", "female", "sofa", "lactate", "treated",
                                "treat_start", "time", "event", "end_reason"]
    assert (ref_frame.person_id.to_numpy() == f.person_id.to_numpy()).all()
    for col in ["age", "female", "sofa", "lactate", "treated", "treat_start", "event"]:
        assert np.allclose(ref_frame[col].astype(float), f[col].astype(float), equal_nan=True), col
    d = f.time - ref_frame.time
    death = f.end_reason == "death"
    assert np.allclose(d[death], 0)
    # Censoring (lost to follow-up) is resolved to the end of the last observed day.
    assert ((d[~death] >= 0) & (d[~death] < 1)).all() and (d[~death] > 0).any()
    assert (f.event == (f.end_reason == "death")).all() and f.time.max() <= 28


def test_synthetic_eligibility_matches_the_frame_filter():
    con = so.write_duckdb(so.to_omop(generate(Scenario(n=600), with_truth=False)))
    ref_frame = so.analysis_frame(con)
    p = _synth_protocol([
        crit("I1", "inclusion", atom(ref("septic shock", "Condition"))),
        crit("I2", "inclusion", atom(ref("age", "Demographic"), value=ValueConstraint(
            comparator=">=", value=65, unit="years"))),
        crit("I3", "inclusion", atom(ref("lactate", "Measurement"), value=gt(2, "mmol/L"),
                                     temporal=within("x_at_time_of_y"))),
        crit("E1", "exclusion", atom(ref("female", "Demographic"))),
    ])
    cc = compile_protocol(p, SYNTH_SPEC)
    assert [c.status for c in cc.criteria] == ["compiled"] * 4
    cohort = con.execute(cc.cohort_sql).df()
    expected = ref_frame[(ref_frame.age >= 65) & (ref_frame.lactate > 2) & (ref_frame.female == 0)]
    assert sorted(cohort.person_id[cohort.eligible]) == sorted(expected.person_id)
    frame = con.execute(cc.frame_sql).df()
    assert sorted(frame.person_id) == sorted(expected.person_id)


def test_nct04134403_compile_report():
    """The real Trialist output: what compiles, and why the rest does not."""
    p = TargetTrialProtocol.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    tmpl = spec_template(p)
    assert "septic shock onset" in tmpl["time_zero"]["note"]
    assert set(tmpl["anchors"]) == {"primary admission", "hospital admission"}
    cc = compile_protocol(p, SYNTH_SPEC.model_copy(update={"concept_sets": {}}))
    st = {c.id: c for c in cc.criteria}
    assert st["I3"].status == st["E1"].status == "compiled"  # age
    assert "look-ahead" in st["I1"].reasons[0]  # ICU admission after septic shock onset = t0
    assert "split the atom" in st["I2"].reasons[0]
    assert "Drug" in st["E8"].reasons[0] and "Condition" in st["E7"].reasons[0]
    assert "primary admission" in " ".join(st["E5"].reasons)
    assert cc.frame_sql is None  # treatments have no approved concept sets yet
    assert any("no approved concept set" in n for n in cc.frame_notes)
    con = so.write_duckdb(so.to_omop(generate(Scenario(n=100), with_truth=False)))
    cohort = con.execute(cc.cohort_sql).df()
    assert cohort.eligible.all() and len(cohort) == 100  # synthetic patients are all adults


def test_cli_compile(tmp_path):
    import json

    from typer.testing import CliRunner

    from agentic_tte.cli import app

    db = tmp_path / "synth.duckdb"
    so.write_duckdb(so.to_omop(generate(Scenario(n=150), with_truth=False)), db).close()
    p = tmp_path / "p.json"
    p.write_text(_synth_protocol([crit("I1", "inclusion", atom(ref("septic shock", "Condition")))])
                 .model_dump_json(), encoding="utf-8")
    run, out = CliRunner().invoke, tmp_path / "out"
    r = run(app, ["compile", str(p), "-o", str(out)])
    assert r.exit_code == 1 and json.loads((out / "spec.json").read_text())["time_zero"]
    spec = tmp_path / "spec.json"
    spec.write_text(SYNTH_SPEC.model_dump_json(), encoding="utf-8")
    r = run(app, ["compile", str(p), "-o", str(out), "--spec", str(spec), "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert "eligible 150" in r.output
    assert (out / "frame.sql").exists() and "Compiled cohort" in (out / "compile.md").read_text()
