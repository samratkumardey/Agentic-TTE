"""Review helpers: `replay design-answer` (the reviewer's arm, criterion and grace-period
decisions written as a checked design.json) and `concepts counts` (a concept review checked
against the data, whole-database aggregates only)."""

import json
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from agentic_tte.cli import app
from agentic_tte.graph.answers import design_answer, parse_duration
from agentic_tte.omop.concept_sets import ConceptSet, ConceptSetBook
from agentic_tte.omop.counts import concept_counts, search, to_markdown
from agentic_tte.protocol import TargetTrialProtocol

PROTOCOL = (Path(__file__).parent / "fixtures" / "protocols"
            / "NCT04134403.final-H100.qwen3-32b.json")
TEMPLATE = {"reviewer": "", "confounders": None, "changes": {}}


def _broken():
    """A draft like APROCCHSS's: every arm a comparator, an extra factorial arm, and a
    component that belongs to another contrast."""
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    arms = p["treatment_strategies"]
    for a in arms:
        a["is_comparator"] = True
    extra = json.loads(json.dumps(arms[0]))
    extra["label"] = "factorial"
    arms.append(extra)
    return p


def test_design_answer_fixes_the_arms():
    p = _broken()
    answer, summary = design_answer(
        p, TEMPLATE, "Dr Test", treatment="Arm 1: Intervention", drop_arms=("factorial",),
        drop_components=("thiamine",), drop_criteria=("E1", "E2"),
        grace=parse_duration("24 hours"))  # fmt: skip
    arms = answer["changes"]["treatment_strategies"]
    assert [(a["label"], a["is_comparator"]) for a in arms] == [
        ("Arm 1: Intervention", False), ("Arm 2: Usual care", True)]
    assert [c["concept"]["name"] for c in arms[0]["components"]] == [
        "hydrocortisone sodium succinate", "ascorbic acid"]
    assert arms[0]["grace_period"] == {"amount": 24.0, "unit": "hour"}
    assert arms[0]["grace_period_origin"] == "user"
    ids = [c["id"] for c in answer["changes"]["eligibility"]]
    assert "E1" not in ids and "E2" not in ids and "I1" in ids
    assert answer["reviewer"] == "Dr Test" and answer["confounders"] is None
    assert not any("WARNING" in s and ("arms" in s or "grace" in s) for s in summary)
    assert "the Clinician's proposal, kept" in summary[-1]
    # the answer is what new_version applies: it must make a valid protocol
    TargetTrialProtocol.model_validate({**p, **answer["changes"]})
    assert p["treatment_strategies"][0]["is_comparator"]  # the input is unchanged


def test_design_answer_refuses_mistakes():
    p = _broken()
    with pytest.raises(ValueError, match="no arm labelled '9'"):
        design_answer(p, TEMPLATE, "Dr Test", treatment="9")
    with pytest.raises(ValueError, match="no component of the kept arms matches"):
        design_answer(p, TEMPLATE, "Dr Test", drop_components=("protein c",))
    with pytest.raises(ValueError, match="no criteria"):
        design_answer(p, TEMPLATE, "Dr Test", drop_criteria=("E99",))
    with pytest.raises(ValueError, match="needs a reviewer"):
        design_answer(p, TEMPLATE, " ")
    with pytest.raises(ValueError):
        parse_duration("24 fortnights")
    _, summary = design_answer(p, TEMPLATE, "Dr Test", treatment="Arm 1: Intervention")
    assert any("2 comparator arms" in s for s in summary)  # factorial kept: warned
    assert any("no grace period" in s for s in summary)


def test_design_answer_cli(tmp_path):
    out = tmp_path / "NCT04134403"
    out.mkdir()
    (out / "pending.txt").write_text("design", encoding="utf-8")
    (out / "protocol.json").write_text(json.dumps(_broken()), encoding="utf-8")
    (out / "design.json").write_text(json.dumps(TEMPLATE), encoding="utf-8")
    args = ["replay", "design-answer", "-o", str(out), "--reviewer", "Dr Test",
            "--treatment", "Arm 1: Intervention", "--drop-arm", "factorial",
            "--drop-criterion", "E1,E2", "--grace", "24 hour"]  # fmt: skip
    r = CliRunner().invoke(app, [*args, "--dry-run"])
    assert r.exit_code == 0, r.output
    assert json.loads((out / "design.json").read_text(encoding="utf-8")) == TEMPLATE
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    written = json.loads((out / "design.json").read_text(encoding="utf-8"))
    assert written["reviewer"] == "Dr Test" and len(written["changes"]["eligibility"]) > 0
    r = CliRunner().invoke(app, [*args[:5], "Dr Test", "--treatment", "nope"])
    assert r.exit_code == 2 and "not written" in r.output
    (out / "pending.txt").write_text("concepts", encoding="utf-8")
    assert CliRunner().invoke(app, args).exit_code == 2


def _cdm(path):
    con = duckdb.connect(str(path))
    con.execute("""CREATE TABLE concept AS SELECT * FROM (VALUES
        (1, 'hydrocortisone', 'Drug', 'S'), (2, 'hydrocortisone 100 MG Injection', 'Drug', 'S'),
        (3, 'hydrocortisone 1% Topical Cream', 'Drug', 'S'),
        (10, 'Platelets [#/volume] in Blood', 'Measurement', 'S'),
        (11, 'Platelet mean volume', 'Measurement', 'S'),
        (20, 'Septic shock', 'Condition', 'S'), (30, 'old code', 'Condition', NULL),
        (8848, 'thousand per microliter', 'Unit', 'S'))
        t(concept_id, concept_name, domain_id, standard_concept)""")
    con.execute("CREATE TABLE concept_ancestor AS SELECT * FROM (VALUES (1, 1), (1, 2), (1, 3), "
                "(10, 10), (11, 11), (20, 20)) t(ancestor_concept_id, descendant_concept_id)")
    con.execute("CREATE TABLE drug_exposure AS SELECT range % 40 AS person_id, "
                "CASE WHEN range < 35 THEN 2 ELSE 3 END AS drug_concept_id FROM range(60)")
    con.execute("CREATE TABLE measurement AS SELECT range AS person_id, "
                "10 AS measurement_concept_id, 150.0 + range AS value_as_number, "
                "8848 AS unit_concept_id, CAST(NULL AS VARCHAR) AS unit_source_value "
                "FROM range(50) UNION ALL SELECT 1, 11, 9.0, NULL, 'fL'")
    con.execute("CREATE TABLE condition_occurrence (person_id BIGINT, condition_concept_id BIGINT)")
    con.execute("CREATE TABLE observation (person_id BIGINT, observation_concept_id BIGINT)")
    con.close()
    return duckdb.connect(str(path), read_only=True)


def test_concept_counts(tmp_path):
    con = _cdm(tmp_path / "cdm.duckdb")
    decisions = {"reviewer": "", "decisions": {
        "T:2:hydrocortisone": {"action": "approve", "concept_ids": [1]},
        "E11:platelet count": {"action": "approve", "concept_ids": [11]},
        "I2:septic shock": {"action": "approve", "concept_ids": [20]},
        "E5:platelets": {"action": "replace", "concept_ids": [10]},
        "E6:old": {"action": "approve", "concept_ids": [20]},
        "C:x": {"action": "surrogate", "concept_ids": [99999]},
        "C:y": {"action": "unmappable", "concept_ids": []}}}
    book = ConceptSetBook(protocol_id="p", protocol_version=1, sets=[
        ConceptSet(key="T:2:hydrocortisone", scope="treatment", criterion_text="t",
                   concept_name="hydrocortisone", domain="Drug"),
        ConceptSet(key="E11:platelet count", scope="eligibility", criterion_text="e",
                   concept_name="platelet count", domain="Observation")])
    rows = {c.key: c for c in concept_counts(con, decisions, book)}
    assert set(rows) == {"T:2:hydrocortisone", "E11:platelet count", "I2:septic shock", "C:x",
                         "E5:platelets", "E6:old"}  # fmt: skip
    hc = rows["T:2:hydrocortisone"]
    assert (hc.records, hc.persons) == (60, 40) and hc.warnings == []  # creams counted too
    plt = rows["E11:platelet count"]  # read from the Observation table: nothing there
    assert plt.records == 0 and any("Measurement concept" in w for w in plt.warnings)
    assert any("inclusion criterion would exclude everyone" in w
               for w in rows["I2:septic shock"].warnings)  # fmt: skip
    assert any("exclusion criterion removes no one" in w for w in rows["E6:old"].warnings)
    assert rows["E5:platelets"].units == ["thousand per microliter: 50 records, median 174.5"]
    assert any("units are not converted" in w for w in rows["E5:platelets"].warnings)
    assert "not in the vocabulary" in rows["C:x"].warnings[0]
    found = search(con, "hydrocortisone", "Drug")
    assert [f[0] for f in found] == [2, 3] and found[0][3] == 35
    assert search(con, "platelet", "Measurement")[1][3] == "<11"  # small cells suppressed
    md = to_markdown(list(rows.values()), {"'platelet'": search(con, "platelet", "Measurement")})
    assert "| `T:2:hydrocortisone` | approve |" in md and "<11" in md


def test_concept_counts_cli(tmp_path):
    _cdm(tmp_path / "cdm.duckdb").close()
    run = tmp_path / "run"
    run.mkdir()
    (run / "decisions.json").write_text(json.dumps({"decisions": {
        "T:2:hydrocortisone": {"action": "approve", "concept_ids": [1]}}}), encoding="utf-8")
    r = CliRunner().invoke(app, ["concepts", "counts", str(run / "decisions.json"), "--db",
                                 str(tmp_path / "cdm.duckdb"), "--search", "hydrocortisone:Drug",
                                 "--out", str(run / "counts.md")])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "hydrocortisone 1% Topical Cream" in (run / "counts.md").read_text(encoding="utf-8")


def test_criterion_edits_fix_what_would_not_compile():
    """APROCCHSS-like defects: a duration drafted as a value on a diagnosis, a dose on a drug,
    a threshold in another unit. The summary warns while the design can still change."""
    from agentic_tte.graph.answers import compile_problems, edit_criteria

    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    crit = [c for c in p["eligibility"] if c["atoms"] and c.get("category") != "administrative"]
    crit = crit[:2]
    a = crit[0]["atoms"][0]
    a["concepts"] = [{**a["concepts"][0], "domain": "Condition"}]
    a["value"] = {"comparator": "<", "value": 24.0, "unit": "hour", "text": "for < 24 h"}
    a["temporal"] = None
    crit[0]["atoms"] = [a, {"concepts": [{"name": "dopamine", "domain": "Drug"}],
                            "value": {"comparator": ">=", "value": 15.0, "unit": "ug/kg/min"}}]
    crit[1]["atoms"] = crit[1]["atoms"][:1]
    crit[1]["atoms"][0]["value"] = None
    crit[1]["atoms"][0]["temporal"] = None
    first = crit[0]["id"]
    assert [(k, c) for k, c, _ in compile_problems(crit)] == [
        (crit[0]["kind"], f"{first}.1"), (crit[0]["kind"], f"{first}.2")]
    edited, notes = edit_criteria(crit, [
        ("no-value", f"{first}.1"), ("window", f"{first}.1=within 24 hour before"),
        ("drop-atom", f"{first}.2"), ("atom-logic", f"{first}=any")])
    e = edited[0]
    assert len(e["atoms"]) == 1 and e["atoms"][0]["value"] is None
    assert e["atoms"][0]["temporal"]["relation"] == "x_within_t_before_y"
    assert e["atoms"][0]["temporal"]["window"] == {"amount": 24.0, "unit": "hour"}
    assert e["atom_logic"] == "any" and "edited at design review" in e["notes"]
    assert compile_problems(edited) == [] and len(crit[0]["atoms"]) == 2  # input unchanged
    assert len(notes) == 1
    v, _ = edit_criteria(crit, [("value", f"{first}.1=< 30 K/uL")])
    assert v[0]["atoms"][0]["value"]["value"] == 30.0
    assert v[0]["atoms"][0]["value"]["unit"] == "K/uL"
    for bad, match in ((("window", f"{first}.1=within 2 day after"), "look-ahead"),
                       (("drop-atom", first), "give"), (("value", f"{first}.9=< 1"), "atom"),
                       (("window", "E99=before"), "no criterion")):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            edit_criteria(crit, [bad])
    _, summary = design_answer({**p, "eligibility": crit}, TEMPLATE, "Dr Test")
    assert any(s.startswith("WARNING") and "will not compile" in s for s in summary)


def _concept_run(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "pending.txt").write_text("concepts", encoding="utf-8")
    (out / "decisions.json").write_text(json.dumps({"reviewer": "", "decisions": {
        "I2:septic shock": {"action": "approve", "concept_ids": [20], "note": ""},
        "E11:platelet count": {"action": "approve", "concept_ids": [11], "note": ""},
        "C:SOFA score": {"action": "approve", "concept_ids": [5], "note": ""},
        "C:source of infection": {"action": "surrogate", "concept_ids": [7],
                                  "note": "Clinician proposes ..."}}}), encoding="utf-8")
    (out / "spec.json").write_text(json.dumps({
        "time_zero": {"domain": "Visit", "concept_ids": [], "occurrence": "first", "note": ""},
        "anchors": {"transfusion": {"domain": "Visit", "concept_ids": [], "occurrence": "last",
                                    "note": ""}},
        "measurement_lookback": {"amount": 24, "unit": "hour"}, "strict": False}),
        encoding="utf-8")  # fmt: skip
    return out


def test_concepts_decide(tmp_path):
    from agentic_tte.graph.answers import concepts_answer

    out = _concept_run(tmp_path)
    dec = json.loads((out / "decisions.json").read_text(encoding="utf-8"))
    spec = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    d, sp, summary = concepts_answer(
        dec, spec, "Dr Test", actions=[("replace", "E11:platelet=3024929"),
                                       ("unmappable", "C:SOFA"), ("reject", "C:source")],
        time_zero="Drug=1321341,1343916", t0_note="first vasopressor")  # fmt: skip
    e = d["decisions"]
    assert e["E11:platelet count"] == {"action": "replace", "concept_ids": [3024929], "note": ""}
    assert e["C:SOFA score"]["action"] == "unmappable" and e["C:SOFA score"]["concept_ids"] == []
    assert e["C:source of infection"]["note"] == ""  # the surrogate note went with it
    assert e["I2:septic shock"]["action"] == "approve" and d["reviewer"] == "Dr Test"
    assert sp["time_zero"]["concept_ids"] == [1321341, 1343916]
    assert sp["time_zero"]["domain"] == "Drug" and sp["anchors"] == {}  # empty anchor removed
    assert any("anchors left empty" in s for s in summary)
    with pytest.raises(ValueError, match="matches 2 keys"):
        concepts_answer(dec, spec, "Dr", actions=[("reject", "C:S")], time_zero="Drug=1")
    with pytest.raises(ValueError, match="time zero has no concept IDs"):
        concepts_answer(dec, spec, "Dr")
    with pytest.raises(ValueError, match="needs concept IDs"):
        concepts_answer(dec, spec, "Dr", actions=[("replace", "E11")], time_zero="Drug=1")
    with pytest.raises(ValueError, match="needs a note"):
        concepts_answer(dec, spec, "Dr", actions=[("surrogate", "C:SOFA=9")], time_zero="Drug=1")

    args = ["concepts", "decide", "-o", str(out), "--reviewer", "Dr Test", "--replace",
            "E11:platelet=3024929", "--time-zero", "Drug=1321341"]  # fmt: skip
    r = CliRunner().invoke(app, [*args, "--dry-run"])
    assert r.exit_code == 0, r.output
    assert json.loads((out / "decisions.json").read_text(encoding="utf-8"))["reviewer"] == ""
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert json.loads((out / "spec.json").read_text(encoding="utf-8"))["time_zero"][
        "concept_ids"] == [1321341]
    assert CliRunner().invoke(app, ["concepts", "decide", "-o", str(out), "--reviewer", "Dr",
                                    "--reject", "nope"]).exit_code == 2  # fmt: skip
