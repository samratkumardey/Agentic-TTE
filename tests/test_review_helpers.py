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
    assert not any("WARNING" in s for s in summary)
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
        (20, 'Septic shock', 'Condition', 'S'), (30, 'old code', 'Condition', NULL))
        t(concept_id, concept_name, domain_id, standard_concept)""")
    con.execute("CREATE TABLE concept_ancestor AS SELECT * FROM (VALUES (1, 1), (1, 2), (1, 3), "
                "(10, 10), (11, 11), (20, 20)) t(ancestor_concept_id, descendant_concept_id)")
    con.execute("CREATE TABLE drug_exposure AS SELECT range % 40 AS person_id, "
                "CASE WHEN range < 30 THEN 2 ELSE 3 END AS drug_concept_id FROM range(60)")
    con.execute("CREATE TABLE measurement AS SELECT range AS person_id, "
                "10 AS measurement_concept_id FROM range(50) UNION ALL SELECT 1, 11")
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
        "C:x": {"action": "surrogate", "concept_ids": [99999]},
        "C:y": {"action": "unmappable", "concept_ids": []}}}
    book = ConceptSetBook(protocol_id="p", protocol_version=1, sets=[
        ConceptSet(key="T:2:hydrocortisone", scope="treatment", criterion_text="t",
                   concept_name="hydrocortisone", domain="Drug"),
        ConceptSet(key="E11:platelet count", scope="eligibility", criterion_text="e",
                   concept_name="platelet count", domain="Observation")])
    rows = {c.key: c for c in concept_counts(con, decisions, book)}
    assert set(rows) == {"T:2:hydrocortisone", "E11:platelet count", "I2:septic shock", "C:x"}
    hc = rows["T:2:hydrocortisone"]
    assert (hc.records, hc.persons) == (60, 40) and hc.warnings == []  # creams counted too
    plt = rows["E11:platelet count"]  # read from the Observation table: nothing there
    assert plt.records == 0 and any("Measurement concept" in w for w in plt.warnings)
    assert any("NO RECORDS" in w for w in rows["I2:septic shock"].warnings)
    assert "not in the vocabulary" in rows["C:x"].warnings[0]
    found = search(con, "hydrocortisone", "Drug")
    assert [f[0] for f in found] == [2, 3] and found[0][3] == 30
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
