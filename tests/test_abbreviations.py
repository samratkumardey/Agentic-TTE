"""Abbreviations in concept search: protocol-text definitions and a reviewer-approved list."""

import json
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from agentic_tte.cli import app
from agentic_tte.omop.abbreviations import (
    PROTOCOL_TEXT,
    Abbreviation,
    AbbreviationList,
    defined_in_text,
    from_protocol,
    merged,
)
from agentic_tte.omop.concept_sets import propose, review_sheet
from agentic_tte.omop.vocab import load_athena, search
from agentic_tte.protocol import (
    ConceptRef,
    Criterion,
    CriterionAtom,
    TargetTrialProtocol,
)

FIXTURES = Path(__file__).parent / "fixtures"
VOCAB = FIXTURES / "vocab"
ASCEND = FIXTURES / "protocols" / "NCT00475852.final-H100.qwen3-32b.json"


@pytest.fixture
def vocab():
    con = duckdb.connect()
    load_athena(con, VOCAB)
    return con


def approved(abbr, expansions, domain=None):
    return Abbreviation(abbr=abbr, expansions=expansions, domain=domain, approved_by="Dr A")


@pytest.mark.parametrize("text,pairs", [
    ("management of acute decompensated heart failure (ADHF) or",
     [("ADHF", "acute decompensated heart failure")]),
    ("Known glucose-6 phosphate dehydrogenase (G-6PD) deficiency",
     [("G-6PD", "glucose-6 phosphate dehydrogenase")]),
    ("MAP (mean arterial pressure) of 65 mmHg", [("MAP", "mean arterial pressure")]),
    ("Patients with limitation of care (i.e. DNR)", []),
    ("serum lactate (>2.0 mmol/L)", []),
    ("Acute Kidney Injury (AKI) defined as", [("AKI", "Acute Kidney Injury")]),
])  # fmt: skip
def test_definitions_in_text(text, pairs):
    assert defined_in_text(text) == pairs


def test_protocol_definitions_from_a_real_draft():
    p = TargetTrialProtocol.model_validate_json(ASCEND.read_text(encoding="utf-8"))
    defs = {a.abbr: a for a in from_protocol(p)}
    assert defs["ADHF"].expansions == ["acute decompensated heart failure"]
    assert defs["ADHF"].source == f"{PROTOCOL_TEXT}: I1" and defs["ADHF"].usable
    assert "ASCEND-HF" not in defs  # the title's acronym names the trial, not a concept


def test_search_expands_only_what_it_may(vocab):
    assert search(vocab, "diagnosis of ADHF", "Condition") == []
    c = search(vocab, "diagnosis of ADHF", "Condition", abbreviations=[
        Abbreviation(abbr="ADHF", expansions=["acute decompensated heart failure"],
                     approved_by=PROTOCOL_TEXT, source=f"{PROTOCOL_TEXT}: I1")])  # fmt: skip
    assert c[0].concept_id == 9000082 and "protocol text: I1" in c[0].abbreviation
    unapproved = Abbreviation(abbr="ADHF", expansions=["acute decompensated heart failure"])
    assert search(vocab, "ADHF", "Condition", abbreviations=[unapproved]) == []


def test_expansion_outranks_a_word_match_and_case_matters(vocab):
    lst = [approved("MAP", ["mean arterial pressure", "mean blood pressure"], "Measurement")]
    # A 3-letter abbreviation gets exact matches only, so the decoy "MAP kinase activity" is
    # not proposed for it; only an approved expansion finds the concept.
    assert search(vocab, "MAP", "Measurement") == []
    c = search(vocab, "MAP", "Measurement", abbreviations=lst)
    assert c[0].concept_id == 9000080 and c[0].match == "exact_name"  # Mean blood pressure
    assert c[0].abbreviation.startswith("MAP = mean blood pressure")
    assert 9000085 not in [x.concept_id for x in c]
    assert search(vocab, "map", "Measurement", abbreviations=lst) == []  # lowercase: no expansion


def test_domain_disambiguates(vocab):
    lst = [approved("IHD", ["intermittent hemodialysis"], "Procedure"),
           approved("IHD", ["ischemic heart disease"], "Condition")]
    assert search(vocab, "IHD", "Procedure", abbreviations=lst)[0].concept_id == 9000084
    assert search(vocab, "history of IHD", "Condition", abbreviations=lst)[0].concept_id \
        == 9000083  # fmt: skip


def test_approval_workflow(tmp_path):
    seed = AbbreviationList.seed()
    assert seed.entries and not seed.approved()  # shipped unapproved
    with pytest.raises(ValueError, match="reviewer"):
        seed.approve("  ")
    assert seed.approve("Dr A", only={"AKI", "map"}) == 2  # case-insensitive keys
    assert {a.abbr for a in seed.approved()} == {"AKI", "MAP"}
    assert seed.approve("Dr B", only={"AKI"}) == 0  # already approved: not overwritten
    seed.save(tmp_path / "a.json")
    back = AbbreviationList.load(tmp_path / "a.json")
    assert {a.abbr: a.approved_by for a in back.approved()} == {"AKI": "Dr A", "MAP": "Dr A"}


def test_protocol_definition_wins_over_the_list():
    p = TargetTrialProtocol(protocol_id="X", eligibility=[Criterion(
        id="I1", kind="inclusion", text="acute kidney injury (AKI)")])
    lst = AbbreviationList(entries=[approved("AKI", ["acute kidney insufficiency"]),
                                    approved("MAP", ["mean arterial pressure"])])
    got = {a.abbr: (a.expansions, a.source) for a in merged(p, lst)}
    assert got["AKI"] == (["acute kidney injury"], f"{PROTOCOL_TEXT}: I1")
    assert got["MAP"][1] == "list"


def test_propose_records_and_shows_abbreviations(vocab):
    p = TargetTrialProtocol(protocol_id="X", eligibility=[
        Criterion(id="I1", kind="inclusion", text="acute decompensated heart failure (ADHF)",
                  atoms=[CriterionAtom(concepts=[ConceptRef(name="diagnosis of ADHF",
                                                            domain="Condition")])]),
        Criterion(id="I2", kind="inclusion", text="MAP below 65",
                  atoms=[CriterionAtom(concepts=[ConceptRef(name="MAP", domain="Measurement")])]),
    ])  # fmt: skip
    plain = propose(vocab, p)
    assert plain.get("I1:diagnosis of ADHF").status == "proposed"  # protocol text defines it
    assert plain.get("I2:MAP").status == "no_match"  # no list given: the decoy is not proposed
    lst = AbbreviationList(entries=[approved("MAP", ["mean blood pressure"], "Measurement")])
    book = propose(vocab, p, abbreviations=lst)
    assert book.get("I2:MAP").concept_ids == [9000080]
    assert {a["abbr"] for a in book.abbreviations} == {"ADHF", "MAP"}
    sheet = review_sheet(book)
    assert "**abbreviation ADHF = acute decompensated heart failure (protocol text: I1)**" in sheet
    assert "- MAP = mean blood pressure (list; approved by Dr A)" in sheet


def test_cli_init_approve_and_propose(tmp_path):
    from agentic_tte.synth.dgp import Scenario, generate
    from agentic_tte.synth.omop import to_omop, write_duckdb

    run = CliRunner().invoke
    lst = tmp_path / "abbr.json"
    assert run(app, ["abbreviations", "init", str(lst)]).exit_code == 0
    assert run(app, ["abbreviations", "init", str(lst)]).exit_code == 2  # never overwritten
    r = run(app, ["abbreviations", "approve", str(lst), "--reviewer", "Dr A", "--only", "MAP"])
    assert r.exit_code == 0 and "1 entries approved" in r.output
    assert "| MAP | mean arterial pressure / mean blood pressure | Measurement | Dr A |" in run(
        app, ["abbreviations", "show", str(lst)]).output  # fmt: skip
    db = tmp_path / "s.duckdb"
    write_duckdb(to_omop(generate(Scenario(n=50), with_truth=False)), db).close()
    p = tmp_path / "p.json"
    p.write_text(TargetTrialProtocol(protocol_id="X", eligibility=[Criterion(
        id="I1", kind="inclusion", text="MAP < 65", atoms=[CriterionAtom(concepts=[ConceptRef(
            name="MAP", domain="Measurement")])])]).model_dump_json(), "utf-8")  # fmt: skip
    r = run(app, ["concepts", "propose", str(p), "--db", str(db), "--vocab", str(VOCAB),
                  "-o", str(tmp_path / "rev"), "--abbreviations", str(lst)])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "1 of 40 entries approved" in r.output
    book = json.loads((tmp_path / "rev" / "book.json").read_text(encoding="utf-8"))
    assert book["sets"][0]["concept_ids"] == [9000080]
