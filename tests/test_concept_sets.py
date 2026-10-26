from pathlib import Path

import duckdb
import pytest

from agentic_tte.omop import DataContext, run_gate
from agentic_tte.omop.concept_sets import (
    DecisionError,
    apply_decisions,
    apply_to_protocol,
    decisions_template,
    propose,
    review_sheet,
)
from agentic_tte.omop.vocab import load_athena, query_variants, search
from agentic_tte.protocol import (
    ConceptRef,
    Criterion,
    CriterionAtom,
    TargetTrialProtocol,
    TreatmentComponent,
    TreatmentStrategy,
    diff,
)
from agentic_tte.synth.dgp import Scenario, generate
from agentic_tte.synth.omop import to_omop, write_duckdb

VOCAB = Path(__file__).parent / "fixtures" / "vocab"


@pytest.fixture
def vocab():
    con = duckdb.connect()
    load_athena(con, VOCAB)
    return con


def ids(cands):
    return [c.concept_id for c in cands]


def test_load_athena_counts(vocab):
    n = vocab.execute("SELECT count(*) FROM concept").fetchone()[0]
    assert n == 19
    with pytest.raises(FileNotFoundError):
        load_athena(duckdb.connect(), VOCAB / "missing")


def test_exact_name_beats_partial_and_deprecated_is_excluded(vocab):
    c = search(vocab, "septic shock", "Condition")
    assert ids(c)[:2] == [196236, 9000001]
    assert c[0].match == "exact_name" and c[1].match == "all_words"
    assert 9000099 not in ids(c)  # deprecated duplicate


def test_synonym_and_maps_to(vocab):
    assert search(vocab, "Shock, septic", "Condition")[0].match == "exact_synonym"
    via = search(vocab, "Severe sepsis with septic shock", "Condition")[0]
    assert (via.concept_id, via.via) == (196236, 9000002)  # ICD-10-CM -> SNOMED standard


def test_abbreviations_salts_and_generic_words(vocab):
    assert ids(search(vocab, "G6PD deficiency", "Condition")) == [9000040]
    assert ids(search(vocab, "glucose-6 phosphate dehydrogenase (G-6PD) deficiency",
                      "Condition")) == [9000040]  # fmt: skip
    drug = search(vocab, "hydrocortisone sodium succinate", "Drug")
    assert drug[0].concept_id == 9000020 and drug[0].concept_class_id == "Ingredient"
    assert drug[0].matched_text == "hydrocortisone"  # the reviewer sees what matched
    assert ids(search(vocab, "serum lactate level", "Measurement")) == [9000031]
    assert "serum lactate" in query_variants("serum lactate level", "Measurement")


def test_domain_is_respected(vocab):
    assert search(vocab, "septic shock", "Drug") == []
    assert search(vocab, "age", "Demographic") == []


def _protocol():
    def crit(cid, kind, name, domain, category="clinical"):
        return Criterion(id=cid, kind=kind, text=f"{cid} text", category=category,
                         atoms=[CriterionAtom(concepts=[ConceptRef(name=name, domain=domain)])])

    return TargetTrialProtocol(
        protocol_id="P", eligibility=[
            crit("I1", "inclusion", "septic shock", "Condition"),
            crit("I2", "inclusion", "age", "Demographic"),
            crit("E1", "exclusion", "G6PD deficiency", "Condition"),
            crit("E2", "exclusion", "prisoner", "Observation", category="administrative"),
            crit("E3", "exclusion", "end-stage disease", "Condition"),
        ],
        treatment_strategies=[TreatmentStrategy(label="HC", components=[TreatmentComponent(
            concept=ConceptRef(name="hydrocortisone sodium succinate", domain="Drug"))]),
            TreatmentStrategy(label="Placebo", is_comparator=True)])  # fmt: skip


def test_propose_review_and_apply(vocab):
    p = _protocol()
    book = propose(vocab, p)
    st = {s.key: s.status for s in book.sets}
    assert st == {"I1:septic shock": "proposed", "I2:age": "not_needed",
                  "E1:G6PD deficiency": "proposed", "E2:prisoner": "not_needed",
                  "E3:end-stage disease": "no_match",
                  "T:HC:hydrocortisone sodium succinate": "proposed"}  # fmt: skip
    assert book.get("I1:septic shock").concept_ids == [196236]
    assert book.gate_sets() == {}  # nothing is used before review
    assert book.gate_sets(include_proposed=True)["I1:septic shock"] == [196236]
    assert not book.fully_reviewed

    sheet = review_sheet(book)
    assert "`I1:septic shock`" in sheet and "196236 Septic shock" in sheet
    tmpl = decisions_template(book)
    assert set(tmpl["decisions"]) == {"I1:septic shock", "E1:G6PD deficiency",
                                      "E3:end-stage disease",
                                      "T:HC:hydrocortisone sodium succinate"}  # fmt: skip

    decisions = {"reviewer": "Dr A", "decisions": {
        "I1:septic shock": {"action": "replace", "concept_ids": [196236, 9000001]},
        "E1:G6PD deficiency": {"action": "approve"},
        "E3:end-stage disease": {"action": "unmappable", "note": "no RWD definition"},
        "T:HC:hydrocortisone sodium succinate": {"action": "approve"}}}  # fmt: skip
    reviewed = apply_decisions(book, decisions)
    assert book.get("I1:septic shock").status == "proposed"  # input unchanged
    s = reviewed.get("I1:septic shock")
    assert (s.status, s.concept_ids, s.reviewer) == ("approved", [196236, 9000001], "Dr A")
    assert reviewed.fully_reviewed
    assert reviewed.gate_sets() == {"I1:septic shock": [196236, 9000001],
                                    "E1:G6PD deficiency": [9000040],
                                    "T:HC:hydrocortisone sodium succinate": [9000020]}  # fmt: skip

    v2 = apply_to_protocol(p, reviewed)
    assert (v2.version, v2.parent_version) == (2, 1)
    shock = v2.eligibility[0].atoms[0].concepts[0]
    assert (shock.omop_concept_id, shock.standard_name, shock.status) == (
        196236, "Septic shock", "mapped")  # fmt: skip
    assert v2.eligibility[4].atoms[0].concepts[0].status == "unmappable"
    assert v2.treatment_strategies[0].components[0].concept.omop_concept_id == 9000020
    assert p.eligibility[0].atoms[0].concepts[0].omop_concept_id is None  # parent untouched
    assert any(c.path.endswith("omop_concept_id") for c in diff(p, v2))
    assert "Dr A" in v2.provenance.notes


@pytest.mark.parametrize("decisions,match", [
    ({"decisions": {"I1:septic shock": {"action": "approve"}}}, "reviewer"),
    ({"reviewer": "x", "decisions": {"nope": {"action": "approve"}}}, "unknown key"),
    ({"reviewer": "x", "decisions": {"I1:septic shock": {"action": "replace"}}}, "concept_ids"),
    ({"reviewer": "x", "decisions": {"I1:septic shock": {"action": "surrogate",
                                                         "concept_ids": [9000010]}}}, "note"),
    ({"reviewer": "x", "decisions": {"I1:septic shock": {"action": "guess"}}}, "unknown action"),
])  # fmt: skip
def test_invalid_decisions(vocab, decisions, match):
    with pytest.raises(DecisionError, match=match):
        apply_decisions(propose(vocab, _protocol()), decisions)


def test_reviewed_sets_feed_the_gate():
    """Synthetic OMOP with the test vocabulary loaded: only approved sets count, and
    descendants are found through concept_ancestor."""
    con = write_duckdb(to_omop(generate(Scenario(n=200), with_truth=False)))
    load_athena(con, VOCAB)  # replaces the local concept table
    p = _protocol()
    book = propose(con, p)
    ctx = DataContext(data_classification="synthetic")
    before = run_gate(con, p, concept_sets=book.gate_sets(), context=ctx)
    rows = {r["concept"]: r["result"] for r in before.check("concepts").values["concepts"]}
    assert rows["septic shock"] == "unmapped"  # proposed but not reviewed
    reviewed = apply_decisions(book, {"reviewer": "Dr A", "decisions": {
        "I1:septic shock": {"action": "approve"}}})  # fmt: skip
    after = run_gate(con, p, concept_sets=reviewed.gate_sets(), context=ctx)
    rows = {r["concept"]: r["result"] for r in after.check("concepts").values["concepts"]}
    assert rows["septic shock"] == "200 patients"
    assert after.check("vocabulary").values["concept_ancestor"] is True


def test_confounders_and_outcome_concepts_are_reviewed_too(vocab):
    from agentic_tte.protocol import AssignmentProcedure, Outcome

    p = _protocol().model_copy(update={
        "assignment": AssignmentProcedure(confounders=[
            ConceptRef(name="serum lactate", domain="Measurement"),
            ConceptRef(name="age", domain="Demographic")]),
        "outcomes": [Outcome(name="shock", primary=True,
                             concept=ConceptRef(name="septic shock", domain="Condition")),
                     Outcome(name="28-day mortality")]})  # fmt: skip
    book = propose(vocab, p)
    st = {s.key: (s.scope, s.status) for s in book.sets}
    assert st["C:serum lactate"] == ("confounder", "proposed")
    assert st["C:age"] == ("confounder", "not_needed")
    assert st["O:shock"] == ("outcome", "proposed")
    assert not any(k.startswith("O:28") for k in st)  # no concept: death is not a concept set
    reviewed = apply_decisions(book, {"reviewer": "Dr A", "decisions": {
        "C:serum lactate": {"action": "approve"}, "O:shock": {"action": "approve"}}})
    v2 = apply_to_protocol(p, reviewed)
    assert v2.assignment.confounders[0].status == "mapped"
    assert v2.outcomes[0].concept.omop_concept_id == 196236
