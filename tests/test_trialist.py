"""General Trialist behaviour on NCT03872011 (failure handling, grounding, registry rules).
The detailed trialist-v2 regressions live in test_nct04134403_regression.py."""

import json
import os

import pytest

from agentic_tte.agents import Trialist
from agentic_tte.agents.trialist import CriterionParse, DesignProposal, TreatmentParse
from agentic_tte.llm import OllamaClient, ScriptedClient
from agentic_tte.llm.base import inline_refs


def _atom(concepts, value=None, temporal=None, logic="all"):
    return {"concepts": concepts, "concept_logic": logic, "value": value, "temporal": temporal,
            "negated": False}  # fmt: skip


def _parse(atoms, complexity="complex"):
    return json.dumps({"atoms": atoms, "atom_logic": "all", "category": "clinical",
                       "complexity": complexity})  # fmt: skip


def _respond(messages, schema):
    """Plays the LLM: picks a canned answer by which schema is requested and by the original
    request (not a retry's correction message)."""
    last = next(m.content for m in reversed(messages)
                if m.role == "user" and not m.content.startswith("Your previous reply"))
    if schema == inline_refs(CriterionParse.model_json_schema()):
        if "allergy" in last:
            return _parse([_atom([{"name": f"allergy to {d}", "domain": "Observation",
                                   "source_text": d}
                                  for d in ("vitamin C", "hydrocortisone", "thiamine")],
                                 logic="any")])  # fmt: skip
        if "Age 18" in last:
            return _parse([_atom([{"name": "age", "domain": "Demographic", "source_text": "Age"}],
                                 value={"comparator": ">=", "value": 18, "upper": None,
                                        "unit": "years", "text": "18 years old or older"})],
                          "simple")  # fmt: skip
        if "Onset of septic shock" in last:
            return _parse([_atom([{"name": "septic shock", "domain": "Condition",
                                   "source_text": "septic shock"}],
                                 temporal={"relation": "x_within_t_before_y", "window_amount": None,
                                           "window_unit": None, "anchor": "randomization",
                                           "text": "within 12 hours"})])  # fmt: skip
        if "Pregnant" in last:
            return "this is not json"  # forces retry, then failure
        return _parse([_atom([{"name": "invented concept", "domain": "Condition",
                               "source_text": "something not in the text"}])], "simple")
    if schema == inline_refs(TreatmentParse.model_json_schema()):
        if "Placebo" in last:
            return json.dumps({"components": [], "is_comparator": False})  # registry wins
        return json.dumps({"components": [
            {"name": "hydrocortisone", "domain": "Drug", "dose": "200 mg/day", "route": "IV",
             "frequency": "continuous infusion", "duration_amount": 5, "duration_unit": "day"}],
            "is_comparator": False})  # fmt: skip
    if schema == inline_refs(DesignProposal.model_json_schema()):
        return json.dumps({"time_zero_anchor": "septic shock onset",
                           "time_zero_offset_amount": None, "time_zero_offset_unit": None,
                           "time_zero_evidence": "Onset of septic shock within 12 hours",
                           "grace_period_amount": 12, "grace_period_unit": "hour",
                           "grace_period_evidence": "Onset of septic shock within 12 hours",
                           "rationale": "Inclusion requires onset of septic shock within 12 h."})
    raise AssertionError("unexpected schema")


def test_draft_from_registry_record(nct03872011_study):
    events = []
    llm = ScriptedClient(_respond, max_retries=1)
    p = Trialist(llm, on_event=lambda k, d: events.append(k)).draft_from_study(nct03872011_study)

    assert p.protocol_id == "NCT03872011-replay"
    assert p.provenance.prompt_version == "trialist-v2"
    assert [c.id for c in p.inclusion] == ["I1", "I2"]
    assert len(p.exclusion) == 9

    age = p.inclusion[0].atoms[0]
    assert age.value.comparator == ">=" and age.value.value == 18
    onset = p.inclusion[1].atoms[0].temporal
    assert (onset.window.amount, onset.window.unit) == (12, "hour")  # filled from the text
    assert onset.anchor == "time_zero"

    allergy = next(c for c in p.exclusion if "allergy" in c.text)
    assert [c.name for c in allergy.concepts] == [
        "allergy to vitamin C", "allergy to hydrocortisone", "allergy to thiamine"]
    assert allergy.atoms[0].concept_logic == "any"

    pregnant = next(c for c in p.exclusion if c.text == "Pregnant")
    assert pregnant.notes.startswith("parse_failed")  # kept, flagged, not dropped
    hallucinated = next(c for c in p.exclusion if c.text == "High-dose steroid therapy")
    assert "not found verbatim" in hallucinated.notes
    refusal = next(c for c in p.exclusion if c.text.startswith("Refusal"))
    assert refusal.category == "administrative"

    experimental, placebo = p.treatment_strategies
    assert not experimental.is_comparator and placebo.is_comparator
    assert experimental.components[0].duration.amount == 5
    # The grace period cites real protocol text here, so it is kept, as an agent proposal.
    assert experimental.grace_period.amount == 12 and placebo.grace_period is None
    assert experimental.grace_period_origin == "agent"

    assert p.time_zero.anchor_event == "septic shock onset"
    assert p.follow_up.horizon.amount == 90
    assert p.causal_contrast_origin == "default"  # NCT03872011 does not state ITT
    assert "parse_failed" in events and "drafted" in events


def test_timeout_on_one_criterion_does_not_lose_the_draft(nct03872011_study):
    """Exit run: one gpt-oss timeout discarded a whole trial. Now it is one failed item."""
    import httpx

    def respond(messages, schema):
        if "Criterion: Known hemachromatosis" in messages[-1].content:
            raise httpx.ReadTimeout("timed out")
        return _respond(messages, schema)

    events = []
    p = Trialist(ScriptedClient(respond, max_retries=1),
                 on_event=lambda k, d: events.append((k, d))).draft_from_study(nct03872011_study)
    hemo = next(c for c in p.eligibility if "hemachromatosis" in c.text)
    assert hemo.notes.startswith("parse_failed") and "ReadTimeout" in hemo.notes
    assert len(p.eligibility) == 11 and p.time_zero is not None


def test_first_requests_match_what_the_draft_sends(nct03872011_study):
    from agentic_tte.agents.trialist import first_requests
    from agentic_tte.sources.ctgov import extract_sections

    llm = ScriptedClient(_respond, max_retries=0)
    Trialist(llm).draft_from_study(nct03872011_study)
    expected = first_requests(extract_sections(nct03872011_study))
    first_attempts = [msgs for msgs, _ in llm.requests if msgs[-1].role == "user"
                      and not msgs[-1].content.startswith("Your previous reply")]  # fmt: skip
    assert [[m.model_dump() for m in r] for r in first_attempts] == \
        [[m.model_dump() for m in r] for r in expected]  # fmt: skip


def test_grounding_check():
    from agentic_tte.agents.trialist import _grounded

    text = "Known allergy to vitamin C, hydrocortisone, or thiamine"
    assert _grounded("Known allergy to thiamine", text)  # distributed phrase is fine
    assert not _grounded("Known allergy to hydrocortoid", text)  # real qwen3:8b copy error
    assert not _grounded("penicillin", text)


def test_null_strings_become_none():
    from agentic_tte.agents.trialist import ParsedComponent

    c = ParsedComponent(name="x", domain="Drug", dose="N/A", route="null", frequency="None",
                        duration_amount=None, duration_unit=None)  # fmt: skip
    assert (c.dose, c.route, c.frequency) == (None, None, None)


@pytest.mark.live
def test_live_ollama_draft(nct03872011_study, tmp_path):
    """Real model, offline registry record: ATTE_OLLAMA_HOST / ATTE_MODEL select the server."""
    llm = OllamaClient(
        os.environ.get("ATTE_MODEL", "qwen3:8b"),
        host=os.environ.get("ATTE_OLLAMA_HOST", "http://localhost:11434"),
    )
    if llm.model_digest() is None:
        pytest.skip(f"{llm.model} not available on {llm.host}")
    p = Trialist(llm).draft_from_study(nct03872011_study)
    assert len(p.eligibility) == 11
    parsed = [c for c in p.eligibility if c.concepts]
    assert len(parsed) >= 9
    assert p.time_zero is not None
    (tmp_path / "protocol.json").write_text(p.model_dump_json(indent=2))
