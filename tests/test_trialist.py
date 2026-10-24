import json
import os

import pytest

from agentic_tte.agents import Trialist
from agentic_tte.agents.trialist import CriterionParse, DesignProposal, TreatmentParse
from agentic_tte.llm import OllamaClient, ScriptedClient
from agentic_tte.llm.base import inline_refs


def _respond(messages, schema):
    """Plays the LLM: picks a canned answer by which schema is requested and by the original
    request (not a retry's correction message)."""
    last = next(m.content for m in reversed(messages)
                if m.role == "user" and not m.content.startswith("Your previous reply"))
    if schema == inline_refs(CriterionParse.model_json_schema()):
        if "allergy" in last:
            concepts = [
                {"name": f"allergy to {d}", "domain": "Observation", "source_text": d}
                for d in ("vitamin C", "hydrocortisone", "thiamine")
            ]
            return json.dumps({"concepts": concepts, "concept_logic": "any", "value": None,
                               "temporal": None, "negated": False, "complexity": "complex"})
        if "Age 18" in last:
            return json.dumps({
                "concepts": [{"name": "age", "domain": "Demographic", "source_text": "Age"}],
                "concept_logic": "all",
                "value": {"comparator": ">=", "value": 18, "upper": None, "unit": "years",
                          "text": "18 years old or older"},
                "temporal": None, "negated": False, "complexity": "simple"})
        if "Onset of septic shock" in last:
            return json.dumps({
                "concepts": [{"name": "septic shock", "domain": "Condition",
                              "source_text": "septic shock"}],
                "concept_logic": "all", "value": None,
                "temporal": {"relation": "x_within_t_before_y", "window_amount": None,
                             "window_unit": None, "anchor": "time_zero",
                             "text": "within 12 hours"},
                "negated": False, "complexity": "complex"})
        if "Pregnant" in last:
            return "this is not json"  # forces retry, then failure
        return json.dumps({
            "concepts": [{"name": "invented concept", "domain": "Condition",
                          "source_text": "something not in the text"}],
            "concept_logic": "any", "value": None, "temporal": None, "negated": False,
            "complexity": "simple"})
    if schema == inline_refs(TreatmentParse.model_json_schema()):
        if "Placebo" in last:
            return json.dumps({"components": [], "is_comparator": False})  # registry wins
        return json.dumps({"components": [
            {"name": "hydrocortisone", "domain": "Drug", "dose": "200 mg/day", "route": "IV",
             "frequency": "continuous infusion", "duration_amount": 5, "duration_unit": "day"}],
            "is_comparator": False})
    if schema == inline_refs(DesignProposal.model_json_schema()):
        return json.dumps({"time_zero_anchor": "septic shock onset",
                           "time_zero_offset_amount": None, "time_zero_offset_unit": None,
                           "grace_period_amount": 12, "grace_period_unit": "hour",
                           "rationale": "Inclusion requires onset of septic shock within 12 h."})
    raise AssertionError("unexpected schema")


def test_draft_from_registry_record(nct03872011_study):
    events = []
    llm = ScriptedClient(_respond, max_retries=1)
    p = Trialist(llm, on_event=lambda k, d: events.append(k)).draft_from_study(nct03872011_study)

    assert p.protocol_id == "NCT03872011-replay"
    assert p.provenance.prompt_version == "trialist-v1"
    assert [c.id for c in p.inclusion] == ["I1", "I2"]
    assert len(p.exclusion) == 9

    age = p.inclusion[0]
    assert age.value.comparator == ">=" and age.value.value == 18
    onset = p.inclusion[1]
    assert onset.temporal.window.amount == 12  # filled deterministically from the text
    assert onset.temporal.window.unit == "hour"

    allergy = next(c for c in p.exclusion if "allergy" in c.text)
    assert [c.name for c in allergy.concepts] == [
        "allergy to vitamin C", "allergy to hydrocortisone", "allergy to thiamine"]
    assert allergy.concept_logic == "any"

    pregnant = next(c for c in p.exclusion if c.text == "Pregnant")
    assert pregnant.notes.startswith("parse_failed")  # kept, flagged, not dropped
    hallucinated = next(c for c in p.exclusion if c.text == "High-dose steroid therapy")
    assert "not found verbatim" in hallucinated.notes

    experimental, placebo = p.treatment_strategies
    assert not experimental.is_comparator and placebo.is_comparator
    assert experimental.components[0].duration.amount == 5
    assert experimental.grace_period.amount == 12 and placebo.grace_period is None

    assert p.time_zero.anchor_event == "septic shock onset"
    assert p.follow_up.horizon.amount == 90
    assert all(p.completeness().values())
    assert "parse_failed" in events and "drafted" in events


def test_grounding_check():
    from agentic_tte.agents.trialist import _grounded

    text = "Known allergy to vitamin C, hydrocortisone, or thiamine"
    assert _grounded("Known allergy to thiamine", text)  # distributed phrase is fine
    assert not _grounded("Known allergy to hydrocortoid", text)  # real qwen3:8b copy error
    assert not _grounded("penicillin", text)


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
