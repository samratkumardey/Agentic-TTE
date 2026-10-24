"""Regression tests from the review of qwen3:32b's trialist-v1 draft of NCT04134403 (STASIS).

The scripted model below reproduces, in the v2 schema, the mistakes qwen3:32b made on
Hellbender (2026-10-24). Each test checks that the code-side fix (C1-C14 in the review) catches
them. Fixtures: the exact CT.gov record used on Hellbender, and the v1 protocol it produced.
"""

import json
from pathlib import Path

import pytest

from agentic_tte.agents import Trialist
from agentic_tte.agents.trialist import CriterionParse, DesignProposal, TreatmentParse
from agentic_tte.llm import ScriptedClient
from agentic_tte.llm.base import inline_refs
from agentic_tte.protocol import TargetTrialProtocol
from agentic_tte.sources.ctgov import extract_sections

FIXTURES = Path(__file__).parent / "fixtures"
STUDY = json.loads((FIXTURES / "ctgov" / "NCT04134403.json").read_text(encoding="utf-8"))
V1_OUTPUT = FIXTURES / "protocols" / "NCT04134403.trialist-v1.qwen3-32b.json"


def concept(name, domain, src=None):
    return {"name": name, "domain": domain, "source_text": src or name}


def atom(concepts, value=None, temporal=None, negated=False, logic="all"):
    return {"concepts": concepts, "concept_logic": logic, "value": value, "temporal": temporal,
            "negated": negated}  # fmt: skip


def value(comparator, v, unit, text):
    return {"comparator": comparator, "value": v, "upper": None, "unit": unit, "text": text}


def temporal(relation, anchor, text, amount=None, unit=None):
    return {"relation": relation, "window_amount": amount, "window_unit": unit, "anchor": anchor,
            "text": text}  # fmt: skip


INVENTED = temporal("x_at_time_of_y", "time_zero", "at time of enrollment")

# Keyed by a phrase of each criterion; mirrors qwen3:32b's v1 behaviour.
CRITERIA = {
    "Septic shock admitted": [
        atom([concept("septic shock", "Condition"),
              concept("ICU admission", "Visit", "admitted to the ICU")],
             temporal=temporal("x_within_t_after_y", "septic shock onset",
                               "admitted to the ICU within 24 hours", 24, "hour")),
        atom([concept("vasopressor", "Drug", "Vasopressor"), concept("MAP", "Measurement")],
             value=value(">=", 65, None, "MAP ≥ 65")),
        atom([concept("serum lactate level", "Measurement", "Serum lactate level")],
             value=value(">", 2.0, "mmol/L", ">2.0 mmol/L"), temporal=INVENTED),
    ],
    "Age ≥ 18": [atom([concept("age", "Demographic", "Age")],
                      value=value(">=", 18, "year", "≥ 18 years"))],
    "Non-pregnant": [atom([concept("pregnancy", "Condition", "Non-pregnant")], temporal=INVENTED,
                          negated=True)],
    "Ability to consent": [atom([concept("ability to consent", "Observation")], logic="any")],
    "Age < 18": [atom([concept("age", "Demographic", "Age")],
                      value=value("<", 18, "year", "< 18 years"))],
    "Pregnant defined": [atom([concept("pregnancy", "Condition", "Pregnant")]),
                         atom([concept("serum HCG", "Measurement", "serum HCG")],
                              value=value("=", None, None, "negative"))],
    "limitation of care": [atom([concept("limitation of care", "Observation"),
                                 concept("DNR", "Observation")], temporal=INVENTED, logic="any")],
    "G6PD": [atom([concept("G6PD deficiency", "Condition")], temporal=INVENTED)],
    "primary admission diagnosis": [atom(
        [concept(n, "Condition") for n in ("Acute stroke", "Acute coronary syndrome",
                                           "Active gastrointestinal bleed", "Burn", "Trauma")],
        logic="any")],
    "Prisoners": [atom([concept("prisoner status", "Observation", "Prisoners")])],
    "episode of sepsis": [atom([concept("sepsis", "Condition", "episode of sepsis")],
                               value=value(">", 1, None, ">1"),
                               temporal=temporal("x_within_t_before_y", "hospital admission",
                                                 "in hospital admission", 1, "day"))],
    "Vasopressor use": [atom([concept("vasopressor", "Drug", "Vasopressor use")],
                             value=value(">", 24, "hour", "for more than 24 hours"),
                             temporal=temporal("x_before_y", "randomization",
                                               "prior to randomization"))],
}  # fmt: skip


def _respond(messages, schema):
    request = next(m.content for m in reversed(messages)
                   if m.role == "user" and not m.content.startswith("Your previous reply"))
    if schema == inline_refs(CriterionParse.model_json_schema()):
        crit = request.split("Criterion: ", 1)[1]
        atoms = next(v for k, v in CRITERIA.items() if k in crit)
        # qwen labelled everything "clinical"; the keyword rule must catch consent/prisoners.
        return json.dumps({"atoms": atoms, "atom_logic": "all", "category": "clinical",
                           "complexity": "complex"})
    if schema == inline_refs(TreatmentParse.model_json_schema()):
        if "Usual care" in request:
            return json.dumps({"components": [], "is_comparator": True})
        comps = [("hydrocortisone sodium succinate", "50 mg", "every 6 hours"),
                 ("thiamine", "200 mg", "every 12 hours"),
                 ("ascorbic acid", "1,500 mg", "every 6 hours")]
        return json.dumps({"components": [
            {"name": n, "domain": "Drug", "dose": d, "route": "null", "frequency": f,
             "duration_amount": None, "duration_unit": None} for n, d, f in comps],
            "is_comparator": False})
    if schema == inline_refs(DesignProposal.model_json_schema()):
        return json.dumps({
            "time_zero_anchor": "septic shock onset", "time_zero_offset_amount": 0,
            "time_zero_offset_unit": "hour",
            "time_zero_evidence": "Septic shock admitted to the ICU within 24 hours",
            "grace_period_amount": 24, "grace_period_unit": "hour",
            "grace_period_evidence": "The grace period of 24 hours reflects the protocol's window",
            "rationale": "Inclusion requires ICU admission within 24 hours of shock onset."})
    raise AssertionError("unexpected schema")  # fmt: skip


@pytest.fixture(scope="module")
def drafted():
    events = []
    p = Trialist(ScriptedClient(_respond), on_event=lambda k, d: events.append((k, d))) \
        .draft_from_study(STUDY)  # fmt: skip
    return p, events


@pytest.fixture(scope="module")
def protocol(drafted):
    return drafted[0]


def crit(p, phrase):
    return next(c for c in p.eligibility if phrase in c.text)


# --- extractor (C9, C11) -------------------------------------------------------------------


def test_definition_merged_and_conjunction_kept():
    s = extract_sections(STUDY)
    assert len(s.inclusion) == 4 and len(s.exclusion) == 8
    i1 = s.inclusion[0]
    assert i1.startswith("Septic shock admitted to the ICU within 24 hours")
    assert "[definition: SEPSIS-3 defines septic shock" in i1
    assert "MAP ≥ 65 AND Serum lactate" in i1 and "AND;" not in i1
    # A criterion with its own definition stays separate.
    assert s.exclusion[1] == "Pregnant defined by negative serum HCG in all females"
    assert "inclusion: 1 definition item(s) merged into the criterion they define" in s.warnings


def test_registry_design_fields_and_no_results_warning():
    s = extract_sections(STUDY)
    src = s.source
    assert (src.overall_status, src.enrollment, src.enrollment_type) == ("UNKNOWN", 80, "ESTIMATED")
    assert (src.allocation, src.intervention_model, src.masking) == (
        "RANDOMIZED", "PARALLEL", "QUADRUPLE")  # fmt: skip
    assert src.has_results is False
    assert any("no results posted" in w for w in s.warnings)
    assert "intention to treat basis" in s.detailed_description


# --- criteria (C1, C5, C6, C12, C13) -----------------------------------------------------------


def test_i1_has_three_atoms_and_invented_time_dropped(protocol):
    i1 = crit(protocol, "Septic shock admitted")
    assert len(i1.atoms) == 3 and i1.atom_logic == "all"
    icu, vaso, lactate = i1.atoms
    assert icu.temporal.window.amount == 24 and icu.temporal.window.unit == "hour"
    assert vaso.value.comparator == ">=" and vaso.value.value == 65  # MAP no longer lost
    assert lactate.value.value == 2.0 and lactate.temporal is None  # "at time of enrollment"
    assert "dropped time constraint not in criterion text: 'at time of enrollment'" in i1.notes


@pytest.mark.parametrize("phrase", ["Non-pregnant", "limitation of care", "G6PD"])
def test_invented_enrollment_times_dropped(protocol, phrase):
    c = crit(protocol, phrase)
    assert all(a.temporal is None for a in c.atoms)
    assert "dropped time constraint" in c.notes


def test_age_stays_a_value_not_a_duration(protocol):
    for phrase in ("Age ≥ 18", "Age < 18"):
        a = crit(protocol, phrase).atoms[0]
        assert a.value.value == 18 and a.temporal is None


def test_e2_negative_hcg_is_qualitative(protocol):
    hcg = crit(protocol, "Pregnant defined").atoms[1]
    assert hcg.value.comparator == "present" and hcg.value.text == "negative"
    assert type(hcg.value.comparator) is str  # stored like every other enum field


@pytest.mark.parametrize("phrase", ["Ability to consent", "Prisoners"])
def test_administrative_criteria_flagged(protocol, phrase):
    c = crit(protocol, phrase)
    assert c.category == "administrative" and not c.emulable
    assert "keyword rule" in c.notes


def test_e7_invented_window_dropped(protocol):
    a = crit(protocol, "episode of sepsis").atoms[0]
    assert a.temporal.window is None
    assert a.temporal.relation == "x_at_time_of_y"  # "in hospital admission" = during it
    assert "dropped window 1 day not in criterion text" in crit(protocol, "episode of sepsis").notes


def test_e8_cumulative_duration_before_time_zero(protocol):
    a = crit(protocol, "Vasopressor use").atoms[0]
    assert a.temporal.relation == "cumulative_duration"
    assert (a.temporal.window.amount, a.temporal.window.unit) == (24, "hour")
    assert a.temporal.anchor == "time_zero"  # "randomization" normalized (C6)
    assert a.value.comparator == ">"


# --- treatments (C7, C8, C14) ------------------------------------------------------------------


def test_treatment_durations_end_events_and_null_route(protocol):
    arm1, arm2 = protocol.treatment_strategies
    got = {c.concept.name: (c.duration.amount, c.duration.unit, c.end_event, c.end_rule, c.route)
           for c in arm1.components}  # fmt: skip
    assert got == {
        "hydrocortisone sodium succinate": (7, "day", "ICU discharge", "first", None),
        "thiamine": (4, "day", "ICU discharge", "first", None),
        "ascorbic acid": (4, "day", "ICU discharge", "first", None),
    }
    assert arm2.is_comparator and arm2.components == []


def test_ungrounded_grace_period_dropped(drafted):
    protocol, events = drafted
    assert all(s.grace_period is None for s in protocol.treatment_strategies)
    assert ("dropped", {"element": "grace_period", "evidence":
            "The grace period of 24 hours reflects the protocol's window"}) in events


def test_time_zero_is_agent_proposed_with_grounded_evidence(protocol):
    tz = protocol.time_zero
    assert tz.origin == "agent"
    assert tz.evidence == "Septic shock admitted to the ICU within 24 hours"


# --- follow-up, outcomes, contrast, origins (C2, C3, C4, C10, C11) ---------------------------


def test_follow_up_ends_at_discharge_and_never_censors_death(protocol):
    fu = protocol.follow_up
    assert (fu.horizon.amount, fu.horizon.unit) == (28, "day")
    assert (fu.end_event, fu.end_rule) == ("hospital discharge", "first")
    assert "death_from_other_causes" not in fu.censoring_events
    assert fu.competing_events == []  # all-cause mortality: death is the outcome
    assert fu.origin == "derived"
    primary = next(o for o in protocol.outcomes if o.primary)
    assert (primary.end_event, primary.end_rule) == ("hospital discharge", "first")


def test_causal_contrast_read_from_registry(protocol):
    assert protocol.causal_contrast == "intention_to_treat"
    assert protocol.causal_contrast_origin == "registry"
    assert "intention to treat basis" in protocol.causal_contrast_evidence


def test_element_status_separates_defaults(protocol):
    assert protocol.element_status() == {
        "eligibility": "registry",
        "treatment_strategies": "registry",
        "assignment": "default",
        "time_zero": "agent",
        "follow_up": "derived",
        "outcomes": "registry",
        "causal_contrast": "registry",
        "analysis_plan": "default",
    }
    done = protocol.completeness()
    assert not done["assignment"] and not done["analysis_plan"]  # v1 reported 8/8
    assert protocol.provenance.prompt_version == "trialist-v2"


# --- backward compatibility ----------------------------------------------------------------


def test_v1_output_still_loads():
    p = TargetTrialProtocol.model_validate_json(V1_OUTPUT.read_text(encoding="utf-8"))
    assert p.schema_version == 1 and p.provenance.prompt_version == "trialist-v1"
    i2 = next(c for c in p.eligibility if c.id == "I2")
    assert len(i2.atoms) == 1 and i2.atoms[0].value.value == 2.0
    e2 = next(c for c in p.eligibility if c.id == "E2")
    assert e2.atoms[0].value.comparator == "present"  # C12 applied on load
    # v1 had no origins: defaults are reported as such, not as complete.
    assert p.element_status()["assignment"] == "default"
    assert p.element_status()["causal_contrast"] == "default"
