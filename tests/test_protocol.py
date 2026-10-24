import pytest
from pydantic import ValidationError

from agentic_tte.protocol import (
    Comparator,
    ConceptRef,
    Criterion,
    Domain,
    Duration,
    FollowUp,
    Outcome,
    TargetTrialProtocol,
    TimeZero,
    TreatmentStrategy,
    ValueConstraint,
    diff,
    new_version,
)


def _protocol() -> TargetTrialProtocol:
    return TargetTrialProtocol(
        protocol_id="NCT00000000-replay",
        eligibility=[
            Criterion(
                id="I1",
                kind="inclusion",
                text="Age 18 years or older",
                concepts=[ConceptRef(name="age", domain=Domain.DEMOGRAPHIC)],
                value=ValueConstraint(comparator=Comparator.GE, value=18, unit="years"),
            ),
            Criterion(id="E1", kind="exclusion", text="Pregnant"),
        ],
        treatment_strategies=[
            TreatmentStrategy(label="Hydrocortisone"),
            TreatmentStrategy(label="Placebo", is_comparator=True),
        ],
        time_zero=TimeZero(anchor_event="septic shock onset"),
        follow_up=FollowUp(horizon=Duration(amount=28, unit="day")),
        outcomes=[Outcome(name="28-day mortality", primary=True)],
    )


def test_completeness_does_not_count_defaults():
    p = _protocol()
    status = p.element_status()
    assert status["time_zero"] == "agent" and status["eligibility"] == "registry"
    assert status["assignment"] == "default" and status["causal_contrast"] == "default"
    done = p.completeness()
    assert done["eligibility"] and done["time_zero"] and done["outcomes"]
    assert not done["assignment"] and not done["analysis_plan"]
    assert [c.id for c in p.inclusion] == ["I1"]
    assert [c.id for c in p.exclusion] == ["E1"]


def test_v1_criterion_shape_is_upgraded_to_one_atom():
    c = _protocol().eligibility[0]
    assert len(c.atoms) == 1 and c.atoms[0].value.value == 18
    assert [x.name for x in c.concepts] == ["age"]
    assert _protocol().eligibility[1].atoms == []  # nothing to wrap


def test_completeness_flags_missing_time_zero():
    p = _protocol().model_copy(update={"time_zero": None})
    assert p.completeness()["time_zero"] is False


def test_content_hash_ignores_provenance_and_version():
    a = _protocol()
    b = new_version(a, created_by="tester")
    assert b.version == 2 and b.parent_version == 1
    assert a.content_hash() == b.content_hash()


def test_new_version_and_diff():
    a = _protocol()
    relaxed = [c.model_copy(deep=True) for c in a.eligibility]
    relaxed[0].atoms[0].value = ValueConstraint(comparator=Comparator.GE, value=16, unit="years")
    b = new_version(a, created_by="redesign", eligibility=relaxed, mode="redesign")
    assert a.eligibility[0].atoms[0].value.value == 18  # parent untouched
    changes = {c.path: (c.before, c.after) for c in diff(a, b)}
    assert changes["eligibility[id=I1].atoms[0].value.value"] == (18, 16)
    assert changes["mode"] == ("replay", "redesign")
    assert a.content_hash() != b.content_hash()


def test_diff_ignores_reordering():
    a = _protocol()
    b = a.model_copy(update={"eligibility": list(reversed(a.eligibility))})
    assert diff(a, b) == []


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        Criterion(id="I1", kind="inclusion", text="x", bogus=1)


def test_roundtrip_json():
    p = _protocol()
    assert TargetTrialProtocol.model_validate_json(p.model_dump_json()) == p
