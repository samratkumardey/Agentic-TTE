import json

import httpx
import pytest
import respx

from agentic_tte.sources.ctgov import (
    CTGovClient,
    CTGovError,
    extract_sections,
    normalize_nct_id,
    split_eligibility,
)
from tests.conftest import EMULATRX_NCT03872011


def test_normalize_nct_id():
    assert normalize_nct_id(" nct03872011 ") == "NCT03872011"
    with pytest.raises(ValueError):
        normalize_nct_id("NCT123")


def test_split_matches_upstream_llm_split(nct03872011_study):
    """Our deterministic splitter must reproduce the items EmulatRx got from GPT-4o."""
    info = json.loads((EMULATRX_NCT03872011 / "info.json").read_text(encoding="utf-8"))
    s = extract_sections(nct03872011_study)
    assert s.inclusion == info["Inclusion Criteria"]
    assert s.exclusion == info["Exclusion Criteria"]
    assert s.warnings == [
        "no results posted on ClinicalTrials.gov; a Replay comparison needs a publication"
    ]


def test_extract_sections_metadata(nct03872011_study):
    s = extract_sections(nct03872011_study)
    assert s.source.nct_id == "NCT03872011"
    assert s.source.phase == ["PHASE3"]
    assert s.source.enrollment == 408
    assert len(s.source.record_sha256) == 64
    assert [a.type for a in s.arms] == ["EXPERIMENTAL", "PLACEBO_COMPARATOR"]
    assert all(a.interventions for a in s.arms)
    primary = [o for o in s.outcomes if o.primary]
    assert [o.measure for o in primary] == ["90-day mortality"]
    assert len(s.outcomes) == 4
    # Age is already in the free text, so no duplicate structured age criterion.
    assert s.structured_criteria == []


def test_structured_age_added_when_text_is_silent(nct03872011_study):
    study = json.loads(json.dumps(nct03872011_study))
    elig = study["protocolSection"]["eligibilityModule"]
    elig["eligibilityCriteria"] = (
        "Inclusion Criteria:\n\n* Septic shock\n\nExclusion Criteria:\n\n* Pregnant"
    )
    elig["maximumAge"] = "80 Years"
    s = extract_sections(study)
    values = [c.atoms[0].value for c in s.structured_criteria]
    assert [(v.comparator, v.value) for v in values] == [
        (">=", 18.0),
        ("<=", 80.0),
    ]


def test_split_nested_bullets_and_continuations():
    text = (
        "Inclusion Criteria:\n\n"
        "1. Adults with one of:\n   * sepsis\n   * septic shock\n"
        "2. Lactate > 2 mmol/L\n  measured within 6 hours\n\n"
        "Exclusion Criteria:\n\n- Pregnancy\n"
    )
    inc, exc, warnings = split_eligibility(text)
    assert inc == [
        "Adults with one of: sepsis; septic shock",
        "Lactate > 2 mmol/L measured within 6 hours",
    ]
    assert exc == ["Pregnancy"]
    assert warnings == []


def test_split_semicolon_paragraph_and_unescape():
    # Shape of NCT00475852 (unbulleted, markdown-escaped).
    text = (
        "Inclusion Criteria:\n\nHospitalized for ADHF\n\nExclusion Criteria:\n\n"
        "At high risk for hypotension; Acute coronary syndrome as primary diagnosis; "
        "Persistent, uncontrolled hypertension (SBP \\[systolic blood pressure\\] \\>180 mmHg)."
    )
    inc, exc, warnings = split_eligibility(text)
    assert inc == ["Hospitalized for ADHF"]
    assert exc == [
        "At high risk for hypotension",
        "Acute coronary syndrome as primary diagnosis",
        "Persistent, uncontrolled hypertension (SBP [systolic blood pressure] >180 mmHg).",
    ]
    assert warnings == ["exclusion: paragraph split on ';' into 3 items"]


def test_split_inline_roman_numerals():
    # Shape of NCT03592693.
    text = (
        "Inclusion Criteria:\n\n* i. Diagnosis of septic shock within 12 hours\n\n"
        "Exclusion Criteria:\n\n* i. Age \\< 18 years ii. Pregnant iii. Patients with AIDS and "
        "a CD4 count of \\< 50 per microliter iv. Known G-6PD deficiency."
    )
    inc, exc, warnings = split_eligibility(text)
    assert inc == ["Diagnosis of septic shock within 12 hours"]
    assert exc == [
        "Age < 18 years",
        "Pregnant",
        "Patients with AIDS and a CD4 count of < 50 per microliter",
        "Known G-6PD deficiency.",
    ]
    assert warnings == ["exclusion: inline i./ii./... list split into 4 items"]


def test_split_without_headers_warns():
    inc, exc, warnings = split_eligibility("* Adults\n* Septic shock")
    assert inc == ["Adults", "Septic shock"] and exc == []
    assert warnings


@respx.mock
def test_client_caches_and_retries(tmp_path, nct03872011_study):
    route = respx.get("https://clinicaltrials.gov/api/v2/studies/NCT03872011").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=nct03872011_study)]
    )
    client = CTGovClient(cache_dir=tmp_path, max_retries=1)
    import agentic_tte.sources.ctgov.client as mod

    mod.time.sleep = lambda s: None  # no real backoff in tests
    first = client.get_study("NCT03872011")
    assert route.call_count == 2
    assert "_retrieved_at" in first
    second = client.get_study("nct03872011")  # served from the cache
    assert route.call_count == 2
    assert second == first


@respx.mock
def test_client_404(tmp_path):
    respx.get("https://clinicaltrials.gov/api/v2/studies/NCT99999999").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(CTGovError, match="not found"):
        CTGovClient(cache_dir=tmp_path).get_study("NCT99999999")
