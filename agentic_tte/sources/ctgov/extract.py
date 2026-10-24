"""Deterministic extraction of TTE-relevant sections from a ClinicalTrials.gov v2 record.

EmulatRx asked an LLM to split eligibility text into items (``EC_EXPORT_PROMPT``). The v2 API
returns lightly structured markdown, so splitting is done here without a model; the LLM only
sees one criterion at a time afterwards.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agentic_tte.protocol.models import (
    Comparator,
    ConceptRef,
    Criterion,
    CriterionAtom,
    Domain,
    TrialSource,
    ValueConstraint,
)
from agentic_tte.protocol.normalize import parse_age
from agentic_tte.sources.ctgov.client import record_sha256


class Intervention(BaseModel):
    type: str | None = None
    name: str
    description: str | None = None
    other_names: list[str] = Field(default_factory=list)


class Arm(BaseModel):
    label: str
    type: str | None = None
    description: str | None = None
    interventions: list[Intervention] = Field(default_factory=list)


class RegistryOutcome(BaseModel):
    measure: str
    description: str | None = None
    time_frame: str | None = None
    primary: bool


class TrialSections(BaseModel):
    source: TrialSource
    brief_summary: str | None = None
    detailed_description: str | None = None
    inclusion: list[str] = Field(default_factory=list)
    exclusion: list[str] = Field(default_factory=list)
    structured_criteria: list[Criterion] = Field(default_factory=list)
    arms: list[Arm] = Field(default_factory=list)
    outcomes: list[RegistryOutcome] = Field(default_factory=list)
    eligibility_text: str | None = None
    warnings: list[str] = Field(default_factory=list)


_HEADER_RE = re.compile(
    r"^\s*(?:key\s+|main\s+|major\s+)?(inclusion|exclusion)\s+criteria\b[^:\n]*:?\s*$",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^(\s*)(?:[*\-•·]|\d{1,2}[.)]|[a-z][.)])\s+(.*)$")


def split_eligibility(text: str | None) -> tuple[list[str], list[str], list[str]]:
    """Split CT.gov eligibility markdown into (inclusion, exclusion, warnings).

    Indented bullets and continuation lines are merged into their parent item, following
    EmulatRx's rule that a line starting with a space continues the previous item.
    """
    sections: dict[str, list[str]] = {"inclusion": [], "exclusion": []}
    warnings: list[str] = []
    current: str | None = None
    unsectioned: list[str] = []
    parent_indent = 0

    for line in (text or "").splitlines():
        if not line.strip():
            continue
        header = _HEADER_RE.match(line)
        if header:
            current = header.group(1).lower()
            continue
        items = sections[current] if current else unsectioned
        bullet = _BULLET_RE.match(line)
        if bullet:
            indent, body = len(bullet.group(1)), bullet.group(2).strip()
            if items and indent > parent_indent:
                items[-1] = _join_child(items[-1], body)
            else:
                items.append(body)
                parent_indent = indent
        elif items and line[:1].isspace():
            items[-1] = f"{items[-1]} {line.strip()}"
        else:
            parts = [p.strip() for p in line.split(";") if p.strip()]
            if len(parts) > 1:
                # Unbulleted registries (e.g. NCT00475852) separate items with semicolons.
                warnings.append(f"{current or 'unsectioned'}: paragraph split on ';' into "
                                f"{len(parts)} items")  # fmt: skip
            items.extend(parts)
            parent_indent = 0

    for key in sections:
        merged = _merge_definitions(sections[key])
        if len(merged) < len(sections[key]):
            warnings.append(
                f"{key}: {len(sections[key]) - len(merged)} definition item(s) merged into the "
                "criterion they define"
            )
        expanded = []
        for item in merged:
            parts = _split_inline_roman(item)
            if len(parts) > 1:
                warnings.append(f"{key}: inline i./ii./... list split into {len(parts)} items")
            expanded.extend(parts)
        sections[key] = [_unescape(i) for i in expanded]
    unsectioned = [_unescape(i) for i in unsectioned]
    if unsectioned:
        if not sections["inclusion"] and not sections["exclusion"]:
            sections["inclusion"] = unsectioned
            warnings.append("no inclusion/exclusion headers; all items treated as inclusion")
        else:
            warnings.append(f"{len(unsectioned)} line(s) before the first header ignored")
    return sections["inclusion"], sections["exclusion"], warnings


_ROMAN = r"(?:x{0,1}(?:ix|iv|v?i{0,3}))"
_ROMAN_MARK = re.compile(rf"(?:^|\s)({_ROMAN})\.\s+")


def _split_inline_roman(item: str) -> list[str]:
    """Split "i. Age < 18 ii. Pregnant iii. ..." (NCT03592693) into items. Only when the item
    starts with "i." and the numerals run i, ii, iii... in order, so "vs. ii" text is safe."""
    if not item.startswith("i. "):
        return [item]
    marks = [m for m in _ROMAN_MARK.finditer(item) if m.group(1)]
    expected = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]
    ordered = []
    for m in marks:
        if len(ordered) < len(expected) and m.group(1) == expected[len(ordered)]:
            ordered.append(m)
    if len(ordered) < 2:
        return [item[3:].strip()]
    bounds = [m.start() for m in ordered[1:]] + [len(item)]
    starts = [m.end() for m in ordered]
    return [item[s:e].strip() for s, e in zip(starts, bounds, strict=True)]


_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!<>|~])")


def _unescape(text: str) -> str:
    """CT.gov v2 returns markdown, so "> 0.3 mg/dL" arrives as "\\> 0.3 mg/dL"."""
    return _MD_ESCAPE.sub(r"\1", text)


_TRAILING_CONJ = re.compile(r"(?::|\b(?:and|or|and/or)\b)\s*$", re.IGNORECASE)


def _join_child(parent: str, child: str) -> str:
    """Append a sub-bullet. A parent ending in ":" or a conjunction ("MAP ≥ 65 AND") reads on
    into the child, so no "; " goes in between (C9)."""
    sep = " " if _TRAILING_CONJ.search(parent) else "; "
    return f"{parent}{sep}{child}"


_DEFINES_RE = re.compile(
    r"^(?:.*?\bdefines\s+(?P<a>.+?)\s+as\b|(?P<b>.+?)\s+(?:is|are)\s+defined\s+as\b)",
    re.IGNORECASE,
)


def _merge_definitions(items: list[str]) -> list[str]:
    """ "SEPSIS-3 defines septic shock as …" after "Septic shock admitted to the ICU…" is the
    definition of a term in the previous item, not a criterion of its own (NCT04134403, C9).
    Only merged when the defined term appears in the previous item, so "Pregnant defined by
    negative serum HCG" (a criterion with its own definition) stays separate."""
    out: list[str] = []
    for item in items:
        m = _DEFINES_RE.match(item)
        term = (m.group("a") or m.group("b")).strip().lower() if m else None
        if out and term and term in out[-1].lower():
            out[-1] = f"{out[-1]} [definition: {item}]"
        else:
            out.append(item)
    return out


_AGE_IN_TEXT = re.compile(r"\bage[ds]?\b|\byears?\s+(?:old|of age)\b|\b\d+\s*(?:years|yrs)\b", re.I)


def structured_criteria(eligibility: dict[str, Any], free_text: list[str]) -> list[Criterion]:
    """Criteria from CT.gov's structured fields (age, sex). Skipped when the free text already
    states the same thing, so the protocol has no duplicates."""
    out: list[Criterion] = []
    has_age_text = any(_AGE_IN_TEXT.search(t) for t in free_text)
    for field, comparator in (("minimumAge", Comparator.GE), ("maximumAge", Comparator.LE)):
        age = parse_age(eligibility.get(field))
        if age and not has_age_text:
            out.append(
                Criterion(
                    id=f"S{len(out) + 1}",
                    kind="inclusion",
                    text=f"{field}: {eligibility[field]}",
                    atoms=[
                        CriterionAtom(
                            concepts=[ConceptRef(name="age", domain=Domain.DEMOGRAPHIC)],
                            value=ValueConstraint(
                                comparator=comparator,
                                value=age[0],
                                unit=age[1],
                                text=eligibility[field],
                            ),
                        )
                    ],
                    complexity="simple",
                    source="registry_structured",
                )
            )
    sex = (eligibility.get("sex") or "ALL").upper()
    if sex in ("FEMALE", "MALE"):
        out.append(
            Criterion(
                id=f"S{len(out) + 1}",
                kind="inclusion",
                text=f"sex: {sex}",
                atoms=[
                    CriterionAtom(
                        concepts=[ConceptRef(name="sex", domain=Domain.DEMOGRAPHIC)],
                        value=ValueConstraint(comparator=Comparator.PRESENT, text=sex.lower()),
                    )
                ],
                complexity="simple",
                source="registry_structured",
            )
        )
    return out


def extract_sections(study: dict[str, Any]) -> TrialSections:
    ps = study.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    design = ps.get("designModule", {})
    elig = ps.get("eligibilityModule", {})
    arms_mod = ps.get("armsInterventionsModule", {})
    out_mod = ps.get("outcomesModule", {})

    inclusion, exclusion, warnings = split_eligibility(elig.get("eligibilityCriteria"))

    interventions = {
        iv.get("name"): Intervention(
            type=iv.get("type"),
            name=iv.get("name", ""),
            description=iv.get("description"),
            other_names=iv.get("otherNames", []),
        )
        for iv in arms_mod.get("interventions", [])
    }
    arms = []
    for ag in arms_mod.get("armGroups", []):
        linked = []
        for ref in ag.get("interventionNames", []):
            name = ref.split(":", 1)[1].strip() if ":" in ref else ref.strip()
            if name in interventions:
                linked.append(interventions[name])
            else:
                warnings.append(f"arm {ag.get('label')!r} names unknown intervention {ref!r}")
        arms.append(
            Arm(
                label=ag.get("label", ""),
                type=ag.get("type"),
                description=ag.get("description"),
                interventions=linked,
            )
        )

    outcomes = [
        RegistryOutcome(
            measure=o.get("measure", ""),
            description=o.get("description"),
            time_frame=o.get("timeFrame"),
            primary=primary,
        )
        for key, primary in (("primaryOutcomes", True), ("secondaryOutcomes", False))
        for o in out_mod.get(key, [])
    ]

    retrieved = study.get("_retrieved_at")
    design_info = design.get("designInfo") or {}
    enrollment = design.get("enrollmentInfo") or {}
    source = TrialSource(
        nct_id=ident.get("nctId"),
        title=ident.get("officialTitle") or ident.get("briefTitle"),
        phase=design.get("phases", []),
        study_type=design.get("studyType"),
        overall_status=ps.get("statusModule", {}).get("overallStatus"),
        allocation=design_info.get("allocation"),
        intervention_model=design_info.get("interventionModel"),
        masking=(design_info.get("maskingInfo") or {}).get("masking"),
        enrollment=enrollment.get("count"),
        enrollment_type=enrollment.get("type"),
        conditions=ps.get("conditionsModule", {}).get("conditions", []),
        has_results=study.get("hasResults"),
        retrieved_at=datetime.fromisoformat(retrieved) if retrieved else None,
        record_sha256=record_sha256({k: v for k, v in study.items() if not k.startswith("_")}),
    )
    if study.get("hasResults") is False:
        warnings.append(
            "no results posted on ClinicalTrials.gov; a Replay comparison needs a publication"
        )
    description = ps.get("descriptionModule", {})
    return TrialSections(
        source=source,
        brief_summary=description.get("briefSummary"),
        detailed_description=description.get("detailedDescription"),
        inclusion=inclusion,
        exclusion=exclusion,
        structured_criteria=structured_criteria(elig, inclusion + exclusion),
        arms=arms,
        outcomes=outcomes,
        eligibility_text=elig.get("eligibilityCriteria"),
        warnings=warnings,
    )
