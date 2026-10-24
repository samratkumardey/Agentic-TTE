"""Trialist agent: ClinicalTrials.gov record -> draft TargetTrialProtocol.

Covers EmulatRx tasks T1 (extract and standardize trial information) and T2 (protocol
generation), ported from ``trialistUtils/main.parse_clinical_trial`` (MIT, TrialLab).

What is deterministic and what uses the LLM:
- deterministic: fetching, splitting eligibility text, structured age/sex, outcomes and their
  horizons, comparator arms by registry arm type, follow-up horizon.
- LLM (schema-constrained, one call per item): concepts/value/temporal per criterion, treatment
  components per arm, and a time-zero proposal.
A failed LLM parse never aborts the draft: the criterion keeps its text and is flagged for review.
OMOP concept mapping is the Informatician's job (Phase 2); concepts leave here ``unmapped``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from agentic_tte.agents import prompts
from agentic_tte.llm.base import LLMClient, LLMOutputError, Message
from agentic_tte.protocol.models import (
    Comparator,
    ConceptRef,
    Criterion,
    Domain,
    Duration,
    FollowUp,
    Outcome,
    Provenance,
    TargetTrialProtocol,
    TemporalConstraint,
    TemporalRelation,
    TimeUnit,
    TimeZero,
    TreatmentComponent,
    TreatmentStrategy,
    ValueConstraint,
)
from agentic_tte.protocol.normalize import parse_duration
from agentic_tte.sources.ctgov.client import CTGovClient
from agentic_tte.sources.ctgov.extract import Arm, TrialSections, extract_sections

EventSink = Callable[[str, dict[str, Any]], None]

# --- LLM-facing schemas: every field required (nullable instead of defaulted), so they work
# with strict guided decoding in vLLM as well as Ollama's format constraint.


class _LLMSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParsedConcept(_LLMSchema):
    name: str
    domain: Domain
    source_text: str


class ParsedValue(_LLMSchema):
    comparator: Comparator
    value: float | None
    upper: float | None
    unit: str | None
    text: str


class ParsedTemporal(_LLMSchema):
    relation: TemporalRelation
    window_amount: float | None
    window_unit: TimeUnit | None
    anchor: str
    text: str


class CriterionParse(_LLMSchema):
    concepts: list[ParsedConcept]
    concept_logic: Literal["all", "any"]
    value: ParsedValue | None
    temporal: ParsedTemporal | None
    negated: bool
    complexity: Literal["simple", "complex"]


class ParsedComponent(_LLMSchema):
    name: str
    domain: Domain
    dose: str | None
    route: str | None
    frequency: str | None
    duration_amount: float | None
    duration_unit: TimeUnit | None


class TreatmentParse(_LLMSchema):
    components: list[ParsedComponent]
    is_comparator: bool


class DesignProposal(_LLMSchema):
    time_zero_anchor: str
    time_zero_offset_amount: float | None
    time_zero_offset_unit: TimeUnit | None
    grace_period_amount: float | None
    grace_period_unit: TimeUnit | None
    rationale: str


_COMPARATOR_ARM_TYPES = {"PLACEBO_COMPARATOR", "SHAM_COMPARATOR", "NO_INTERVENTION",
                         "ACTIVE_COMPARATOR"}  # fmt: skip


def _duration(amount: float | None, unit: TimeUnit | str | None) -> Duration | None:
    if amount is None or unit is None:
        return None
    return Duration(amount=amount, unit=unit)


_STOPWORDS = {"a", "an", "and", "or", "of", "to", "the", "in", "on", "with", "for", "at", "by",
              "not", "no", "any"}  # fmt: skip


def _grounded(source_text: str, criterion_text: str) -> bool:
    """Every content word of ``source_text`` occurs in the criterion. Exact substrings are too
    strict: splitting "allergy to X, Y, or Z" yields "allergy to Y", which is not contiguous."""
    words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))  # noqa: E731
    return (words(source_text) - _STOPWORDS) <= words(criterion_text)


class Trialist:
    name = "trialist"

    def __init__(self, llm: LLMClient, ctgov: CTGovClient | None = None,
                 on_event: EventSink | None = None):  # fmt: skip
        self.llm = llm
        self.ctgov = ctgov
        self.on_event = on_event or (lambda kind, payload: None)

    # -- entry points -----------------------------------------------------------------------

    def draft(self, nct_id: str, *, refresh: bool = False) -> TargetTrialProtocol:
        if self.ctgov is None:
            raise RuntimeError("Trialist.draft needs a CTGovClient")
        study = self.ctgov.get_study(nct_id, refresh=refresh)
        self.on_event("fetched", {"nct_id": nct_id})
        return self.draft_from_study(study)

    def draft_from_study(self, study: dict[str, Any]) -> TargetTrialProtocol:
        sections = extract_sections(study)
        for w in sections.warnings:
            self.on_event("warning", {"message": w})
        return self.draft_from_sections(sections)

    def draft_from_sections(self, s: TrialSections) -> TargetTrialProtocol:
        eligibility = list(s.structured_criteria)
        for kind, items, prefix in (("inclusion", s.inclusion, "I"),
                                    ("exclusion", s.exclusion, "E")):  # fmt: skip
            for i, text in enumerate(items, 1):
                eligibility.append(self.parse_criterion(f"{prefix}{i}", kind, text))

        strategies = [self.parse_arm(arm) for arm in s.arms]
        outcomes = [
            Outcome(
                name=o.measure,
                description=o.description,
                time_frame=o.time_frame,
                primary=o.primary,
                horizon=parse_duration(o.time_frame) or parse_duration(o.measure),
            )
            for o in s.outcomes
        ]
        primary = next((o for o in outcomes if o.primary), None)
        time_zero, grace = self.propose_time_zero(s)
        if grace is not None:
            for st in strategies:
                if not st.is_comparator:
                    st.grace_period = grace

        nct = s.source.nct_id or "unregistered"
        protocol = TargetTrialProtocol(
            protocol_id=f"{nct}-replay",
            mode="replay",
            source=s.source,
            eligibility=eligibility,
            treatment_strategies=strategies,
            time_zero=time_zero,
            follow_up=FollowUp(horizon=primary.horizon if primary else None),
            outcomes=outcomes,
            provenance=Provenance(
                created_by=self.name,
                llm_model=self.llm.model,
                llm_model_digest=self.llm.model_digest(),
                prompt_version=prompts.PROMPT_VERSION,
            ),
        )
        self.on_event("drafted", {"protocol_id": protocol.protocol_id,
                                  "completeness": protocol.completeness()})  # fmt: skip
        return protocol

    # -- steps ------------------------------------------------------------------------------

    def parse_criterion(self, cid: str, kind: Literal["inclusion", "exclusion"],
                        text: str) -> Criterion:  # fmt: skip
        messages = [
            Message(role="system", content=prompts.CRITERION_SYSTEM),
            Message(role="user", content=prompts.CRITERION_EXAMPLE_USER),
            Message(role="assistant", content=prompts.CRITERION_EXAMPLE_ASSISTANT),
            Message(role="user", content=f"Criterion type: {kind}\nCriterion: {text}"),
        ]
        try:
            p = self.llm.structured(messages, CriterionParse)
        except LLMOutputError as exc:
            self.on_event("parse_failed", {"criterion": cid, "error": str(exc)})
            return Criterion(id=cid, kind=kind, text=text, notes=f"parse_failed: {exc}")

        notes = []
        concepts = []
        for c in p.concepts:
            if not _grounded(c.source_text, text):
                notes.append(f"concept {c.name!r} not found verbatim in criterion text")
            concepts.append(ConceptRef(name=c.name, domain=c.domain, source_text=c.source_text))
        temporal = None
        if p.temporal:
            window = _duration(p.temporal.window_amount, p.temporal.window_unit)
            temporal = TemporalConstraint(
                relation=p.temporal.relation,
                window=window or parse_duration(p.temporal.text),
                anchor=p.temporal.anchor or "time_zero",
                text=p.temporal.text,
            )
        value = ValueConstraint(**p.value.model_dump()) if p.value else None
        if not concepts:
            notes.append("no concepts extracted")
        self.on_event("criterion_parsed", {"criterion": cid, "concepts": len(concepts)})
        return Criterion(
            id=cid,
            kind=kind,
            text=text,
            concepts=concepts,
            concept_logic=p.concept_logic,
            value=value,
            temporal=temporal,
            negated=p.negated,
            complexity=p.complexity,
            notes="; ".join(notes) or None,
        )

    def parse_arm(self, arm: Arm) -> TreatmentStrategy:
        details = "\n".join(
            f"- {iv.type or 'INTERVENTION'}: {iv.name}"
            + (f" (also: {', '.join(iv.other_names)})" if iv.other_names else "")
            + (f"\n  {iv.description}" if iv.description else "")
            for iv in arm.interventions
        )
        user = (
            f"Arm label: {arm.label}\nArm type: {arm.type or 'unknown'}\n"
            f"Arm description: {arm.description or '-'}\nInterventions:\n{details or '-'}"
        )
        by_type = arm.type in _COMPARATOR_ARM_TYPES if arm.type else None
        try:
            p = self.llm.structured(
                [Message(role="system", content=prompts.TREATMENT_SYSTEM),
                 Message(role="user", content=user)],
                TreatmentParse,
            )  # fmt: skip
        except LLMOutputError as exc:
            self.on_event("parse_failed", {"arm": arm.label, "error": str(exc)})
            return TreatmentStrategy(
                label=arm.label,
                arm_type=arm.type,
                description=arm.description,
                is_comparator=bool(by_type),
            )
        components = [
            TreatmentComponent(
                concept=ConceptRef(name=c.name, domain=c.domain, source_text=c.name),
                dose=c.dose,
                route=c.route,
                frequency=c.frequency,
                duration=_duration(c.duration_amount, c.duration_unit),
            )
            for c in p.components
        ]
        return TreatmentStrategy(
            label=arm.label,
            arm_type=arm.type,
            description=arm.description,
            components=components,
            # The registry arm type is authoritative when present.
            is_comparator=by_type if by_type is not None else p.is_comparator,
        )

    def propose_time_zero(self, s: TrialSections) -> tuple[TimeZero | None, Duration | None]:
        primary = next((o for o in s.outcomes if o.primary), None)
        user = (
            f"Title: {s.source.title}\nConditions: {', '.join(s.source.conditions)}\n"
            "Inclusion criteria:\n" + "\n".join(f"- {c}" for c in s.inclusion)
            + "\nArms:\n" + "\n".join(f"- {a.label} ({a.type}): {a.description}" for a in s.arms)
            + (f"\nPrimary outcome: {primary.measure}; time frame: {primary.time_frame}"
               if primary else "")
        )  # fmt: skip
        try:
            p = self.llm.structured(
                [Message(role="system", content=prompts.DESIGN_SYSTEM),
                 Message(role="user", content=user)],
                DesignProposal,
            )  # fmt: skip
        except LLMOutputError as exc:
            self.on_event("parse_failed", {"element": "time_zero", "error": str(exc)})
            return None, None
        tz = TimeZero(
            anchor_event=p.time_zero_anchor,
            offset=_duration(p.time_zero_offset_amount, p.time_zero_offset_unit),
            rationale=p.rationale,
        )
        return tz, _duration(p.grace_period_amount, p.grace_period_unit)
