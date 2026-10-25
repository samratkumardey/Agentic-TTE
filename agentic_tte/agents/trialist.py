"""Trialist agent: ClinicalTrials.gov record -> draft TargetTrialProtocol.

Covers EmulatRx tasks T1 (extract and standardize trial information) and T2 (protocol
generation), ported from ``trialistUtils/main.parse_clinical_trial`` (MIT, TrialLab).

What is deterministic and what uses the LLM:
- deterministic: fetching, splitting eligibility text, structured age/sex, outcomes and their
  horizons and end events, comparator arms by registry arm type, treatment course durations
  when the model leaves them out, follow-up and censoring, the causal contrast when the
  registry states it, anchor normalization, and grounding checks on everything the model writes.
- LLM (schema-constrained, one call per item): atoms/concepts/values/time constraints and
  category per criterion, treatment components per arm, and a time-zero proposal.
A failed LLM parse never aborts the draft: the criterion keeps its text and is flagged for review.
Text the model wrote that the criterion never said is dropped and noted (trialist-v2, C5).
OMOP concept mapping is the Informatician's job (Phase 2); concepts leave here ``unmapped``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from agentic_tte.agents import prompts
from agentic_tte.llm.base import LLMClient, LLMOutputError, Message
from agentic_tte.protocol.models import (
    CausalContrast,
    Comparator,
    ConceptRef,
    Criterion,
    CriterionAtom,
    Domain,
    Duration,
    FollowUp,
    Origin,
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
from agentic_tte.protocol.normalize import (
    as_time_unit,
    durations_in,
    grounded_fraction,
    normalize_anchor,
    parse_course_duration,
    parse_duration,
    parse_time_frame,
)
from agentic_tte.sources.ctgov.client import CTGovClient
from agentic_tte.sources.ctgov.extract import Arm, TrialSections, extract_sections

EventSink = Callable[[str, dict[str, Any]], None]

# --- LLM-facing schemas: every field required (nullable instead of defaulted), so they work
# with strict guided decoding in vLLM as well as Ollama's format constraint.

_NULL_STRINGS = {"null", "none", "n/a", "na", "not stated", "not specified", "unknown", ""}


class _LLMSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _null_strings(cls, v: Any) -> Any:
        # Models sometimes write the string "null" instead of JSON null (C7).
        if isinstance(v, str) and v.strip().lower() in _NULL_STRINGS:
            return None
        return v


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


class ParsedAtom(_LLMSchema):
    concepts: list[ParsedConcept]
    concept_logic: Literal["all", "any"]
    value: ParsedValue | None
    temporal: ParsedTemporal | None
    negated: bool


class CriterionParse(_LLMSchema):
    atoms: list[ParsedAtom]
    atom_logic: Literal["all", "any"]
    category: Literal["clinical", "administrative"]
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
    time_zero_evidence: str | None
    grace_period_amount: float | None
    grace_period_unit: TimeUnit | None
    grace_period_evidence: str | None
    rationale: str


_COMPARATOR_ARM_TYPES = {"PLACEBO_COMPARATOR", "SHAM_COMPARATOR", "NO_INTERVENTION",
                         "ACTIVE_COMPARATOR"}  # fmt: skip
_ADMINISTRATIVE_RE = re.compile(
    r"\bconsent\b|\bprisoners?\b|\bincarcerat|\brefus(?:al|e)\b|\bwilling(?:ness)?\b|"
    r"\bable to comply\b|\blegally authori[sz]ed representative\b",
    re.IGNORECASE,
)
_ALL_CAUSE_DEATH_RE = re.compile(r"\bmortality\b|\bdeath\b|\bsurvival\b", re.IGNORECASE)
_CAUSE_SPECIFIC_RE = re.compile(
    r"cause[- ]specific|cardiovascular|cardiac death|related (?:death|mortality)", re.IGNORECASE
)
_ITT_RE = re.compile(r"intention[- ]to[- ]treat|\bITT\b", re.IGNORECASE)
_PER_PROTOCOL_RE = re.compile(r"\bper[- ]protocol\b", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_DURING_RE = re.compile(r"^\s*(?:in|during|throughout)\b", re.I)
_DURATION_WORDING = re.compile(r"\bfor\b|\blasting\b|\bduration\b|\bconsecutive\b", re.I)

# Concepts must be fully grounded; value/temporal text may paraphrase a little (the anchor is
# often worded differently) but must mostly come from the criterion.
_TEXT_GROUNDING = 0.5
_EVIDENCE_GROUNDING = 0.8


def _duration(amount: float | None, unit: TimeUnit | str | None) -> Duration | None:
    if amount is None or unit is None:
        return None
    return Duration(amount=amount, unit=unit)


def _grounded(source_text: str, criterion_text: str) -> bool:
    """Every content word of ``source_text`` occurs in the criterion. Exact substrings are too
    strict: splitting "allergy to X, Y, or Z" yields "allergy to Y", which is not contiguous."""
    return grounded_fraction(source_text, criterion_text) == 1.0


def _sentence_with(pattern: re.Pattern[str], *texts: str | None) -> str | None:
    for text in texts:
        for sentence in _SENTENCE_RE.findall(text or ""):
            if pattern.search(sentence):
                return sentence.strip()
    return None


_NEEDS_WINDOW = {TemporalRelation.X_WITHIN_T_BEFORE_Y, TemporalRelation.X_WITHIN_T_AFTER_Y,
                 TemporalRelation.CUMULATIVE_DURATION}  # fmt: skip


def _unit_grounded(unit: str, text: str) -> bool:
    """Time units are normalized ("years" -> "year"), so only other units are checked; plural
    forms count as present."""
    if as_time_unit(unit):
        return True
    have = {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower())}
    return all(w.rstrip("s") in have for w in re.findall(r"[a-z0-9]+", unit.lower()))


def _without_window(relation: str, text: str | None) -> str | None:
    """The relation left once an invented window is removed. "in hospital admission" means
    during the admission (E7), not before it."""
    if relation != TemporalRelation.CUMULATIVE_DURATION and _DURING_RE.search(text or ""):
        return TemporalRelation.X_AT_Y
    return {
        TemporalRelation.X_WITHIN_T_BEFORE_Y: TemporalRelation.X_BEFORE_Y,
        TemporalRelation.X_WITHIN_T_AFTER_Y: TemporalRelation.X_AFTER_Y,
        TemporalRelation.CUMULATIVE_DURATION: None,
    }.get(relation, relation)


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
        outcomes = []
        for o in s.outcomes:
            horizon, end_event, end_rule = parse_time_frame(o.time_frame)
            outcomes.append(
                Outcome(
                    name=o.measure,
                    description=o.description,
                    time_frame=o.time_frame,
                    primary=o.primary,
                    horizon=horizon or parse_duration(o.measure),
                    end_event=end_event,
                    end_rule=end_rule,
                )
            )
        primary = next((o for o in outcomes if o.primary), None)

        time_zero, grace, grace_evidence = self.propose_time_zero(s)
        if grace is not None:
            for st in strategies:
                if not st.is_comparator:
                    st.grace_period = grace
                    st.grace_period_origin = Origin.AGENT
                    st.grace_period_evidence = grace_evidence

        contrast, contrast_origin, contrast_evidence = self._causal_contrast(s)
        nct = s.source.nct_id or "unregistered"
        protocol = TargetTrialProtocol(
            protocol_id=f"{nct}-replay",
            mode="replay",
            source=s.source,
            eligibility=eligibility,
            treatment_strategies=strategies,
            time_zero=time_zero,
            follow_up=self._follow_up(primary),
            outcomes=outcomes,
            causal_contrast=contrast,
            causal_contrast_origin=contrast_origin,
            causal_contrast_evidence=contrast_evidence,
            provenance=Provenance(
                created_by=self.name,
                llm_model=self.llm.model,
                llm_model_digest=self.llm.model_digest(),
                prompt_version=prompts.PROMPT_VERSION,
            ),
        )
        self.on_event("drafted", {"protocol_id": protocol.protocol_id,
                                  "elements": protocol.element_status()})  # fmt: skip
        return protocol

    # -- steps ------------------------------------------------------------------------------

    def parse_criterion(self, cid: str, kind: Literal["inclusion", "exclusion"],
                        text: str) -> Criterion:  # fmt: skip
        category = "administrative" if _ADMINISTRATIVE_RE.search(text) else "clinical"
        try:
            p = self.llm.structured(criterion_messages(kind, text), CriterionParse)
        except LLMOutputError as exc:
            self.on_event("parse_failed", {"criterion": cid, "error": str(exc)})
            return Criterion(id=cid, kind=kind, text=text, category=category,
                             notes=f"parse_failed: {exc}")  # fmt: skip

        notes: list[str] = []
        atoms = [self._atom(a, text, notes) for a in p.atoms]
        if not any(a.concepts for a in atoms):
            notes.append("no concepts extracted")
        if category == "administrative" and p.category != "administrative":
            notes.append("category set to administrative by keyword rule")
        elif p.category == "administrative":
            category = "administrative"
        self.on_event("criterion_parsed",
                      {"criterion": cid, "atoms": len(atoms),
                       "concepts": sum(len(a.concepts) for a in atoms)})  # fmt: skip
        return Criterion(
            id=cid,
            kind=kind,
            text=text,
            atoms=atoms,
            atom_logic=p.atom_logic,
            category=category,
            complexity=p.complexity,
            notes="; ".join(notes) or None,
        )

    def _atom(self, a: ParsedAtom, text: str, notes: list[str]) -> CriterionAtom:
        concepts = []
        for c in a.concepts:
            if not _grounded(c.source_text, text):
                notes.append(f"concept {c.name!r} not found verbatim in criterion text")
            concepts.append(ConceptRef(name=c.name, domain=c.domain, source_text=c.source_text))

        value = None
        if a.value:
            if grounded_fraction(a.value.text, text) < _TEXT_GROUNDING:
                notes.append(f"dropped value not in criterion text: {a.value.text!r}")
            else:
                value = ValueConstraint(**a.value.model_dump())
                if value.unit and not _unit_grounded(value.unit, text):
                    # Kept: often a correct clinical inference ("MAP ≥ 65" -> mmHg), but the
                    # registry did not say it, so a reviewer should see it.
                    notes.append(f"unit {value.unit!r} not in criterion text")

        temporal = None
        if a.temporal:
            temporal = self._temporal(a.temporal, text, notes)

        # A duration written as a value ("> 24 hour" for "vasopressor use … for more than 24
        # hours") is a cumulative duration (E8). Ages in years are not.
        if (
            value
            and value.value is not None
            and as_time_unit(value.unit)
            and not any(c.domain == Domain.DEMOGRAPHIC for c in concepts)
        ):
            relation = temporal.relation if temporal else None
            ordering = relation in (TemporalRelation.X_BEFORE_Y, TemporalRelation.X_AFTER_Y,
                                    TemporalRelation.CUMULATIVE_DURATION)  # fmt: skip
            if ordering or (relation is None and _DURATION_WORDING.search(value.text or "")):
                temporal = TemporalConstraint(
                    relation=TemporalRelation.CUMULATIVE_DURATION,
                    window=Duration(amount=value.value, unit=as_time_unit(value.unit)),
                    anchor=temporal.anchor if temporal else "time_zero",
                    text=temporal.text if temporal else value.text,
                )
                if relation != TemporalRelation.CUMULATIVE_DURATION:
                    notes.append("duration value read as cumulative_duration")
        return CriterionAtom(
            concepts=concepts,
            concept_logic=a.concept_logic,
            value=value,
            temporal=temporal,
            negated=a.negated,
        )

    def _temporal(
        self, t: ParsedTemporal, text: str, notes: list[str]
    ) -> TemporalConstraint | None:
        if grounded_fraction(t.text, text) < _TEXT_GROUNDING:
            notes.append(f"dropped time constraint not in criterion text: {t.text!r}")
            return None
        window = _duration(t.window_amount, t.window_unit) or parse_duration(t.text)
        relation: str | None = t.relation
        if window is not None and window not in durations_in(text):
            notes.append(f"dropped window {window.amount:g} {window.unit} not in criterion text")
            window = None
        if window is None and relation in _NEEDS_WINDOW:
            # "within t" with no t (E7 on Hellbender: "in hospital admission").
            fixed = _without_window(relation, t.text)
            notes.append(f"{relation} without a window read as {fixed or 'no time constraint'}")
            relation = fixed
            if relation is None:
                return None
        return TemporalConstraint(
            relation=relation,
            window=window,
            anchor=normalize_anchor(t.anchor),
            text=t.text,
        )

    def parse_arm(self, arm: Arm) -> TreatmentStrategy:
        by_type = arm.type in _COMPARATOR_ARM_TYPES if arm.type else None
        try:
            p = self.llm.structured(arm_messages(arm), TreatmentParse)
        except LLMOutputError as exc:
            self.on_event("parse_failed", {"arm": arm.label, "error": str(exc)})
            return TreatmentStrategy(
                label=arm.label,
                arm_type=arm.type,
                description=arm.description,
                is_comparator=bool(by_type),
            )
        arm_text = " ".join(
            filter(None, [iv.description for iv in arm.interventions] + [arm.description])
        )
        names = [c.name for c in p.components]
        components = []
        for c in p.components:
            segment = _component_segment(c.name, names, arm_text)
            duration = _duration(c.duration_amount, c.duration_unit)
            if duration is None:
                duration = parse_course_duration(segment)  # C8 fallback
            _, end_event, end_rule = parse_time_frame(segment)
            components.append(
                TreatmentComponent(
                    concept=ConceptRef(name=c.name, domain=c.domain, source_text=c.name),
                    dose=c.dose,
                    route=c.route,
                    frequency=c.frequency,
                    duration=duration,
                    end_event=end_event,
                    end_rule=end_rule,
                )
            )
        return TreatmentStrategy(
            label=arm.label,
            arm_type=arm.type,
            description=arm.description,
            components=components,
            # The registry arm type is authoritative when present.
            is_comparator=by_type if by_type is not None else p.is_comparator,
        )

    def propose_time_zero(
        self, s: TrialSections
    ) -> tuple[TimeZero | None, Duration | None, str | None]:
        messages = design_messages(s)
        user = messages[-1].content
        try:
            p = self.llm.structured(messages, DesignProposal)
        except LLMOutputError as exc:
            self.on_event("parse_failed", {"element": "time_zero", "error": str(exc)})
            return None, None, None

        evidence = p.time_zero_evidence
        if evidence and grounded_fraction(evidence, user) < _EVIDENCE_GROUNDING:
            self.on_event("dropped", {"element": "time_zero_evidence", "text": evidence})
            evidence = None
        tz = TimeZero(
            anchor_event=p.time_zero_anchor,
            offset=_duration(p.time_zero_offset_amount, p.time_zero_offset_unit),
            rationale=p.rationale,
            origin=Origin.AGENT,
            evidence=evidence,
        )
        grace = _duration(p.grace_period_amount, p.grace_period_unit)
        grace_evidence = p.grace_period_evidence
        if grace is not None and (
            not grace_evidence or grounded_fraction(grace_evidence, user) < _EVIDENCE_GROUNDING
        ):
            # C14: a grace period nobody wrote down is an assumption, not a protocol element.
            self.on_event("dropped", {"element": "grace_period", "evidence": grace_evidence})
            grace, grace_evidence = None, None
        return tz, grace, grace_evidence

    # -- deterministic elements ---------------------------------------------------------------

    @staticmethod
    def _causal_contrast(s: TrialSections) -> tuple[CausalContrast, Origin, str | None]:
        """Read the contrast from the registry text when it is stated (C11)."""
        texts = (s.detailed_description, s.brief_summary)
        itt = _sentence_with(_ITT_RE, *texts)
        if itt:
            return CausalContrast.INTENTION_TO_TREAT, Origin.REGISTRY, itt
        pp = _sentence_with(_PER_PROTOCOL_RE, *texts)
        if pp:
            return CausalContrast.PER_PROTOCOL, Origin.REGISTRY, pp
        return CausalContrast.INTENTION_TO_TREAT, Origin.DEFAULT, None

    @staticmethod
    def _follow_up(primary: Outcome | None) -> FollowUp:
        """Horizon and end event from the primary outcome's time frame. Death is never a
        censoring event for an all-cause mortality outcome (C4); for other outcomes it is a
        competing event, left to the Statistician to model."""
        if primary is None:
            return FollowUp()
        all_cause_death = bool(
            _ALL_CAUSE_DEATH_RE.search(primary.name) and not _CAUSE_SPECIFIC_RE.search(primary.name)
        )
        return FollowUp(
            horizon=primary.horizon,
            end_event=primary.end_event,
            end_rule=primary.end_rule,
            censoring_events=["loss_to_follow_up", "end_of_data"],
            competing_events=[] if all_cause_death else ["death"],
            origin=Origin.DERIVED if primary.horizon else Origin.DEFAULT,
        )


# --- prompts as data: also used by eval/check_exit_run.py to rebuild a run's exact requests ---


def criterion_messages(kind: str, text: str) -> list[Message]:
    return [
        Message(role="system", content=prompts.CRITERION_SYSTEM),
        Message(role="user", content=prompts.CRITERION_EXAMPLE_USER),
        Message(role="assistant", content=prompts.CRITERION_EXAMPLE_ASSISTANT),
        Message(role="user", content=f"Criterion type: {kind}\nCriterion: {text}"),
    ]


def arm_messages(arm: Arm) -> list[Message]:
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
    return [Message(role="system", content=prompts.TREATMENT_SYSTEM),
            Message(role="user", content=user)]  # fmt: skip


def design_messages(s: TrialSections) -> list[Message]:
    primary = next((o for o in s.outcomes if o.primary), None)
    user = (
        f"Title: {s.source.title}\nConditions: {', '.join(s.source.conditions)}\n"
        "Inclusion criteria:\n" + "\n".join(f"- {c}" for c in s.inclusion)
        + "\nExclusion criteria:\n" + "\n".join(f"- {c}" for c in s.exclusion)
        + "\nArms:\n" + "\n".join(f"- {a.label} ({a.type}): {a.description}" for a in s.arms)
        + (f"\nPrimary outcome: {primary.measure}; time frame: {primary.time_frame}"
           if primary else "")
        + (f"\nDetailed description: {s.detailed_description}"
           if s.detailed_description else "")
    )  # fmt: skip
    return [Message(role="system", content=prompts.DESIGN_SYSTEM),
            Message(role="user", content=user)]  # fmt: skip


def first_requests(s: TrialSections) -> list[list[Message]]:
    """The first-attempt request of every LLM call a draft makes, in order. Retries add to
    these conversations, so they never coincide with another first attempt."""
    return (
        [criterion_messages("inclusion", t) for t in s.inclusion]
        + [criterion_messages("exclusion", t) for t in s.exclusion]
        + [arm_messages(a) for a in s.arms]
        + [design_messages(s)]
    )


def _component_segment(name: str, all_names: list[str], text: str) -> str:
    """The part of the intervention text about one component: from its name to the next
    component's name ("Hydrocortisone …, 7 days or until ICU discharge Thiamine …")."""
    low = text.lower()

    def find(n: str) -> int:
        i = low.find(n.lower())
        if i == -1 and n.split():
            i = low.find(n.split()[0].lower())
        return i

    start = find(name)
    if start == -1:
        return ""
    later = [i for n in all_names if n != name and (i := find(n)) > start]
    end = min(later) if later else min(len(text), start + 250)
    return text[start:end]
