"""The Target Trial Protocol: the central, versioned object every agent reads and writes.

It mirrors the standard target-trial specification (Hernán & Robins 2016; Hernán, Wang & Leaf
2022): eligibility, treatment strategies, assignment, time zero, follow-up, outcomes, causal
contrast and analysis plan.

Ported from and extending EmulatRx ``trialist_utils.py`` (``TrialInfo``,
``TrialEligibilityCriterion``, ``TrialTreatment``, ``TrialOutcome``; MIT, TrialLab). Upstream
kept criteria as free strings (``x_category``/``x_concept_name``/``temporal_value``...); here they
are typed, so a deterministic compiler can turn them into SQL later.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


# --- vocabulary ---------------------------------------------------------------------------------


class Domain(StrEnum):
    """Concept domains, following Criteria2Query 3.0 / OMOP CDM as used by EmulatRx."""

    DEMOGRAPHIC = "Demographic"
    CONDITION = "Condition"
    DEVICE = "Device"
    PROCEDURE = "Procedure"
    DRUG = "Drug"
    MEASUREMENT = "Measurement"
    OBSERVATION = "Observation"
    VISIT = "Visit"


class MappingStatus(StrEnum):
    UNMAPPED = "unmapped"  # not yet looked up in the OMOP vocabulary
    MAPPED = "mapped"
    UNMAPPABLE = "unmappable"  # no RWD equivalent; must be reported, never silently dropped
    SURROGATE = "surrogate"  # replaced by a clinically justified proxy


class Comparator(StrEnum):
    LT = "<"
    LE = "<="
    EQ = "="
    GE = ">="
    GT = ">"
    BETWEEN = "between"
    PRESENT = "present"  # qualitative: positive, high-dose, etc.; detail in ValueConstraint.text


class TimeUnit(StrEnum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class TemporalRelation(StrEnum):
    """Temporal patterns from the EmulatRx TEMPORAL_PROMPT, aligned with the Time Event Ontology."""

    X_WITHIN_T_BEFORE_Y = "x_within_t_before_y"
    X_WITHIN_T_AFTER_Y = "x_within_t_after_y"
    X_BEFORE_Y = "x_before_y"
    X_AFTER_Y = "x_after_y"
    X_AT_Y = "x_at_time_of_y"
    CUMULATIVE_DURATION = "cumulative_duration"


class CausalContrast(StrEnum):
    INTENTION_TO_TREAT = "intention_to_treat"  # observational analogue
    PER_PROTOCOL = "per_protocol"


class Estimand(StrEnum):
    HAZARD_RATIO = "hazard_ratio"
    RISK_DIFFERENCE = "risk_difference"
    RMST_DIFFERENCE = "rmst_difference"


# --- building blocks ----------------------------------------------------------------------------


class Duration(_Model):
    amount: float = Field(ge=0)
    unit: TimeUnit


class ConceptRef(_Model):
    name: str
    domain: Domain
    source_text: str | None = None
    omop_concept_id: int | None = None
    standard_name: str | None = None
    status: MappingStatus = MappingStatus.UNMAPPED


class ValueConstraint(_Model):
    comparator: Comparator
    value: float | None = None
    upper: float | None = None  # for BETWEEN
    unit: str | None = None
    text: str | None = None  # original wording, e.g. "18 years old or older"


class TemporalConstraint(_Model):
    relation: TemporalRelation
    window: Duration | None = None
    anchor: str = "time_zero"  # event Y; "time_zero" or a named event such as "ICU admission"
    text: str | None = None


class Criterion(_Model):
    id: str
    kind: Literal["inclusion", "exclusion"]
    text: str
    concepts: list[ConceptRef] = Field(default_factory=list)
    concept_logic: Literal["all", "any"] = "any"  # "allergy to X, Y, or Z" -> any
    value: ValueConstraint | None = None
    temporal: TemporalConstraint | None = None
    negated: bool = False
    complexity: Literal["simple", "complex"] | None = None  # Ross et al. 2010 heuristic
    source: Literal["registry_structured", "registry_text", "user", "agent"] = "registry_text"
    notes: str | None = None


class TreatmentComponent(_Model):
    concept: ConceptRef
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: Duration | None = None


class TreatmentStrategy(_Model):
    label: str
    arm_type: str | None = None  # EXPERIMENTAL, PLACEBO_COMPARATOR, ACTIVE_COMPARATOR, ...
    description: str | None = None
    components: list[TreatmentComponent] = Field(default_factory=list)
    is_comparator: bool = False
    grace_period: Duration | None = None  # window after time zero in which initiation counts


class Outcome(_Model):
    name: str
    description: str | None = None
    time_frame: str | None = None
    primary: bool = False
    concept: ConceptRef | None = None
    horizon: Duration | None = None


class TimeZero(_Model):
    anchor_event: str  # e.g. "septic shock onset", "ICU admission"
    offset: Duration | None = None
    rationale: str | None = None


class FollowUp(_Model):
    start: str = "time_zero"
    horizon: Duration | None = None
    censoring_events: list[str] = Field(
        default_factory=lambda: ["death_from_other_causes", "loss_to_follow_up", "end_of_data"]
    )


class AssignmentProcedure(_Model):
    """How randomization is emulated: confounders to balance on and the balancing method."""

    method: Literal["auto", "psm", "iptw", "overlap_weights", "none"] = "auto"
    confounders: list[ConceptRef] = Field(default_factory=list)


class AnalysisPlan(_Model):
    estimand: Estimand = Estimand.HAZARD_RATIO
    outcome_model: Literal["auto", "cox", "kaplan_meier", "parametric", "rsf", "doubly_robust"] = (
        "auto"
    )
    immortal_time_handling: Literal["auto", "clone_censor_weight", "landmark", "none"] = "auto"
    prespecified_subgroups: list[str] = Field(default_factory=list)
    sensitivity_analyses: list[str] = Field(default_factory=list)


class TrialSource(_Model):
    registry: str = "ClinicalTrials.gov"
    nct_id: str | None = None
    title: str | None = None
    phase: list[str] = Field(default_factory=list)
    enrollment: int | None = None
    conditions: list[str] = Field(default_factory=list)
    has_results: bool | None = None
    retrieved_at: datetime | None = None
    record_sha256: str | None = None  # hash of the raw registry JSON used


class Provenance(_Model):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by: str = "agentic-tte"  # agent name or user id
    llm_model: str | None = None
    llm_model_digest: str | None = None
    prompt_version: str | None = None
    notes: str | None = None


# --- the protocol -------------------------------------------------------------------------------

TTE_ELEMENTS = (
    "eligibility",
    "treatment_strategies",
    "assignment",
    "time_zero",
    "follow_up",
    "outcomes",
    "causal_contrast",
    "analysis_plan",
)


class TargetTrialProtocol(_Model):
    protocol_id: str
    version: int = 1
    parent_version: int | None = None
    mode: Literal["replay", "rehearse", "redesign"] = "replay"
    status: Literal["draft", "approved"] = "draft"
    source: TrialSource = Field(default_factory=TrialSource)

    eligibility: list[Criterion] = Field(default_factory=list)
    treatment_strategies: list[TreatmentStrategy] = Field(default_factory=list)
    assignment: AssignmentProcedure = Field(default_factory=AssignmentProcedure)
    time_zero: TimeZero | None = None
    follow_up: FollowUp = Field(default_factory=FollowUp)
    outcomes: list[Outcome] = Field(default_factory=list)
    causal_contrast: CausalContrast = CausalContrast.INTENTION_TO_TREAT
    analysis_plan: AnalysisPlan = Field(default_factory=AnalysisPlan)

    provenance: Provenance = Field(default_factory=Provenance)

    @property
    def inclusion(self) -> list[Criterion]:
        return [c for c in self.eligibility if c.kind == "inclusion"]

    @property
    def exclusion(self) -> list[Criterion]:
        return [c for c in self.eligibility if c.kind == "exclusion"]

    def content_hash(self) -> str:
        """Hash of the scientific content only (ignores version, status and provenance)."""
        payload = self.model_dump(
            mode="json", exclude={"version", "parent_version", "status", "provenance"}
        )
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def completeness(self) -> dict[str, bool]:
        """Which of the TTE elements are specified. Defaults with ``auto`` count as specified;
        the Auditor (Phase 3) checks their quality."""
        return {
            "eligibility": bool(self.eligibility),
            "treatment_strategies": len(self.treatment_strategies) >= 2,
            "assignment": True,
            "time_zero": self.time_zero is not None,
            "follow_up": self.follow_up.horizon is not None,
            "outcomes": any(o.primary for o in self.outcomes),
            "causal_contrast": True,
            "analysis_plan": True,
        }
