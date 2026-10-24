"""The Target Trial Protocol: the central, versioned object every agent reads and writes.

It mirrors the standard target-trial specification (Hernán & Robins 2016; Hernán, Wang & Leaf
2022): eligibility, treatment strategies, assignment, time zero, follow-up, outcomes, causal
contrast and analysis plan.

Ported from and extending EmulatRx ``trialist_utils.py`` (``TrialInfo``,
``TrialEligibilityCriterion``, ``TrialTreatment``, ``TrialOutcome``; MIT, TrialLab). Upstream
kept criteria as free strings (``x_category``/``x_concept_name``/``temporal_value``...); here they
are typed, so a deterministic compiler can turn them into SQL later.

Schema v2 (trialist-v2): a criterion is a list of atoms, each with its own value and temporal
constraint; every protocol element records where it came from (``Origin``); time frames can end
at an event ("28 days or discharge, whichever comes first"). v1 files are upgraded on load.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 2


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


class Origin(StrEnum):
    """Where a protocol element's content came from. Scoring and the UI rely on this to keep
    registry facts apart from agent proposals and silent defaults."""

    REGISTRY = "registry"  # stated in the registry record
    DERIVED = "derived"  # computed deterministically from registry content
    AGENT = "agent"  # proposed by an LLM agent; needs review
    USER = "user"  # set by a trialist in Rehearse/Redesign
    DEFAULT = "default"  # nobody specified it


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

    @model_validator(mode="after")
    def _qualitative_without_number(self) -> ValueConstraint:
        # "= negative" or "> high" carry no number: they are qualitative (C12).
        if self.comparator != Comparator.PRESENT and self.value is None and self.upper is None:
            self.comparator = Comparator.PRESENT.value  # plain str, as use_enum_values stores
        return self


class TemporalConstraint(_Model):
    relation: TemporalRelation
    window: Duration | None = None
    anchor: str = "time_zero"  # event Y; "time_zero" or a named event such as "ICU admission"
    text: str | None = None


class CriterionAtom(_Model):
    """One independently checkable condition: "serum lactate > 2.0 mmol/L", or "vasopressor
    use for more than 24 hours before time zero"."""

    concepts: list[ConceptRef] = Field(default_factory=list)
    concept_logic: Literal["all", "any"] = "any"  # "allergy to X, Y, or Z" -> any
    value: ValueConstraint | None = None
    temporal: TemporalConstraint | None = None
    negated: bool = False


_V1_ATOM_KEYS = ("concepts", "concept_logic", "value", "temporal", "negated")


class Criterion(_Model):
    id: str
    kind: Literal["inclusion", "exclusion"]
    text: str
    atoms: list[CriterionAtom] = Field(default_factory=list)
    atom_logic: Literal["all", "any"] = "all"
    category: Literal["clinical", "administrative"] = "clinical"
    complexity: Literal["simple", "complex"] | None = None  # Ross et al. 2010 heuristic
    source: Literal["registry_structured", "registry_text", "user", "agent"] = "registry_text"
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _upgrade_v1(cls, data: Any) -> Any:
        """Schema v1 kept one value/temporal per criterion; wrap it in a single atom."""
        if isinstance(data, dict) and any(k in data for k in _V1_ATOM_KEYS):
            data = dict(data)
            atom = {k: data.pop(k) for k in _V1_ATOM_KEYS if k in data}
            if "atoms" not in data and (atom.get("concepts") or atom.get("value")
                                        or atom.get("temporal")):  # fmt: skip
                data["atoms"] = [atom]
        return data

    @property
    def concepts(self) -> list[ConceptRef]:
        return [c for atom in self.atoms for c in atom.concepts]

    @property
    def emulable(self) -> bool:
        """Administrative criteria (consent, prisoners) have no RWD counterpart; TTE drops them
        with a stated reason."""
        return self.category == "clinical"


class TreatmentComponent(_Model):
    concept: ConceptRef
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: Duration | None = None
    end_event: str | None = None  # "ICU discharge" in "7 days or until ICU discharge"
    end_rule: Literal["first", "last"] | None = None


class TreatmentStrategy(_Model):
    label: str
    arm_type: str | None = None  # EXPERIMENTAL, PLACEBO_COMPARATOR, ACTIVE_COMPARATOR, ...
    description: str | None = None
    components: list[TreatmentComponent] = Field(default_factory=list)
    is_comparator: bool = False
    grace_period: Duration | None = None  # window after time zero in which initiation counts
    grace_period_origin: Origin | None = None
    grace_period_evidence: str | None = None


class Outcome(_Model):
    name: str
    description: str | None = None
    time_frame: str | None = None
    primary: bool = False
    concept: ConceptRef | None = None
    horizon: Duration | None = None
    end_event: str | None = None  # "hospital discharge" in "28 days or discharge"
    end_rule: Literal["first", "last"] | None = None


class TimeZero(_Model):
    anchor_event: str  # e.g. "all eligibility criteria first met", "ICU admission"
    offset: Duration | None = None
    rationale: str | None = None
    origin: Origin = Origin.AGENT
    evidence: str | None = None


class FollowUp(_Model):
    start: str = "time_zero"
    horizon: Duration | None = None
    end_event: str | None = None
    end_rule: Literal["first", "last"] | None = None
    # No defaults: censoring depends on the outcome (death is never censored for all-cause
    # mortality) and is set by the Trialist or the Statistician.
    censoring_events: list[str] = Field(default_factory=list)
    competing_events: list[str] = Field(default_factory=list)
    origin: Origin = Origin.DEFAULT


class AssignmentProcedure(_Model):
    """How randomization is emulated: confounders to balance on and the balancing method."""

    method: Literal["auto", "psm", "iptw", "overlap_weights", "none"] = "auto"
    confounders: list[ConceptRef] = Field(default_factory=list)
    origin: Origin = Origin.DEFAULT


class AnalysisPlan(_Model):
    estimand: Estimand = Estimand.HAZARD_RATIO
    outcome_model: Literal["auto", "cox", "kaplan_meier", "parametric", "rsf", "doubly_robust"] = (
        "auto"
    )
    immortal_time_handling: Literal["auto", "clone_censor_weight", "landmark", "none"] = "auto"
    prespecified_subgroups: list[str] = Field(default_factory=list)
    sensitivity_analyses: list[str] = Field(default_factory=list)
    origin: Origin = Origin.DEFAULT


class TrialSource(_Model):
    registry: str = "ClinicalTrials.gov"
    nct_id: str | None = None
    title: str | None = None
    phase: list[str] = Field(default_factory=list)
    study_type: str | None = None
    overall_status: str | None = None
    allocation: str | None = None
    intervention_model: str | None = None
    masking: str | None = None
    enrollment: int | None = None
    enrollment_type: str | None = None  # ACTUAL or ESTIMATED
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
_SPECIFIED = {Origin.REGISTRY, Origin.DERIVED, Origin.AGENT, Origin.USER}


class TargetTrialProtocol(_Model):
    schema_version: int = SCHEMA_VERSION
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
    causal_contrast_origin: Origin = Origin.DEFAULT
    causal_contrast_evidence: str | None = None
    analysis_plan: AnalysisPlan = Field(default_factory=AnalysisPlan)

    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_v1(cls, data: Any) -> Any:
        # v1 files carry no schema_version and had registry-unaware defaults.
        if isinstance(data, dict) and "schema_version" not in data:
            data = {**data, "schema_version": 1}
        return data

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

    def element_status(self) -> dict[str, str]:
        """For each TTE element: "missing", or the ``Origin`` of its content."""
        return {
            "eligibility": Origin.REGISTRY if self.eligibility else "missing",
            "treatment_strategies": Origin.REGISTRY
            if len(self.treatment_strategies) >= 2
            else "missing",
            "assignment": self.assignment.origin,
            "time_zero": self.time_zero.origin if self.time_zero else "missing",
            "follow_up": self.follow_up.origin if self.follow_up.horizon else "missing",
            "outcomes": Origin.REGISTRY if any(o.primary for o in self.outcomes) else "missing",
            "causal_contrast": self.causal_contrast_origin,
            "analysis_plan": self.analysis_plan.origin,
        }

    def completeness(self) -> dict[str, bool]:
        """True where an element is actually specified. Defaults do not count (C3): a
        ``method: auto`` assignment is a placeholder, not a design decision."""
        return {k: v in _SPECIFIED for k, v in self.element_status().items()}
