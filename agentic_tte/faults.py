"""Fault injection for E1 (eval/SCORING.md, Phase 3 changelog item 11c).

Each fault takes a correct design (protocol + compile spec) and returns a copy with one known
error planted, so the ground truth of every test is known by construction. The Protocol Auditor
(``agentic_tte.agents.auditor``) should catch each one; unmodified designs measure its false
alarms. Faults never touch data.

| Fault | Planted error | Auditor check |
|---|---|---|
| F-a | immortal time: immortal-time handling set to "none" with treatment after time zero | A-a |
| F-b | look-ahead: an eligibility atom moved to the 24 hours after time zero | A-b |
| F-c | time zero moved 2 days later, after treatment start | A-c |
| F-d | treatment and comparator arms swapped | A-d |
| F-e | outcome horizon changed (28 <-> 90 days) | A-e |
| F-f | the exclusion of prior treatment removed (prevalent users) | A-f |
| F-g | death made a censoring event for a mortality outcome | A-g |
| F-h | the treatment added as a confounder (post-baseline information) | A-h |
| F-i | the grace period removed though the trial states an initiation window | A-i |
| F-j | the treatment concept set replaced by another concept | A-j |

F-h: the compiler reads every confounder at or before time zero by construction, so "a
confounder measured after time zero" cannot be expressed in a compiled design; the fault plants
the nearest expressible error, a confounder that is the treatment itself (changelog item 14).
"""

from __future__ import annotations

from agentic_tte.omop.compiler import CompileSpec, _death_only, _norm
from agentic_tte.omop.gate import concept_key
from agentic_tte.protocol.models import (
    ConceptRef,
    Criterion,
    CriterionAtom,
    Domain,
    Duration,
    TargetTrialProtocol,
    TemporalConstraint,
)

FAULTS = {
    "F-a": ("immortal time", "A-a"),
    "F-b": ("look-ahead in eligibility", "A-b"),
    "F-c": ("time zero after treatment start", "A-c"),
    "F-d": ("arms swapped", "A-d"),
    "F-e": ("outcome horizon changed", "A-e"),
    "F-f": ("prevalent users", "A-f"),
    "F-g": ("death censored", "A-g"),
    "F-h": ("treatment as a confounder", "A-h"),
    "F-i": ("grace period removed", "A-i"),
    "F-j": ("wrong concept set", "A-j"),
}


class NotApplicable(ValueError):
    """The design has nothing this fault can change (reported, not counted)."""


def _treatment_arm(p: TargetTrialProtocol):
    arms = [a for a in p.treatment_strategies if not a.is_comparator]
    if len(arms) != 1:
        raise NotApplicable("needs exactly one treatment arm")
    return arms[0]


Design = tuple[TargetTrialProtocol, CompileSpec, str]


def inject(fault: str, protocol: TargetTrialProtocol, spec: CompileSpec, *,
           wrong_concept: int | None = None) -> Design:
    """Return (protocol, spec, description) with ``fault`` planted. ``wrong_concept``: the
    concept F-j puts in place of the treatment (another drug or class)."""
    p = protocol.model_copy(deep=True)
    s = spec.model_copy(deep=True)
    if fault == "F-a":
        arm = _treatment_arm(p)
        if arm.grace_period is None:
            raise NotApplicable("no grace period: treatment is meant to start at time zero")
        p.analysis_plan.immortal_time_handling = "none"
        return p, s, "immortal-time handling set to 'none'"
    if fault == "F-b":
        crit = next((c for c in p.eligibility if c.kind == "inclusion" and c.atoms), None)
        if crit is None:
            raise NotApplicable("no inclusion criterion with atoms")
        crit.atoms[0].temporal = TemporalConstraint(
            relation="x_within_t_after_y", window=Duration(amount=24, unit="hour"),
            anchor="time_zero", text="within 24 hours after time zero")
        return p, s, f"{crit.id} now reads the 24 hours after time zero"
    if fault == "F-c":
        if p.time_zero is None:
            raise NotApplicable("no time zero")
        p.time_zero.offset = Duration(amount=2, unit="day")
        return p, s, "time zero moved 2 days later"
    if fault == "F-d":
        if len(p.treatment_strategies) != 2:
            raise NotApplicable("needs two arms")
        for a in p.treatment_strategies:
            a.is_comparator = not a.is_comparator
        return p, s, "treatment and comparator flags swapped"
    if fault == "F-e":
        o = next((o for o in p.outcomes if o.primary), None)
        if o is None or o.horizon is None:
            raise NotApplicable("no primary outcome horizon")
        new = 90 if o.horizon.amount != 90 else 28
        o.horizon = Duration(amount=new, unit="day")
        return p, s, f"primary outcome horizon set to {new} days"
    if fault == "F-f":
        names = {_norm(c.concept.name) for c in _treatment_arm(p).components}
        keep = [c for c in p.eligibility if not (c.kind == "exclusion" and any(
            _norm(r.name) in names for r in c.concepts))]  # fmt: skip
        if len(keep) == len(p.eligibility):
            raise NotApplicable("no exclusion of prior treatment to remove")
        p.eligibility = keep
        return p, s, "exclusion of prior treatment removed"
    if fault == "F-g":
        o = next((o for o in p.outcomes if o.primary), None)
        if o is None or not (_death_only(o.name) or (s.outcome and s.outcome.death)):
            raise NotApplicable("the primary outcome does not include death")
        p.follow_up.censoring_events = [*p.follow_up.censoring_events, "death"]
        return p, s, "death added to the censoring events"
    if fault == "F-h":
        arm = _treatment_arm(p)
        comp = arm.components[0]
        ids = s.concept_sets.get(f"T:{arm.label}:{comp.concept.name}")
        if not ids:
            raise NotApplicable("the treatment has no concept set")
        name = f"{comp.concept.name} exposure"
        p.assignment.confounders.append(ConceptRef(name=name, domain=comp.concept.domain))
        s.concept_sets[f"C:{name}"] = list(ids)
        return p, s, f"the treatment added as confounder '{name}'"
    if fault == "F-i":
        arm = _treatment_arm(p)
        if arm.grace_period is None:
            raise NotApplicable("no grace period to remove")
        g = arm.grace_period
        arm.grace_period_evidence = arm.grace_period_evidence or \
            f"initiation within {g.amount:g} {g.unit}s of time zero"  # fmt: skip
        arm.grace_period = None
        return p, s, "grace period removed (the trial's initiation window is kept)"
    if fault == "F-j":
        arm = _treatment_arm(p)
        comp = arm.components[0]
        key = f"T:{arm.label}:{comp.concept.name}"
        if key not in s.concept_sets or wrong_concept is None:
            raise NotApplicable("no treatment concept set, or no wrong concept given")
        s.concept_sets[key] = [wrong_concept]
        return p, s, f"treatment concept set replaced by concept {wrong_concept}"
    raise KeyError(fault)


def new_user_exclusion(p: TargetTrialProtocol, cid: str = "E-new-user") -> Criterion:
    """The exclusion a new-user design needs: any record of the treatment before time zero.
    Concept set key: ``concept_key(criterion, name)``."""
    arm = _treatment_arm(p)
    comp = arm.components[0]
    return Criterion(id=cid, kind="exclusion",
                     text=f"{comp.concept.name} before time zero (new users only)",
                     atoms=[CriterionAtom(concepts=[ConceptRef(
                         name=comp.concept.name, domain=Domain(comp.concept.domain))],
                         temporal=TemporalConstraint(relation="x_before_y",
                                                     anchor="time_zero"))])  # fmt: skip


__all__ = ["FAULTS", "NotApplicable", "concept_key", "inject", "new_user_exclusion"]
