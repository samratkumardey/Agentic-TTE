"""Protocol Auditor: checks a target-trial design for the errors that break emulations
(PLAN.md §3, §17; eval/SCORING.md Phase 3 changelog item 11).

Deterministic rules, one per fault type of the E1 fault catalogue (item 11c). Each reads the
protocol, the compile spec and, where the question is about the data, aggregate shares computed
on the data side (``data_summary``); no patient row leaves it.

| Check | Catches | Reads |
|---|---|---|
| A-a | immortal time: treatment after time zero, ignored by the analysis | plan, grace, data |
| A-b | look-ahead: eligibility uses records after time zero | criteria, compile report |
| A-c | time zero after treatment start | data |
| A-d | arms swapped: comparator flags against the registry arm types | strategies |
| A-e | outcome horizon different from the trial's time frame | outcomes |
| A-f | prevalent users: no exclusion of prior use of the treatment | criteria |
| A-g | death censored for a mortality outcome | follow-up, outcome spec |
| A-h | a confounder that is the treatment or the outcome (post-baseline information) | spec |
| A-i | an initiation window stated by the trial but no grace period | strategies |
| A-j | a concept set that is not the reviewed one, or names another concept | spec |

A finding is ``fail`` (the design is wrong as it stands) or ``warn`` (a check could not be
made, or needs a person). The Auditor never edits the protocol: a fail goes back to design
review.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from agentic_tte.omop.compiler import (
    EVENT_TABLE,
    CompiledCohort,
    CompileSpec,
    _death_only,
    _in_set,
    _is_placebo,
    _norm,
    _time,
)
from agentic_tte.protocol.models import (
    Domain,
    TargetTrialProtocol,
    TemporalRelation,
    TreatmentStrategy,
)
from agentic_tte.protocol.normalize import durations_in, parse_time_frame

CHECKS = {
    "A-a": "immortal time",
    "A-b": "look-ahead in eligibility",
    "A-c": "time zero after treatment start",
    "A-d": "treatment and comparator arms",
    "A-e": "outcome horizon",
    "A-f": "new users (prior treatment excluded)",
    "A-g": "death not censored for a mortality outcome",
    "A-h": "confounders measured before treatment",
    "A-i": "grace period for a stated initiation window",
    "A-j": "reviewed concept sets",
}
LATE_START_SHARE = 0.05  # A-a: treated who start after time zero
BEFORE_T0_SHARE = 0.05  # A-c: indexed patients with the treatment before time zero
BEFORE_T0_DAYS = 7  # A-c: how far back "just before time zero" looks
_DAYS = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30.44, "year": 365.25}
_AFTER = (TemporalRelation.X_AFTER_Y, TemporalRelation.X_WITHIN_T_AFTER_Y)
_T0 = {"time zero", "time 0", "randomization", "randomisation", "enrollment", "enrolment",
       "baseline", "study entry", "inclusion", "index"}  # fmt: skip
_NO_TREATMENT_ARM = ("PLACEBO", "SHAM", "NO_INTERVENTION")


class Finding(BaseModel):
    check: str
    severity: Literal["fail", "warn"]
    element: str
    message: str


class AuditReport(BaseModel):
    protocol_id: str
    checks_run: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)

    @property
    def failed(self) -> list[str]:
        return sorted({f.check for f in self.findings if f.severity == "fail"})

    @property
    def passed(self) -> bool:
        return not self.failed

    def to_markdown(self) -> str:
        lines = [f"## Protocol audit: {self.protocol_id}", "",
                 "**Passed**" if self.passed else f"**Failed:** {', '.join(self.failed)}", "",
                 "| Check | Result |", "|---|---|"]  # fmt: skip
        for c in self.checks_run:
            fs = [f for f in self.findings if f.check == c]
            res = "; ".join(f"{f.severity.upper()}: {f.message}" for f in fs) or "ok"
            lines.append(f"| {c} {CHECKS[c]} | {res} |")
        return "\n".join(lines) + "\n"


def _days(d) -> float:
    return d.amount * _DAYS[d.unit]


def _arms(p: TargetTrialProtocol) -> tuple[list[TreatmentStrategy], list[TreatmentStrategy]]:
    return ([a for a in p.treatment_strategies if not a.is_comparator],
            [a for a in p.treatment_strategies if a.is_comparator])  # fmt: skip


def _treatment_sets(p: TargetTrialProtocol, spec: CompileSpec) -> list[tuple[Domain, list[int]]]:
    arms, _ = _arms(p)
    out = []
    for arm in arms:
        for comp in arm.components:
            ids = spec.concept_sets.get(f"T:{arm.label}:{comp.concept.name}") or (
                [comp.concept.omop_concept_id] if comp.concept.omop_concept_id else [])
            if ids:
                out.append((Domain(comp.concept.domain), ids))
    return out


def _is_t0(anchor: str, p: TargetTrialProtocol) -> bool:
    n = _norm(anchor)
    tz = _norm(p.time_zero.anchor_event) if p.time_zero else ""
    return n in _T0 or n == tz or n.replace(" ", "_") == "time_zero"


def data_summary(con, protocol: TargetTrialProtocol, spec: CompileSpec,
                 compiled: CompiledCohort) -> dict[str, float | int | None]:  # fmt: skip
    """Aggregate shares the data checks need, computed where the data are:
    - ``late_start_share``: treated patients (analysis frame) whose treatment starts after
      time zero;
    - ``before_t0_share``: indexed patients (before eligibility) with a treatment record in the
      ``BEFORE_T0_DAYS`` days before time zero."""
    out: dict[str, float | int | None] = {"late_start_share": None, "before_t0_share": None}
    if compiled.frame_sql:
        f = con.execute(f"SELECT count(*) FILTER (WHERE treated = 1), count(*) FILTER "
                        f"(WHERE treated = 1 AND treat_start > 1e-6) FROM ({compiled.frame_sql})"
                        ).fetchone()  # fmt: skip
        out["treated"] = f[0]
        out["late_start_share"] = f[1] / f[0] if f[0] else None
    sets = _treatment_sets(protocol, spec)
    if sets:
        exists = " OR ".join(
            f"EXISTS (SELECT 1 FROM {EVENT_TABLE[dom][0]} r WHERE r.person_id = b.person_id "
            f"AND {_in_set(f'r.{EVENT_TABLE[dom][1]}', ids, spec.descendants)} "
            f"AND {_time('r', dom)} < b.t0 AND {_time('r', dom)} >= b.t0 - "
            f"INTERVAL '{BEFORE_T0_DAYS} days')" for dom, ids in sets)  # fmt: skip
        n, k = con.execute(f"WITH {compiled.ctes} SELECT count(*), count(*) FILTER (WHERE "
                           f"{exists}) FROM base b").fetchone()  # fmt: skip
        out["indexed"] = n
        out["before_t0_share"] = k / n if n else None
    return out


def audit(protocol: TargetTrialProtocol, spec: CompileSpec, *,
          compiled: CompiledCohort | None = None, summary: dict | None = None,
          approved_sets: dict[str, list[int]] | None = None,
          concept_names: dict[int, str] | None = None,
          surrogate_keys: set[str] | None = None) -> AuditReport:  # fmt: skip
    """Run every check. ``approved_sets`` are the reviewed concept sets
    (``ConceptSetBook.gate_sets()``); ``concept_names`` maps concept IDs to names (the
    vocabulary), for the name check of A-j when no reviewed sets are given; sets a reviewer
    approved as surrogates (``surrogate_keys``) are exempt from that name check."""
    p = protocol
    rep = AuditReport(protocol_id=p.protocol_id, checks_run=list(CHECKS))
    add = rep.findings.append
    arms, comparators = _arms(p)
    plan = p.analysis_plan

    # A-a immortal time
    late = (summary or {}).get("late_start_share")
    grace = arms[0].grace_period if len(arms) == 1 else None
    if late is None:
        add(Finding(check="A-a", severity="warn", element="analysis_plan",
                    message="no analysis frame: the start of treatment was not checked"))
    elif late > LATE_START_SHARE:
        if plan.immortal_time_handling == "none":
            add(Finding(check="A-a", severity="fail", element="analysis_plan",
                        message=f"treatment starts after time zero for {100 * late:.0f}% of "
                                "the treated, and immortal-time handling is 'none'"))
        elif grace is None and plan.immortal_time_handling != "clone_censor_weight":
            add(Finding(check="A-a", severity="fail", element="treatment_strategies",
                        message=f"treatment starts after time zero for {100 * late:.0f}% of "
                                "the treated, with no grace period: patients would be "
                                "classified by future treatment"))

    # A-b look-ahead
    for c in p.eligibility:
        for a in c.atoms:
            t = a.temporal
            if t and TemporalRelation(t.relation) in _AFTER and _is_t0(t.anchor, p):
                add(Finding(check="A-b", severity="fail", element="eligibility",
                            message=f"{c.id} uses records after time zero ({c.text[:60]})"))
                break
    for cc in (compiled.criteria if compiled else []):
        if any("look-ahead" in r for r in cc.reasons) and not any(
                f.check == "A-b" and f.message.startswith(cc.id) for f in rep.findings):
            add(Finding(check="A-b", severity="fail", element="eligibility",
                        message=f"{cc.id}: the compiler found look-ahead"))

    # A-c time zero after treatment start
    before = (summary or {}).get("before_t0_share")
    if before is None:
        add(Finding(check="A-c", severity="warn", element="time_zero",
                    message="no treatment concept set or data: not checked"))
    elif before > BEFORE_T0_SHARE:
        add(Finding(check="A-c", severity="fail", element="time_zero",
                    message=f"{100 * before:.0f}% of indexed patients have the treatment in "
                            f"the {BEFORE_T0_DAYS} days before time zero: time zero is after "
                            "treatment start"))

    # A-d arms
    if len(arms) != 1 or len(comparators) != 1:
        add(Finding(check="A-d", severity="fail", element="treatment_strategies",
                    message=f"{len(arms)} treatment and {len(comparators)} comparator arms; "
                            "exactly one of each is needed"))
    for a in p.treatment_strategies:
        kind = (a.arm_type or "").upper()
        names = [c.concept.name for c in a.components]
        if any(k in kind for k in _NO_TREATMENT_ARM) and not a.is_comparator:
            add(Finding(check="A-d", severity="fail", element="treatment_strategies",
                        message=f"'{a.label}' is a {kind} arm in the registry but is the "
                                "treatment arm here"))
        if "EXPERIMENTAL" in kind and a.is_comparator:
            add(Finding(check="A-d", severity="fail", element="treatment_strategies",
                        message=f"'{a.label}' is the EXPERIMENTAL arm in the registry but is "
                                "the comparator here"))
        if not a.is_comparator and names and all(_is_placebo(n) for n in names):
            add(Finding(check="A-d", severity="fail", element="treatment_strategies",
                        message=f"the treatment arm '{a.label}' has only placebo components"))

    # A-e horizon
    primary = next((o for o in p.outcomes if o.primary), None)
    if primary is None:
        add(Finding(check="A-e", severity="fail", element="outcomes",
                    message="no primary outcome"))
    else:
        stated, _, _ = parse_time_frame(primary.time_frame)
        h = primary.horizon or p.follow_up.horizon
        if stated is None or h is None:
            add(Finding(check="A-e", severity="warn", element="outcomes",
                        message="the trial's time frame or the horizon is missing: not compared"))
        elif abs(_days(stated) - _days(h)) > 0.01 * _days(stated):
            add(Finding(check="A-e", severity="fail", element="outcomes",
                        message=f"horizon {h.amount:g} {h.unit}(s) but the trial's time frame "
                                f"is {stated.amount:g} {stated.unit}(s)"))

    # A-f new users
    treat_names = {_norm(c.concept.name) for a in arms for c in a.components}
    new_user = any(
        c.kind == "exclusion" and any(
            _norm(r.name) in treat_names and (
                a.temporal is None or TemporalRelation(a.temporal.relation) not in _AFTER)
            for a in c.atoms for r in a.concepts)
        for c in p.eligibility)  # fmt: skip
    if treat_names and not new_user:
        add(Finding(check="A-f", severity="fail", element="eligibility",
                    message="no exclusion of prior use of the treatment before time zero: "
                            "prevalent users can enter"))

    # A-g death censoring
    if primary is not None:
        death_outcome = _death_only(primary.name) or bool(spec.outcome and spec.outcome.death)
        if death_outcome and any("death" in _norm(e) or "died" in _norm(e)
                                 for e in p.follow_up.censoring_events):  # fmt: skip
            add(Finding(check="A-g", severity="fail", element="follow_up",
                        message="death is a censoring event for an outcome that includes "
                                "death"))
        if spec.outcome is not None and not spec.outcome.death and _death_only(primary.name):
            add(Finding(check="A-g", severity="fail", element="outcomes",
                        message="the outcome spec excludes death for a mortality outcome"))

    # A-h confounders
    treat_ids = {i for _, ids in _treatment_sets(p, spec) for i in ids}
    events = spec.outcome.events if spec.outcome else []
    outcome_ids = {i for ev in events for i in ev.concept_ids}
    if primary is not None and primary.concept is not None:
        outcome_ids |= set(spec.concept_sets.get(f"O:{primary.name}", []))
    for ref in p.assignment.confounders:
        ids = set(spec.concept_sets.get(f"C:{ref.name}", []))
        what = ("the treatment" if (ids & treat_ids or _norm(ref.name) in treat_names)
                else "the outcome" if ids & outcome_ids else None)  # fmt: skip
        if what:
            add(Finding(check="A-h", severity="fail", element="assignment",
                        message=f"confounder '{ref.name}' is {what}: it carries information "
                                "from after treatment starts"))

    # A-i grace period for a stated initiation window
    for a in arms:
        text = " ".join(x for x in (a.grace_period_evidence, a.description) if x)
        if a.grace_period is None and durations_in(text) and re.search(
                r"\b(within|after|before|by|initiat|start)", text, re.I):  # fmt: skip
            add(Finding(check="A-i", severity="fail", element="treatment_strategies",
                        message=f"'{a.label}' states an initiation window ({text[:60]}) but "
                                "has no grace period"))

    # A-j concept sets
    if approved_sets is not None:
        for key, ids in spec.concept_sets.items():
            ok = approved_sets.get(key)
            if ok is None:
                add(Finding(check="A-j", severity="fail", element="concepts",
                            message=f"'{key}' has no reviewed concept set"))
            elif sorted(set(ok)) != sorted(set(ids)):
                add(Finding(check="A-j", severity="fail", element="concepts",
                            message=f"'{key}' differs from its reviewed concept set"))
    elif concept_names is not None:
        for a in arms:
            for comp in a.components:
                key = f"T:{a.label}:{comp.concept.name}"
                if key in (surrogate_keys or set()):
                    continue  # a reviewed surrogate is named differently on purpose
                ids = spec.concept_sets.get(key, [])
                want = set(_norm(comp.concept.name).split())
                bad = [i for i in ids if not want & set(_norm(concept_names.get(i, "")).split())]
                if bad:
                    add(Finding(check="A-j", severity="fail", element="concepts",
                                message=f"treatment '{comp.concept.name}': concepts {bad} "
                                        "have other names"))
    else:
        add(Finding(check="A-j", severity="warn", element="concepts",
                    message="no reviewed sets or vocabulary given: not checked"))
    return rep
