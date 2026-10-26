"""Attrition table and feasibility: who the compiled criteria keep, and whether that is enough.

``attrition(con, compiled)`` runs on the data side and returns aggregates only:
- the flow from persons in the database to eligible patients: index event, observation period
  and alive at time zero, then each compiled criterion in protocol order, with the number
  remaining and removed, and how many patients meet each criterion on its own (which criterion
  is binding);
- the criteria that were not applied (not compiled or not emulable), with their reasons;
- for the analysis frame: patients, events and person-time per arm, end-of-follow-up reasons
  and the pooled Kaplan-Meier risk at the horizon.

``feasibility(report, protocol, design_hr=...)`` turns that into the gate's last check
(PLAN.md §14): red when no eligible patients, an empty arm or no events; amber when counts are
small, events fall short of the Schoenfeld requirement for the design HR, there are fewer than
10 events per confounder, or criteria were left out; green otherwise.

Counts below 11 are suppressed in everything shown or sent on (``public()``, ``to_markdown``,
the gate check). The raw counts stay in the report object on the data side. Complementary
suppression (a suppressed cell recovered from two neighbouring totals) is not done yet.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field

from agentic_tte.omop.compiler import CompiledCohort
from agentic_tte.omop.gate import SMALL_CELL, GateCheck, GateStatus, suppress
from agentic_tte.protocol.models import TargetTrialProtocol
from agentic_tte.stats.design import schoenfeld_events


class AttritionStep(BaseModel):
    label: str
    criterion: str | None = None  # criterion id, or None for the base steps
    kind: str | None = None
    remaining: int
    removed: int = 0
    met_alone: int | None = None  # among index persons in observation, ignoring other criteria


class ArmSummary(BaseModel):
    arm: str
    patients: int
    events: int
    person_days: float


class AttritionReport(BaseModel):
    protocol_id: str
    protocol_hash: str
    steps: list[AttritionStep]
    not_applied: list[dict] = Field(default_factory=list)
    eligible: int
    arms: list[ArmSummary] | None = None  # None when the frame was not compiled
    end_reasons: dict[str, int] = Field(default_factory=dict)
    horizon_days: float | None = None
    risk_at_horizon: float | None = None  # pooled Kaplan-Meier risk
    confounders: int = 0

    @property
    def events(self) -> int | None:
        return sum(a.events for a in self.arms) if self.arms is not None else None

    def public(self) -> dict:
        """The report with every count below 11 suppressed: what may leave the data side."""
        d = self.model_dump()
        for s in d["steps"]:
            for k in ("remaining", "removed", "met_alone"):
                if s[k] is not None:
                    s[k] = suppress(s[k])
        d["eligible"] = suppress(d["eligible"])
        for a in d["arms"] or []:
            a["patients"], a["events"] = suppress(a["patients"]), suppress(a["events"])
            a["person_days"] = round(a["person_days"]) if a["patients"] != f"<{SMALL_CELL}" \
                else None  # fmt: skip
        d["end_reasons"] = {k: suppress(v) for k, v in d["end_reasons"].items()}
        return d

    def to_markdown(self) -> str:
        p = self.public()
        lines = [f"# Attrition: {self.protocol_id}", "",
                 "| Step | Remaining | Removed | Meeting it alone |", "|---|---|---|---|"]
        for s in p["steps"]:
            alone = "–" if s["met_alone"] is None else s["met_alone"]
            removed = "–" if s["criterion"] is None and s["removed"] == 0 else s["removed"]
            lines.append(f"| {s['label']} | {s['remaining']} | {removed} | {alone} |")
        if p["not_applied"]:
            lines += ["", "Not applied (the cohort is wider than the protocol by these):", ""]
            lines += [f"- {c['id']} ({c['status']}): {'; '.join(c['reasons'])}"
                      for c in p["not_applied"]]  # fmt: skip
        if p["arms"] is not None:
            lines += ["", "| Arm | Patients | Events | Person-days |", "|---|---|---|---|"]
            for a in p["arms"]:
                pd_ = "–" if a["person_days"] is None else f"{a['person_days']:,}"
                lines.append(f"| {a['arm']} | {a['patients']} | {a['events']} | {pd_} |")
            reasons = ", ".join(f"{k} {v}" for k, v in p["end_reasons"].items())
            lines += ["", f"End of follow-up: {reasons}."]
            if self.risk_at_horizon is not None:
                lines.append(f"Pooled Kaplan-Meier risk at {self.horizon_days:g} days: "
                             f"{100 * self.risk_at_horizon:.1f}%.")
        else:
            lines += ["", "Analysis frame not compiled: events are not known."]
        return "\n".join(lines) + "\n"


def _km_risk(time: np.ndarray, event: np.ndarray, horizon: float) -> float | None:
    if len(time) == 0:
        return None
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(time, event)
    return float(1 - km.survival_function_at_times(horizon).iloc[0])


def attrition(con, compiled: CompiledCohort) -> AttritionReport:
    """Run the compiled queries and return the attrition aggregates."""
    applied = [c for c in compiled.criteria if c.column]
    conds = [c.column if c.kind == "inclusion" else f"NOT {c.column}" for c in applied]
    prefix = [" AND ".join(conds[: i + 1]) for i in range(len(conds))]
    sel = ["(SELECT count(*) FROM person)", "(SELECT count(*) FROM idx)", "count(*)"]
    sel += [f"count(*) FILTER (WHERE {p})" for p in prefix]
    sel += [f"count(*) FILTER (WHERE {c.column})" for c in applied]
    row = con.execute(f"WITH {compiled.ctes}\nSELECT {', '.join(sel)} FROM cohort").fetchone()
    persons, index, base = row[:3]
    after = row[3:3 + len(applied)]
    alone = row[3 + len(applied):]
    steps = [AttritionStep(label="Persons in the database", remaining=persons),
             AttritionStep(label="With the index (time-zero) event", remaining=index,
                           removed=persons - index),
             AttritionStep(label="Index inside an observation period, alive at time zero",
                           remaining=base, removed=index - base)]  # fmt: skip
    prev = base
    for c, n, a in zip(applied, after, alone, strict=True):
        verb = "meets" if c.kind == "inclusion" else "does not meet"
        steps.append(AttritionStep(label=f"{c.id} ({c.kind}, {verb}): {c.text[:60]}",
                                   criterion=c.id, kind=c.kind, remaining=n,
                                   removed=prev - n, met_alone=a))  # fmt: skip
        prev = n
    rep = AttritionReport(
        protocol_id=compiled.protocol_id, protocol_hash=compiled.protocol_hash, steps=steps,
        not_applied=[{"id": c.id, "status": c.status, "reasons": c.reasons}
                     for c in compiled.criteria if not c.column],
        eligible=prev, horizon_days=compiled.horizon_days,
        confounders=max(len(compiled.frame_columns) - 6, 0))  # fmt: skip
    if compiled.frame_sql is None:
        return rep
    frame = con.execute(f"SELECT treated, time, event, end_reason FROM ({compiled.frame_sql})"
                        ).df()  # fmt: skip
    rep.arms = []
    for treated, label in ((1, "treatment"), (0, "comparator")):
        g = frame[frame.treated == treated]
        rep.arms.append(ArmSummary(arm=label, patients=len(g), events=int(g.event.sum()),
                                   person_days=float(g.time.sum())))  # fmt: skip
    rep.end_reasons = {k: int(v) for k, v in frame.end_reason.value_counts().sort_index().items()}
    if compiled.horizon_days:
        rep.risk_at_horizon = _km_risk(frame.time.to_numpy(), frame.event.to_numpy(),
                                       compiled.horizon_days)  # fmt: skip
    return rep


def feasibility(rep: AttritionReport, protocol: TargetTrialProtocol | None = None, *,
                design_hr: float | None = None, alpha: float = 0.05,
                power: float = 0.8) -> GateCheck:  # fmt: skip
    """The gate's feasibility check from an attrition report (aggregates only)."""
    values: dict = {"eligible": suppress(rep.eligible), "design_hr": design_hr}
    notes: list[str] = []
    status = GateStatus.GREEN

    def worse(s: GateStatus) -> None:
        nonlocal status
        order = [GateStatus.GREEN, GateStatus.AMBER, GateStatus.RED]
        if order.index(s) > order.index(status):
            status = s

    if rep.eligible == 0:
        return GateCheck(id="feasibility", title="Feasibility", status=GateStatus.RED,
                         detail="no eligible patients", values=values)
    if rep.eligible < SMALL_CELL:
        worse(GateStatus.AMBER)
    notes.append(f"{suppress(rep.eligible)} eligible")
    enrolled = protocol.source.enrollment if protocol else None
    if enrolled:
        notes.append(f"the trial enrolled {enrolled}")
        values["trial_enrollment"] = enrolled
    if rep.not_applied:
        worse(GateStatus.AMBER)
        k = len(rep.not_applied)
        notes.append(f"{k} {'criterion' if k == 1 else 'criteria'} not applied, so the cohort "
                     "is wider than the protocol")
    if rep.arms is None:
        worse(GateStatus.AMBER)
        notes.append("events unknown: the analysis frame was not compiled")
        return GateCheck(id="feasibility", title="Feasibility", status=status,
                         detail="; ".join(notes), values=values)
    arms = {a.arm: a for a in rep.arms}
    values["arms"] = {k: {"patients": suppress(a.patients), "events": suppress(a.events)}
                      for k, a in arms.items()}  # fmt: skip
    notes.append(", ".join(f"{k}: {suppress(a.patients)} patients, {suppress(a.events)} events"
                           for k, a in arms.items()))
    if any(a.patients == 0 for a in rep.arms):
        worse(GateStatus.RED)
        notes.append("an arm is empty")
    if rep.events == 0:
        worse(GateStatus.RED)
        notes.append("no outcome events")
    elif any(a.events < SMALL_CELL for a in rep.arms):
        worse(GateStatus.AMBER)
        notes.append("fewer than 11 events in an arm")
    if rep.confounders and rep.events is not None and rep.events < 10 * rep.confounders:
        worse(GateStatus.AMBER)
        notes.append(f"fewer than 10 events per confounder ({rep.confounders} confounders)")
    if design_hr is not None and rep.events:
        share = arms["treatment"].patients / rep.eligible
        if 0 < share < 1:
            need = schoenfeld_events(design_hr, alpha=alpha, power=power, allocation=share)
            values["events_required"] = need
            notes.append(f"{need} events needed for HR {design_hr:g} at {100 * power:.0f}% power "
                         f"(alpha {alpha:g}, {100 * share:.0f}% treated)")
            if rep.events < need:
                worse(GateStatus.AMBER)
                notes.append("not enough events for that HR")
            if rep.risk_at_horizon:
                values["n_required"] = math.ceil(need / rep.risk_at_horizon)
    elif design_hr is None:
        notes.append("no design HR given: power not assessed")
    return GateCheck(id="feasibility", title="Feasibility", status=status,
                     detail="; ".join(notes), values=values)
