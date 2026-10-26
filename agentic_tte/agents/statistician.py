"""Statistician: analysis frame + protocol -> effect estimates and diagnostics (PLAN.md §3).

Deterministic: the method comes from the protocol's analysis plan through a fixed menu, and
every number from ``agentic_tte.stats``. An LLM may later choose among the menu's options; it
never computes a number (PLAN.md §1).

Menu:
- primary: clone-censor-weight when the treatment strategy has a grace period (or the plan
  asks for it), otherwise IPTW (stabilized ATE weights, marginal Cox with robust SEs), or
  propensity-score matching when the plan says ``psm`` (an ATT, with pair-clustered SEs);
- always: the crude Cox HR, IPTW as a secondary estimate, covariate balance (SMD before and
  after weighting), the proportional-hazards test, IPTW-weighted Kaplan-Meier risks at the
  horizon and the E-value of the primary estimate.

Runs where the data are; only aggregates come out. Patients with a missing confounder are left
out (complete case), and the number is reported. No estimate is made when an arm has fewer than
11 patients or there are fewer than 11 events.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from agentic_tte.omop.gate import SMALL_CELL, suppress
from agentic_tte.protocol.models import TargetTrialProtocol
from agentic_tte.stats import (
    SMD_THRESHOLD,
    ccw_hr,
    cox_hr,
    evalue_hr,
    iptw_weights,
    match_nearest,
    ph_test,
    propensity_scores,
    risk_difference,
    smd,
)

_DAYS = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30.44, "year": 365.25}


class Estimate(BaseModel):
    method: str
    hr: float
    lower: float
    upper: float
    p: float
    n: int
    events: int


class AnalysisResult(BaseModel):
    primary_method: str | None = None
    estimates: list[Estimate] = Field(default_factory=list)
    n_frame: int
    n_analysed: int
    dropped_missing: int
    confounders: list[str]
    arms: dict[str, dict] = Field(default_factory=dict)  # suppressed counts
    smd_before: dict[str, float] = Field(default_factory=dict)
    smd_after: dict[str, float] = Field(default_factory=dict)
    risks: dict[str, float] = Field(default_factory=dict)
    ph_p: float | None = None
    evalue: dict[str, float] = Field(default_factory=dict)
    horizon_days: float | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def primary(self) -> Estimate | None:
        return next((e for e in self.estimates if e.method == self.primary_method), None)

    def to_markdown(self) -> str:
        lines = ["## Effect estimates", ""]
        if not self.estimates:
            lines += ["No estimate: " + "; ".join(self.notes), ""]
            return "\n".join(lines)
        lines += ["| Method | HR | 95% CI | p | N | Events |", "|---|---|---|---|---|---|"]
        for e in self.estimates:
            star = " (primary)" if e.method == self.primary_method else ""
            lines.append(f"| {e.method}{star} | {e.hr:.3f} | {e.lower:.3f}–{e.upper:.3f} | "
                         f"{e.p:.3g} | {suppress(e.n)} | {suppress(e.events)} |")  # fmt: skip
        lines.append("")
        if self.risks:
            lines.append(f"IPTW-weighted Kaplan-Meier risk at {self.horizon_days:g} days: "
                         f"treated {100 * self.risks['risk_treated']:.1f}%, untreated "
                         f"{100 * self.risks['risk_untreated']:.1f}% (difference "
                         f"{100 * self.risks['risk_difference']:+.1f} points).")
        if self.smd_after:
            worst = max(self.smd_after.values())
            lines.append(f"Balance: max SMD {max(self.smd_before.values()):.3f} before, "
                         f"{worst:.3f} after weighting "
                         f"({'balanced' if worst < SMD_THRESHOLD else 'IMBALANCED'}; "
                         f"threshold {SMD_THRESHOLD}).")
        if self.ph_p is not None:
            lines.append(f"Proportional hazards test (crude model): p = {self.ph_p:.3g}.")
        if self.evalue:
            ci = f", CI limit {self.evalue['evalue_ci']:.2f}" if "evalue_ci" in self.evalue else ""
            lines.append(f"E-value of the primary estimate: {self.evalue['evalue']:.2f}{ci}.")
        lines.append(f"Analysed {suppress(self.n_analysed)} of {suppress(self.n_frame)} "
                     f"({suppress(self.dropped_missing)} left out for a missing confounder).")
        lines += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def _est(hr, method: str) -> Estimate:
    return Estimate(method=method, hr=hr.hr, lower=hr.lower, upper=hr.upper, p=hr.p, n=hr.n,
                    events=hr.events)  # fmt: skip


def analyze(frame: pd.DataFrame, protocol: TargetTrialProtocol, *, confounders: list[str],
            horizon_days: float | None = None) -> AnalysisResult:  # fmt: skip
    """Estimate the treatment effect in an analysis frame (``treated``, ``treat_start``,
    ``time``, ``event`` and the confounder columns)."""
    d = frame.dropna(subset=confounders).reset_index(drop=True)
    res = AnalysisResult(n_frame=len(frame), n_analysed=len(d),
                         dropped_missing=len(frame) - len(d), confounders=confounders,
                         horizon_days=horizon_days)  # fmt: skip
    for label, g in (("treatment", d[d.treated == 1]), ("comparator", d[d.treated == 0])):
        res.arms[label] = {"patients": suppress(len(g)), "events": suppress(int(g.event.sum()))}
    n1, n0 = int(d.treated.sum()), int((1 - d.treated).sum())
    if min(n1, n0) < SMALL_CELL or d.event.sum() < SMALL_CELL:
        res.notes.append(f"fewer than {SMALL_CELL} patients in an arm or events in total")
        return res
    if not confounders:
        res.notes.append("no confounders: estimates are unadjusted")

    arm = next((a for a in protocol.treatment_strategies if not a.is_comparator), None)
    grace = arm.grace_period if arm else None
    grace_days = grace.amount * _DAYS[grace.unit] if grace else None
    plan = protocol.analysis_plan
    if plan.immortal_time_handling == "clone_censor_weight" or (
            plan.immortal_time_handling == "auto" and grace_days):  # fmt: skip
        primary = "clone-censor-weight"
    elif protocol.assignment.method == "psm":
        primary = "psm"
    else:
        primary = "iptw"
        if protocol.assignment.method == "overlap_weights":
            res.notes.append("overlap weights are not implemented yet: IPTW used instead")
    res.primary_method = primary

    res.estimates.append(_est(cox_hr(d), "crude"))
    if confounders:
        ps = propensity_scores(d, confounders, "treated")
        d["iptw"] = iptw_weights(ps, d["treated"])
        res.estimates.append(_est(cox_hr(d, weights="iptw"), "iptw"))
        res.smd_before = smd(d, confounders, "treated").round(4).to_dict()
        res.smd_after = smd(d, confounders, "treated", weights=d["iptw"]).round(4).to_dict()
        if horizon_days:
            res.risks = risk_difference(d, horizon_days, weights="iptw")
        if primary == "psm":
            m = match_nearest(d, ps, "treated")
            res.estimates.append(_est(cox_hr(m.data, cluster="pair_id"), "psm"))
            res.notes.append(f"PSM matched {m.n_matched_treated} of {m.n_treated} treated "
                             "(estimates the ATT)")
    elif primary in ("iptw", "psm"):
        res.primary_method = "crude"
    if primary == "clone-censor-weight":
        if confounders:
            res.estimates.append(_est(ccw_hr(d, grace=grace_days, covariates=confounders),
                                      "clone-censor-weight"))  # fmt: skip
            res.notes.append(f"clone-censor-weight with a {grace_days:g}-day grace period "
                             "(initiation within grace vs never)")
        else:
            res.primary_method = "crude"
            res.notes.append("clone-censor-weight needs confounders: crude estimate is primary")
    try:
        res.ph_p = ph_test(d)
    except Exception as e:  # noqa: BLE001 - a diagnostic, never fatal
        res.notes.append(f"PH test failed: {e}")
    p = res.primary
    res.evalue = evalue_hr(p.hr, p.lower, p.upper)
    if res.smd_after and max(res.smd_after.values()) >= SMD_THRESHOLD:
        res.notes.append("residual imbalance after weighting: interpret with caution")
    return res
