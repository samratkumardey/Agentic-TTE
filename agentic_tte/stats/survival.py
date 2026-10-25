"""Survival analysis: weighted Cox models, Kaplan-Meier risks, PH checks, subgroups.

Ported from EmulatRx ``emulation_utils.py`` (``calculate_hazard_ratios``, ``run_cox_model``,
``run_kaplan_meier``; MIT, TrialLab). Changes: the default Cox model is unpenalized and
marginal (treatment only, after weighting or matching), with robust (sandwich) SEs and
optional clustering on matched pairs. Upstream fitted ``CoxPHFitter(penalizer=0.1)`` with all
covariates, a ridge penalty that shrinks the treatment effect toward HR 1; ``penalizer`` and
``covariates`` reproduce that for comparison. Risks come from (weighted) Kaplan-Meier at the
horizon instead of crude outcome incidence, which ignores censoring.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import StatisticalWarning
from lifelines.statistics import proportional_hazard_test


@dataclass
class HazardRatio:
    hr: float
    lower: float
    upper: float
    p: float
    n: int
    events: int
    method: str = ""

    def covers(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def as_dict(self) -> dict:
        return asdict(self)


def cox_hr(df: pd.DataFrame, *, time: str = "time", event: str = "event",
           treatment: str = "treated", weights: str | None = None,
           covariates: list[str] | None = None, cluster: str | None = None,
           penalizer: float = 0.0, method: str = "") -> HazardRatio:  # fmt: skip
    cols = [time, event, treatment, *(covariates or [])]
    cols += [c for c in (weights, cluster) if c]
    data = df[cols].dropna()
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(data, duration_col=time, event_col=event, weights_col=weights,
            cluster_col=cluster, robust=weights is not None or cluster is not None)  # fmt: skip
    row = cph.summary.loc[treatment]
    return HazardRatio(
        hr=float(row["exp(coef)"]),
        lower=float(row["exp(coef) lower 95%"]),
        upper=float(row["exp(coef) upper 95%"]),
        p=float(row["p"]),
        n=len(data),
        events=int(data[event].sum()),
        method=method,
    )


def km_risk(df: pd.DataFrame, horizon: float, *, time: str = "time", event: str = "event",
            weights: str | None = None) -> float:  # fmt: skip
    """Cumulative incidence 1 - S(horizon) from a (weighted) Kaplan-Meier fit. Point estimate
    only: with IPTW weights the KM variance is not valid, so CIs need the bootstrap."""
    kmf = KaplanMeierFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StatisticalWarning)  # non-integer weights: expected
        kmf.fit(df[time], df[event], weights=None if weights is None else df[weights])
    return float(1 - kmf.survival_function_at_times(horizon).iloc[0])


def risk_difference(df: pd.DataFrame, horizon: float, *, treatment: str = "treated",
                    time: str = "time", event: str = "event",
                    weights: str | None = None) -> dict[str, float]:  # fmt: skip
    r1 = km_risk(df[df[treatment] == 1], horizon, time=time, event=event, weights=weights)
    r0 = km_risk(df[df[treatment] == 0], horizon, time=time, event=event, weights=weights)
    return {"risk_treated": r1, "risk_untreated": r0, "risk_difference": r1 - r0}


def ph_test(df: pd.DataFrame, *, time: str = "time", event: str = "event",
            treatment: str = "treated", covariates: list[str] | None = None) -> float:  # fmt: skip
    """p-value of the Schoenfeld-residual test of proportional hazards for the treatment
    (rank time transform)."""
    data = df[[time, event, treatment, *(covariates or [])]].dropna()
    cph = CoxPHFitter().fit(data, time, event)
    res = proportional_hazard_test(cph, data, time_transform="rank")
    return float(res.summary.loc[treatment, "p"])


def subgroup_hrs(df: pd.DataFrame, mask: pd.Series, *, labels: tuple[str, str] = ("in", "out"),
                 **cox_kwargs) -> dict[str, HazardRatio]:  # fmt: skip
    """HR within a subgroup and its complement (e.g. SOFA < 8 vs >= 8). Exploratory unless the
    subgroup was pre-specified in the protocol (eval/SCORING.md; PLAN.md §1)."""
    return {labels[0]: cox_hr(df[mask], **cox_kwargs), labels[1]: cox_hr(df[~mask], **cox_kwargs)}


def interaction_p(df: pd.DataFrame, modifier: str, *, time: str = "time", event: str = "event",
                  treatment: str = "treated", weights: str | None = None) -> float:  # fmt: skip
    """p-value of the treatment x modifier interaction: the test that belongs with a subgroup
    claim, rather than comparing two subgroup p-values."""
    d = df[[time, event, treatment, modifier, *([weights] if weights else [])]].dropna().copy()
    d["_ix"] = d[treatment] * d[modifier]
    cph = CoxPHFitter().fit(d, time, event, weights_col=weights, robust=weights is not None)
    return float(cph.summary.loc["_ix", "p"])


def as_frame(results: dict[str, HazardRatio]) -> pd.DataFrame:
    return pd.DataFrame({k: v.as_dict() for k, v in results.items()}).T.astype(
        {"hr": float, "lower": float, "upper": float, "p": float}
    )


def log_error(estimate: float, truth: float) -> float:
    return float(abs(np.log(estimate) - np.log(truth)))
