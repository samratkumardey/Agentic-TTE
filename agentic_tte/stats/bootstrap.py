"""Risk measures at a horizon with bootstrap CIs (eval/SCORING.md R4).

Weighted Kaplan-Meier risks have no valid analytic variance once the weights are estimated,
so the CIs come from a nonparametric bootstrap over patients that refits the weights in every
resample: 500 resamples and seed 20261027 by default, as pre-registered.

Two weightings, matching the Statistician's primary methods:
- ``iptw``: stabilized ATE weights from a propensity model; risks by weighted KM per arm;
- ``ccw``: clone-censor-weight with a grace period; risks by weighted KM on each arm's clones
  in counting-process form (entry at the row's start).

Measures, treatment vs comparator: risk ratio (RR), odds ratio of the cumulative risks (OR) and
risk difference (RD). CIs are percentile intervals; ratios are bootstrapped on the log scale,
which gives the same percentile limits. Resamples where a measure is undefined (no event in an
arm) are dropped and counted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.exceptions import StatisticalWarning
from pydantic import BaseModel

from agentic_tte.stats.balance import iptw_weights, propensity_scores
from agentic_tte.stats.ccw import clone_censor_weight
from agentic_tte.stats.survival import km_risk

N_BOOTSTRAP = 500
SEED = 20261027


class RiskMeasures(BaseModel):
    method: str
    horizon_days: float
    risk_treated: float
    risk_untreated: float
    rr: float | None
    or_: float | None
    rd: float
    ci: dict[str, tuple[float, float]] = {}  # measure -> (lower, upper), 95% percentile
    n_bootstrap: int = 0
    failed: int = 0  # resamples dropped (a measure undefined or a fit failure)
    seed: int | None = None


def _km_entry(long: pd.DataFrame, horizon: float) -> float:
    kmf = KaplanMeierFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StatisticalWarning)
        kmf.fit(long["stop"], long["event"], entry=long["start"], weights=long["weight"])
    return float(1 - kmf.survival_function_at_times(horizon).iloc[0])


def point_risks(df: pd.DataFrame, horizon: float, *, method: str, covariates: list[str],
                grace: float | None = None) -> tuple[float, float]:  # fmt: skip
    """(risk under treatment, risk under comparator) at ``horizon``."""
    if method == "iptw":
        d = df.copy()
        if covariates:
            d["_w"] = iptw_weights(propensity_scores(d, covariates, "treated"), d["treated"])
        else:
            d["_w"] = 1.0
        return (km_risk(d[d.treated == 1], horizon, weights="_w"),
                km_risk(d[d.treated == 0], horizon, weights="_w"))  # fmt: skip
    if method == "ccw":
        if grace is None:
            raise ValueError("clone-censor-weight needs a grace period")
        long = clone_censor_weight(df, grace=grace, covariates=covariates)
        return (_km_entry(long[long.arm == 1], horizon),
                _km_entry(long[long.arm == 0], horizon))  # fmt: skip
    raise ValueError(f"unknown method '{method}'")


def _measures(r1: float, r0: float) -> dict[str, float | None]:
    rr = r1 / r0 if r0 > 0 else None
    odds = (r1 / (1 - r1), r0 / (1 - r0)) if 0 < r1 < 1 and 0 < r0 < 1 else None
    return {"rr": rr, "or_": odds[0] / odds[1] if odds else None, "rd": r1 - r0}


def risk_measures(df: pd.DataFrame, horizon: float, *, method: str, covariates: list[str],
                  grace: float | None = None, n_bootstrap: int = N_BOOTSTRAP,
                  seed: int = SEED) -> RiskMeasures:  # fmt: skip
    """Risks, RR, OR and RD at ``horizon`` with bootstrap CIs that refit the weights."""
    d = df.dropna(subset=covariates).reset_index(drop=True)
    r1, r0 = point_risks(d, horizon, method=method, covariates=covariates, grace=grace)
    m = _measures(r1, r0)
    out = RiskMeasures(method=method, horizon_days=horizon, risk_treated=r1, risk_untreated=r0,
                       rr=m["rr"], or_=m["or_"], rd=m["rd"], n_bootstrap=n_bootstrap,
                       seed=seed if n_bootstrap else None)  # fmt: skip
    if not n_bootstrap:
        return out
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"rr": [], "or_": [], "rd": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(d), len(d))
        b = d.iloc[idx].reset_index(drop=True)
        b["person_id"] = np.arange(len(b))  # each draw is its own patient (CCW clusters)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mb = _measures(*point_risks(b, horizon, method=method, covariates=covariates,
                                            grace=grace))  # fmt: skip
        except Exception:  # noqa: BLE001 - a failed refit is counted, not fatal
            out.failed += 1
            continue
        if mb["rr"] is None or mb["or_"] is None or mb["rr"] <= 0 or mb["or_"] <= 0:
            out.failed += 1
            continue
        for k in draws:
            draws[k].append(mb[k])
    for k, v in draws.items():
        if len(v) >= 0.9 * n_bootstrap:  # too many failures: no CI rather than a biased one
            arr = np.log(v) if k != "rd" else np.asarray(v)
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out.ci[k] = (float(np.exp(lo)), float(np.exp(hi))) if k != "rd" else (float(lo),
                                                                                  float(hi))
    return out
