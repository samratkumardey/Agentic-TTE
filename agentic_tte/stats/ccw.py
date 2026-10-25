"""Clone-censor-weight (CCW) for treatment strategies with a grace period.

Target trial: strategy 1 "start treatment within ``grace`` days of time zero", strategy 0
"do not start within ``grace`` days". Every patient is cloned into both arms at time zero;
a clone is artificially censored when its data stop being compatible with its arm:
- arm 0 clones are censored when treatment starts within the grace period;
- arm 1 clones are censored at the end of the grace period if treatment has not started.
Inverse probability of censoring weights (IPCW), estimated from the confounders, remove the
selection this censoring creates. Patients who die inside the grace period without starting
treatment are consistent with both strategies and count in both arms.

EmulatRx describes an "extended clone-censor-weight methodology" (Methods, Statistical
analysis) without code in the public repository; this is a standard implementation (Hernán
et al. 2016; Maringe et al. 2020). Weights:
- arm 0, interval k of the grace period: 1 / prod_{j<k} (1 - h_j(X)), where h_j is the
  probability of starting treatment in interval j among those still untreated and at risk
  (pooled logistic regression with interval effects);
- arm 1, after the grace period: 1 / P(started by the end of grace | X, at risk at that time);
  before it, weight 1 (no one is censored yet).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from agentic_tte.stats.survival import HazardRatio


def _fit_predict(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    if y.min() == y.max():  # no variation: probability is the observed constant
        return np.full(len(x_new), float(y[0]))
    m = LogisticRegression(C=np.inf, max_iter=2000).fit(x, y)
    return m.predict_proba(x_new)[:, 1]


def _standardize(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].astype(float)
    return ((x - x.mean()) / x.std(ddof=0).replace(0, 1)).to_numpy()


def clone_censor_weight(df: pd.DataFrame, *, grace: float, covariates: list[str],
                        time: str = "time", event: str = "event",
                        treat_start: str = "treat_start", interval: float = 0.5,
                        weighted: bool = True, truncate: float | None = None) -> pd.DataFrame:
    """Long (counting-process) data: person_id, arm, start, stop, event, weight.

    ``truncate`` (opt-in) clips weights at that quantile of the rows. Off by default: in the
    synthetic check (true HR 0.5) clipping at 0.995 moved the estimate from 0.616 to 0.646
    against a truth of 0.596, because the row quantile is dominated by arm-0 rows and mostly
    clips the large arm-1 weights after the grace period. If used, report it."""
    d = df.reset_index(drop=True)
    pid = d["person_id"].to_numpy() if "person_id" in d else np.arange(len(d))
    t = d[time].to_numpy(dtype=float)
    ev = d[event].to_numpy().astype(int)
    u = d[treat_start].to_numpy(dtype=float)  # NaN = never started
    started = ~np.isnan(u) & (u < np.minimum(grace, t))
    x = _standardize(d, covariates)
    edges = np.arange(0, grace + 1e-9, interval)
    if edges[-1] < grace:
        edges = np.append(edges, grace)
    k_count = len(edges) - 1

    # Arm 0 censoring model: discrete hazard of starting treatment per grace interval.
    stop0 = np.where(started, u, np.minimum(t, grace))  # end of time at risk of starting
    pi_rows, pi_k, pi_y = [], [], []
    for k in range(k_count):
        at_risk = stop0 > edges[k]
        idx = np.flatnonzero(at_risk)
        pi_rows.append(idx)
        pi_k.append(np.full(len(idx), k))
        pi_y.append((started[idx] & (u[idx] >= edges[k]) & (u[idx] < edges[k + 1])).astype(int))
    rows, ks, ys = np.concatenate(pi_rows), np.concatenate(pi_k), np.concatenate(pi_y)
    dummies = np.eye(k_count)[ks]
    design = np.hstack([x[rows], dummies[:, 1:]])
    hazard = np.zeros((len(d), k_count))
    if weighted:
        all_rows = np.repeat(np.arange(len(d)), k_count)
        all_k = np.tile(np.arange(k_count), len(d))
        new = np.hstack([x[all_rows], np.eye(k_count)[all_k][:, 1:]])
        hazard = _fit_predict(design, ys, new).reshape(len(d), k_count)
    surv_untreated = np.cumprod(np.hstack([np.ones((len(d), 1)), 1 - hazard]), axis=1)
    w0 = 1 / np.clip(surv_untreated, 1e-6, None)  # w0[:, k] = weight at the start of interval k

    # Arm 1 censoring model: started by the end of grace, among those at risk then.
    at_grace = t > grace
    p_started = np.ones(len(d))
    if weighted and at_grace.any():
        p_started = _fit_predict(x[at_grace], started[at_grace].astype(int), x)

    out = []
    for i in range(len(d)):
        # --- arm 0: no start within grace
        end0 = u[i] if started[i] else t[i]
        for k in range(k_count):
            a, b = edges[k], min(edges[k + 1], end0)
            if b <= a:
                break
            out.append((pid[i], 0, a, b, int(ev[i] and not started[i] and t[i] <= b and b == t[i]),
                        w0[i, k]))  # fmt: skip
        if not started[i] and t[i] > grace:
            out.append((pid[i], 0, grace, t[i], ev[i], w0[i, k_count]))
        # --- arm 1: start within grace
        first = min(t[i], grace)
        out.append((pid[i], 1, 0.0, first, int(ev[i] and t[i] <= grace), 1.0))
        if started[i] and t[i] > grace:
            out.append((pid[i], 1, grace, t[i], ev[i], 1 / max(p_started[i], 1e-6)))
    long = pd.DataFrame(out, columns=["person_id", "arm", "start", "stop", "event", "weight"])
    long = long[long["stop"] > long["start"]].reset_index(drop=True)
    if truncate is not None and weighted:
        long["weight"] = long["weight"].clip(upper=long["weight"].quantile(truncate))
    return long


def ccw_hr(df: pd.DataFrame, *, grace: float, covariates: list[str], weighted: bool = True,
           **kwargs) -> HazardRatio:  # fmt: skip
    """Marginal HR of strategy 1 vs 0 from the cloned data: weighted Cox model in
    counting-process form, robust SEs clustered on the patient (each patient has two clones)."""
    from lifelines import CoxPHFitter

    long = clone_censor_weight(df, grace=grace, covariates=covariates, weighted=weighted,
                               **kwargs)  # fmt: skip
    cph = CoxPHFitter().fit(long, duration_col="stop", event_col="event", entry_col="start",
                            weights_col="weight", cluster_col="person_id", robust=True,
                            formula="arm")  # fmt: skip
    row = cph.summary.loc["arm"]
    return HazardRatio(
        hr=float(row["exp(coef)"]),
        lower=float(row["exp(coef) lower 95%"]),
        upper=float(row["exp(coef) upper 95%"]),
        p=float(row["p"]),
        n=int(long["person_id"].nunique()),
        events=int(long["event"].sum()),
        method="clone-censor-weight" if weighted else "clone-censor (unweighted)",
    )
