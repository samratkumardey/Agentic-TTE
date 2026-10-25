"""Confounder balancing: propensity scores, IPTW, matching and standardized mean differences.

Ported from EmulatRx ``emulation_utils.py`` (``calculate_propensity_scores``,
``match_pairs``, ``matching_propensity``, ``check_balance_after_matching``, ``perform_iptw``;
MIT, TrialLab), with these deliberate changes:
- the propensity model uses every confounder, not only those with SMD > 0 before matching;
- the matching caliper defaults to 0.2 SD of the logit propensity score (Austin 2011); upstream
  used 1 x the MAD of all pairwise distances, kept as ``caliper="upstream_mad"``;
- treated patients are matched in random (seeded) order rather than data order;
- IPTW weights are stabilized and optionally truncated; upstream used unstabilized weights;
- SMDs can be weighted, so balance after IPTW is checked on the weighted sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SMD_THRESHOLD = 0.1  # EmulatRx and common practice


def _design(df: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    x = df[covariates].astype(float)
    return ((x - x.mean()) / x.std(ddof=0).replace(0, 1)).to_numpy()


def propensity_scores(df: pd.DataFrame, covariates: list[str], treatment: str) -> pd.Series:
    """Unpenalized logistic regression of treatment on the covariates (main effects)."""
    model = LogisticRegression(C=np.inf, max_iter=2000)
    model.fit(_design(df, covariates), df[treatment].to_numpy())
    return pd.Series(model.predict_proba(_design(df, covariates))[:, 1], index=df.index, name="ps")


def iptw_weights(ps: pd.Series, treated: pd.Series, *, stabilized: bool = True,
                 truncate: tuple[float, float] | None = (0.01, 0.99)) -> pd.Series:  # fmt: skip
    """ATE weights. Stabilized: P(A=a) / P(A=a | X). Truncation clips weights at the given
    quantiles to limit the influence of near-positivity violations."""
    a = treated.to_numpy().astype(bool)
    p = ps.to_numpy()
    w = np.where(a, 1 / p, 1 / (1 - p))
    if stabilized:
        w = w * np.where(a, a.mean(), 1 - a.mean())
    if truncate is not None:
        lo, hi = np.quantile(w, truncate)
        w = np.clip(w, lo, hi)
    return pd.Series(w, index=ps.index, name="iptw")


def smd(df: pd.DataFrame, covariates: list[str], treatment: str,
        weights: pd.Series | None = None) -> pd.Series:  # fmt: skip
    """Absolute standardized mean differences (weighted if weights are given)."""
    a = df[treatment].to_numpy().astype(bool)
    w = np.ones(len(df)) if weights is None else weights.to_numpy()
    out = {}
    for c in covariates:
        x = df[c].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        m1, m0 = (np.average(x[ok & g], weights=w[ok & g]) for g in (a, ~a))
        v1, v0 = (np.average((x[ok & g] - m) ** 2, weights=w[ok & g])
                  for g, m in ((a, m1), (~a, m0)))  # fmt: skip
        pooled = np.sqrt((v1 + v0) / 2)
        out[c] = abs(m1 - m0) / pooled if pooled > 0 else 0.0
    return pd.Series(out, name="smd")


@dataclass
class MatchResult:
    data: pd.DataFrame  # matched rows, with a "pair_id" column for clustered SEs
    n_treated: int
    n_matched_treated: int
    caliper: float | None


def match_nearest(df: pd.DataFrame, ps: pd.Series, treatment: str, *, ratio: int = 1,
                  caliper: float | str | None = 0.2, seed: int = 0) -> MatchResult:  # fmt: skip
    """Greedy nearest-neighbour matching on the logit propensity score, without replacement.
    ``caliper``: SD multiple of the logit PS (default 0.2), ``"upstream_mad"`` for EmulatRx's
    rule (1 x MAD of all pairwise PS distances), or None. Estimates the ATT."""
    logit = np.log(ps / (1 - ps)).to_numpy()
    a = df[treatment].to_numpy().astype(bool)
    treated_pos = np.flatnonzero(a)
    control_pos = np.flatnonzero(~a)
    if caliper == "upstream_mad":
        d = np.abs(ps.to_numpy()[treated_pos][:, None] - ps.to_numpy()[control_pos][None, :])
        width = float(np.median(np.abs(d - np.median(d))))
        score = ps.to_numpy()
    else:
        width = None if caliper is None else float(caliper) * float(np.std(logit))
        score = logit
    rng = np.random.default_rng(seed)
    order = rng.permutation(treated_pos)
    available = np.ones(len(control_pos), dtype=bool)
    rows, pair_ids, n_matched = [], [], 0
    for pair, t in enumerate(order):
        matched = []
        for _ in range(ratio):
            if not available.any():
                break
            dist = np.abs(score[control_pos] - score[t])
            dist[~available] = np.inf
            j = int(np.argmin(dist))
            if width is not None and dist[j] > width:
                break
            matched.append(control_pos[j])
            available[j] = False
        if matched:
            n_matched += 1
            rows += [t, *matched]
            pair_ids += [pair] * (1 + len(matched))
    data = df.iloc[rows].copy()
    data["pair_id"] = pair_ids
    return MatchResult(data, len(treated_pos), n_matched, width)
