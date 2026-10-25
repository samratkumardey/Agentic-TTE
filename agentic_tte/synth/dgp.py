"""Synthetic ICU cohorts with a known data-generating process (Phase 1).

The only data where an estimator can be checked against the truth. Each scenario plants:
- confounding by indication: SOFA, age and lactate raise both the chance of treatment and the
  hazard of death, so an unadjusted comparison is biased;
- a treatment effect: a conditional hazard ratio, constant or modified by SOFA;
- optionally, delayed treatment start within a grace period, the setting where classifying
  patients by "ever treated" creates immortal-time bias.

Ground truth is defined as a trial would define it: every patient of a large truth sample is
followed under both treatment strategies (potential outcomes, no censoring). From these come
the true marginal risks, risk difference and marginal hazard ratio, i.e. the quantities an
emulation estimates. The planted conditional HR differs from the marginal HR whenever
covariates affect the hazard (non-collapsibility); EmulatRx's synthetic check (Table 3)
compared against the conditional value.

Covariate scales follow the EmulatRx synthetic design (1,000 subjects, SOFA- and age-driven
treatment, exponential hazards with baseline 0.1) but in days with a 28-day horizon.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd

TreatmentTiming = Literal["at_time_zero", "delayed"]


@dataclass(frozen=True)
class Scenario:
    n: int = 2000
    seed: int = 20261024
    true_hr: float = 0.5  # conditional HR of treatment (ignored when an effect modifier is set)
    baseline_hazard: float = 0.02  # per day, at average covariates
    horizon_days: float = 28.0
    censor_rate: float = 0.005  # per day, loss to follow-up
    confounding: float = 1.0  # scales covariate effects on treatment and on death
    n_noise: int = 7  # extra N(0, 1) covariates without any effect
    modifier_threshold: float | None = None  # SOFA cut-off for effect modification
    modifier_log_hr: float = 0.7  # log HR +m below the cut-off, -m at or above it
    timing: TreatmentTiming = "at_time_zero"
    grace_days: float = 3.0  # "delayed": initiation time ~ Uniform(0, grace)
    missing_lactate: float = 0.0  # share of patients without a lactate measurement
    truth_n: int = 60_000

    def with_(self, **changes) -> Scenario:
        return replace(self, **changes)


@dataclass
class Cohort:
    scenario: Scenario
    frame: pd.DataFrame
    truth: dict[str, float] = field(default_factory=dict)
    # Potential death times of the observed patients (never shown to an analysis), so the
    # truth can be computed for whatever population an analysis ends up estimating:
    #   t1  start at time zero ("at_time_zero") or at the patient's own time U ("delayed")
    #   t1_deadline  natural initiators start at U, everyone else at the end of grace
    #   t0  never treated
    potential: pd.DataFrame | None = None

    def truth_in(self, person_ids, strategy: str = "t1") -> float:
        """True marginal HR (strategy vs never) among the given patients."""
        p = self.potential.set_index("person_id").loc[list(person_ids)]
        return _marginal_hr(p[strategy].to_numpy(), p["t0"].to_numpy(), self.scenario.horizon_days)

    @property
    def covariates(self) -> list[str]:
        noise = [f"x{k}" for k in range(1, self.scenario.n_noise + 1)]
        return ["age", "female", "sofa", "lactate", *noise]

    @property
    def confounders(self) -> list[str]:
        return ["age", "sofa", "lactate"]


# --- covariates and mechanisms ----------------------------------------------------------------

_LOG_LACTATE_MU, _LOG_LACTATE_SD = np.log(2.5), 0.5


def _covariates(rng: np.random.Generator, s: Scenario, n: int) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "age": np.clip(rng.normal(62, 14, n), 18, 95).round(1),
            "female": rng.binomial(1, 0.45, n),
            "sofa": np.clip(rng.normal(8, 3, n), 0, 24).round(),
            "lactate": np.exp(rng.normal(_LOG_LACTATE_MU, _LOG_LACTATE_SD, n)).round(2),
        }
    )
    for k in range(1, s.n_noise + 1):
        df[f"x{k}"] = rng.normal(0, 1, n)
    return df


def _z(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        (df["sofa"].to_numpy() - 8) / 3,
        (df["age"].to_numpy() - 62) / 14,
        (np.log(df["lactate"].to_numpy()) - _LOG_LACTATE_MU) / _LOG_LACTATE_SD,
    )


def propensity(df: pd.DataFrame, s: Scenario) -> np.ndarray:
    """True probability of (initiating) treatment."""
    zs, za, zl = _z(df)
    logit = -0.2 + s.confounding * (0.8 * zs + 0.4 * za + 0.3 * zl)
    return 1 / (1 + np.exp(-logit))


def _outcome_lp(df: pd.DataFrame, s: Scenario) -> np.ndarray:
    zs, za, zl = _z(df)
    return s.confounding * (0.6 * zs + 0.4 * za + 0.3 * zl)


def log_hr(df: pd.DataFrame, s: Scenario) -> np.ndarray:
    """Conditional log hazard ratio of treatment for each patient."""
    if s.modifier_threshold is None:
        return np.full(len(df), np.log(s.true_hr))
    below = df["sofa"].to_numpy() < s.modifier_threshold
    return np.where(below, s.modifier_log_hr, -s.modifier_log_hr)


def _deadline_start(s: Scenario, initiator: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Grace-period strategy as clone-censor-weight estimates it: patients who would start
    anyway do so at their natural time; everyone else starts at the deadline (end of grace).
    Differs from "everyone starts at a random time in the grace period" because a non-starter
    is only recognisably non-adherent once the grace period is over."""
    if s.timing == "at_time_zero":
        return start
    return np.where(initiator, start, s.grace_days)


def _death_times(df, s, treated: np.ndarray, start: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Death times from unit-exponential draws ``e`` (shared across potential outcomes).
    Hazard is rate0 before treatment start and rate0 * HR after it; ``start`` is 0 for
    treatment at time zero and inf for never treated."""
    rate0 = s.baseline_hazard * np.exp(_outcome_lp(df, s))
    hr = np.exp(log_hr(df, s))
    start = np.where(treated, start, np.inf)
    before = rate0 * np.minimum(start, 1e12)  # cumulative hazard reached at treatment start
    return np.where(e <= before, e / rate0, start + (e - before) / (rate0 * hr))


# --- observed cohort ------------------------------------------------------------------------


def generate(s: Scenario, *, with_truth: bool = True) -> Cohort:
    rng = np.random.default_rng(s.seed)
    df = _covariates(rng, s, s.n)
    initiator = rng.binomial(1, propensity(df, s)).astype(bool)
    if s.timing == "at_time_zero":
        start = np.zeros(s.n)
    else:
        start = rng.uniform(0, s.grace_days, s.n)
    e = rng.exponential(1.0, s.n)
    death = _death_times(df, s, initiator, start, e)
    everyone = np.ones(s.n, dtype=bool)
    potential = pd.DataFrame({
        "person_id": np.arange(1, s.n + 1),
        "t1": _death_times(df, s, everyone, start, e),
        "t1_deadline": _death_times(df, s, everyone, _deadline_start(s, initiator, start), e),
        "t0": _death_times(df, s, ~everyone, start, e),
    })  # fmt: skip
    censor = rng.exponential(1 / s.censor_rate, s.n) if s.censor_rate > 0 else np.full(s.n, np.inf)
    end = np.minimum.reduce([death, censor, np.full(s.n, s.horizon_days)])
    # A delayed initiator only gets treated if still alive and under follow-up at the start.
    treated = initiator & (start < end)
    df.insert(0, "person_id", np.arange(1, s.n + 1))
    df["treated"] = treated.astype(int)
    df["treat_start"] = np.where(treated, start, np.nan)
    df["time"] = end
    df["event"] = (death <= end).astype(int)
    df["end_reason"] = np.select(
        [death <= end, censor <= end], ["death", "lost_to_follow_up"], default="end_of_follow_up"
    )
    if s.missing_lactate > 0:
        df.loc[rng.random(s.n) < s.missing_lactate, "lactate"] = np.nan
    cohort = Cohort(s, df, potential=potential)
    if with_truth:
        cohort.truth = ground_truth(s)
    return cohort


# --- ground truth ------------------------------------------------------------------------------


def _marginal_hr(t1: np.ndarray, t0: np.ndarray, horizon: float) -> float:
    """Marginal HR between two potential-outcome arms of the same patients, censored only at
    the horizon: the HR a perfect randomized trial would estimate."""
    from lifelines import CoxPHFitter

    t = np.concatenate([t1, t0])
    d = pd.DataFrame({"time": np.minimum(t, horizon), "event": (t <= horizon).astype(int),
                      "arm": np.r_[np.ones(len(t1)), np.zeros(len(t0))]})  # fmt: skip
    return float(np.exp(CoxPHFitter().fit(d, "time", "event").params_["arm"]))


def ground_truth(s: Scenario) -> dict[str, float]:
    rng = np.random.default_rng(s.seed + 1_000_003)
    df = _covariates(rng, s, s.truth_n)
    e = rng.exponential(1.0, s.truth_n)
    if s.timing == "at_time_zero":
        start = np.zeros(s.truth_n)
    else:
        start = rng.uniform(0, s.grace_days, s.truth_n)
    everyone = np.ones(s.truth_n, dtype=bool)
    t1 = _death_times(df, s, everyone, start, e)  # strategy 1: (initiate within grace)
    t0 = _death_times(df, s, ~everyone, start, e)  # strategy 0: never initiate
    h = s.horizon_days
    assigned = rng.binomial(1, propensity(df, s)).astype(bool)
    t1_deadline = _death_times(df, s, everyone, _deadline_start(s, assigned, start), e)
    truth = {
        "conditional_hr": float(np.exp(log_hr(df, s)).mean()) if s.modifier_threshold is None
        else float("nan"),
        "marginal_hr": _marginal_hr(t1, t0, h),
        "marginal_hr_att": _marginal_hr(t1[assigned], t0[assigned], h),
        "marginal_hr_grace_deadline": _marginal_hr(t1_deadline, t0, h),
        "risk_treated": float((t1 <= h).mean()),
        "risk_untreated": float((t0 <= h).mean()),
    }  # fmt: skip
    truth["risk_difference"] = truth["risk_treated"] - truth["risk_untreated"]
    if s.modifier_threshold is not None:
        below = df["sofa"].to_numpy() < s.modifier_threshold
        truth["share_below_threshold"] = float(below.mean())
        truth["marginal_hr_below"] = _marginal_hr(t1[below], t0[below], h)
        truth["marginal_hr_above"] = _marginal_hr(t1[~below], t0[~below], h)
        truth["conditional_hr_below"] = float(np.exp(s.modifier_log_hr))
        truth["conditional_hr_above"] = float(np.exp(-s.modifier_log_hr))
    return truth


def describe(c: Cohort) -> dict:
    """Scenario, truth and observed summary, for reports and run manifests."""
    f = c.frame
    return {
        "scenario": asdict(c.scenario),
        "truth": c.truth,
        "observed": {
            "n": len(f),
            "treated": int(f["treated"].sum()),
            "events": int(f["event"].sum()),
            "lost_to_follow_up": int((f["end_reason"] == "lost_to_follow_up").sum()),
        },
    }
