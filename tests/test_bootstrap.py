"""Risk measures with bootstrap CIs (eval/SCORING.md R4) against potential-outcome truth on
synthetic cohorts: IPTW for treatment at time zero, clone-censor-weight for a grace period."""

import numpy as np
import pytest

from agentic_tte.agents.statistician import analyze
from agentic_tte.protocol import Duration, TargetTrialProtocol, TreatmentStrategy
from agentic_tte.stats import risk_difference, risk_measures
from agentic_tte.stats.balance import iptw_weights, propensity_scores
from agentic_tte.stats.bootstrap import SEED, _measures
from agentic_tte.synth.dgp import Scenario, generate


def _sample_truth(c, deadline: bool) -> tuple[float, float]:
    """Risks if every patient of this sample followed each strategy (potential outcomes)."""
    h = c.scenario.horizon_days
    t1 = c.potential.t1_deadline if deadline else c.potential.t1
    return float((t1 <= h).mean()), float((c.potential.t0 <= h).mean())


@pytest.mark.parametrize("timing,method", [("at_time_zero", "iptw"), ("delayed", "ccw")])
def test_bootstrap_cis_cover_the_truth(timing, method):
    c = generate(Scenario(n=1500, seed=5, timing=timing, grace_days=3.0))
    r1, r0 = _sample_truth(c, deadline=timing == "delayed")
    m = risk_measures(c.frame, c.scenario.horizon_days, method=method,
                      covariates=c.confounders, grace=3.0, n_bootstrap=100)  # fmt: skip
    assert m.failed == 0 and set(m.ci) == {"rr", "or_", "rd"}
    lo, hi = m.ci["rd"]
    assert lo <= r1 - r0 <= hi and lo < m.rd < hi
    lo, hi = m.ci["rr"]
    assert lo <= r1 / r0 <= hi
    assert abs(m.risk_treated - r1) < 0.05 and abs(m.risk_untreated - r0) < 0.05


def test_iptw_point_estimate_matches_the_weighted_km():
    c = generate(Scenario(n=800, seed=3), with_truth=False)
    d = c.frame.copy()
    d["w"] = iptw_weights(propensity_scores(d, c.confounders, "treated"), d["treated"])
    ref = risk_difference(d, 28.0, weights="w")
    m = risk_measures(c.frame, 28.0, method="iptw", covariates=c.confounders, n_bootstrap=0)
    assert m.rd == pytest.approx(ref["risk_difference"]) and m.ci == {} and m.seed is None


def test_the_bootstrap_is_reproducible_with_its_seed():
    c = generate(Scenario(n=600, seed=9), with_truth=False)
    a, b = (risk_measures(c.frame, 28.0, method="iptw", covariates=c.confounders,
                          n_bootstrap=30) for _ in range(2))  # fmt: skip
    assert a.ci == b.ci and a.seed == SEED
    other = risk_measures(c.frame, 28.0, method="iptw", covariates=c.confounders,
                          n_bootstrap=30, seed=1)  # fmt: skip
    assert other.ci != a.ci


def test_undefined_measures():
    assert _measures(0.2, 0.0) == {"rr": None, "or_": None, "rd": 0.2}
    m = _measures(0.3, 0.2)
    assert m["rr"] == pytest.approx(1.5) and m["or_"] == pytest.approx((0.3 / 0.7) / (0.2 / 0.8))
    with pytest.raises(ValueError, match="grace"):
        risk_measures(generate(Scenario(n=100), with_truth=False).frame, 28.0, method="ccw",
                      covariates=["sofa"], n_bootstrap=0)  # fmt: skip


def test_the_statistician_reports_risk_measures():
    c = generate(Scenario(n=800, seed=4, timing="delayed", grace_days=3.0), with_truth=False)
    p = TargetTrialProtocol(protocol_id="B", treatment_strategies=[
        TreatmentStrategy(label="Rx", grace_period=Duration(amount=3, unit="day")),
        TreatmentStrategy(label="None", is_comparator=True)])  # fmt: skip
    res = analyze(c.frame, p, confounders=c.confounders, horizon_days=28.0, n_bootstrap=20)
    rm = res.risk_measures
    assert res.primary_method == "clone-censor-weight" and rm["method"] == "ccw"
    assert rm["n_bootstrap"] == 20 and np.isfinite(rm["ci"]["rd"]).all()
    assert "Risks at 28 days (ccw weighting)" in res.to_markdown()
    assert analyze(c.frame, p, confounders=c.confounders, horizon_days=28.0).risk_measures is None
