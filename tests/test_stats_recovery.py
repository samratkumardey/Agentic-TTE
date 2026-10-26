"""Phase 1 exit criterion: the statistics recover planted effects (PLAN.md §9).

Tolerances are fixed from the expected precision, not tuned to these seeds: with roughly
2,000 events the SE of a log HR is about 0.05, so estimates must lie within 0.15 on the log
scale (about 3 SE) and their 95% CI must cover the truth. The truth is the marginal effect from
potential outcomes (agentic_tte/synth/dgp.py), except where a test says otherwise.
"""

import math

import pytest

from agentic_tte.stats import (
    SMD_THRESHOLD,
    ccw_hr,
    cox_hr,
    evalue_hr,
    interaction_p,
    iptw_weights,
    match_nearest,
    ph_test,
    propensity_scores,
    risk_difference,
    sample_size,
    schoenfeld_events,
    smd,
    subgroup_hrs,
)
from agentic_tte.stats.survival import log_error
from agentic_tte.synth.dgp import Scenario, generate

pytestmark = pytest.mark.slow  # ~2 min; skip with -m "not slow and not live"

TOL = 0.15
N = 6000
TRUTH_N = 40_000


@pytest.fixture(scope="module", params=[0.5, 1.0, 2.0, 3.0], ids=lambda h: f"HR{h}")
def point(request):
    c = generate(Scenario(n=N, true_hr=request.param, truth_n=TRUTH_N))
    f = c.frame.copy()
    ps = propensity_scores(f, c.confounders, "treated")
    f["w"] = iptw_weights(ps, f["treated"])
    return c, f, ps


def test_unadjusted_comparison_is_confounded(point):
    c, f, _ = point
    assert log_error(cox_hr(f).hr, c.truth["marginal_hr"]) > 0.3


def test_iptw_balances_and_recovers_marginal_hr(point):
    c, f, _ = point
    assert smd(f, c.confounders, "treated").max() > 0.3  # imbalanced before
    assert smd(f, c.confounders, "treated", f["w"]).max() < SMD_THRESHOLD
    est = cox_hr(f, weights="w")
    assert log_error(est.hr, c.truth["marginal_hr"]) < TOL
    assert est.covers(c.truth["marginal_hr"])


def test_psm_recovers_marginal_hr_in_the_treated(point):
    c, f, ps = point
    m = match_nearest(f, ps, "treated", seed=1)
    assert m.n_matched_treated > 0.6 * m.n_treated
    assert smd(m.data, c.confounders, "treated").max() < SMD_THRESHOLD
    est = cox_hr(m.data, cluster="pair_id")
    # The caliper drops the sickest treated patients, so the estimand is the effect in the
    # matched treated, not in all treated: with prognostic covariates (non-collapsibility)
    # the two marginal HRs differ.
    truth = c.truth_in(m.data.loc[m.data["treated"] == 1, "person_id"])
    assert log_error(est.hr, truth) < TOL
    assert est.covers(truth)


def test_adjusted_cox_recovers_conditional_hr(point):
    c, f, _ = point
    est = cox_hr(f, covariates=c.confounders)
    assert log_error(est.hr, c.truth["conditional_hr"]) < TOL
    assert est.covers(c.truth["conditional_hr"])


def test_weighted_km_recovers_risk_difference(point):
    c, f, _ = point
    rd = risk_difference(f, c.scenario.horizon_days, weights="w")
    assert abs(rd["risk_difference"] - c.truth["risk_difference"]) < 0.04


def test_upstream_ridge_penalty_shrinks_effects_toward_null():
    """EmulatRx fitted CoxPHFitter(penalizer=0.1) with all covariates. On the same weighted
    data the penalty pulls the treatment effect toward HR 1, part of the attenuation in the
    paper's Table 3 (0.6345 for a planted 0.5)."""
    c = generate(Scenario(n=N, true_hr=0.5), with_truth=False)
    f = c.frame.copy()
    f["w"] = iptw_weights(propensity_scores(f, c.confounders, "treated"), f["treated"])
    plain = cox_hr(f, covariates=c.covariates, weights="w")
    ridge = cox_hr(f, covariates=c.covariates, weights="w", penalizer=0.1)
    assert abs(math.log(ridge.hr)) < 0.85 * abs(math.log(plain.hr))


def test_ph_test_quiet_when_hazards_are_proportional():
    c = generate(Scenario(n=4000, true_hr=0.5), with_truth=False)
    assert ph_test(c.frame, covariates=c.confounders) > 0.01


# --- immortal time: treatment starts within a grace period --------------------------------


def test_ever_treated_comparison_has_immortal_time_bias_and_ccw_removes_it():
    """No confounding, no effect: any difference is immortal time. Treated patients had to
    survive until treatment started, so "ever treated" looks protective."""
    s = Scenario(n=N, true_hr=1.0, timing="delayed", grace_days=7.0, confounding=0.0,
                 truth_n=TRUTH_N)  # fmt: skip
    c = generate(s)
    naive = cox_hr(c.frame)
    assert naive.upper < 1.0  # spurious benefit
    est = ccw_hr(c.frame, grace=s.grace_days, covariates=c.confounders)
    assert c.truth["marginal_hr_grace_deadline"] == pytest.approx(1.0)  # null either way
    assert log_error(est.hr, 1.0) < TOL
    assert est.covers(1.0)


def test_grace_period_estimand_differs_from_random_start():
    """With a real effect, "start at a random time within grace" and "start naturally, or at
    the deadline" are different strategies; CCW targets the second. Documented so that a
    report states which grace-period strategy an emulation estimates."""
    t = generate(Scenario(n=10, true_hr=0.5, timing="delayed", grace_days=7.0,
                          truth_n=TRUTH_N)).truth  # fmt: skip
    assert t["marginal_hr_grace_deadline"] > t["marginal_hr"] + 0.03  # later start: less benefit


@pytest.mark.parametrize("hr", [1.0, 0.5])
def test_ccw_with_confounded_initiation(hr):
    """Confounded start of treatment: only the weighted version recovers the strategy effect;
    unweighted cloning censors informatively. The estimand is the grace-period strategy CCW
    targets: start at the natural time if you would have started, otherwise at the deadline
    (agentic_tte/synth/dgp.py, _deadline_start)."""
    s = Scenario(n=N, true_hr=hr, timing="delayed", grace_days=3.0, truth_n=TRUTH_N)
    c = generate(s)
    truth = c.truth["marginal_hr_grace_deadline"]
    weighted = ccw_hr(c.frame, grace=s.grace_days, covariates=c.confounders)
    unweighted = ccw_hr(c.frame, grace=s.grace_days, covariates=c.confounders, weighted=False)
    assert log_error(weighted.hr, truth) < TOL and weighted.covers(truth)
    assert log_error(unweighted.hr, truth) > log_error(weighted.hr, truth) + 0.1


def test_ccw_when_treatment_starts_at_time_zero():
    """Regression (found by the end-to-end Replay): starts exactly at time zero must count in
    the first grace interval's censoring model. They did not, so arm-0 weights were all 1 and
    the estimate stayed confounded (0.73 against a truth of 0.53)."""
    s = Scenario(n=N, true_hr=0.5, truth_n=TRUTH_N)  # timing at_time_zero: treat_start = 0
    c = generate(s)
    truth = c.truth["marginal_hr"]
    est = ccw_hr(c.frame, grace=0.5, covariates=c.confounders)
    assert log_error(est.hr, truth) < TOL and est.covers(truth)


# --- effect modification --------------------------------------------------------------------


def test_subgroups_recover_opposite_effects():
    """SOFA modifies the effect (harm below 8, benefit at or above). Subgroup analysis should
    find both, and the interaction test should be decisive."""
    c = generate(Scenario(n=N, modifier_threshold=8, truth_n=TRUTH_N))
    f = c.frame.copy()
    f["w"] = iptw_weights(propensity_scores(f, c.confounders, "treated"), f["treated"])
    f["low_sofa"] = (f["sofa"] < 8).astype(int)
    sg = subgroup_hrs(f, f["sofa"] < 8, labels=("below", "above"), weights="w")
    for side in ("below", "above"):
        truth = c.truth[f"marginal_hr_{side}"]
        assert log_error(sg[side].hr, truth) < TOL and sg[side].covers(truth)
    assert sg["below"].lower > 1 and sg["above"].upper < 1
    assert interaction_p(f, "low_sofa", weights="w") < 1e-6


# --- design calculations (closed form) ------------------------------------------------------


def test_schoenfeld_sample_size():
    # (1.96 + 0.8416)^2 / (0.25 * ln(0.7)^2) = 246.8 -> 247 events
    assert schoenfeld_events(0.7) == 247
    assert sample_size(0.7, 0.3) == {"events": 247, "n": 824}
    assert schoenfeld_events(0.7, allocation=1 / 3) > 247  # unequal allocation needs more
    with pytest.raises(ValueError):
        schoenfeld_events(1.0)


def test_evalues():
    # Rare outcome: E = 1/HR + sqrt(1/HR * (1/HR - 1)) (VanderWeele & Ding 2017)
    e = evalue_hr(0.5, 0.4, 0.7, rare_outcome=True)
    assert e["evalue"] == pytest.approx(2 + math.sqrt(2))
    assert e["evalue_ci"] == pytest.approx(1 / 0.7 + math.sqrt(1 / 0.7 * (1 / 0.7 - 1)))
    assert evalue_hr(0.5, 0.4, 1.1)["evalue_ci"] == 1.0  # CI includes the null
    assert evalue_hr(0.5)["evalue"] < e["evalue"]  # common outcome: smaller E-value
