"""Deterministic statistics for the Statistician agent. The LLM chooses among these methods;
it never computes a number itself (PLAN.md §1)."""

from agentic_tte.stats.balance import (
    SMD_THRESHOLD,
    MatchResult,
    iptw_weights,
    match_nearest,
    propensity_scores,
    smd,
)
from agentic_tte.stats.bootstrap import RiskMeasures, risk_measures
from agentic_tte.stats.ccw import ccw_hr, clone_censor_weight
from agentic_tte.stats.design import evalue_hr, sample_size, schoenfeld_events
from agentic_tte.stats.survival import (
    HazardRatio,
    cox_hr,
    interaction_p,
    km_risk,
    ph_test,
    risk_difference,
    subgroup_hrs,
)

__all__ = [
    "SMD_THRESHOLD",
    "HazardRatio",
    "MatchResult",
    "RiskMeasures",
    "ccw_hr",
    "clone_censor_weight",
    "cox_hr",
    "evalue_hr",
    "interaction_p",
    "iptw_weights",
    "km_risk",
    "match_nearest",
    "ph_test",
    "propensity_scores",
    "risk_difference",
    "risk_measures",
    "sample_size",
    "schoenfeld_events",
    "smd",
    "subgroup_hrs",
]
