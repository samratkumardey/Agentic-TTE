"""Trial-design calculations: Schoenfeld sample size and E-values.

The sample size follows EmulatRx's adaptive sample size step (Methods: Schoenfeld formula for
the Cox model, then events divided by the cumulative event rate observed in the RWD). The
E-value (VanderWeele & Ding 2017) is an addition: how strong unmeasured confounding would have
to be to explain away an emulated effect.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def schoenfeld_events(hr: float, *, alpha: float = 0.05, power: float = 0.8,
                      allocation: float = 0.5) -> int:  # fmt: skip
    """Events needed to detect ``hr`` with a two-sided log-rank / Cox test."""
    if hr <= 0 or hr == 1:
        raise ValueError("hr must be positive and different from 1")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(z**2 / (allocation * (1 - allocation) * math.log(hr) ** 2))


def sample_size(hr: float, event_probability: float, *, alpha: float = 0.05,
                power: float = 0.8, allocation: float = 0.5) -> dict[str, float]:  # fmt: skip
    """Total participants: required events / probability of an event during follow-up. In an
    emulation, take the event probability from the RWD cohort (e.g. a pooled KM risk at the
    horizon), which already reflects censoring and follow-up length."""
    if not 0 < event_probability <= 1:
        raise ValueError("event_probability must be in (0, 1]")
    events = schoenfeld_events(hr, alpha=alpha, power=power, allocation=allocation)
    return {"events": events, "n": math.ceil(events / event_probability)}


def _e(rr: float) -> float:
    rr = 1 / rr if rr < 1 else rr
    return rr + math.sqrt(rr * (rr - 1))


def evalue_hr(hr: float, lower: float | None = None, upper: float | None = None, *,
              rare_outcome: bool = False) -> dict[str, float]:  # fmt: skip
    """E-values for a hazard ratio and its confidence limit closest to 1. For a common outcome
    (> 15% by the end of follow-up) the HR is first converted to an approximate risk ratio,
    RR = (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)) (VanderWeele 2017)."""

    def to_rr(h: float) -> float:
        if rare_outcome:
            return h
        return (1 - 0.5 ** math.sqrt(h)) / (1 - 0.5 ** math.sqrt(1 / h))

    out = {"evalue": _e(to_rr(hr))}
    if lower is not None and upper is not None:
        if lower <= 1 <= upper:
            out["evalue_ci"] = 1.0
        else:
            out["evalue_ci"] = _e(to_rr(upper if hr < 1 else lower))
    return out
