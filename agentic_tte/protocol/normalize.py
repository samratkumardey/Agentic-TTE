"""Deterministic normalizers for value, duration and temporal phrases.

These run before (and as a check on) any LLM output, so simple cases such as "18 Years" or
"90 days after randomization" never depend on the model.
"""

from __future__ import annotations

import re

from agentic_tte.protocol.models import (
    Comparator,
    Duration,
    TemporalConstraint,
    TemporalRelation,
    TimeUnit,
    ValueConstraint,
)

_NUM = r"(\d+(?:\.\d+)?)"
_UNIT_ALIASES = {
    "min": TimeUnit.MINUTE, "mins": TimeUnit.MINUTE, "minute": TimeUnit.MINUTE,
    "minutes": TimeUnit.MINUTE,
    "h": TimeUnit.HOUR, "hr": TimeUnit.HOUR, "hrs": TimeUnit.HOUR, "hour": TimeUnit.HOUR,
    "hours": TimeUnit.HOUR,
    "d": TimeUnit.DAY, "day": TimeUnit.DAY, "days": TimeUnit.DAY,
    "wk": TimeUnit.WEEK, "wks": TimeUnit.WEEK, "week": TimeUnit.WEEK, "weeks": TimeUnit.WEEK,
    "mo": TimeUnit.MONTH, "month": TimeUnit.MONTH, "months": TimeUnit.MONTH,
    "y": TimeUnit.YEAR, "yr": TimeUnit.YEAR, "yrs": TimeUnit.YEAR, "year": TimeUnit.YEAR,
    "years": TimeUnit.YEAR,
}  # fmt: skip
_UNIT_RE = "|".join(sorted(_UNIT_ALIASES, key=len, reverse=True))
_DURATION_RE = re.compile(
    rf"{_NUM}\s*-?\s*(?:(?:consecutive|continuous|full|calendar)\s+)?({_UNIT_RE})\b",
    re.IGNORECASE,
)


def parse_duration(text: str | None) -> Duration | None:
    """First "<number> <time unit>" in ``text``: "90 days after randomization" -> 90 day,
    "28-day mortality" -> 28 day. Returns None when there is no explicit duration."""
    if not text:
        return None
    m = _DURATION_RE.search(text)
    if not m:
        return None
    return Duration(amount=float(m.group(1)), unit=_UNIT_ALIASES[m.group(2).lower()])


_GE = r"(?:or (?:older|more|greater|above|higher)|at least|minimum|≥|>=|no (?:less|fewer) than)"
_LE = (
    r"(?:or (?:younger|less|fewer|below|lower)|at most|maximum|up to|≤|<="
    r"|no (?:more|greater) than)"
)
_GT = r"(?:\b(?:greater than|more than|older than|over|above|exceed(?:s|ing)?)\b|>)"
_LT = r"(?:\b(?:less than|fewer than|younger than|under|below)\b|<)"


def parse_value(text: str | None) -> ValueConstraint | None:
    """Parse a numeric comparison: "18 years old or older" -> >= 18 years; "SBP < 90 mmHg" ->
    < 90 mmHg; "between 18 and 65 years" -> between. Qualitative values ("high-dose") give
    PRESENT. Returns None for empty text."""
    if not text or not text.strip():
        return None
    t = text.strip()
    low = t.lower()

    m = re.search(rf"between\s+{_NUM}\s*(?:and|-|to)\s*{_NUM}\s*([^\s,;.)]*)", low)
    if m:
        return ValueConstraint(
            comparator=Comparator.BETWEEN,
            value=float(m.group(1)),
            upper=float(m.group(2)),
            unit=m.group(3) or None,
            text=t,
        )
    num = re.search(rf"{_NUM}\s*([a-zA-Z/%µ][\w/%µ^.]*)?", t)
    if not num:
        return ValueConstraint(comparator=Comparator.PRESENT, text=t)
    value = float(num.group(1))
    unit = num.group(2)
    if unit and unit.lower() in {"or", "and", "old", "at"}:
        unit = None
    for pattern, comp in ((_GE, Comparator.GE), (_LE, Comparator.LE), (_GT, Comparator.GT),
                          (_LT, Comparator.LT)):  # fmt: skip
        if re.search(pattern, low):
            return ValueConstraint(comparator=comp, value=value, unit=unit, text=t)
    return ValueConstraint(comparator=Comparator.EQ, value=value, unit=unit, text=t)


def parse_age(text: str | None) -> tuple[float, str] | None:
    """CT.gov structured ages: "18 Years" -> (18.0, "years")."""
    if not text:
        return None
    m = re.match(rf"\s*{_NUM}\s*(\w+)", text)
    if not m:
        return None
    return float(m.group(1)), m.group(2).lower()


def parse_temporal(texts: list[str]) -> TemporalConstraint | None:
    """Combine annotated temporal fragments ("within the last 3 months", "before") into one
    constraint. A bare "within t" in an eligibility criterion means the event happened within
    t before time zero."""
    joined = " ".join(t for t in texts if t).strip()
    if not joined:
        return None
    low = joined.lower()
    window = parse_duration(joined)
    after = re.search(r"\bafter\b|\bfollowing\b", low)
    before = re.search(r"\bbefore\b|\bprior\b|\blast\b|\bpast\b|\bprevious\b|\bpreceding\b", low)
    cumulative = re.search(r"\bfor at least\b|\bcumulative\b|\bconsecutive\b|\bduration\b", low)
    if cumulative and window:
        relation = TemporalRelation.CUMULATIVE_DURATION
    elif window:
        relation = (
            TemporalRelation.X_WITHIN_T_AFTER_Y
            if after and not before
            else TemporalRelation.X_WITHIN_T_BEFORE_Y
        )
    elif after and not before:
        relation = TemporalRelation.X_AFTER_Y
    elif before:
        relation = TemporalRelation.X_BEFORE_Y
    elif re.search(r"\bat (?:the )?time of\b|\bat admission\b|\bduring\b", low):
        relation = TemporalRelation.X_AT_Y
    else:
        return None
    return TemporalConstraint(relation=relation, window=window, text=joined)
