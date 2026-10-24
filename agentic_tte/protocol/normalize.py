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


def durations_in(text: str | None) -> list[Duration]:
    """Every "<number> <time unit>" in ``text``."""
    return [
        Duration(amount=float(m.group(1)), unit=_UNIT_ALIASES[m.group(2).lower()])
        for m in _DURATION_RE.finditer(text or "")
    ]


def as_time_unit(unit: str | None) -> TimeUnit | None:
    """"hours" -> hour; None for non-time units such as "mmol/L"."""
    return _UNIT_ALIASES.get((unit or "").strip().lower())


_END_EVENT_RE = re.compile(
    r"\bor\s+(?:until\s+)?(?:(icu|hospital)\s+)?discharge(?:\s+from\s+(?:the\s+)?(icu|hospital))?",
    re.IGNORECASE,
)


def parse_time_frame(text: str | None) -> tuple[Duration | None, str | None, str | None]:
    """"28 days or discharge (whichever comes first)" -> (28 day, "hospital discharge", "first").
    Returns (horizon, end_event, end_rule); end_rule defaults to "first" for "or until X"."""
    horizon = parse_duration(text)
    if not text:
        return horizon, None, None
    m = _END_EVENT_RE.search(text)
    if not m:
        return horizon, None, None
    place = (m.group(1) or m.group(2) or "hospital").lower()
    event = "ICU discharge" if place == "icu" else "hospital discharge"
    rule = "last" if re.search(r"whichever (?:comes |is )?(?:last|later)", text, re.I) else "first"
    return horizon, event, rule


_FREQUENCY_BEFORE = re.compile(r"(?:\bevery|\bq|\bper|/)\s*$", re.IGNORECASE)


def parse_course_duration(text: str | None) -> Duration | None:
    """Treatment course length, skipping dosing intervals: in "50 mg, every 6 hours, 7 days"
    the course is 7 days, not 6 hours."""
    if not text:
        return None
    for m in _DURATION_RE.finditer(text):
        if _FREQUENCY_BEFORE.search(text[max(0, m.start() - 8) : m.start()]):
            continue
        return Duration(amount=float(m.group(1)), unit=_UNIT_ALIASES[m.group(2).lower()])
    return None


_TIME_ZERO_ANCHORS = re.compile(
    r"^\s*(?:the\s+)?(?:time\s+of\s+)?(?:randomi[sz]ation|enrol?l?ment|study\s+entry|"
    r"inclusion|screening|baseline|time[_ ]zero|study\s+start)\s*$",
    re.IGNORECASE,
)


def normalize_anchor(anchor: str | None) -> str:
    """Randomization, enrollment and similar anchors all mean time zero in an emulation (C6)."""
    if not anchor or _TIME_ZERO_ANCHORS.match(anchor):
        return "time_zero"
    return anchor.strip()


_STOPWORDS = {"a", "an", "and", "or", "of", "to", "the", "in", "on", "with", "for", "at", "by",
              "not", "no", "any", "is", "as", "be"}  # fmt: skip


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def grounded_fraction(span: str | None, source: str) -> float:
    """Share of the content words in ``span`` that occur in ``source`` (1.0 for empty spans).
    Used to catch text the model wrote that the criterion never said (C5)."""
    words = _words(span or "") - _STOPWORDS
    if not words:
        return 1.0
    return len(words & _words(source)) / len(words)


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
