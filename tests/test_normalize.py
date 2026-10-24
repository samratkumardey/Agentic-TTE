import pytest

from agentic_tte.protocol.annotation import parse_annotation
from agentic_tte.protocol.normalize import parse_age, parse_duration, parse_temporal, parse_value


@pytest.mark.parametrize(
    ("text", "amount", "unit"),
    [
        ("90 days after randomization", 90, "day"),
        ("28-day mortality", 28, "day"),
        ("within 12 hours", 12, "hour"),
        ("within the last 3 months", 3, "month"),
        ("6 wks", 6, "week"),
    ],
)
def test_parse_duration(text, amount, unit):
    d = parse_duration(text)
    assert (d.amount, d.unit) == (amount, unit)


def test_parse_duration_none():
    assert parse_duration("at ICU discharge") is None


@pytest.mark.parametrize(
    ("text", "comparator", "value", "unit"),
    [
        ("18 years old or older", ">=", 18, "years"),
        ("< 90 mmHg", "<", 90, "mmHg"),
        ("<= 30000 per cubic millimeter", "<=", 30000, "per"),
        ("greater than 2.0 mmol/L", ">", 2.0, "mmol/L"),
        ("at least 65", ">=", 65, None),
        ("under 18 years", "<", 18, "years"),
    ],
)
def test_parse_value(text, comparator, value, unit):
    v = parse_value(text)
    assert (v.comparator, v.value, v.unit) == (comparator, value, unit)


def test_parse_value_between_and_qualitative():
    v = parse_value("between 18 and 65 years")
    assert (v.comparator, v.value, v.upper) == ("between", 18, 65)
    assert parse_value("High-dose").comparator == "present"
    # "underlying" must not read as "under"
    assert parse_value("underlying 5 conditions").comparator == "="


def test_parse_age():
    assert parse_age("18 Years") == (18.0, "years")
    assert parse_age(None) is None


def test_parse_temporal():
    t = parse_temporal(["within the last 3 months", "before"])
    assert t.relation == "x_within_t_before_y" and t.window.amount == 3
    t = parse_temporal(["within 24 hours after ICU admission"])
    assert t.relation == "x_within_t_after_y"
    t = parse_temporal(["for at least 6 consecutive hours"])
    assert t.relation == "cumulative_duration"
    assert parse_temporal([]) is None


def test_parse_annotation_tags_and_malformed_close():
    ann = parse_annotation(
        "[Annotation] Patient <Demographic>aged<Demographic> > <Value>65 years old</Value> "
        "<Negation_cue>no</Negation_cue> <Condition>T2DM (Type 2 Diabetes Mellitus)</Condition> "
        "<Provider>enrolling physician</Provider>"
    )
    assert [(c.name, c.domain) for c in ann.concepts] == [
        ("aged", "Demographic"),
        ("T2DM (Type 2 Diabetes Mellitus)", "Condition"),
    ]
    assert ann.values == ["65 years old"]
    assert ann.negations == ["no"]
    assert ann.other == {"Provider": ["enrolling physician"]}
