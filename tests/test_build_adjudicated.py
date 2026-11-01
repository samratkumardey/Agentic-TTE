"""Adjudicated references (eval/build_adjudicated_references.py): the investigator's reference
with each recorded decision applied, and nothing else changed."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).parents[1] / "eval"
sys.path.insert(0, str(EVAL))
_spec = importlib.util.spec_from_file_location("build_adjudicated_references",
                                               EVAL / "build_adjudicated_references.py")
bar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bar)
from make_reference_templates import template  # noqa: E402


def _pair(study):
    t = {k: v for k, v in template(study).items() if not k.startswith("_")}
    for i, a in enumerate(t["treatment_strategies"]):
        a["is_comparator"] = i > 0
        if i == 0:
            a["components"] = [{"concept": {"name": "hydrocortisone", "domain": "Drug"}}]
    t["time_zero"]["anchor_event"] = "septic shock"
    for c in t["eligibility"]:
        c["atoms"] = [{"concepts": [{"name": "septic shock", "domain": "Condition"}]}]
    t["outcomes"][0]["horizon"] = {"amount": 90, "unit": "day"}
    t["assignment"]["confounders"] = [{"name": "age", "domain": "Demographic"},
                                      {"name": "time to treatment", "domain": "Observation"}]
    ai = copy.deepcopy(t)
    ai["eligibility"][0]["atoms"] = [{"concepts": [{"name": "vasopressor", "domain": "Drug"}]}]
    ai["treatment_strategies"][0]["grace_period"] = {"amount": 6, "unit": "hour"}
    ai["outcomes"][0]["end_event"] = "hospital discharge"
    return t, ai


def _d(i, el, item, dec):
    return {"id": i, "nct_id": "NCT03872011", "element": el, "item": item, "decision": dec}


def test_decisions_are_applied_and_nothing_else(nct03872011_study):
    skd, ai = _pair(nct03872011_study)
    arm = skd["treatment_strategies"][0]["label"]
    cid = skd["eligibility"][0]["id"]
    out = bar.apply(skd, ai, [
        _d(0, "eligibility", f"{cid} concepts", "AI"),
        _d(1, "treatment", f"arm '{arm}' grace period (hours)", "AI"),
        _d(2, "outcome", "end_event", "AI"),
        _d(3, "confounders", "only one annotator listed",
           {"add": [{"name": "SOFA score", "domain": "Measurement"}],
            "remove": ["time to treatment"]}),
        _d(4, "time_zero", "anchor event", "SKD")])
    assert out["eligibility"][0]["atoms"] == ai["eligibility"][0]["atoms"]
    assert out["eligibility"][1:] == skd["eligibility"][1:]
    assert out["treatment_strategies"][0]["grace_period"] == {"amount": 6, "unit": "hour"}
    assert out["outcomes"][0]["end_event"] == "hospital discharge"
    assert [c["name"] for c in out["assignment"]["confounders"]] == ["age", "SOFA score"]
    assert out["time_zero"] == skd["time_zero"]
    assert skd["eligibility"][0]["atoms"][0]["concepts"][0]["name"] == "septic shock"  # unchanged


def test_open_or_unknown_decisions_are_refused(nct03872011_study):
    skd, ai = _pair(nct03872011_study)
    with pytest.raises(ValueError, match="still open"):
        bar.apply(skd, ai, [_d(0, "outcome", "end_event", None)])
    with pytest.raises(ValueError, match="not listed"):
        bar.apply(skd, ai, [_d(0, "confounders", "x", {"remove": ["no such confounder"]})])
