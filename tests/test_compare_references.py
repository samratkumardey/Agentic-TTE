"""E3 reference agreement (eval/compare_references.py): identical references agree on every
element; notation differences that compile the same are not counted as disagreements by the
compile-equivalent measure; real differences are listed for adjudication."""

import copy
import importlib.util
import sys
from pathlib import Path

from agentic_tte.protocol import TargetTrialProtocol

EVAL = Path(__file__).parents[1] / "eval"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mrt = _load("make_reference_templates")
cr = _load("compare_references")


def _filled(study) -> dict:
    t = {k: v for k, v in mrt.template(study).items() if not k.startswith("_")}
    for i, a in enumerate(t["treatment_strategies"]):
        a["is_comparator"] = i > 0
        if i == 0:
            a["components"] = [{"concept": {"name": "hydrocortisone", "domain": "Drug"}}]
    t["time_zero"]["anchor_event"] = "ICU admission with septic shock"
    for c in t["eligibility"]:
        c["atoms"] = [{"concepts": [{"name": "Septic shock", "domain": "Condition"}],
                       "value": {"comparator": ">=", "value": 18}}]
    t["outcomes"][0]["horizon"] = {"amount": 90, "unit": "day"}
    t["assignment"]["confounders"] = [{"name": "age", "domain": "Demographic"},
                                      {"name": "SOFA score", "domain": "Measurement"}]
    return t


def test_identical_references_agree(nct03872011_study):
    p = TargetTrialProtocol.model_validate(_filled(nct03872011_study))
    r = cr.compare(p, p, {})
    assert r["eligibility"]["strict"].f1 == 1 and r["eligibility"]["compiled"].f1 == 1
    assert r["treatment"]["comparator_agree"] and r["treatment"]["grace_agree"]
    assert all(r["outcome"].values()) and r["confounders"].f1 == 1
    # Only the time-zero rating, which is never automatic.
    assert [d["element"] for d in r["disagreements"]] == ["time_zero"]


def test_notation_vs_substance(nct03872011_study):
    a = _filled(nct03872011_study)
    b = copy.deepcopy(a)
    for c in b["eligibility"]:  # explicit "at time zero" and a unit: compiles the same
        c["atoms"][0]["temporal"] = {"relation": "x_at_time_of_y"}
        c["atoms"][0]["value"]["unit"] = "year"
        c["atoms"][0]["concepts"][0]["name"] = "septic shock"
    b["treatment_strategies"][0]["grace_period"] = {"amount": 0, "unit": "hour"}
    pa, pb = (TargetTrialProtocol.model_validate(x) for x in (a, b))
    r = cr.compare(pa, pb, {})
    assert r["eligibility"]["strict"].f1 == 0 and r["eligibility"]["compiled"].f1 == 1
    assert r["treatment"]["grace_agree"]  # 0 h is no grace period

    b["treatment_strategies"][0]["grace_period"] = {"amount": 6, "unit": "hour"}
    b["outcomes"][0]["horizon"] = {"amount": 28, "unit": "day"}
    b["assignment"]["confounders"] = [{"name": "age", "domain": "Demographic"}]
    r = cr.compare(pa, TargetTrialProtocol.model_validate(b), {})
    items = {(d["element"], d["item"]) for d in r["disagreements"]}
    assert ("outcome", "horizon_hours") in items and ("confounders",
                                                      "only one annotator listed") in items
    assert any(e == "treatment" and "grace" in i for e, i in items)


def test_lenient_names():
    assert cr.lenient("PaO2/FiO2 ratio", "pao2 fio2 ratio")
    assert cr.lenient("Invasive mechanical ventilation", "invasive mechanical ventilation")
    assert cr.lenient("serum lactate", "arterial serum lactate")
    assert not cr.lenient("age", "pregnancy")
