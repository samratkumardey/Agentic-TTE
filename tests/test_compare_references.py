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


def test_comparator_error_does_not_hide_the_treatment_arm(nct03872011_study):
    # A draft that calls every arm a comparator (registry ACTIVE_COMPARATOR labels) is wrong on
    # the comparator, and its treatment arm is still scored: no grace period, no components.
    a = _filled(nct03872011_study)
    a["treatment_strategies"][0]["grace_period"] = {"amount": 6, "unit": "hour"}
    b = copy.deepcopy(a)
    for arm in b["treatment_strategies"]:
        arm["is_comparator"], arm["grace_period"], arm["components"] = True, None, []
    r = cr.compare(*(TargetTrialProtocol.model_validate(x) for x in (a, b)), {})
    t = r["treatment"]
    assert not t["comparator_agree"] and t["grace_agree"] is False
    assert t["components"].recall == 0


def test_e3_eligibility_grade(nct03872011_study):
    se = _load("score_e3")
    a = _filled(nct03872011_study)
    p = TargetTrialProtocol.model_validate(a)
    assert se.eligibility_grade(p, p)["grade_strict"] == "correct"
    b = copy.deepcopy(a)
    for c in b["eligibility"]:
        c["atoms"][0]["concepts"][0]["name"] = "cardiogenic shock"
    g = se.eligibility_grade(p, TargetTrialProtocol.model_validate(b))
    assert g["strict"] == 0 and g["grade_strict"] == "wrong"


def test_e3_time_zero_summary_waits_for_every_rating():
    se = _load("score_e3")
    sheet = [{"rating": "same"}, {"rating": "different"}, {"rating": None}]
    assert "not reported" in se.t0_summary(sheet)
    sheet[2]["rating"] = "same"
    assert se.t0_summary(sheet) == "same 2/3, acceptable alternative 0/3, different 1/3"
