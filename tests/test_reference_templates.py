"""E3 reference-protocol templates (eval/make_reference_templates.py): built from the registry
record only, rejected while blank, accepted once every element is filled."""

import importlib.util
import json
import sys
from pathlib import Path

EVAL = Path(__file__).parents[1] / "eval"
_spec = importlib.util.spec_from_file_location("make_reference_templates",
                                               EVAL / "make_reference_templates.py")
mrt = importlib.util.module_from_spec(_spec)
sys.modules["make_reference_templates"] = mrt
_spec.loader.exec_module(mrt)


def test_template_holds_registry_facts_only(nct03872011_study):
    t = mrt.template(nct03872011_study)
    assert t["source"]["nct_id"] == "NCT03872011"
    assert t["eligibility"] and all(c["atoms"] == [] for c in t["eligibility"])
    assert all(a["components"] == [] and a["is_comparator"] is None
               for a in t["treatment_strategies"])  # fmt: skip
    assert t["time_zero"]["anchor_event"] is None and t["assignment"]["confounders"] == []
    assert t["outcomes"][0]["time_frame"] and t["outcomes"][0]["horizon"] is None


def test_check_lists_what_is_missing_then_accepts(nct03872011_study, tmp_path):
    t = mrt.template(nct03872011_study)
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(t), encoding="utf-8")
    blank = mrt.check(path)
    assert any("is_comparator" in x for x in blank) and "time zero: no anchor event" in blank

    for i, a in enumerate(t["treatment_strategies"]):
        a["is_comparator"] = i > 0
        if i == 0:
            a["components"] = [{"concept": {"name": "hydrocortisone", "domain": "Drug"}}]
    t["time_zero"]["anchor_event"] = "ICU admission with septic shock"
    path.write_text(json.dumps(t), encoding="utf-8")
    todo = mrt.check(path)
    assert any("no atoms" in x for x in todo) and "primary outcome: no horizon" in todo
    assert "assignment: no confounders" in todo

    for c in t["eligibility"]:
        c["atoms"] = [{"concepts": [{"name": "septic shock", "domain": "Condition"}]}]
    t["outcomes"][0]["horizon"] = {"amount": 90, "unit": "day"}
    t["assignment"]["confounders"] = [{"name": "age", "domain": "Demographic"}]
    path.write_text(json.dumps(t), encoding="utf-8")
    assert mrt.check(path) == []


def test_core_trials_come_from_the_selection():
    trials = mrt.core_trials()
    assert len(trials) == 12 and "NCT01448109" in trials  # ADRENAL
    assert sorted(p.stem for p in (EVAL / "reference_protocols" / "templates").glob("*.json")) \
        == sorted(trials)  # fmt: skip
