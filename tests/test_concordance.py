"""The Concordance agent (agentic_tte/agents/concordance.py): the R4 agreement metrics of
eval/SCORING.md, measure and orientation handling, failures counted as no agreement, and the
set summaries (Clopper-Pearson, REML pooled ratio of ratios)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from agentic_tte.agents.concordance import (
    Emulation,
    Reference,
    agree,
    clopper_pearson,
    measure_kind,
    orientation,
    reml,
    summarize,
    to_markdown,
)
from agentic_tte.graph.replay import result
from tests.test_clinician import drafted  # noqa: F401

Z = 1.959964


def _emu(kind="HR", v=0.70, lo=0.55, hi=0.89, status="signed_off", t=("Cisatracurium",),
         c=("Placebo",), boot=(500, 20261027)) -> Emulation:  # fmt: skip
    analysis = {"primary_method": "iptw", "estimates": [
        {"method": "iptw", "hr": v, "lower": lo, "upper": hi, "p": 0.01, "n": 900,
         "events": 300}]}  # fmt: skip
    if kind != "HR":
        key = {"RR": "rr", "OR": "or_", "RD": "rd"}[kind]
        analysis["risk_measures"] = {"rr": None, "or_": None, "rd": 0.0, key: v,
                                     "ci": {key: [lo, hi]}, "n_bootstrap": boot[0],
                                     "seed": boot[1]}  # fmt: skip
    return Emulation(nct_id="NCT0", status=status, treatment=list(t), comparator=list(c),
                     analysis=analysis)  # fmt: skip


ACURASYS = Reference(nct_id="NCT00299650", trial="ACURASYS", measure="HR (adjusted)", value=0.68,
                     ci=(0.48, 0.98), contrast="cisatracurium vs placebo")  # fmt: skip


def test_measures_and_orientation():
    assert measure_kind("HR (adjusted for PaO2:FiO2)") == ("HR", False)
    assert measure_kind("RD in survival (percentage points)") == ("RD", True)
    assert measure_kind("RD (percentage points, adjusted for site)") == ("RD", False)
    assert measure_kind("median days") == (None, False)
    assert orientation("cisatracurium vs placebo", ["Cisatracurium"], ["Placebo"]) == "same"
    assert orientation("saline vs bicarbonate (as reported)", ["sodium bicarbonate"],
                       ["saline"]) == "reversed"  # fmt: skip
    assert orientation("ECLS vs usual care", ["ECLS"], ["No ECLS"]) == "same"
    assert orientation("accelerated vs standard RRT initiation",
                       ["Accelerated RRT initiation"], ["Standard RRT initiation"]) == "same"
    # No treatment arm in the protocol (a broken draft): never guessed.
    assert orientation("hydrocortisone vs placebo", [], ["Hydrocortisone"]) == "unconfirmed"
    assert orientation("drug A or B", ["A"], ["B"]) == "unconfirmed"


def test_hr_agreement_by_hand():
    r = agree(ACURASYS, _emu())
    te, se_e = math.log(0.70), (math.log(0.89) - math.log(0.55)) / (2 * Z)
    tr, se_r = math.log(0.68), (math.log(0.98) - math.log(0.48)) / (2 * Z)
    assert r.scored and r.orientation == "same"
    assert r.z == pytest.approx((te - tr) / math.sqrt(se_e**2 + se_r**2), rel=1e-4)
    assert r.standardized and r.estimate and r.regulatory and r.direction
    assert r.ror[0] == pytest.approx(0.70 / 0.68, rel=1e-6)
    assert r.ci_width == pytest.approx(0.89 / 0.55, rel=1e-3)
    # Significant harm against significant benefit: no agreement of any kind.
    r = agree(ACURASYS, _emu(v=1.40, lo=1.15, hi=1.70))
    assert r.scored and not (r.standardized or r.estimate or r.regulatory or r.direction)
    # Both null on opposite sides: direction agrees (both CIs contain the null).
    null = ACURASYS.model_copy(update={"value": 0.95, "ci": (0.80, 1.13)})
    r = agree(null, _emu(v=1.05, lo=0.90, hi=1.22))
    assert r.direction and r.regulatory and r.standardized


def test_risk_differences_in_points_reversed_and_survival():
    hydrarea = Reference(nct_id="NCT01636089", measure="RD (percentage points)", value=-1.8,
                         ci=(-12.3, 8.9), contrast="saline vs bicarbonate")  # fmt: skip
    r = agree(hydrarea, _emu("RD", 0.02, -0.03, 0.07, t=("bicarbonate",), c=("saline",)))
    assert r.orientation == "reversed" and r.theta_e == pytest.approx(-2.0)
    assert r.estimate and "reverse" in " ".join(r.notes) and r.ror is None
    evk = Reference(nct_id="NCT02643381", measure="RD in survival (percentage points)",
                    value=-7.8, ci=(-13.0, -2.4), contrast="etomidate vs ketamine")
    r = agree(evk, _emu("RD", 0.06, 0.02, 0.10, t=("Etomidate",), c=("Ketamine",)))
    assert r.theta_e == pytest.approx(-6.0) and r.estimate and r.regulatory
    assert "survival" in " ".join(r.notes)


def test_failures_and_unconfirmed_orientation():
    rr = Reference(nct_id="NCT1", measure="RR", value=0.88, ci=(0.78, 0.99),
                   contrast="hydrocortisone + fludrocortisone vs placebo")
    rows = [agree(rr, None),
            agree(rr, _emu(status="stopped").model_copy(update={"stop_reason": "audit"})),
            agree(rr, _emu()),  # an HR-only analysis: no bootstrap risk measures
            agree(rr, _emu("RR", 0.9, 0.8, 1.0, t=(), c=("Placebo",))),
            agree(rr, _emu("RR", 0.9, 0.8, 1.0, t=(), c=("Placebo",)), confirmed="same"),
            agree(rr, _emu("RR", 0.9, 0.8, 1.0, t=("Hydrocortisone",), boot=(100, 1)))]
    reasons = [r.reason for r in rows]
    assert reasons[0] == "no run" and "status stopped: audit" in reasons[1]
    assert "bootstrap not run" in reasons[2] and "orientation unconfirmed" in reasons[3]
    for r in rows[:4]:
        assert not (r.scored or r.standardized or r.estimate or r.regulatory or r.direction)
    assert rows[4].scored and "confirmed by a person" in " ".join(rows[4].notes)
    assert rows[5].scored and "not the pre-registered" in " ".join(rows[5].notes)
    s = summarize("core", rows)
    assert s.n == 6 and s.unconfirmed == 1 and s.failures == 3
    assert s.standardized == (2, 5)  # the unconfirmed trial waits outside the denominator
    md = to_markdown({"core": rows}, run="runs/test")
    assert "2/5 (40%)" in md and "orientation awaiting confirmation 1" in md


def test_set_statistics():
    lo, hi = clopper_pearson(5, 10)
    assert lo == pytest.approx(0.1871, abs=1e-4) and hi == pytest.approx(0.8129, abs=1e-4)
    assert clopper_pearson(0, 4)[0] == 0.0 and clopper_pearson(4, 4)[1] == 1.0
    mu, se, t2 = reml(np.array([0.1, 0.1, 0.1]), np.array([0.01, 0.02, 0.04]))
    assert t2 == 0.0 and mu == pytest.approx(0.1)
    w = 1 / np.array([0.01, 0.02, 0.04])
    assert se == pytest.approx(math.sqrt(1 / w.sum()))
    _, _, t2 = reml(np.array([-0.6, 0.0, 0.6, 0.9]), np.full(4, 0.01))
    assert t2 > 0.2  # clear heterogeneity
    rows = [agree(ACURASYS, _emu()), agree(ACURASYS, _emu(v=0.9, lo=0.7, hi=1.15))]
    s = summarize("core", rows)
    assert s.n_ror == 2 and s.pooled_ror[1] < s.pooled_ror[0] < s.pooled_ror[2]


def test_result_json_from_a_run(drafted):  # noqa: F811
    r = result({"protocol": drafted.model_dump(mode="json"), "status": "signed_off",
                "analysis": {"estimates": []}})  # fmt: skip
    arms = {a.label: a.is_comparator for a in drafted.treatment_strategies}
    assert r["nct_id"] == drafted.source.nct_id and r["status"] == "signed_off"
    assert [n for n in r["treatment"] if n in arms] == [k for k, v in arms.items() if not v]
    assert Emulation.model_validate(r).protocol_hash == drafted.content_hash()
    assert result({"nct_id": "NCT9", "status": "stopped"})["treatment"] == []


def test_sets_come_from_the_selection_file():
    sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
    from score_concordance import references, sets

    refs = references()
    s = sets(refs)
    core = s["Core set (primary agreement set)"]
    assert len(core) == 12 and "NCT00299650" in core and "NCT02856698" not in core
    assert len(s["Breadth set (secondary, exploratory)"]) == 10
    assert all(measure_kind(refs[n].measure)[0] for ids in s.values() for n in ids)
