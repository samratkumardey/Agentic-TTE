"""Protocol Auditor and fault injection (E1, eval/SCORING.md Phase 3 changelog items 11 and 14)
on synthetic OMOP: every planted fault is caught by its own check, the correct design raises no
alarm, and the faults that bias the estimate do so in the expected direction."""

import importlib.util
import sys
from pathlib import Path

import pytest

from agentic_tte.agents.auditor import audit, data_summary
from agentic_tte.faults import FAULTS, NotApplicable, inject
from agentic_tte.omop.compiler import compile_protocol
from agentic_tte.synth import omop as so
from agentic_tte.synth.dgp import Scenario, generate

EVAL = Path(__file__).parents[1] / "eval"
_spec = importlib.util.spec_from_file_location("fault_injection", EVAL / "fault_injection.py")
fi = importlib.util.module_from_spec(_spec)
sys.modules["fault_injection"] = fi
_spec.loader.exec_module(fi)


@pytest.fixture(scope="module")
def con():
    c = generate(Scenario(n=800, seed=7, timing="delayed", grace_days=fi.GRACE_DAYS),
                 with_truth=False)  # fmt: skip
    return so.write_duckdb(so.to_omop(c))


def _audit(con, p, spec, approved):
    cc = compile_protocol(p, spec)
    return audit(p, spec, compiled=cc, summary=data_summary(con, p, spec, cc),
                 approved_sets=approved)  # fmt: skip


def test_the_correct_design_raises_no_alarm(con):
    p, spec = fi.synthetic_design()
    rep = _audit(con, p, spec, dict(spec.concept_sets))
    assert rep.passed, rep.to_markdown()
    assert not [f for f in rep.findings if f.severity == "warn"]
    s = data_summary(con, p, spec, compile_protocol(p, spec))
    assert s["before_t0_share"] == 0 and s["late_start_share"] > 0.5  # delayed starts


@pytest.mark.parametrize("fault", list(FAULTS))
def test_each_fault_is_caught_by_its_check(con, fault):
    p, spec = fi.synthetic_design()
    fp, fs, _ = inject(fault, p, spec, wrong_concept=fi.WRONG_CONCEPT)
    rep = _audit(con, fp, fs, dict(spec.concept_sets))
    assert FAULTS[fault][1] in rep.failed, rep.to_markdown()


def test_faults_leave_the_original_design_unchanged():
    p, spec = fi.synthetic_design()
    before = (p.content_hash(), spec.model_dump_json())
    for f in FAULTS:
        inject(f, p, spec, wrong_concept=fi.WRONG_CONCEPT)
    assert (p.content_hash(), spec.model_dump_json()) == before


def test_inapplicable_faults_say_so():
    p, spec = fi.synthetic_design()
    p.treatment_strategies[0].grace_period = None
    for f in ("F-a", "F-i"):
        with pytest.raises(NotApplicable):
            inject(f, p, spec)
    with pytest.raises(NotApplicable):
        inject("F-j", *fi.synthetic_design())  # no wrong concept given


def test_immortal_time_faults_bias_toward_benefit():
    r = fi.run(n=1500, seed=11)
    base = r["rows"]["baseline"]["estimate"][0]
    for f in ("F-a", "F-i"):
        assert r["rows"][f]["estimate"][0] < base  # "ever treated" looks protective
        assert r["rows"][f]["method"] == "iptw"
    assert r["rows"]["baseline"]["failed"] == []
    assert all(FAULTS[f][1] in r["rows"][f]["failed"] for f in FAULTS)


def test_concept_names_check_without_reviewed_sets(con):
    p, spec = fi.synthetic_design()
    names = {so.STUDY_DRUG: "Study drug (synthetic)", fi.WRONG_CONCEPT: "Noise covariate x1"}
    assert "A-j" not in audit(p, spec, concept_names=names).failed
    fp, fs, _ = inject("F-j", p, spec, wrong_concept=fi.WRONG_CONCEPT)
    assert "A-j" in audit(fp, fs, concept_names=names).failed
    assert any(f.check == "A-j" and f.severity == "warn" for f in audit(p, spec).findings)


def test_missing_time_frame_is_a_warning_not_a_pass(con):
    p, spec = fi.synthetic_design()
    p.outcomes[0].time_frame = None
    rep = audit(p, spec, approved_sets=dict(spec.concept_sets))
    assert any(f.check == "A-e" and f.severity == "warn" for f in rep.findings)
    assert any(f.check == "A-a" and f.severity == "warn" for f in rep.findings)  # no frame
