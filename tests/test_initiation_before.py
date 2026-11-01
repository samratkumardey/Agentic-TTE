"""A treatment strategy that starts shortly before the event defining time zero (HYDRAREA:
hydration from 1 hour before contrast; SCORING.md Phase 3 changelog item 26). Initiation in the
stated window before time zero counts as initiation at time zero; earlier records still count
as treatment before time zero (Auditor A-c)."""

import importlib.util
import sys
from pathlib import Path

import pytest

from agentic_tte.agents.auditor import audit, data_summary
from agentic_tte.omop.compiler import compile_protocol
from agentic_tte.protocol.models import Duration
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


def _design(window: Duration | None):
    """The synthetic design with time zero 2 days after ICU admission, so most treatment
    starts fall before it. The new-user exclusion is dropped: it would exclude them."""
    p, spec = fi.synthetic_design()
    p.eligibility = [c for c in p.eligibility if c.id != "E-new-user"]
    p.time_zero.offset = Duration(amount=2, unit="day")
    p.treatment_strategies[0].initiation_before = window
    return p, spec


def _run(con, window):
    p, spec = _design(window)
    cc = compile_protocol(p, spec)
    frame = con.execute(cc.frame_sql).df()
    summary = data_summary(con, p, spec, cc)
    rep = audit(p, spec, compiled=cc, summary=summary, approved_sets=dict(spec.concept_sets))
    return frame, summary, rep, cc


def test_window_counts_earlier_starts_as_treated_at_time_zero(con):
    f0, s0, r0, _ = _run(con, None)
    f1, s1, r1, cc = _run(con, Duration(amount=2, unit="day"))
    assert len(f1) == len(f0)  # the window changes who is treated, not who is eligible
    assert f1.treated.sum() > f0.treated.sum()
    assert (f1.treat_start.dropna() >= 0).all()  # never before time zero in the frame
    assert any("before time zero counts as initiation" in n for n in cc.frame_notes)
    # A-c: without the window, treatment before time zero is flagged; inside it, it is not.
    assert "A-c" in r0.failed and s0["before_t0_share"] > 0.05
    assert "A-c" not in r1.failed and s1["before_t0_share"] == 0


def test_starts_before_the_window_are_still_flagged(con):
    _, s, rep, _ = _run(con, Duration(amount=6, unit="hour"))
    assert "A-c" in rep.failed and s["before_t0_share"] > 0.05


def test_no_window_leaves_the_frame_unchanged(con):
    p, spec = fi.synthetic_design()
    assert p.treatment_strategies[0].initiation_before is None
    cc = compile_protocol(p, spec)
    assert not any("before time zero counts" in n for n in cc.frame_notes)
