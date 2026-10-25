import numpy as np
import pandas as pd
import pytest

from agentic_tte.synth.dgp import Scenario, generate
from agentic_tte.synth.omop import (
    SEPTIC_SHOCK,
    STUDY_DRUG,
    analysis_frame,
    to_omop,
    write_duckdb,
)


def test_reproducible_from_seed():
    a = generate(Scenario(n=300), with_truth=False).frame
    b = generate(Scenario(n=300), with_truth=False).frame
    pd.testing.assert_frame_equal(a, b)
    c = generate(Scenario(n=300, seed=1), with_truth=False).frame
    assert not a["time"].equals(c["time"])


def test_frame_is_well_formed():
    f = generate(Scenario(n=2000, timing="delayed", missing_lactate=0.2), with_truth=False).frame
    assert (f["time"] > 0).all() and (f["time"] <= 28).all()
    assert set(f["event"]) <= {0, 1}
    treated = f["treated"] == 1
    assert f.loc[treated, "treat_start"].between(0, 3).all()
    assert (f.loc[treated, "treat_start"] < f.loc[treated, "time"]).all()  # alive at start
    assert f.loc[~treated, "treat_start"].isna().all()
    assert 0.15 < f["lactate"].isna().mean() < 0.25


def test_confounding_by_indication_is_planted():
    f = generate(Scenario(n=4000), with_truth=False).frame
    assert f.loc[f.treated == 1, "sofa"].mean() > f.loc[f.treated == 0, "sofa"].mean() + 1


@pytest.mark.parametrize("hr", [0.5, 3.0])
def test_marginal_truth_is_closer_to_null_than_conditional(hr):
    """Non-collapsibility: with prognostic covariates the marginal HR lies between the
    conditional HR and 1."""
    t = generate(Scenario(n=10, true_hr=hr, truth_n=40_000)).truth
    assert t["conditional_hr"] == pytest.approx(hr)
    lo, hi = sorted((hr, 1.0))
    assert lo < t["marginal_hr"] < hi
    assert np.sign(t["risk_difference"]) == np.sign(np.log(hr))


def test_null_truth_is_exactly_null():
    t = generate(Scenario(n=10, true_hr=1.0, timing="delayed", truth_n=20_000)).truth
    assert t["marginal_hr"] == pytest.approx(1.0) and t["risk_difference"] == 0.0


def test_omop_round_trip(tmp_path):
    c = generate(Scenario(n=400, timing="delayed", missing_lactate=0.1), with_truth=False)
    tables = to_omop(c)
    con = write_duckdb(tables, tmp_path / "synth.duckdb")
    a, f = analysis_frame(con), c.frame
    assert list(a["person_id"]) == list(f["person_id"])
    for col in ("age", "sofa", "lactate", "treat_start", "time"):
        assert (a[col].isna() == f[col].isna()).all()
        np.testing.assert_allclose(a[col].fillna(-1), f[col].fillna(-1), atol=1e-4)
    for col in ("female", "treated", "event"):
        assert (a[col].to_numpy() == f[col].to_numpy()).all()
    concepts = set(tables["concept"]["concept_id"])
    assert {SEPTIC_SHOCK, STUDY_DRUG, 8507, 8532} <= concepts
    local = tables["concept"][tables["concept"]["concept_id"] >= 2_000_000_000]
    assert local["standard_concept"].isna().all()  # local IDs never pose as standard
