"""E2 (a) on a made-up MIMIC-shaped OMOP database and made-up mimic-code exports: agreement,
each kind of disagreement, and patients with a time zero on one side only."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from agentic_tte.synth import omop as so

EVAL = Path(__file__).parents[1] / "eval"
_spec = importlib.util.spec_from_file_location("e2a_mimic_code", EVAL / "e2a_mimic_code.py")
e2a = importlib.util.module_from_spec(_spec)
sys.modules["e2a_mimic_code"] = e2a
_spec.loader.exec_module(e2a)

C = e2a.CONCEPTS
T = pd.Timestamp("2150-01-01 10:00")
H = pd.Timedelta(hours=1)


def _cdm():
    """1: positive on everything; 2: values outside the windows, norepinephrine from a
    prescription only; 3: 64 by year of birth (65 by mimic-code's anchor age); 4: dies 5 hours
    after ICU admission."""
    t = so._empty_events()
    birth = {1: T - pd.Timedelta(days=365.25 * 70), 2: T - pd.Timedelta(days=365.25 * 50),
             3: T - pd.Timedelta(days=365.25 * 64.5), 4: T - pd.Timedelta(days=365.25 * 80)}
    t["person"] = pd.DataFrame([{"person_id": i, "gender_concept_id": 8507,
                                 "year_of_birth": b.year, "month_of_birth": b.month,
                                 "day_of_birth": b.day, "birth_datetime": b}
                                for i, b in birth.items()])  # fmt: skip
    t["observation_period"] = pd.DataFrame({
        "person_id": list(birth), "observation_period_start_date": (T - 400 * 24 * H).date(),
        "observation_period_end_date": (T + 400 * 24 * H).date()})  # fmt: skip
    t["death"] = pd.DataFrame([{"person_id": 4, "death_date": (T + 5 * H).date(),
                                "death_datetime": T + 5 * H}])  # fmt: skip
    t["visit_occurrence"] = pd.DataFrame(
        {"person_id": pd.Series(dtype="int64"), "visit_concept_id": pd.Series(dtype="int64"),
         "visit_start_datetime": pd.Series(dtype="datetime64[ns]"),
         "visit_start_date": pd.Series(dtype="datetime64[ns]")})  # fmt: skip
    t["visit_detail"] = pd.DataFrame([{
        "visit_detail_id": 100 + i, "person_id": i, "visit_detail_concept_id": C["icu"],
        "visit_detail_start_datetime": T, "visit_detail_start_date": T.normalize()}
        for i in birth])  # fmt: skip

    def meas(p, concept, when, v, src):
        return {"person_id": p, "measurement_concept_id": concept, "measurement_datetime": when,
                "measurement_date": when.normalize(), "value_as_number": v,
                "measurement_source_value": src, "visit_occurrence_id": 200 + p}

    t["measurement"] = pd.DataFrame([
        meas(1, C["creatinine"], T + 2 * H, 2.5, "50912"),
        meas(1, C["lactate"], T + 2 * H, 1.5, "50813"),
        meas(1, C["lactate"], T + H, 3.0, "225668"),  # charted lactate: not in first_day_bg
        meas(2, C["creatinine"], T - 10 * H, 2.5, "50912"),
        meas(2, C["lactate"], T + 30 * H, 3.0, "50813")])

    def vent(p, when):
        return {"person_id": p, "procedure_concept_id": C["invasive_ventilation"],
                "procedure_datetime": when, "procedure_date": when.normalize(),
                "procedure_source_value": "225792"}

    t["procedure_occurrence"] = pd.DataFrame([vent(1, T + H), vent(2, T + 30 * H),
                                              vent(3, T + 2 * H)])
    t["drug_exposure"] = pd.DataFrame([
        {"person_id": p, "drug_concept_id": C["norepinephrine"],
         "drug_exposure_start_datetime": T + 3 * H if p == 1 else pd.NaT,
         "drug_exposure_start_date": (T + 3 * H).normalize() + (pd.Timedelta(0) if p == 1
                                                                else pd.Timedelta(days=1)),
         "drug_source_value": src} for p, src in ((1, "221906"), (2, "Norepinephrine"))])
    return so.write_duckdb(t)


@pytest.fixture
def derived(tmp_path):
    pd.DataFrame({"subject_id": [1, 2, 3, 4], "stay_id": [101, 102, 103, 104],
                  "icu_intime": [T] * 4, "admission_age": [70, 50, 65, 80],
                  "first_icu_stay": [True] * 4}).to_csv(tmp_path / "icustay_detail.csv.gz",
                                                        index=False)  # fmt: skip
    pd.DataFrame({"stay_id": [101, 102, 103, 104], "creatinine_max": [2.5, 1.0, 0.9, 3.0]}
                 ).to_csv(tmp_path / "first_day_lab.csv.gz", index=False)  # fmt: skip
    pd.DataFrame({"stay_id": [101, 102], "lactate_max": [1.5, 1.0]}).to_csv(
        tmp_path / "first_day_bg.csv.gz", index=False)
    pd.DataFrame({"stay_id": [101, 102, 103], "starttime": [T + H, T + 2 * H, T - 5 * H],
                  "endtime": [T + 10 * H] * 3,
                  "ventilation_status": ["InvasiveVent", "InvasiveVent", "InvasiveVent"]}
                 ).to_csv(tmp_path / "ventilation.csv.gz", index=False)  # fmt: skip
    pd.DataFrame({"stay_id": [101], "starttime": [T + 3 * H], "endtime": [T + 9 * H]}).to_csv(
        tmp_path / "norepinephrine.csv.gz", index=False)
    return tmp_path


def test_agreement_and_each_disagreement(derived):
    rows = {r["definition"]: r for r in e2a.compare(e2a.omop_flags(_cdm()),
                                                     e2a.reference_flags(derived))}
    assert rows["age65"] == {**rows["age65"], "both": 2, "omop_only": 0, "reference_only": 1,
                             "neither": 1, "compared": 4}  # patient 3: anchor age
    c = rows["creat2"]  # windows match: only patient 1 positive
    assert (c["both"], c["omop_only"], c["reference_only"]) == (1, 0, 0)
    assert c["compared"] == 3 and c["reference_without_omop"] == 1  # patient 4 died on day 1
    v = rows["ivent24"]  # 2: ours starts after 24 h; 3: mimic-code's episode began earlier
    assert (v["both"], v["omop_only"], v["reference_only"]) == (1, 1, 1)
    assert (rows["lact2"]["both"], rows["lact2"]["omop_only"]) == (0, 1)  # charted lactate
    assert rows["norepi24"]["omop_only"] == 1  # a prescription mimic-code does not count
    assert rows["creat2"]["kappa"] == pytest.approx(1.0)


def test_public_output_suppresses_small_counts(derived):
    rows = e2a.compare(e2a.omop_flags(_cdm()), e2a.reference_flags(derived))
    pub = e2a.public(rows)
    assert pub[0]["both"] == "<11" and pub[0]["omop_only"] == 0
    md = e2a.to_markdown(rows)
    assert "| age65 | <11 |" in md and "mimic-code only" in md


def test_breakdown_names_the_source_of_each_disagreement(derived):
    rows = e2a.breakdown(_cdm(), derived)
    got = {(r["definition"], r["group"], r["category"]): r["patients"] for r in rows}
    assert got[("lact2", "ours only", "qualifying values from: ICU item 225668")] == 1
    assert got[("lact2", "ours only", "mimic-code lactate_max present and <= 2")] == 1
    assert got[("ivent24", "ours only",
                "mimic-code invasive episode already running at ICU admission")] == 1
    assert got[("ivent24", "mimic-code only",
                "our invasive ventilation starts after 24 h")] == 1
    assert got[("ivent24", "mimic-code only", "no invasive ventilation record of ours")] == 0
    assert got[("norepi24", "ours only",
                "day-1 records from: prescriptions (date only: placed at midnight)")] == 1
    md = e2a.breakdown_markdown(rows)
    assert "| lact2 | ours only | qualifying values from: ICU item 225668 | <11 |" in md
