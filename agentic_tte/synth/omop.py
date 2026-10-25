"""Write a synthetic cohort as OMOP CDM v5.4 tables (DuckDB), and read it back.

Everything downstream (the Informatician's SQL in Phase 2, the statistics) sees only OMOP, so
the synthetic data has to be OMOP too. Tables: person, observation_period, visit_occurrence,
condition_occurrence, measurement, drug_exposure, death, plus a small concept table.

Concept IDs: gender uses the standard OMOP concepts (8507 male, 8532 female); septic shock uses
196236, the concept EmulatRx's Athena lookup returned for NCT03872011. Everything else uses
local concept IDs (>= 2,000,000,000, OMOP's range for site-specific concepts) until the Phase 2
vocabulary step maps them to standard concepts. The concept table says which is which.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_tte.synth.dgp import Cohort

GENDER = {0: 8507, 1: 8532}  # OMOP standard: MALE, FEMALE
SEPTIC_SHOCK = 196236
LOCAL = 2_000_000_000
ICU_VISIT = LOCAL + 1
SOFA = LOCAL + 2
LACTATE = LOCAL + 3
STUDY_DRUG = LOCAL + 4
NOISE_BASE = LOCAL + 100  # x1 -> LOCAL + 101, ...

INDEX_START = datetime(2015, 1, 1)


def concept_table(n_noise: int) -> pd.DataFrame:
    rows = [
        (8507, "MALE", "Gender", "Gender", "S"),
        (8532, "FEMALE", "Gender", "Gender", "S"),
        (SEPTIC_SHOCK, "Septic shock", "Condition", "SNOMED", "S"),
        (ICU_VISIT, "Intensive care unit stay (synthetic)", "Visit", "ATTE-SYNTH", None),
        (SOFA, "SOFA score (synthetic)", "Measurement", "ATTE-SYNTH", None),
        (LACTATE, "Lactate [mmol/L] (synthetic)", "Measurement", "ATTE-SYNTH", None),
        (STUDY_DRUG, "Study drug (synthetic)", "Drug", "ATTE-SYNTH", None),
    ] + [(NOISE_BASE + k, f"Noise covariate x{k} (synthetic)", "Measurement", "ATTE-SYNTH", None)
         for k in range(1, n_noise + 1)]  # fmt: skip
    return pd.DataFrame(rows, columns=["concept_id", "concept_name", "domain_id",
                                       "vocabulary_id", "standard_concept"])  # fmt: skip


def to_omop(cohort: Cohort) -> dict[str, pd.DataFrame]:
    f = cohort.frame
    s = cohort.scenario
    rng = np.random.default_rng(s.seed + 7)
    n = len(f)
    pid = f["person_id"].to_numpy()
    index = pd.to_datetime(INDEX_START) + pd.to_timedelta(rng.integers(0, 5 * 365, n), unit="D") \
        + pd.to_timedelta(rng.integers(0, 24 * 60, n), unit="min")  # fmt: skip
    birth = index - pd.to_timedelta(f["age"].to_numpy() * 365.25, unit="D")
    end = index + pd.to_timedelta(f["time"].to_numpy(), unit="D")

    person = pd.DataFrame({
        "person_id": pid,
        "gender_concept_id": f["female"].map(GENDER).to_numpy(),
        "year_of_birth": birth.year,
        "month_of_birth": birth.month,
        "day_of_birth": birth.day,
        "birth_datetime": birth,
        "race_concept_id": 0,
        "ethnicity_concept_id": 0,
    })  # fmt: skip
    observation_period = pd.DataFrame({
        "observation_period_id": pid,
        "person_id": pid,
        "observation_period_start_date": (index - pd.to_timedelta(365, unit="D")).date,
        "observation_period_end_date": end.date,
        "period_type_concept_id": 32817,  # EHR
    })  # fmt: skip
    visit = pd.DataFrame({
        "visit_occurrence_id": pid,
        "person_id": pid,
        "visit_concept_id": ICU_VISIT,
        "visit_start_datetime": index,
        "visit_start_date": index.date,
        "visit_end_datetime": end,
        "visit_end_date": end.date,
        "visit_type_concept_id": 32817,
    })  # fmt: skip
    condition = pd.DataFrame({
        "condition_occurrence_id": pid,
        "person_id": pid,
        "condition_concept_id": SEPTIC_SHOCK,
        "condition_start_datetime": index,
        "condition_start_date": index.date,
        "condition_type_concept_id": 32817,
        "visit_occurrence_id": pid,
    })  # fmt: skip

    meas = []
    numeric = [("sofa", SOFA), ("lactate", LACTATE)] + [
        (f"x{k}", NOISE_BASE + k) for k in range(1, s.n_noise + 1)
    ]
    for col, cid in numeric:
        keep = f[col].notna().to_numpy()
        meas.append(pd.DataFrame({
            "person_id": pid[keep],
            "measurement_concept_id": cid,
            "measurement_datetime": index[keep],
            "measurement_date": index[keep].date,
            "value_as_number": f[col].to_numpy()[keep],
            "measurement_type_concept_id": 32817,
            "visit_occurrence_id": pid[keep],
        }))  # fmt: skip
    measurement = pd.concat(meas, ignore_index=True)
    measurement.insert(0, "measurement_id", np.arange(1, len(measurement) + 1))

    treated = f["treated"].to_numpy() == 1
    start = index[treated] + pd.to_timedelta(f["treat_start"].to_numpy()[treated], unit="D")
    drug = pd.DataFrame({
        "drug_exposure_id": np.arange(1, treated.sum() + 1),
        "person_id": pid[treated],
        "drug_concept_id": STUDY_DRUG,
        "drug_exposure_start_datetime": start,
        "drug_exposure_start_date": start.date,
        "drug_exposure_end_datetime": end[treated],
        "drug_exposure_end_date": end[treated].date,
        "drug_type_concept_id": 32817,
        "visit_occurrence_id": pid[treated],
    })  # fmt: skip
    dead = f["event"].to_numpy() == 1
    death = pd.DataFrame({
        "person_id": pid[dead],
        "death_datetime": end[dead],
        "death_date": end[dead].date,
        "death_type_concept_id": 32817,
    })  # fmt: skip
    return {
        "person": person,
        "observation_period": observation_period,
        "visit_occurrence": visit,
        "condition_occurrence": condition,
        "measurement": measurement,
        "drug_exposure": drug,
        "death": death,
        "concept": concept_table(s.n_noise),
        # CDM metadata, read by the Data Support Gate.
        "cdm_source": pd.DataFrame([{
            "cdm_source_name": "Agentic-TTE synthetic ICU cohort",
            "cdm_source_abbreviation": "ATTE-SYNTH",
            "cdm_version": "5.4",
            "vocabulary_version": "ATTE-SYNTH local concepts",
            "source_description": f"agentic_tte.synth, seed {s.seed}, n {s.n}",
        }]),  # fmt: skip
    }


def write_duckdb(tables: dict[str, pd.DataFrame], path: str | Path = ":memory:"):
    """Write tables into a DuckDB database and return the open connection."""
    import duckdb

    con = duckdb.connect(str(path))
    for name, df in tables.items():
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    return con


ANALYSIS_SQL = f"""
WITH idx AS (
  SELECT person_id, visit_start_datetime AS t0, visit_end_datetime AS t_end
  FROM visit_occurrence WHERE visit_concept_id = {ICU_VISIT}
),
m AS (
  SELECT person_id,
         max(CASE WHEN measurement_concept_id = {SOFA} THEN value_as_number END) AS sofa,
         max(CASE WHEN measurement_concept_id = {LACTATE} THEN value_as_number END) AS lactate
  FROM measurement GROUP BY person_id
),
d AS (
  SELECT person_id, min(drug_exposure_start_datetime) AS rx_start
  FROM drug_exposure WHERE drug_concept_id = {STUDY_DRUG} GROUP BY person_id
)
SELECT p.person_id,
       date_diff('second', p.birth_datetime, idx.t0) / 86400.0 / 365.25 AS age,
       CAST(p.gender_concept_id = 8532 AS INTEGER) AS female,
       m.sofa, m.lactate,
       CAST(d.rx_start IS NOT NULL AS INTEGER) AS treated,
       date_diff('second', idx.t0, d.rx_start) / 86400.0 AS treat_start,
       date_diff('second', idx.t0, idx.t_end) / 86400.0 AS time,
       CAST(de.person_id IS NOT NULL AS INTEGER) AS event
FROM person p
JOIN idx USING (person_id)
LEFT JOIN m USING (person_id)
LEFT JOIN d USING (person_id)
LEFT JOIN death de USING (person_id)
ORDER BY p.person_id
"""


def analysis_frame(con) -> pd.DataFrame:
    """Rebuild the analysis frame from OMOP tables with SQL: the hand-written reference that the
    Phase 2 IR-to-SQL compiler must reproduce."""
    return con.execute(ANALYSIS_SQL).df()
