"""Write a synthetic cohort as OMOP CDM v5.4 tables (DuckDB), and read it back.

Everything downstream (the Informatician's SQL in Phase 2, the statistics) sees only OMOP, so
the synthetic data has to be OMOP too. Tables: person, observation_period, visit_occurrence,
condition_occurrence, measurement, drug_exposure, procedure_occurrence, observation,
device_exposure, death, plus a small concept table. A ``Profile`` names what the patients are
(index event, measurements, treatment) so different trials can be replayed; the default is the
septic-shock ICU cohort.

Concept IDs: gender uses the standard OMOP concepts (8507 male, 8532 female); septic shock uses
196236, the concept EmulatRx's Athena lookup returned for NCT03872011. Everything else uses
local concept IDs (>= 2,000,000,000, OMOP's range for site-specific concepts) until the Phase 2
vocabulary step maps them to standard concepts. The concept table says which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class Profile:
    """What the synthetic patients are, as OMOP concepts. The data-generating process is the
    same for every profile (dgp.py: a severity score and a lab drive treatment and death); a
    profile only names the index event, the two measurements and the treatment, so that a
    trial's protocol can be replayed on data that looks like its population.

    Concept slots: index visit ``ICU_VISIT``, severity ``SOFA``, lab ``LACTATE``, first
    treatment concept ``STUDY_DRUG``; further treatment concepts, a non-default index
    condition and a prior procedure get their own local IDs. With ``standard_local`` the local
    concepts are standard in their own vocabulary (allowed for site vocabularies), so the
    vocabulary search can find them; a reviewer still approves every set."""

    name: str = "septic-shock-icu"
    index_visit: str = "Intensive care unit stay (synthetic)"
    index_condition: tuple[int, str, str] = (SEPTIC_SHOCK, "Septic shock", "SNOMED")
    severity: str = "SOFA score (synthetic)"
    lab: str = "Lactate [mmol/L] (synthetic)"
    treatment: tuple[str, ...] = ("Study drug (synthetic)",)
    treatment_domain: str = "Drug"  # or "Procedure"
    prior_procedure: str | None = None  # recorded before the index event
    prior_procedure_hours: float = 6.0
    standard_local: bool = False


DEFAULT = Profile()
TREATMENT_EXTRA = LOCAL + 10  # second, third ... treatment concept
INDEX_CONDITION_LOCAL = LOCAL + 20
PRIOR_PROCEDURE = LOCAL + 30


def treatment_ids(profile: Profile = DEFAULT) -> list[int]:
    return [STUDY_DRUG] + [TREATMENT_EXTRA + k for k in range(1, len(profile.treatment))]


def index_condition_id(profile: Profile = DEFAULT) -> int:
    cid = profile.index_condition[0]
    return cid if cid else INDEX_CONDITION_LOCAL


def concept_table(n_noise: int, profile: Profile = DEFAULT) -> pd.DataFrame:
    std = "S" if profile.standard_local else None
    tclass = "Ingredient" if profile.treatment_domain == "Drug" else "Procedure"
    cid, cname, cvocab = profile.index_condition
    rows = [
        (8507, "MALE", "Gender", "Gender", "Gender", "S"),
        (8532, "FEMALE", "Gender", "Gender", "Gender", "S"),
        (index_condition_id(profile), cname, "Condition", cvocab, "Clinical Finding",
         "S" if cid else std),
        (ICU_VISIT, profile.index_visit, "Visit", "ATTE-SYNTH", "Visit", std),
        (SOFA, profile.severity, "Measurement", "ATTE-SYNTH", "Clinical Observation", std),
        (LACTATE, profile.lab, "Measurement", "ATTE-SYNTH", "Lab Test", std),
    ]
    rows += [(tid, name, profile.treatment_domain, "ATTE-SYNTH", tclass, std)
             for tid, name in zip(treatment_ids(profile), profile.treatment, strict=True)]
    if profile.prior_procedure:
        rows.append((PRIOR_PROCEDURE, profile.prior_procedure, "Procedure", "ATTE-SYNTH",
                     "Procedure", std))  # fmt: skip
    rows += [(NOISE_BASE + k, f"Noise covariate x{k} (synthetic)", "Measurement", "ATTE-SYNTH",
              "Lab Test", None) for k in range(1, n_noise + 1)]  # fmt: skip
    df = pd.DataFrame(rows, columns=["concept_id", "concept_name", "domain_id", "vocabulary_id",
                                     "concept_class_id", "standard_concept"])  # fmt: skip
    df["concept_code"] = df["concept_id"].astype(str)
    df["invalid_reason"] = None
    return df


def to_omop(cohort: Cohort, profile: Profile = DEFAULT) -> dict[str, pd.DataFrame]:
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
        "condition_concept_id": index_condition_id(profile),
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
    # One exposure per treatment concept (a combination arm), all starting together.
    exposures = pd.concat([pd.DataFrame({"person_id": pid[treated], "concept_id": tid,
                                         "start": start, "end": end[treated]})
                           for tid in treatment_ids(profile)], ignore_index=True)  # fmt: skip
    events = _empty_events()
    rx = exposures if profile.treatment_domain == "Drug" else exposures.iloc[:0]
    drug = pd.DataFrame({
        "drug_exposure_id": np.arange(1, len(rx) + 1),
        "person_id": rx["person_id"].to_numpy(),
        "drug_concept_id": rx["concept_id"].to_numpy(),
        "drug_exposure_start_datetime": rx["start"].to_numpy(),
        "drug_exposure_start_date": pd.to_datetime(rx["start"]).dt.normalize().to_numpy(),
        "drug_exposure_end_datetime": rx["end"].to_numpy(),
        "drug_exposure_end_date": pd.to_datetime(rx["end"]).dt.normalize().to_numpy(),
        "drug_type_concept_id": 32817,
        "visit_occurrence_id": rx["person_id"].to_numpy(),
    }) if len(rx) else events["drug_exposure"]  # fmt: skip
    procs = []
    if profile.treatment_domain == "Procedure":
        procs.append(exposures[["person_id", "concept_id", "start"]])
    if profile.prior_procedure:
        procs.append(pd.DataFrame({"person_id": pid, "concept_id": PRIOR_PROCEDURE,
                                   "start": index - pd.to_timedelta(
                                       profile.prior_procedure_hours, unit="h")}))  # fmt: skip
    if procs:
        pr = pd.concat(procs, ignore_index=True)
        events["procedure_occurrence"] = pd.DataFrame({
            "procedure_occurrence_id": np.arange(1, len(pr) + 1),
            "person_id": pr["person_id"].to_numpy(),
            "procedure_concept_id": pr["concept_id"].to_numpy(),
            "procedure_datetime": pr["start"].to_numpy(),
            "procedure_date": pd.to_datetime(pr["start"]).dt.normalize().to_numpy(),
            "procedure_type_concept_id": 32817,
            "visit_occurrence_id": pr["person_id"].to_numpy(),
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
        "procedure_occurrence": events["procedure_occurrence"],
        "observation": events["observation"],
        "device_exposure": events["device_exposure"],
        "death": death,
        "concept": concept_table(s.n_noise, profile),
        # CDM metadata, read by the Data Support Gate.
        "cdm_source": pd.DataFrame([{
            "cdm_source_name": f"Agentic-TTE synthetic cohort ({profile.name})"
            if profile != DEFAULT else "Agentic-TTE synthetic ICU cohort",
            "cdm_source_abbreviation": "ATTE-SYNTH",
            "cdm_version": "5.4",
            "vocabulary_version": "ATTE-SYNTH local concepts",
            "source_description": f"agentic_tte.synth, seed {s.seed}, n {s.n}",
        }]),  # fmt: skip
    }


def _empty_events() -> dict[str, pd.DataFrame]:
    """Typed empty event tables: a CDM has every table, and compiled SQL may query any."""
    ts, i8 = "datetime64[ns]", "int64"
    def t(cols):
        return pd.DataFrame({c: pd.Series(dtype=d) for c, d in cols})
    return {
        "drug_exposure": t([("drug_exposure_id", i8), ("person_id", i8), ("drug_concept_id", i8),
                            ("drug_exposure_start_datetime", ts),
                            ("drug_exposure_start_date", ts),
                            ("drug_exposure_end_datetime", ts), ("drug_exposure_end_date", ts),
                            ("drug_type_concept_id", i8), ("visit_occurrence_id", i8)]),
        "procedure_occurrence": t([("procedure_occurrence_id", i8), ("person_id", i8),
                                   ("procedure_concept_id", i8), ("procedure_datetime", ts),
                                   ("procedure_date", ts), ("procedure_type_concept_id", i8),
                                   ("visit_occurrence_id", i8)]),
        "observation": t([("observation_id", i8), ("person_id", i8),
                          ("observation_concept_id", i8), ("observation_datetime", ts),
                          ("observation_date", ts), ("value_as_number", "float64"),
                          ("observation_type_concept_id", i8)]),
        "device_exposure": t([("device_exposure_id", i8), ("person_id", i8),
                              ("device_concept_id", i8), ("device_exposure_start_datetime", ts),
                              ("device_exposure_start_date", ts),
                              ("device_type_concept_id", i8)]),
    }  # fmt: skip


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
