"""MIMIC-IV 3.1 (raw PhysioNet release) -> OMOP CDM 5.4 in DuckDB (PLAN.md §16).

Runs where the data are (Hellbender). Reads the gzipped CSVs directly, never copies patient
rows anywhere else, and writes one DuckDB file holding the CDM tables and the vocabulary.
What leaves the run is the mapping report: aggregate counts, with small cells suppressed.

Source to standard concepts only through the vocabulary: the source code's concept
(ICD9CM, ICD10CM, ICD9Proc, ICD10PCS, NDC) and its 'Maps to' relationship. A record goes to
the table of its standard concept's domain (an ICD-10-CM code mapped to an Observation
concept lands in ``observation``). Unmapped records are kept with concept 0 and their source
value, and counted in the report; nothing is dropped silently.

Built so far: person, observation_period, visit_occurrence (hospital admissions),
visit_detail (ICU stays), death, and from diagnoses_icd, procedures_icd and prescriptions:
condition_occurrence, procedure_occurrence, drug_exposure, observation, measurement.
chartevents and procedureevents are converted through the MIT-LCP curated item maps
(``mimic-code/mimic-iv/concepts/concept_map``, SSSOM files with author and reviewer ORCIDs;
``concept_map``), read where they are, never copied into this repository. Each curated target
is checked against the loaded vocabulary: a standard, valid concept is used; one that is no
longer standard is followed through 'Maps to'; one that is missing is rejected. All are
reported, and the map rows used go into ``source_to_concept_map``. Chart and procedure items
without a curated map are not converted, and are counted by item in the report.
labevents and inputevents have no curated map: they are converted through reviewed item
maps (``item_maps``; agentic_tte/etl/item_maps.py), approved per item. Not converted: the ED
and note modules (not in this copy of the release).

MIMIC specifics, stated in ``cdm_source`` and the report:
- Dates are shifted per patient: only within-patient times mean anything. Age comes from
  ``anchor_age`` at ``anchor_year`` (year_of_birth = anchor_year - anchor_age); ages over 89
  are top-coded by MIMIC.
- Diagnosis codes have no time. They are coded at discharge; ``diagnosis_time="admission"``
  (default, the usual convention) dates them at admission, a mild look-ahead within the index
  admission; ``"discharge"`` dates them when they were known.
- Deaths include out-of-hospital deaths from state records up to one year after the last
  discharge (``patients.dod``, a date). Each observation period therefore ends
  ``death_followup_days`` (365) after the last discharge, or at death; outside hospital stays
  only death is observed.
- Persons without any hospital admission are left out (counted in the report): this CDM is
  for inpatient emulations.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agentic_tte.omop.gate import suppress
from agentic_tte.omop.vocab import load_athena

# OMOP concepts used by the ETL (standard concepts in the Visit, Type and Gender vocabularies).
MALE, FEMALE = 8507, 8532
INPATIENT, ER_AND_INPATIENT, INTENSIVE_CARE = 9201, 262, 32037
EHR, EHR_BILLING, EHR_PRESCRIPTION = 32817, 32821, 32838
PRIMARY_DX, SECONDARY_DX = 32902, 32908

SOURCE_TABLES = {
    "hosp": ["patients", "admissions", "diagnoses_icd", "procedures_icd", "prescriptions"],
    "icu": ["icustays"],
}

# CDM 5.4 tables: the columns the ETL fills or the compiler and gate read, typed.
CDM = {
    "person": "person_id BIGINT, gender_concept_id INTEGER, year_of_birth INTEGER, "
              "month_of_birth INTEGER, day_of_birth INTEGER, birth_datetime TIMESTAMP, "
              "race_concept_id INTEGER, ethnicity_concept_id INTEGER, location_id BIGINT, "
              "provider_id BIGINT, care_site_id BIGINT, person_source_value VARCHAR, "
              "gender_source_value VARCHAR, gender_source_concept_id INTEGER, "
              "race_source_value VARCHAR, race_source_concept_id INTEGER, "
              "ethnicity_source_value VARCHAR, ethnicity_source_concept_id INTEGER",
    "observation_period": "observation_period_id BIGINT, person_id BIGINT, "
                          "observation_period_start_date DATE, "
                          "observation_period_end_date DATE, period_type_concept_id INTEGER",
    "visit_occurrence": "visit_occurrence_id BIGINT, person_id BIGINT, "
                        "visit_concept_id INTEGER, visit_start_date DATE, "
                        "visit_start_datetime TIMESTAMP, visit_end_date DATE, "
                        "visit_end_datetime TIMESTAMP, visit_type_concept_id INTEGER, "
                        "provider_id BIGINT, care_site_id BIGINT, visit_source_value VARCHAR, "
                        "visit_source_concept_id INTEGER, admitted_from_concept_id INTEGER, "
                        "admitted_from_source_value VARCHAR, discharged_to_concept_id INTEGER, "
                        "discharged_to_source_value VARCHAR, "
                        "preceding_visit_occurrence_id BIGINT",
    "visit_detail": "visit_detail_id BIGINT, person_id BIGINT, visit_detail_concept_id INTEGER, "
                    "visit_detail_start_date DATE, visit_detail_start_datetime TIMESTAMP, "
                    "visit_detail_end_date DATE, visit_detail_end_datetime TIMESTAMP, "
                    "visit_detail_type_concept_id INTEGER, visit_detail_source_value VARCHAR, "
                    "visit_detail_source_concept_id INTEGER, visit_occurrence_id BIGINT",
    "condition_occurrence": "condition_occurrence_id BIGINT, person_id BIGINT, "
                            "condition_concept_id INTEGER, condition_start_date DATE, "
                            "condition_start_datetime TIMESTAMP, condition_end_date DATE, "
                            "condition_end_datetime TIMESTAMP, "
                            "condition_type_concept_id INTEGER, "
                            "condition_status_concept_id INTEGER, visit_occurrence_id BIGINT, "
                            "condition_source_value VARCHAR, "
                            "condition_source_concept_id INTEGER, "
                            "condition_status_source_value VARCHAR",
    "procedure_occurrence": "procedure_occurrence_id BIGINT, person_id BIGINT, "
                            "procedure_concept_id INTEGER, procedure_date DATE, "
                            "procedure_datetime TIMESTAMP, procedure_type_concept_id INTEGER, "
                            "visit_occurrence_id BIGINT, procedure_source_value VARCHAR, "
                            "procedure_source_concept_id INTEGER",
    "drug_exposure": "drug_exposure_id BIGINT, person_id BIGINT, drug_concept_id INTEGER, "
                     "drug_exposure_start_date DATE, drug_exposure_start_datetime TIMESTAMP, "
                     "drug_exposure_end_date DATE, drug_exposure_end_datetime TIMESTAMP, "
                     "drug_type_concept_id INTEGER, quantity DOUBLE, "
                     "dose_unit_source_value VARCHAR, route_source_value VARCHAR, "
                     "visit_occurrence_id BIGINT, drug_source_value VARCHAR, "
                     "drug_source_concept_id INTEGER",
    "measurement": "measurement_id BIGINT, person_id BIGINT, measurement_concept_id INTEGER, "
                   "measurement_date DATE, measurement_datetime TIMESTAMP, "
                   "measurement_type_concept_id INTEGER, value_as_number DOUBLE, "
                   "value_as_concept_id INTEGER, unit_concept_id INTEGER, "
                   "visit_occurrence_id BIGINT, measurement_source_value VARCHAR, "
                   "measurement_source_concept_id INTEGER, unit_source_value VARCHAR, "
                   "value_source_value VARCHAR",
    "observation": "observation_id BIGINT, person_id BIGINT, observation_concept_id INTEGER, "
                   "observation_date DATE, observation_datetime TIMESTAMP, "
                   "observation_type_concept_id INTEGER, value_as_number DOUBLE, "
                   "visit_occurrence_id BIGINT, observation_source_value VARCHAR, "
                   "observation_source_concept_id INTEGER",
    "device_exposure": "device_exposure_id BIGINT, person_id BIGINT, device_concept_id INTEGER, "
                       "device_exposure_start_date DATE, "
                       "device_exposure_start_datetime TIMESTAMP, "
                       "device_type_concept_id INTEGER, visit_occurrence_id BIGINT, "
                       "device_source_value VARCHAR, device_source_concept_id INTEGER",
    "death": "person_id BIGINT, death_date DATE, death_datetime TIMESTAMP, "
             "death_type_concept_id INTEGER, cause_concept_id INTEGER, "
             "cause_source_value VARCHAR, cause_source_concept_id INTEGER",
    "source_to_concept_map": "source_code VARCHAR, source_concept_id INTEGER, "
                             "source_vocabulary_id VARCHAR, source_code_description VARCHAR, "
                             "target_concept_id INTEGER, target_vocabulary_id VARCHAR, "
                             "valid_start_date DATE, valid_end_date DATE, "
                             "invalid_reason VARCHAR",
    "cdm_source": "cdm_source_name VARCHAR, cdm_source_abbreviation VARCHAR, "
                  "cdm_holder VARCHAR, source_description VARCHAR, "
                  "source_documentation_reference VARCHAR, cdm_etl_reference VARCHAR, "
                  "source_release_date DATE, cdm_release_date DATE, cdm_version VARCHAR, "
                  "cdm_version_concept_id INTEGER, vocabulary_version VARCHAR",
}  # fmt: skip

# Where a routed record goes, by the domain of its standard concept: table, id column and the
# SELECT list that fills it from the staging table ``stg``.
_ROUTES = {
    "Condition": ("condition_occurrence", """
        person_id, concept_id AS condition_concept_id, event_date AS condition_start_date,
        event_datetime AS condition_start_datetime, type_concept_id AS condition_type_concept_id,
        status_concept_id AS condition_status_concept_id, visit_occurrence_id,
        source_value AS condition_source_value, source_concept_id AS condition_source_concept_id,
        status_source_value AS condition_status_source_value"""),
    "Procedure": ("procedure_occurrence", """
        person_id, concept_id AS procedure_concept_id, event_date AS procedure_date,
        event_datetime AS procedure_datetime, type_concept_id AS procedure_type_concept_id,
        visit_occurrence_id, source_value AS procedure_source_value,
        source_concept_id AS procedure_source_concept_id"""),
    "Drug": ("drug_exposure", """
        person_id, concept_id AS drug_concept_id, event_date AS drug_exposure_start_date,
        event_datetime AS drug_exposure_start_datetime,
        coalesce(end_date, event_date) AS drug_exposure_end_date,
        coalesce(end_datetime, event_datetime) AS drug_exposure_end_datetime,
        type_concept_id AS drug_type_concept_id, value_as_number AS quantity,
        unit_source_value AS dose_unit_source_value, route_source_value, visit_occurrence_id,
        source_value AS drug_source_value, source_concept_id AS drug_source_concept_id"""),
    "Measurement": ("measurement", """
        person_id, concept_id AS measurement_concept_id, event_date AS measurement_date,
        event_datetime AS measurement_datetime, type_concept_id AS measurement_type_concept_id,
        visit_occurrence_id, source_value AS measurement_source_value,
        source_concept_id AS measurement_source_concept_id, value_as_number,
        unit_source_value, value_source_value"""),
    "Observation": ("observation", """
        person_id, concept_id AS observation_concept_id, event_date AS observation_date,
        event_datetime AS observation_datetime, type_concept_id AS observation_type_concept_id,
        visit_occurrence_id, source_value AS observation_source_value,
        source_concept_id AS observation_source_concept_id, value_as_number"""),
    "Device": ("device_exposure", """
        person_id, concept_id AS device_concept_id, event_date AS device_exposure_start_date,
        event_datetime AS device_exposure_start_datetime,
        type_concept_id AS device_type_concept_id, visit_occurrence_id,
        source_value AS device_source_value, source_concept_id AS device_source_concept_id"""),
}  # fmt: skip
_ID = {"condition_occurrence": "condition_occurrence_id",
       "procedure_occurrence": "procedure_occurrence_id", "drug_exposure": "drug_exposure_id",
       "measurement": "measurement_id", "observation": "observation_id",
       "device_exposure": "device_exposure_id"}  # fmt: skip


class EtlConfig(BaseModel):
    source: Path  # mimic-iv-3.1 directory with hosp/ and icu/
    vocab: Path  # Athena vocabulary download (tab-separated CSVs)
    out: Path  # DuckDB file to create
    sample_percent: float | None = Field(default=None, gt=0, le=100)  # pilot: subject subset
    diagnosis_time: Literal["admission", "discharge"] = "admission"
    death_followup_days: int = 365
    threads: int | None = None
    memory_limit: str | None = None  # e.g. "100GB"
    temp_directory: str | None = None  # node-local scratch for spills
    concept_map: Path | None = None  # mimic-code mimic-iv/concepts/concept_map (SSSOM)
    item_maps: list[Path] = Field(default_factory=list)  # reviewed maps (labs, inputs)


class EtlReport(BaseModel):
    config: dict
    started_at: datetime
    finished_at: datetime | None = None
    duckdb_version: str | None = None
    vocabulary_version: str | None = None
    source_rows: dict[str, int] = Field(default_factory=dict)
    cdm_rows: dict[str, int] = Field(default_factory=dict)
    persons_without_admission: int = 0
    mapping: dict[str, dict] = Field(default_factory=dict)  # per origin: rows, mapped, top unmapped
    item_maps: dict[str, dict] = Field(default_factory=dict)  # curated map checks, per source
    notes: list[str] = Field(default_factory=list)

    def public(self) -> dict:
        """Counts below 11 suppressed: what may be copied off the cluster."""
        d = json.loads(self.model_dump_json())
        for k in ("source_rows", "cdm_rows"):
            d[k] = {t: suppress(n) for t, n in d[k].items()}
        d["persons_without_admission"] = suppress(d["persons_without_admission"])
        for m in d["mapping"].values():
            m["rows"], m["mapped"] = suppress(m["rows"]), suppress(m["mapped"])
            m["top_unmapped"] = [[v, suppress(n)] for v, n in m["top_unmapped"]]
        return d  # item_maps hold map-file facts (item IDs, labels), not patient counts

    def to_markdown(self) -> str:
        p = self.public()
        opts = {k: self.config.get(k)
                for k in ("sample_percent", "diagnosis_time", "death_followup_days")}
        lines = ["# MIMIC-IV to OMOP CDM 5.4: ETL report", "",
                 f"Started {self.started_at:%Y-%m-%d %H:%M} UTC; vocabulary "
                 f"{self.vocabulary_version or 'unknown'}; DuckDB {self.duckdb_version}. "
                 f"Options: {json.dumps(opts)}.",
                 "", "| CDM table | Rows |", "|---|---|"]  # fmt: skip
        lines += [f"| {t} | {n} |" for t, n in p["cdm_rows"].items()]
        lines += ["", f"Persons without a hospital admission (left out): "
                      f"{p['persons_without_admission']}.", "",
                  "| Source | Rows | Mapped to a standard concept | Share | Top unmapped "
                  "source values |", "|---|---|---|---|---|"]  # fmt: skip
        for origin, m in p["mapping"].items():
            raw = self.mapping[origin]
            share = f"{100 * raw['mapped'] / raw['rows']:.1f}%" if raw["rows"] else "–"
            top = "; ".join(f"{v} ({n})" for v, n in m["top_unmapped"][:8]) or "–"
            lines.append(f"| {origin} | {m['rows']} | {m['mapped']} | {share} | {top} |")
        for source, m in self.item_maps.items():
            lines += ["", f"### Curated item map: {source} ({m['file']})", "",
                      f"{m['rows_in_file']} rows: {m['accepted']} accepted, {m['remapped']} "
                      f"followed through 'Maps to', {len(m['rejected'])} rejected. Predicates: "
                      f"{json.dumps(m['predicates'])}."]  # fmt: skip
            lines += [f"- rejected: {i} {label} -> {cid} ({why})"
                      for i, label, cid, why in m["rejected"]]
        lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def _src(cfg: EtlConfig, module: str, table: str) -> str:
    """A raw MIMIC table as DuckDB reads it: every column as text, cast explicitly below, so
    type detection on a sample can never mis-type a column."""
    path = (cfg.source / module / f"{table}.csv.gz").as_posix()
    return f"read_csv('{path}', header = true, all_varchar = true)"


def _vocabulary_version(con) -> str | None:
    try:
        row = con.execute("SELECT vocabulary_version FROM vocabulary "
                          "WHERE vocabulary_id = 'None'").fetchone()  # fmt: skip
        return row[0] if row else None
    except Exception:  # noqa: BLE001 - VOCABULARY.csv is optional
        return None


def build(cfg: EtlConfig, *, git_commit: str | None = None) -> EtlReport:
    """Build the CDM database. Returns the report (raw counts; use ``public()`` to share)."""
    import duckdb

    if cfg.out.exists():
        raise FileExistsError(f"{cfg.out} exists; the ETL never overwrites a CDM")
    for module, tables in SOURCE_TABLES.items():
        for t in tables:
            f = cfg.source / module / f"{t}.csv.gz"
            if not f.exists():
                raise FileNotFoundError(f)
    rep = EtlReport(config=json.loads(cfg.model_dump_json()), started_at=datetime.now(UTC))
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(cfg.out))
    for setting, value in (("threads", cfg.threads), ("memory_limit", cfg.memory_limit),
                           ("temp_directory", cfg.temp_directory)):  # fmt: skip
        if value is not None:
            con.execute(f"SET {setting} = '{value}'")
    rep.duckdb_version = con.execute("SELECT version()").fetchone()[0]

    load_athena(con, cfg.vocab)
    rep.vocabulary_version = _vocabulary_version(con)
    for table, cols in CDM.items():
        con.execute(f"CREATE TABLE {table} ({cols})")

    # --- persons and admissions -----------------------------------------------------------
    pct = cfg.sample_percent
    keep = f"WHERE (CAST(subject_id AS BIGINT) % 10000) < {int(round(pct * 100))}" if pct \
        else ""  # fmt: skip
    con.execute(f"""CREATE TEMP TABLE pat AS
        SELECT CAST(subject_id AS BIGINT) AS subject_id, gender,
               CAST(anchor_age AS INTEGER) AS anchor_age,
               CAST(anchor_year AS INTEGER) AS anchor_year, CAST(dod AS DATE) AS dod
        FROM {_src(cfg, 'hosp', 'patients')} {keep}""")
    con.execute(f"""CREATE TEMP TABLE adm AS
        SELECT CAST(a.subject_id AS BIGINT) AS subject_id, CAST(hadm_id AS BIGINT) AS hadm_id,
               CAST(admittime AS TIMESTAMP) AS admittime,
               CAST(dischtime AS TIMESTAMP) AS dischtime,
               CAST(nullif(deathtime, '') AS TIMESTAMP) AS deathtime,
               CAST(nullif(edregtime, '') AS TIMESTAMP) AS edregtime,
               admission_type, admission_location, discharge_location, race
        FROM {_src(cfg, 'hosp', 'admissions')} a
        WHERE CAST(a.subject_id AS BIGINT) IN (SELECT subject_id FROM pat)""")
    rep.source_rows["patients"] = con.execute("SELECT count(*) FROM pat").fetchone()[0]
    rep.source_rows["admissions"] = con.execute("SELECT count(*) FROM adm").fetchone()[0]
    rep.persons_without_admission = con.execute(
        "SELECT count(*) FROM pat WHERE subject_id NOT IN (SELECT subject_id FROM adm)"
    ).fetchone()[0]

    con.execute(f"""INSERT INTO person BY NAME
        SELECT p.subject_id AS person_id,
               CASE p.gender WHEN 'M' THEN {MALE} WHEN 'F' THEN {FEMALE} ELSE 0 END
                   AS gender_concept_id,
               p.anchor_year - p.anchor_age AS year_of_birth,
               0 AS race_concept_id, 0 AS ethnicity_concept_id,
               CAST(p.subject_id AS VARCHAR) AS person_source_value,
               p.gender AS gender_source_value, 0 AS gender_source_concept_id,
               r.race AS race_source_value, 0 AS race_source_concept_id,
               0 AS ethnicity_source_concept_id
        FROM pat p
        JOIN (SELECT subject_id, arg_max(race, admittime) AS race FROM adm GROUP BY 1) r
          USING (subject_id)""")  # fmt: skip
    con.execute("""CREATE TEMP TABLE dth AS
        SELECT p.subject_id AS person_id,
               coalesce(CAST(min(a.deathtime) AS DATE), p.dod) AS death_date,
               min(a.deathtime) AS death_datetime
        FROM pat p LEFT JOIN adm a USING (subject_id)
        WHERE p.subject_id IN (SELECT subject_id FROM adm)
        GROUP BY p.subject_id, p.dod
        HAVING min(a.deathtime) IS NOT NULL OR p.dod IS NOT NULL""")
    con.execute(f"""INSERT INTO death BY NAME
        SELECT person_id, death_date, death_datetime, {EHR} AS death_type_concept_id,
               0 AS cause_concept_id FROM dth""")
    f = int(cfg.death_followup_days)
    con.execute(f"""INSERT INTO observation_period BY NAME
        SELECT row_number() OVER (ORDER BY a.subject_id) AS observation_period_id,
               a.subject_id AS person_id,
               CAST(min(least(a.admittime, coalesce(a.edregtime, a.admittime))) AS DATE)
                   AS observation_period_start_date,
               CASE WHEN d.death_date IS NOT NULL
                    THEN greatest(d.death_date, CAST(max(a.dischtime) AS DATE))
                    ELSE CAST(CAST(max(a.dischtime) AS DATE) + INTERVAL '{f} days' AS DATE)
               END
                   AS observation_period_end_date,
               {EHR} AS period_type_concept_id
        FROM adm a LEFT JOIN dth d ON d.person_id = a.subject_id
        GROUP BY a.subject_id, d.death_date""")  # fmt: skip
    con.execute(f"""INSERT INTO visit_occurrence BY NAME
        SELECT hadm_id AS visit_occurrence_id, subject_id AS person_id,
               CASE WHEN edregtime IS NOT NULL THEN {ER_AND_INPATIENT} ELSE {INPATIENT} END
                   AS visit_concept_id,
               CAST(admittime AS DATE) AS visit_start_date, admittime AS visit_start_datetime,
               CAST(dischtime AS DATE) AS visit_end_date, dischtime AS visit_end_datetime,
               {EHR} AS visit_type_concept_id, admission_type AS visit_source_value,
               0 AS visit_source_concept_id, 0 AS admitted_from_concept_id,
               admission_location AS admitted_from_source_value,
               0 AS discharged_to_concept_id, discharge_location AS discharged_to_source_value
        FROM adm""")
    icu = _src(cfg, "icu", "icustays")
    con.execute(f"""INSERT INTO visit_detail BY NAME
        SELECT CAST(stay_id AS BIGINT) AS visit_detail_id,
               CAST(subject_id AS BIGINT) AS person_id,
               {INTENSIVE_CARE} AS visit_detail_concept_id,
               CAST(CAST(intime AS TIMESTAMP) AS DATE) AS visit_detail_start_date,
               CAST(intime AS TIMESTAMP) AS visit_detail_start_datetime,
               CAST(CAST(outtime AS TIMESTAMP) AS DATE) AS visit_detail_end_date,
               CAST(outtime AS TIMESTAMP) AS visit_detail_end_datetime,
               {EHR} AS visit_detail_type_concept_id,
               first_careunit AS visit_detail_source_value,
               0 AS visit_detail_source_concept_id,
               CAST(hadm_id AS BIGINT) AS visit_occurrence_id
        FROM {icu}
        WHERE CAST(hadm_id AS BIGINT) IN (SELECT hadm_id FROM adm)""")

    # --- coded events, through the vocabulary -------------------------------------------------
    # Source concepts by vocabulary and code with the dots removed (MIMIC stores "R6521",
    # the vocabulary "R65.21"), and their standard targets ('Maps to').
    con.execute("""CREATE TEMP TABLE codemap AS
        SELECT s.vocabulary_id, replace(s.concept_code, '.', '') AS code,
               s.concept_id AS source_concept_id, t.concept_id AS target_concept_id,
               t.domain_id AS target_domain
        FROM concept s
        LEFT JOIN concept_relationship r ON r.concept_id_1 = s.concept_id
             AND r.relationship_id = 'Maps to'
             AND (r.invalid_reason IS NULL OR r.invalid_reason = '')
        LEFT JOIN concept t ON t.concept_id = r.concept_id_2 AND t.standard_concept = 'S'
        WHERE s.vocabulary_id IN ('ICD9CM', 'ICD10CM', 'ICD9Proc', 'ICD10PCS', 'NDC')""")
    con.execute("""CREATE TEMP TABLE stg (origin VARCHAR, home VARCHAR, person_id BIGINT,
        visit_occurrence_id BIGINT, event_date DATE, event_datetime TIMESTAMP,
        end_date DATE, end_datetime TIMESTAMP, concept_id INTEGER, domain VARCHAR,
        source_value VARCHAR, source_concept_id INTEGER, type_concept_id INTEGER,
        status_concept_id INTEGER, status_source_value VARCHAR, route_source_value VARCHAR,
        value_as_number DOUBLE, unit_source_value VARCHAR, value_source_value VARCHAR)""")
    when = "a.admittime" if cfg.diagnosis_time == "admission" else "a.dischtime"
    dx = _src(cfg, "hosp", "diagnoses_icd")
    con.execute(f"""INSERT INTO stg
        SELECT 'diagnoses_icd', 'Condition', a.subject_id, a.hadm_id,
               CAST({when} AS DATE), {when}, NULL, NULL,
               coalesce(m.target_concept_id, 0), m.target_domain, d.icd_code,
               coalesce(m.source_concept_id, 0), {EHR_BILLING},
               CASE WHEN d.seq_num = '1' THEN {PRIMARY_DX} ELSE {SECONDARY_DX} END,
               d.seq_num, NULL, NULL, NULL, NULL
        FROM {dx} d
        JOIN adm a ON a.hadm_id = CAST(d.hadm_id AS BIGINT)
        LEFT JOIN codemap m ON m.code = d.icd_code
             AND m.vocabulary_id = CASE d.icd_version WHEN '9' THEN 'ICD9CM' ELSE 'ICD10CM' END
        """)  # fmt: skip
    px = _src(cfg, "hosp", "procedures_icd")
    con.execute(f"""INSERT INTO stg
        SELECT 'procedures_icd', 'Procedure', a.subject_id, a.hadm_id,
               CAST(p.chartdate AS DATE), NULL, NULL, NULL,
               coalesce(m.target_concept_id, 0), m.target_domain, p.icd_code,
               coalesce(m.source_concept_id, 0), {EHR_BILLING}, NULL, p.seq_num, NULL,
               NULL, NULL, NULL
        FROM {px} p
        JOIN adm a ON a.hadm_id = CAST(p.hadm_id AS BIGINT)
        LEFT JOIN codemap m ON m.code = p.icd_code
             AND m.vocabulary_id = CASE p.icd_version WHEN '9' THEN 'ICD9Proc'
                                                      ELSE 'ICD10PCS' END""")  # fmt: skip
    rx = _src(cfg, "hosp", "prescriptions")
    con.execute(f"""INSERT INTO stg
        SELECT 'prescriptions', 'Drug', a.subject_id, a.hadm_id,
               CAST(CAST(nullif(r.starttime, '') AS TIMESTAMP) AS DATE),
               CAST(nullif(r.starttime, '') AS TIMESTAMP),
               CAST(CAST(nullif(r.stoptime, '') AS TIMESTAMP) AS DATE),
               CAST(nullif(r.stoptime, '') AS TIMESTAMP),
               coalesce(m.target_concept_id, 0), m.target_domain,
               coalesce(nullif(r.ndc, ''), '0') || ':' || coalesce(r.drug, ''),
               coalesce(m.source_concept_id, 0), {EHR_PRESCRIPTION}, NULL, NULL, r.route,
               NULL, NULL, NULL
        FROM {rx} r
        JOIN adm a ON a.hadm_id = CAST(r.hadm_id AS BIGINT)
        LEFT JOIN codemap m ON m.vocabulary_id = 'NDC' AND m.code = r.ndc
        WHERE nullif(r.starttime, '') IS NOT NULL""")
    n_rx_no_start = con.execute(f"""SELECT count(*) FROM {rx} r
        JOIN adm a ON a.hadm_id = CAST(r.hadm_id AS BIGINT)
        WHERE nullif(r.starttime, '') IS NULL""").fetchone()[0]
    if n_rx_no_start:
        rep.notes.append(f"prescriptions without a start time left out: {suppress(n_rx_no_start)}")

    if cfg.concept_map is not None:
        _item_events(con, cfg, rep)
    else:
        rep.notes.append("chartevents and procedureevents not converted: no concept_map given")
    _reviewed_events(con, cfg, rep)

    # Route by the standard concept's domain; unmapped records stay in their home table.
    for domain, (table, select) in _ROUTES.items():
        con.execute(f"""INSERT INTO {table} BY NAME
            SELECT row_number() OVER (ORDER BY person_id, event_datetime, event_date,
                                      source_value) AS {_ID[table]}, {select}
            FROM stg WHERE coalesce(domain, home) = '{domain}'""")  # fmt: skip
    other = con.execute("""SELECT coalesce(domain, home), count(*) FROM stg
        WHERE coalesce(domain, home) NOT IN ('Condition', 'Procedure', 'Drug', 'Measurement',
                                             'Observation', 'Device') GROUP BY 1""").fetchall()
    for dom, n in other:
        rep.notes.append(f"{suppress(n)} records map to the {dom} domain, which has no table "
                         "here; left out")  # fmt: skip

    for origin in ("diagnoses_icd", "procedures_icd", "prescriptions"):
        rows, mapped = con.execute(f"""SELECT count(*), count(*) FILTER (WHERE concept_id <> 0)
            FROM stg WHERE origin = '{origin}'""").fetchone()  # fmt: skip
        top = con.execute(f"""SELECT source_value, count(*) AS n FROM stg
            WHERE origin = '{origin}' AND concept_id = 0
            GROUP BY 1 ORDER BY n DESC, 1 LIMIT 20""").fetchall()  # fmt: skip
        rep.mapping[origin] = {"rows": rows, "mapped": mapped,
                               "top_unmapped": [[v, n] for v, n in top]}  # fmt: skip
        rep.source_rows[origin] = rows

    desc = ("MIMIC-IV 3.1 (PhysioNet), hosp and icu modules"
            + (f", {pct:g}% subject sample (subject_id mod 10000)" if pct else "")
            + f"; diagnoses dated at {cfg.diagnosis_time}; observation periods run "
              f"{f} days past the last discharge (death ascertainment); dates are shifted "
              "per patient")  # fmt: skip
    con.execute("INSERT INTO cdm_source VALUES (?, ?, ?, ?, ?, ?, NULL, ?, '5.4', 0, ?)", [
        "MIMIC-IV 3.1 (Agentic-TTE ETL)", "MIMIC-IV", "PhysioNet / MIT-LCP", desc,
        "https://physionet.org/content/mimiciv/3.1/",
        f"agentic_tte.etl.mimic_omop {git_commit or 'unknown commit'}",
        datetime.now(UTC).date(), rep.vocabulary_version or "unknown"])  # fmt: skip
    rep.notes.append(desc)
    for table in CDM:
        rep.cdm_rows[table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    con.execute("CHECKPOINT")
    con.close()
    rep.finished_at = datetime.now(UTC)
    return rep


# Curated MIT-LCP maps: source table, map file, home domain, start and end time columns.
ITEM_MAPS = {
    "chartevents": ("chartevents_to_omop.csv", "Measurement", "charttime", None),
    "procedureevents": ("procedureevents_to_omop.csv", "Procedure", "starttime", "endtime"),
}


def _checked_map(con, name: str, raw_select: str) -> None:
    """``map_<name>`` from rows (itemid, label, predicate, curated_id, reviewer): a target is
    used if standard and valid, followed through 'Maps to' if it is no longer standard, and
    rejected if the vocabulary does not have it."""
    con.execute(f"""CREATE TEMP TABLE map_{name} AS
        WITH raw AS ({raw_select}),
        checked AS (
            SELECT raw.*, c.concept_id AS found, c.standard_concept AS std,
                   c.invalid_reason AS inv,
                   (SELECT min(r.concept_id_2) FROM concept_relationship r
                    JOIN concept t ON t.concept_id = r.concept_id_2 AND t.standard_concept = 'S'
                    WHERE r.concept_id_1 = raw.curated_id AND r.relationship_id = 'Maps to'
                      AND (r.invalid_reason IS NULL OR r.invalid_reason = '')) AS successor
            FROM raw LEFT JOIN concept c ON c.concept_id = raw.curated_id)
        SELECT itemid, label, predicate, curated_id, reviewer,
               CASE WHEN found IS NULL THEN NULL
                    WHEN std = 'S' AND coalesce(inv, '') = '' THEN curated_id
                    ELSE successor END AS concept_id,
               CASE WHEN found IS NULL THEN 'not in the vocabulary'
                    WHEN std = 'S' AND coalesce(inv, '') = '' THEN 'accepted'
                    WHEN successor IS NOT NULL THEN 'remapped'
                    ELSE 'not standard and no Maps to' END AS status
        FROM checked""")  # fmt: skip


def _map_report(con, rep: EtlReport, name: str, file: str) -> None:
    stats = dict(con.execute(f"SELECT status, count(*) FROM map_{name} GROUP BY 1").fetchall())
    rejected = con.execute(f"""SELECT itemid, label, curated_id, status FROM map_{name}
        WHERE concept_id IS NULL ORDER BY itemid""").fetchall()  # fmt: skip
    preds = dict(con.execute(f"SELECT predicate, count(*) FROM map_{name} GROUP BY 1"
                             ).fetchall())  # fmt: skip
    rep.item_maps[name] = {
        "file": file, "rows_in_file": sum(stats.values()),
        "accepted": stats.get("accepted", 0), "remapped": stats.get("remapped", 0),
        "rejected": [list(r) for r in rejected], "predicates": preds}  # fmt: skip
    con.execute(f"""INSERT INTO source_to_concept_map
        SELECT CAST(m.itemid AS VARCHAR), 0, 'MIMIC-IV {name}', m.label, m.concept_id,
               c.vocabulary_id, DATE '1970-01-01', DATE '2099-12-31', NULL
        FROM map_{name} m JOIN concept c ON c.concept_id = m.concept_id""")  # fmt: skip


# Event sources: module, home domain, start and end time, value columns (number, unit, raw),
# how rows join the admitted cohort, and the item dictionary for labels.
EVENTS = {
    "chartevents": ("icu", "Measurement", "charttime", None,
                    "TRY_CAST(e.valuenum AS DOUBLE), e.valueuom, e.value", "hadm", "d_items"),
    "procedureevents": ("icu", "Procedure", "starttime", "endtime", "NULL, NULL, NULL", "hadm",
                        "d_items"),
    # Labs join on the patient: outpatient labs (no admission) are kept, with no visit.
    "labevents": ("hosp", "Measurement", "charttime", None,
                  "TRY_CAST(e.valuenum AS DOUBLE), e.valueuom, e.value", "subject",
                  "d_labitems"),
    "inputevents": ("icu", "Drug", "starttime", "endtime",
                    "TRY_CAST(e.amount AS DOUBLE), e.amountuom, e.amount", "hadm", "d_items"),
}  # fmt: skip

# Curated MIT-LCP maps (SSSOM) for the sources that have one.
ITEM_MAPS = {"chartevents": "chartevents_to_omop.csv",
             "procedureevents": "procedureevents_to_omop.csv"}


def _convert_events(con, cfg: EtlConfig, rep: EtlReport, source: str, how: str) -> None:
    module, home, start, end, value, join, dictionary = EVENTS[source]
    for t in (source, dictionary):
        mod = "hosp" if t == "d_labitems" else module
        if not (cfg.source / mod / f"{t}.csv.gz").exists():
            raise FileNotFoundError(cfg.source / mod / f"{t}.csv.gz")
    ev = _src(cfg, module, source)
    if join == "hadm":
        link = "JOIN adm a ON a.hadm_id = TRY_CAST(e.hadm_id AS BIGINT)"
        person, visit = "a.subject_id", "a.hadm_id"
    else:
        link = ("JOIN (SELECT DISTINCT subject_id FROM adm) a "
                "ON a.subject_id = TRY_CAST(e.subject_id AS BIGINT)")  # fmt: skip
        person, visit = "a.subject_id", "TRY_CAST(nullif(e.hadm_id, '') AS BIGINT)"
    t_end = f"CAST(nullif(e.{end}, '') AS TIMESTAMP)" if end else "NULL"
    con.execute(f"""INSERT INTO stg
        SELECT '{source}', '{home}', {person}, {visit},
               CAST(CAST(e.{start} AS TIMESTAMP) AS DATE), CAST(e.{start} AS TIMESTAMP),
               CAST({t_end} AS DATE), {t_end}, m.concept_id, c.domain_id,
               e.itemid, 0, {EHR}, NULL, NULL, NULL, {value}
        FROM {ev} e {link}
        JOIN map_{source} m ON m.itemid = TRY_CAST(e.itemid AS BIGINT)
             AND m.concept_id IS NOT NULL
        JOIN concept c ON c.concept_id = m.concept_id
        WHERE nullif(e.{start}, '') IS NOT NULL""")  # fmt: skip
    converted = con.execute(f"SELECT count(*) FROM stg WHERE origin = '{source}'").fetchone()[0]
    top = con.execute(f"""
        SELECT TRY_CAST(e.itemid AS BIGINT) AS itemid, count(*) AS n FROM {ev} e {link}
        WHERE TRY_CAST(e.itemid AS BIGINT) NOT IN
              (SELECT itemid FROM map_{source} WHERE concept_id IS NOT NULL)
        GROUP BY 1 ORDER BY n DESC, 1""").fetchall()  # fmt: skip
    dict_mod = "hosp" if dictionary == "d_labitems" else module
    labels = dict(con.execute(f"SELECT TRY_CAST(itemid AS BIGINT), label "
                              f"FROM {_src(cfg, dict_mod, dictionary)}").fetchall())
    unconverted = sum(n for _, n in top)
    rep.mapping[source] = {
        "rows": converted + unconverted, "mapped": converted,
        "top_unmapped": [[f"{i} {labels.get(i, '?')}", n] for i, n in top[:20]]}
    rep.source_rows[source] = converted + unconverted
    rep.notes.append(f"{source}: items without {how} are not converted "
                     f"({suppress(unconverted)} rows, {len(top)} items; see the mapping "
                     "table)")  # fmt: skip


def _item_events(con, cfg: EtlConfig, rep: EtlReport) -> None:
    for source, file in ITEM_MAPS.items():
        path = (cfg.concept_map / file).as_posix()
        _checked_map(con, source, f"""
            SELECT TRY_CAST(replace(subject_id, 'mimic:', '') AS BIGINT) AS itemid,
                   subject_label AS label, predicate_id AS predicate,
                   TRY_CAST(replace(object_id, 'omop:', '') AS INTEGER) AS curated_id,
                   coalesce(reviewer_id, '') AS reviewer
            FROM read_csv('{path}', header = true, all_varchar = true)
            WHERE object_id LIKE 'omop:%'""")  # fmt: skip
        _map_report(con, rep, source, file)
        _convert_events(con, cfg, rep, source, "a curated map")


def _reviewed_events(con, cfg: EtlConfig, rep: EtlReport) -> None:
    """labevents and inputevents through reviewed item maps (agentic_tte.etl.item_maps)."""
    from agentic_tte.etl.item_maps import ItemMap

    seen = set()
    for path in cfg.item_maps:
        m = ItemMap.model_validate_json(Path(path).read_text(encoding="utf-8"))
        if m.kind in seen:
            raise ValueError(f"two item maps for {m.kind}")
        seen.add(m.kind)
        approved = [(e.itemid, e.label, e.concept_id, e.reviewer or "")
                    for e in m.entries if e.status == "approved" and e.concept_id]  # fmt: skip
        con.execute(f"CREATE TEMP TABLE raw_{m.kind} (itemid BIGINT, label VARCHAR, "
                    f"curated_id INTEGER, reviewer VARCHAR)")
        if approved:
            con.executemany(f"INSERT INTO raw_{m.kind} VALUES (?, ?, ?, ?)", approved)
        _checked_map(con, m.kind, f"SELECT itemid, label, 'reviewed' AS predicate, "
                                  f"curated_id, reviewer FROM raw_{m.kind}")
        _map_report(con, rep, m.kind, Path(path).name)
        rep.item_maps[m.kind]["reviewers"] = sorted({a[3] for a in approved if a[3]})
        rep.item_maps[m.kind]["not_reviewed"] = sum(e.status in ("proposed", "no_match")
                                                    for e in m.entries)  # fmt: skip
        _convert_events(con, cfg, rep, m.kind, "an approved item map")
    for kind in ("labevents", "inputevents"):
        if kind not in seen:
            rep.notes.append(f"{kind} not converted: no reviewed item map given")


def sha256sums_digest(source: Path) -> str | None:
    """Hash of the release's SHA256SUMS.txt: identifies the exact source in the manifest."""
    import hashlib

    f = source / "SHA256SUMS.txt"
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None

