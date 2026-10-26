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
Not yet: labevents, chartevents, inputevents and procedureevents, which need reviewed item
maps (itemid -> LOINC or standard concept), and the empty ED and note modules.

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
                     "drug_type_concept_id INTEGER, route_source_value VARCHAR, "
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
        type_concept_id AS drug_type_concept_id, route_source_value, visit_occurrence_id,
        source_value AS drug_source_value, source_concept_id AS drug_source_concept_id"""),
    "Measurement": ("measurement", """
        person_id, concept_id AS measurement_concept_id, event_date AS measurement_date,
        event_datetime AS measurement_datetime, type_concept_id AS measurement_type_concept_id,
        visit_occurrence_id, source_value AS measurement_source_value,
        source_concept_id AS measurement_source_concept_id"""),
    "Observation": ("observation", """
        person_id, concept_id AS observation_concept_id, event_date AS observation_date,
        event_datetime AS observation_datetime, type_concept_id AS observation_type_concept_id,
        visit_occurrence_id, source_value AS observation_source_value,
        source_concept_id AS observation_source_concept_id"""),
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
        return d

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
        status_concept_id INTEGER, status_source_value VARCHAR, route_source_value VARCHAR)""")
    when = "a.admittime" if cfg.diagnosis_time == "admission" else "a.dischtime"
    dx = _src(cfg, "hosp", "diagnoses_icd")
    con.execute(f"""INSERT INTO stg
        SELECT 'diagnoses_icd', 'Condition', a.subject_id, a.hadm_id,
               CAST({when} AS DATE), {when}, NULL, NULL,
               coalesce(m.target_concept_id, 0), m.target_domain, d.icd_code,
               coalesce(m.source_concept_id, 0), {EHR_BILLING},
               CASE WHEN d.seq_num = '1' THEN {PRIMARY_DX} ELSE {SECONDARY_DX} END,
               d.seq_num, NULL
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
               coalesce(m.source_concept_id, 0), {EHR_BILLING}, NULL, p.seq_num, NULL
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
               coalesce(m.source_concept_id, 0), {EHR_PRESCRIPTION}, NULL, NULL, r.route
        FROM {rx} r
        JOIN adm a ON a.hadm_id = CAST(r.hadm_id AS BIGINT)
        LEFT JOIN codemap m ON m.vocabulary_id = 'NDC' AND m.code = r.ndc
        WHERE nullif(r.starttime, '') IS NOT NULL""")
    n_rx_no_start = con.execute(f"""SELECT count(*) FROM {rx} r
        JOIN adm a ON a.hadm_id = CAST(r.hadm_id AS BIGINT)
        WHERE nullif(r.starttime, '') IS NULL""").fetchone()[0]
    if n_rx_no_start:
        rep.notes.append(f"prescriptions without a start time left out: {suppress(n_rx_no_start)}")

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
    rep.notes += [desc, "not yet converted: labevents, chartevents, inputevents, "
                        "procedureevents (need reviewed item maps)"]  # fmt: skip
    for table in CDM:
        rep.cdm_rows[table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    con.execute("CHECKPOINT")
    con.close()
    rep.finished_at = datetime.now(UTC)
    return rep


def sha256sums_digest(source: Path) -> str | None:
    """Hash of the release's SHA256SUMS.txt: identifies the exact source in the manifest."""
    import hashlib

    f = source / "SHA256SUMS.txt"
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None

