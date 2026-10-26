"""MIMIC-IV -> OMOP CDM 5.4 ETL on a tiny MADE-UP dataset shaped like the MIMIC-IV 3.1 tables
(written at test time; no MIMIC rows exist in the repository)."""

from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from agentic_tte.etl.mimic_omop import EtlConfig, build
from agentic_tte.omop import DataContext, run_gate
from agentic_tte.omop.compiler import CompileSpec, EventDef, compile_protocol
from agentic_tte.protocol import (
    ConceptRef,
    Criterion,
    CriterionAtom,
    Duration,
    Outcome,
    TargetTrialProtocol,
    TreatmentComponent,
    TreatmentStrategy,
)

VOCAB = Path(__file__).parent / "fixtures" / "vocab"
S1, S2, S3, S4 = 10000001, 10000002, 10000003, 10000004


def _write(root: Path, module: str, name: str, rows: list[dict], columns: list[str]):
    d = root / module
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(d / f"{name}.csv.gz", index=False,
                                               compression="gzip")  # fmt: skip


def made_up_mimic(root: Path) -> Path:
    """Four invented patients. S1: ED then ICU, septic shock (ICD-10), dialysis, hydrocortisone,
    dies 7 days after discharge. S2: two admissions, an unmapped code, a prescription without
    a start time. S3: no admission. S4: septic shock (ICD-9), dies in hospital."""
    _write(root, "hosp", "patients", [
        {"subject_id": S1, "gender": "M", "anchor_age": 60, "anchor_year": 2150,
         "anchor_year_group": "2008 - 2010", "dod": "2150-03-15"},
        {"subject_id": S2, "gender": "F", "anchor_age": 45, "anchor_year": 2160,
         "anchor_year_group": "2011 - 2013", "dod": None},
        {"subject_id": S3, "gender": "M", "anchor_age": 70, "anchor_year": 2170,
         "anchor_year_group": "2014 - 2016", "dod": None},
        {"subject_id": S4, "gender": "F", "anchor_age": 80, "anchor_year": 2155,
         "anchor_year_group": "2017 - 2019", "dod": "2155-05-02"},
    ], ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"])
    adm_cols = ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                "admission_type", "admit_provider_id", "admission_location",
                "discharge_location", "insurance", "language", "marital_status", "race",
                "edregtime", "edouttime", "hospital_expire_flag"]
    _write(root, "hosp", "admissions", [
        {"subject_id": S1, "hadm_id": 20000001, "admittime": "2150-03-01 10:00:00",
         "dischtime": "2150-03-08 12:00:00", "admission_type": "EW EMER.",
         "admission_location": "EMERGENCY ROOM", "discharge_location": "HOME", "race": "WHITE",
         "edregtime": "2150-03-01 08:00:00", "hospital_expire_flag": 0},
        {"subject_id": S2, "hadm_id": 20000002, "admittime": "2160-01-01 09:00:00",
         "dischtime": "2160-01-05 10:00:00", "admission_type": "ELECTIVE", "race": "ASIAN",
         "hospital_expire_flag": 0},
        {"subject_id": S2, "hadm_id": 20000003, "admittime": "2160-06-01 09:00:00",
         "dischtime": "2160-06-03 10:00:00", "admission_type": "ELECTIVE",
         "race": "ASIAN - CHINESE", "hospital_expire_flag": 0},
        {"subject_id": S4, "hadm_id": 20000004, "admittime": "2155-04-28 09:00:00",
         "dischtime": "2155-05-02 13:00:00", "deathtime": "2155-05-02 13:00:00",
         "admission_type": "URGENT", "race": "BLACK", "hospital_expire_flag": 1},
    ], adm_cols)
    _write(root, "icu", "icustays", [
        {"subject_id": S1, "hadm_id": 20000001, "stay_id": 30000001,
         "first_careunit": "Medical Intensive Care Unit (MICU)", "last_careunit": "MICU",
         "intime": "2150-03-01 12:00:00", "outtime": "2150-03-04 12:00:00", "los": 3.0},
        {"subject_id": S4, "hadm_id": 20000004, "stay_id": 30000004,
         "first_careunit": "MICU", "last_careunit": "MICU", "intime": "2155-04-28 12:00:00",
         "outtime": "2155-05-02 13:00:00", "los": 4.0},
    ], ["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime",
        "outtime", "los"])
    _write(root, "hosp", "diagnoses_icd", [
        {"subject_id": S1, "hadm_id": 20000001, "seq_num": 1, "icd_code": "R6521",
         "icd_version": 10},
        {"subject_id": S1, "hadm_id": 20000001, "seq_num": 2, "icd_code": "Z992",
         "icd_version": 10},
        {"subject_id": S4, "hadm_id": 20000004, "seq_num": 1, "icd_code": "78552",
         "icd_version": 9},
        {"subject_id": S2, "hadm_id": 20000002, "seq_num": 1, "icd_code": "Q999",
         "icd_version": 10},
    ], ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
    _write(root, "hosp", "procedures_icd", [
        {"subject_id": S1, "hadm_id": 20000001, "seq_num": 1, "chartdate": "2150-03-02",
         "icd_code": "5A1D70Z", "icd_version": 10},
    ], ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"])
    rx_cols = ["subject_id", "hadm_id", "pharmacy_id", "poe_id", "poe_seq",
               "order_provider_id", "starttime", "stoptime", "drug_type", "drug",
               "formulary_drug_cd", "gsn", "ndc", "prod_strength", "form_rx", "dose_val_rx",
               "dose_unit_rx", "form_val_disp", "form_unit_disp", "doses_per_24_hrs", "route"]
    _write(root, "hosp", "prescriptions", [
        {"subject_id": S1, "hadm_id": 20000001, "starttime": "2150-03-01 13:00:00",
         "stoptime": "2150-03-05 13:00:00", "drug_type": "MAIN",
         "drug": "Hydrocortisone Na Succ.", "ndc": "00009001101", "route": "IV"},
        {"subject_id": S1, "hadm_id": 20000001, "starttime": "2150-03-01 13:00:00",
         "stoptime": None, "drug_type": "BASE", "drug": "Sodium Chloride 0.9%", "ndc": "0",
         "route": "IV"},
        {"subject_id": S2, "hadm_id": 20000002, "starttime": None, "stoptime": None,
         "drug_type": "MAIN", "drug": "Acetaminophen", "ndc": "0", "route": "PO"},
    ], rx_cols)  # fmt: skip
    return root


@pytest.fixture
def cdm(tmp_path):
    src = made_up_mimic(tmp_path / "mimic-iv-3.1")
    out = tmp_path / "omop" / "mimic.duckdb"
    rep = build(EtlConfig(source=src, vocab=VOCAB, out=out), git_commit="test")
    return duckdb.connect(str(out), read_only=True), rep


def rows(con, sql):
    return con.execute(sql).fetchall()


def test_persons_visits_deaths_and_observation(cdm):
    con, rep = cdm
    assert rows(con, "SELECT person_id, gender_concept_id, year_of_birth, race_source_value "
                     "FROM person ORDER BY 1") == [  # fmt: skip
        (S1, 8507, 2090, "WHITE"), (S2, 8532, 2115, "ASIAN - CHINESE"), (S4, 8532, 2075, "BLACK")]
    assert rep.persons_without_admission == 1  # S3
    assert rows(con, "SELECT visit_occurrence_id, visit_concept_id FROM visit_occurrence "
                     "ORDER BY 1") == [(20000001, 262), (20000002, 9201), (20000003, 9201),
                                       (20000004, 9201)]  # fmt: skip
    assert rows(con, "SELECT visit_detail_id, visit_detail_concept_id, visit_occurrence_id "
                     "FROM visit_detail ORDER BY 1") == [(30000001, 32037, 20000001),
                                                          (30000004, 32037, 20000004)]
    # S1 died after discharge (date only, from state records); S4 in hospital (with time).
    assert rows(con, "SELECT person_id, death_date, death_datetime FROM death ORDER BY 1") == [
        (S1, date(2150, 3, 15), None), (S4, date(2155, 5, 2), datetime(2155, 5, 2, 13))]
    op = dict(rows(con, "SELECT person_id, observation_period_end_date FROM observation_period"))
    assert op[S1] == date(2150, 3, 15)  # ends at death
    assert op[S2] == date(2161, 6, 3)  # last discharge + 365 days: death ascertainment
    start = rows(con, f"SELECT observation_period_start_date FROM observation_period "
                      f"WHERE person_id = {S1}")[0][0]
    assert start == date(2150, 3, 1)


def test_codes_go_through_the_vocabulary_and_follow_their_domain(cdm):
    con, rep = cdm
    assert sorted(rows(con, "SELECT person_id, condition_concept_id, condition_source_concept_id,"
                            " condition_status_concept_id FROM condition_occurrence")) == [
        (S1, 196236, 9000002, 32902), (S2, 0, 0, 32902), (S4, 196236, 9000003, 32902)]
    # Z99.2 maps to an Observation concept, so it lands in observation.
    assert rows(con, "SELECT person_id, observation_concept_id, observation_source_value "
                     "FROM observation") == [(S1, 9000093, "Z992")]
    assert rows(con, "SELECT procedure_concept_id, procedure_date, procedure_datetime "
                     "FROM procedure_occurrence") == [(9000070, date(2150, 3, 2), None)]
    drugs = sorted(rows(con, "SELECT drug_concept_id, drug_source_value, "
                             "drug_exposure_end_datetime FROM drug_exposure"))
    assert drugs[0][0] == 0 and drugs[0][1] == "0:Sodium Chloride 0.9%"
    assert drugs[0][2] == datetime(2150, 3, 1, 13)  # no stop time: ends at its start
    assert drugs[1][:2] == (9000021, "00009001101:Hydrocortisone Na Succ.")
    m = rep.mapping
    assert (m["diagnoses_icd"]["rows"], m["diagnoses_icd"]["mapped"]) == (4, 3)
    assert m["diagnoses_icd"]["top_unmapped"] == [["Q999", 1]]
    assert (m["prescriptions"]["rows"], m["prescriptions"]["mapped"]) == (2, 1)
    assert any("without a start time" in n for n in rep.notes)
    pub = rep.public()
    assert pub["mapping"]["diagnoses_icd"]["rows"] == "<11"
    assert pub["mapping"]["diagnoses_icd"]["top_unmapped"] == [["Q999", "<11"]]
    md = rep.to_markdown()
    assert "| diagnoses_icd | <11 | <11 | 75.0% | Q999 (<11) |" in md
    src = rows(con, "SELECT cdm_version, cdm_source_abbreviation, source_description "
                    "FROM cdm_source")[0]  # fmt: skip
    assert src[:2] == ("5.4", "MIMIC-IV") and "diagnoses dated at admission" in src[2]


def test_diagnosis_time_and_sample(tmp_path):
    src = made_up_mimic(tmp_path / "m")
    out = tmp_path / "d.duckdb"
    build(EtlConfig(source=src, vocab=VOCAB, out=out, diagnosis_time="discharge"))
    con = duckdb.connect(str(out), read_only=True)
    t = rows(con, f"SELECT condition_start_datetime FROM condition_occurrence "
                  f"WHERE person_id = {S1}")[0][0]
    assert t == datetime(2150, 3, 8, 12)  # known at discharge
    out2 = tmp_path / "s.duckdb"
    rep = build(EtlConfig(source=src, vocab=VOCAB, out=out2, sample_percent=0.02))
    assert rows(duckdb.connect(str(out2), read_only=True), "SELECT person_id FROM person") == [
        (S1,)]  # subject_id mod 10000 < 2
    assert "0.02% subject sample" in " ".join(rep.notes)


def test_refuses_to_overwrite_or_run_without_sources(tmp_path, cdm):
    src = made_up_mimic(tmp_path / "x")
    out = tmp_path / "o.duckdb"
    out.write_bytes(b"")
    with pytest.raises(FileExistsError):
        build(EtlConfig(source=src, vocab=VOCAB, out=out))
    (src / "hosp" / "prescriptions.csv.gz").unlink()
    with pytest.raises(FileNotFoundError, match="prescriptions"):
        build(EtlConfig(source=src, vocab=VOCAB, out=tmp_path / "p.duckdb"))


def test_the_gate_and_the_compiler_run_on_the_output(cdm):
    """The CDM the ETL writes is what the Phase 2 code expects: the gate passes its structure
    checks, and a compiled cohort and frame run on it."""
    con, _ = cdm
    r = run_gate(con, None, context=DataContext(data_classification="deidentified",
                                                approved_by_rcss=True))  # fmt: skip
    assert r.check("structure").status == "green" and r.check("cdm_metadata").status == "green"
    assert r.check("death").status == "green"
    p = TargetTrialProtocol(
        protocol_id="T", eligibility=[Criterion(id="I1", kind="inclusion", text="adults",
            atoms=[CriterionAtom(concepts=[ConceptRef(name="age", domain="Demographic")],
                                 value={"comparator": ">=", "value": 18})])],
        treatment_strategies=[
            TreatmentStrategy(label="HC", components=[TreatmentComponent(concept=ConceptRef(
                name="hydrocortisone", domain="Drug", omop_concept_id=9000020))]),
            TreatmentStrategy(label="none", is_comparator=True)],
        outcomes=[Outcome(name="28-day mortality", primary=True,
                          horizon=Duration(amount=28, unit="day"))])  # fmt: skip
    cc = compile_protocol(p, CompileSpec(time_zero=EventDef(domain="Condition",
                                                            concept_ids=[196236])))
    cohort = con.execute(cc.cohort_sql).df()
    assert sorted(cohort.person_id) == [S1, S4] and cohort.eligible.all()
    f = con.execute(cc.frame_sql).df().set_index("person_id")
    assert f.loc[S1, "treated"] == 1  # hydrocortisone (descendant of the ingredient)
    assert f.loc[S1, "event"] == 1 and f.loc[S4, "event"] == 1
    # Time zero: the septic-shock code, dated at admission (10:00). S1's death is a date only
    # (state record), placed at noon; S4 died in hospital at 13:00 on day 4.
    assert f.loc[S1, "time"] == pytest.approx(14 + 2 / 24)
    assert f.loc[S4, "time"] == pytest.approx(4 + 4 / 24)


def test_cli_writes_cdm_report_and_manifest(tmp_path):
    from typer.testing import CliRunner

    from agentic_tte.cli import app

    src = made_up_mimic(tmp_path / "mimic-iv-3.1")
    (src / "SHA256SUMS.txt").write_text("made-up checksums\n", encoding="utf-8")
    out = tmp_path / "omop" / "mimic.duckdb"
    r = CliRunner().invoke(app, ["etl", "mimic", "--source", str(src), "--vocab", str(VOCAB),
                                 "--out", str(out), "--threads", "2"])  # fmt: skip
    assert r.exit_code == 0, r.output
    manifest = (tmp_path / "omop" / "mimic.manifest.txt").read_text(encoding="utf-8")
    assert "source_sha256sums_digest=" in manifest and "duckdb=v" in manifest
    report = (tmp_path / "omop" / "mimic.report.md").read_text(encoding="utf-8")
    assert "<11" in report and "Q999" in report
    assert CliRunner().invoke(app, ["etl", "mimic", "--source", str(src), "--vocab",
                                    str(VOCAB), "--out", str(out)]).exit_code != 0  # fmt: skip
