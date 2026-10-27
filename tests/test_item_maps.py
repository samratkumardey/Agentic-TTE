"""Reviewed item maps for labevents and inputevents: propose -> review -> apply -> ETL, on
made-up items and events (no MIMIC rows in the repository)."""

import json
from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from agentic_tte.cli import app
from agentic_tte.etl.item_maps import (
    DecisionError,
    ItemMap,
    apply_decisions,
    decisions_template,
    item_counts,
    propose_items,
    review_sheet,
)
from agentic_tte.etl.mimic_omop import EtlConfig, build
from agentic_tte.omop.vocab import load_athena
from tests.test_etl_mimic import S1, S2, S4, VOCAB, _write, made_up_mimic

LACTATE, CREATININE, MADE_UP = 50813, 50912, 59999
NOREPINEPHRINE = 221906


def made_up_labs_and_inputs(mimic: Path) -> Path:
    _write(mimic, "hosp", "d_labitems", [
        {"itemid": LACTATE, "label": "Lactate", "fluid": "Blood", "category": "Blood Gas"},
        {"itemid": CREATININE, "label": "Creatinine", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": MADE_UP, "label": "Made-up assay", "fluid": "Blood", "category": "Other"},
    ], ["itemid", "label", "fluid", "category"])
    le = ["labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid", "order_provider_id",
          "charttime", "storetime", "value", "valuenum", "valueuom", "ref_range_lower",
          "ref_range_upper", "flag", "priority", "comments"]
    rows = [{"subject_id": S1, "hadm_id": 20000001, "itemid": LACTATE,
             "charttime": f"2150-03-01 {h:02d}:00:00", "value": f"{1 + h / 10:.1f}",
             "valuenum": 1 + h / 10, "valueuom": "mmol/L"} for h in range(10, 22)]
    rows += [{"subject_id": S4, "hadm_id": 20000004, "itemid": LACTATE,
              "charttime": "2155-04-28 10:00:00", "value": "4.0", "valuenum": 4.0,
              "valueuom": "mmol/L"},
             {"subject_id": S2, "hadm_id": None, "itemid": CREATININE,  # outpatient lab
              "charttime": "2160-03-01 08:00:00", "value": "1.1", "valuenum": 1.1,
              "valueuom": "mg/dL"},
             {"subject_id": S1, "hadm_id": 20000001, "itemid": MADE_UP,
              "charttime": "2150-03-01 12:00:00", "value": "x"}]
    _write(mimic, "hosp", "labevents", rows, le)
    _write(mimic, "icu", "d_items", [
        {"itemid": NOREPINEPHRINE, "label": "Norepinephrine", "linksto": "inputevents",
         "category": "Medications", "unitname": "mg"},
    ], ["itemid", "label", "abbreviation", "linksto", "category", "unitname", "param_type",
        "lownormalvalue", "highnormalvalue"])
    ie = ["subject_id", "hadm_id", "stay_id", "caregiver_id", "starttime", "endtime",
          "storetime", "itemid", "amount", "amountuom", "rate", "rateuom", "orderid",
          "linkorderid", "ordercategoryname", "secondaryordercategoryname",
          "ordercomponenttypedescription", "ordercategorydescription", "patientweight",
          "totalamount", "totalamountuom", "isopenbag", "continueinnextdept",
          "statusdescription", "originalamount", "originalrate"]
    _write(mimic, "icu", "inputevents", [
        {"subject_id": S1, "hadm_id": 20000001, "stay_id": 30000001,
         "starttime": "2150-03-01 12:00:00", "endtime": "2150-03-01 18:00:00",
         "itemid": NOREPINEPHRINE, "amount": 2.0, "amountuom": "mg", "rate": 0.1,
         "rateuom": "mcg/kg/min"}], ie)
    return mimic


def _csv(mimic: Path, module: str, table: str) -> str:
    return f"read_csv('{(mimic / module / f'{table}.csv.gz').as_posix()}', header = true, " \
           "all_varchar = true)"  # fmt: skip


@pytest.fixture
def setup(tmp_path):
    mimic = made_up_labs_and_inputs(made_up_mimic(tmp_path / "mimic-iv-3.1"))
    con = duckdb.connect()
    load_athena(con, VOCAB)
    return mimic, con


def _labs(mimic, con) -> ItemMap:
    items, total = item_counts(con, "labevents", _csv(mimic, "hosp", "labevents"),
                               _csv(mimic, "hosp", "d_labitems"), min_rows=1)  # fmt: skip
    return propose_items(con, "labevents", items, total_rows=total)


def test_proposals_from_labels_most_frequent_first(setup):
    mimic, con = setup
    m = _labs(mimic, con)
    assert [e.itemid for e in m.entries] == [LACTATE, CREATININE, MADE_UP]
    lac = m.entries[0]
    assert lac.rows == 13 and lac.specimen == "Blood" and lac.status == "proposed"
    assert {9000030, 9000031} <= {c.concept_id for c in lac.candidates}  # Blood, Serum/Plasma
    assert lac.candidates[0].concept_id != 9000145  # the panel is only an alternative
    assert m.entries[1].rows == "<11"  # suppressed in the file too
    assert m.entries[1].status == "proposed"  # creatinine is in the fixture vocabulary
    assert m.entries[2].status == "no_match"  # the made-up assay is not
    sheet = review_sheet(m)
    assert "covering 100.0% of labevents rows" in sheet and "| 50813 | Lactate |" in sheet
    tmpl = decisions_template(m)
    # Nothing pre-filled: every item needs an explicit decision. The proposal is shown only.
    assert {d["action"] for d in tmpl["decisions"].values()} == {""}
    lac_d = tmpl["decisions"][str(LACTATE)]
    assert lac_d["proposed_concept_id"] in (9000030, 9000031) and lac_d["concept_id"] is None
    assert lac_d["label"] == "Lactate" and "_actions" in tmpl
    assert tmpl["decisions"][str(MADE_UP)]["proposed_concept_id"] is None
    small = item_counts(con, "labevents", _csv(mimic, "hosp", "labevents"),
                        _csv(mimic, "hosp", "d_labitems"))[0]  # fmt: skip
    assert [i["itemid"] for i in small] == [LACTATE]  # default: at least 11 events


def test_decisions_are_validated(setup):
    mimic, con = setup
    m = _labs(mimic, con)
    with pytest.raises(DecisionError, match="reviewer"):
        apply_decisions(m, {"decisions": {}})
    with pytest.raises(DecisionError, match="unknown item"):
        apply_decisions(m, {"reviewer": "Dr A", "decisions": {"1": {"action": "approve"}}})
    with pytest.raises(DecisionError, match="replace needs"):
        apply_decisions(m, {"reviewer": "Dr A",
                            "decisions": {str(LACTATE): {"action": "replace"}}})
    with pytest.raises(DecisionError, match="nothing to approve"):
        apply_decisions(m, {"reviewer": "Dr A",
                            "decisions": {str(MADE_UP): {"action": "approve"}}})
    with pytest.raises(DecisionError, match="unknown action"):
        apply_decisions(m, {"reviewer": "Dr A",
                            "decisions": {str(LACTATE): {"action": "aprove"}}})


def test_signing_the_template_unread_approves_nothing(setup):
    """The failure the blank template prevents: a reviewer adds a name and applies the file
    as generated. Every item stays undecided and nothing is converted."""
    mimic, con = setup
    m = _labs(mimic, con)
    tmpl = decisions_template(m)
    tmpl["reviewer"] = "Dr A"
    out = apply_decisions(m, tmpl)
    assert out.approved() == {} and not out.fully_reviewed
    assert {e.status for e in out.entries} == {"proposed", "no_match"}
    assert all(e.reviewer is None for e in out.entries)  # nobody is recorded as deciding
    tmpl["decisions"][str(LACTATE)]["action"] = "approve"  # one explicit decision
    out = apply_decisions(m, tmpl)
    assert out.approved() == {LACTATE: m.entries[0].concept_id}
    assert out.entries[0].reviewer == "Dr A" and out.entries[1].status == "proposed"


def test_reviewed_maps_drive_the_etl(setup, tmp_path):
    mimic, con = setup
    labs = apply_decisions(_labs(mimic, con), {"reviewer": "Dr A", "decisions": {
        str(LACTATE): {"action": "replace", "concept_id": 9000030, "note": "blood gas"},
        str(CREATININE): {"action": "unmappable"}, str(MADE_UP): {"action": "reject"}}})
    assert labs.fully_reviewed and labs.approved() == {LACTATE: 9000030}
    items, total = item_counts(con, "inputevents", _csv(mimic, "icu", "inputevents"),
                               _csv(mimic, "icu", "d_items"), min_rows=1)  # fmt: skip
    inputs = propose_items(con, "inputevents", items, total_rows=total)
    assert inputs.entries[0].candidates[0].concept_id == 9000060  # ingredient
    inputs = apply_decisions(inputs, {"reviewer": "Dr B", "decisions": {
        str(NOREPINEPHRINE): {"action": "approve"}}})
    paths = []
    for name, m in (("labs", labs), ("inputs", inputs)):
        f = tmp_path / f"{name}.reviewed.json"
        f.write_text(m.model_dump_json(), encoding="utf-8")
        paths.append(f)
    out = tmp_path / "cdm.duckdb"
    rep = build(EtlConfig(source=mimic, vocab=VOCAB, out=out, item_maps=paths))
    cdm = duckdb.connect(str(out), read_only=True)
    lab = cdm.execute("SELECT person_id, count(*), min(value_as_number), max(unit_source_value)"
                      " FROM measurement WHERE measurement_concept_id = 9000030 GROUP BY 1 "
                      "ORDER BY 1").fetchall()  # fmt: skip
    assert lab == [(S1, 12, 2.0, "mmol/L"), (S4, 1, 4.0, "mmol/L")]
    drug = cdm.execute("SELECT drug_concept_id, drug_exposure_start_datetime, "
                       "drug_exposure_end_datetime, quantity, dose_unit_source_value "
                       "FROM drug_exposure WHERE drug_concept_id = 9000060").fetchall()
    assert drug == [(9000060, datetime(2150, 3, 1, 12), datetime(2150, 3, 1, 18), 2.0, "mg")]
    assert (rep.mapping["labevents"]["rows"], rep.mapping["labevents"]["mapped"]) == (15, 13)
    assert rep.mapping["labevents"]["top_unmapped"][0] == [f"{CREATININE} Creatinine", 1]
    assert rep.item_maps["labevents"]["reviewers"] == ["Dr A"]
    assert rep.item_maps["inputevents"]["predicates"] == {"reviewed": 1}
    assert "labevents: items without an approved item map are not converted" in " ".join(
        rep.notes)


def test_a_reviewed_target_is_still_checked_against_the_vocabulary(setup, tmp_path):
    mimic, con = setup
    labs = apply_decisions(_labs(mimic, con), {"reviewer": "Dr A", "decisions": {
        str(LACTATE): {"action": "replace", "concept_id": 9000090},  # an NDC: not standard
        str(CREATININE): {"action": "replace", "concept_id": 123}}})  # not in the vocabulary
    f = tmp_path / "labs.json"
    f.write_text(labs.model_dump_json(), encoding="utf-8")
    rep = build(EtlConfig(source=mimic, vocab=VOCAB, out=tmp_path / "x.duckdb", item_maps=[f]))
    m = rep.item_maps["labevents"]
    assert m["remapped"] == 1 and m["rejected"] == [[CREATININE, "Creatinine", 123,
                                                     "not in the vocabulary"]]
    with pytest.raises(ValueError, match="two item maps"):
        build(EtlConfig(source=mimic, vocab=VOCAB, out=tmp_path / "y.duckdb",
                        item_maps=[f, f]))


def test_cli_propose_apply_and_etl(setup, tmp_path):
    mimic, _ = setup
    run = CliRunner().invoke
    rev = tmp_path / "review"
    r = run(app, ["etl", "items", "propose", "--source", str(mimic), "--vocab", str(VOCAB),
                  "--kind", "labevents", "-o", str(rev), "--min-rows", "1"])  # fmt: skip
    assert r.exit_code == 0, r.output
    d = json.loads((rev / "decisions.json").read_text(encoding="utf-8"))
    r = run(app, ["etl", "items", "apply", str(rev / "map.json"), str(rev / "decisions.json"),
                  "-o", str(rev / "reviewed.json")])  # fmt: skip
    assert r.exit_code == 2 and "reviewer" in r.output  # not filled in yet
    d["reviewer"] = "Dr A"
    (rev / "decisions.json").write_text(json.dumps(d), encoding="utf-8")
    r = run(app, ["etl", "items", "apply", str(rev / "map.json"), str(rev / "decisions.json"),
                  "-o", str(rev / "reviewed.json")])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "3 items undecided (not converted)" in r.output  # signed, but nothing decided
    d["decisions"][str(LACTATE)].update(action="replace", concept_id=9000030)
    (rev / "decisions.json").write_text(json.dumps(d), encoding="utf-8")
    r = run(app, ["etl", "items", "apply", str(rev / "map.json"), str(rev / "decisions.json"),
                  "-o", str(rev / "reviewed.json")])  # fmt: skip
    assert r.exit_code == 0 and "2 items undecided" in r.output, r.output
    r = run(app, ["etl", "mimic", "--source", str(mimic), "--vocab", str(VOCAB), "--out",
                  str(tmp_path / "c.duckdb"), "--item-map", str(rev / "reviewed.json")])
    assert r.exit_code == 0, r.output
    assert "item_maps=" in (tmp_path / "c.manifest.txt").read_text(encoding="utf-8")
    cdm = duckdb.connect(str(tmp_path / "c.duckdb"), read_only=True)
    assert cdm.execute("SELECT count(*) FROM measurement WHERE measurement_concept_id = "
                       "9000030").fetchone()[0] == 13  # only the decided item  # fmt: skip


def test_cli_explains_an_unset_path(setup, tmp_path):
    """An unset environment variable gives an empty path ("."): say so, before any slow load."""
    mimic, _ = setup
    run = CliRunner().invoke
    r = run(app, ["etl", "items", "propose", "--source", "", "--vocab", str(VOCAB),
                  "--kind", "inputevents", "-o", str(tmp_path / "x")])  # fmt: skip
    assert r.exit_code == 2 and "ATTE_MIMIC_SOURCE" in r.output
    r = run(app, ["etl", "items", "propose", "--source", str(mimic), "--vocab", "",
                  "--kind", "inputevents", "-o", str(tmp_path / "x")])  # fmt: skip
    assert r.exit_code == 2 and "source hpc/env.sh" in r.output


def test_specimen_follows_the_lab_category(setup):
    """MIMIC labels chemistry specimens "Blood", but they are serum or plasma; blood gas is
    whole blood."""
    mimic, con = setup
    m = propose_items(con, "labevents", [
        {"itemid": 1, "label": "Creatinine", "specimen": "Blood", "category": "Chemistry",
         "rows": 50},
        {"itemid": 2, "label": "Lactate", "specimen": "Blood", "category": "Blood Gas",
         "rows": 40}], total_rows=90)  # fmt: skip
    assert m.entries[0].concept_id == 9000135  # Creatinine in Serum or Plasma
    assert m.entries[1].concept_id == 9000030  # Lactate in Blood


def test_lab_items_carry_their_most_common_unit(setup):
    mimic, con = setup
    items, _ = item_counts(con, "labevents", _csv(mimic, "hosp", "labevents"),
                           _csv(mimic, "hosp", "d_labitems"), min_rows=1)  # fmt: skip
    units = {it["itemid"]: it["unit"] for it in items}
    assert units[LACTATE] == "mmol/L" and units[CREATININE] == "mg/dL"
    assert units[MADE_UP] is None


def test_unit_rank_prefers_the_matching_loinc_property():
    """A g/dL item is a mass concentration, not [Presence]; a % item is a fraction, not a
    count; an unknown or missing unit ranks nothing."""
    from agentic_tte.etl.item_maps import _unit_rank
    from agentic_tte.omop.vocab import Candidate

    def c(name):
        return Candidate(concept_id=1, concept_name=name, vocabulary_id="LOINC",
                         domain_id="Measurement", standard=True, match="all_words",
                         matched_text=name, score=0.5)  # fmt: skip

    assert _unit_rank(c("Hemoglobin [Mass/volume] in Blood"), "g/dL") == 0
    assert _unit_rank(c("Hemoglobin [Presence] in Blood"), "g/dL") == 1
    assert _unit_rank(c("Neutrophils/100 leukocytes in Blood"), "%") == 0
    assert _unit_rank(c("Neutrophils [#/volume] in Blood"), "%") == 1
    assert _unit_rank(c("Neutrophils [#/volume] in Blood"), "K/uL") == 0
    assert _unit_rank(c("Anything"), "furlongs") == 0
    assert _unit_rank(c("Anything"), None) == 0


def test_lab_queries_match_plurals_and_skip_count():
    from agentic_tte.omop.vocab import Domain, _other_number, query_variants

    assert _other_number("platelet") == "platelets" and _other_number("platelets") == "platelet"
    assert _other_number("ph") == "ph" and _other_number("24") == "24"
    variants = query_variants("Platelet Count", Domain.MEASUREMENT)
    assert "platelet" in [v.lower() for v in variants]
