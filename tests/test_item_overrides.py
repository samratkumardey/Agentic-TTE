"""Reviewed overrides for the curated chartevents and procedureevents maps, on made-up data:
the MIMIC pilot (2026-10-26) found curated targets the 2026 vocabulary no longer has as
standard, among them Invasive Ventilation (225792)."""

import json
from datetime import datetime

import duckdb
from typer.testing import CliRunner

from agentic_tte.cli import app
from agentic_tte.etl.item_maps import ItemEntry, ItemMap
from agentic_tte.etl.mimic_omop import EtlConfig, build
from tests.test_etl_mimic import S1, VOCAB, _write, made_up_item_maps, made_up_mimic

VENT = 225792
PE_COLS = ["subject_id", "hadm_id", "stay_id", "caregiver_id", "starttime", "endtime",
           "storetime", "itemid", "value", "valueuom", "location", "locationcategory",
           "orderid", "linkorderid", "ordercategoryname", "ordercategorydescription",
           "patientweight", "isopenbag", "continueinnextdept", "statusdescription",
           "originalamount", "originalrate"]  # fmt: skip


def _setup(tmp_path):
    """Made-up MIMIC with a ventilation event whose curated target (9000099) is deprecated
    and has no replacement, as Invasive Ventilation's is in the 2026 vocabulary."""
    src = made_up_mimic(tmp_path / "mimic-iv-3.1")
    maps = made_up_item_maps(tmp_path / "concept_map", src)
    with (maps / "procedureevents_to_omop.csv").open("a", encoding="utf-8") as f:
        f.write('mimic:225792,Invasive Ventilation,skos:closeMatch,omop:9000099,Old concept,'
                'HumanCurated,,"orcid:a|orcid:b",\n')
    base = {"subject_id": S1, "hadm_id": 20000001, "stay_id": 30000001}
    _write(src, "icu", "procedureevents", [
        {**base, "starttime": "2150-03-02 08:00:00", "endtime": "2150-03-02 12:00:00",
         "itemid": 225441, "value": 240, "valueuom": "min"},
        {**base, "starttime": "2150-03-01 13:00:00", "endtime": "2150-03-03 09:00:00",
         "itemid": VENT, "value": 2640, "valueuom": "min"}], PE_COLS)
    d_items = [(220045, "Heart Rate"), (220052, "Arterial Blood Pressure mean"),
               (220046, "Made-up alarm"), (220047, "Made-up shock flag"),
               (999999, "Made-up unmapped item"), (225441, "Hemodialysis"),
               (VENT, "Invasive Ventilation")]  # fmt: skip
    _write(src, "icu", "d_items", [{"itemid": i, "label": label} for i, label in d_items],
           ["itemid", "label", "abbreviation", "linksto", "category", "unitname",
            "param_type", "lownormalvalue", "highnormalvalue"])  # fmt: skip
    return src, maps


def _reviewed(kind, decisions, tmp_path):
    entries = [ItemEntry(itemid=i, label=label, rows="<11", concept_id=cid, status=status,
                         reviewer="Dr A" if status != "proposed" else None)
               for i, label, cid, status in decisions]  # fmt: skip
    f = tmp_path / f"{kind}.reviewed.json"
    f.write_text(ItemMap(kind=kind, total_rows=10, listed_rows=10, entries=entries)
                 .model_dump_json(), encoding="utf-8")  # fmt: skip
    return f


def test_reviewed_decisions_override_the_curated_map(tmp_path):
    src, maps = _setup(tmp_path)
    proc = _reviewed("procedureevents", [
        (VENT, "Invasive Ventilation", 9000120, "approved"),  # fixes a rejected curated target
        (225441, "Hemodialysis", None, "rejected"),  # turns a working curated mapping off
    ], tmp_path)
    chart = _reviewed("chartevents", [
        (220046, "Made-up alarm", 9000100, "approved"),  # curated target not in vocabulary
        (220052, "Arterial Blood Pressure mean", None, "unmappable"),  # curated turned off
        (220045, "Heart Rate", 9000080, "proposed"),  # undecided: the curated map stands
    ], tmp_path)
    out = tmp_path / "cdm.duckdb"
    rep = build(EtlConfig(source=src, vocab=VOCAB, out=out, concept_map=maps,
                          item_maps=[proc, chart]))  # fmt: skip
    con = duckdb.connect(str(out), read_only=True)
    procs = con.execute("SELECT procedure_concept_id, procedure_datetime FROM procedure_occurrence"
                        " WHERE procedure_source_value IN ('225441', '225792')").fetchall()
    assert procs == [(9000120, datetime(2150, 3, 1, 13))]  # ventilation; dialysis turned off
    meas = sorted(con.execute("SELECT measurement_source_value, measurement_concept_id "
                              "FROM measurement").fetchall())  # fmt: skip
    assert meas == [("220045", 9000100), ("220046", 9000100)]  # HR curated; MAP turned off
    p = rep.item_maps["procedureevents"]
    assert p["overrides"] == 2 and p["reviewers"] == ["Dr A"]
    assert p["predicates"]["reviewed"] == 1 and p["rejected"] == []  # 225792 now reviewed
    assert p["file"] == "procedureevents_to_omop.csv + procedureevents.reviewed.json"
    assert rep.item_maps["chartevents"]["overrides"] == 2  # undecided items do not count
    assert "items without a curated or reviewed map are not converted" in " ".join(rep.notes)


def test_a_reviewed_map_alone_converts_only_its_approved_items(tmp_path):
    src, _ = _setup(tmp_path)
    proc = _reviewed("procedureevents", [(VENT, "Invasive Ventilation", 9000120, "approved")],
                     tmp_path)  # fmt: skip
    rep = build(EtlConfig(source=src, vocab=VOCAB, out=tmp_path / "c.duckdb", item_maps=[proc]))
    assert (rep.mapping["procedureevents"]["rows"],
            rep.mapping["procedureevents"]["mapped"]) == (2, 1)  # fmt: skip
    assert any("chartevents not converted" in n for n in rep.notes)


def test_propose_lists_rejected_curated_items_and_skips_working_ones(tmp_path):
    src, maps = _setup(tmp_path)
    run = CliRunner().invoke
    for kind, min_rows, expected in (("procedureevents", 1, [VENT]),
                                     ("chartevents", 2, [999999, 220046])):  # fmt: skip
        out = tmp_path / kind
        r = run(app, ["etl", "items", "propose", "--source", str(src), "--vocab", str(VOCAB),
                      "--kind", kind, "-o", str(out), "--min-rows", str(min_rows),
                      "--concept-map", str(maps)])  # fmt: skip
        assert r.exit_code == 0, r.output
        m = json.loads((out / "map.json").read_text(encoding="utf-8"))
        assert [e["itemid"] for e in m["entries"]] == expected
        # A rejected curated item is listed even below --min-rows, with the reason.
        curated = {e["itemid"]: e["curated"] for e in m["entries"]}
        if kind == "procedureevents":
            assert curated[VENT] == "9000099 (not standard and no Maps to)"
            assert "curated map rejected: 9000099" in (out / "review.md").read_text("utf-8")
        else:
            assert curated[220046] == "123 (not in the vocabulary)" and curated[999999] is None


def test_top_keeps_the_most_frequent_items_and_every_rejected_curated_one(tmp_path):
    src, maps = _setup(tmp_path)
    out = tmp_path / "chart"
    r = CliRunner().invoke(app, ["etl", "items", "propose", "--source", str(src), "--vocab",
                                 str(VOCAB), "--kind", "chartevents", "-o", str(out),
                                 "--min-rows", "1", "--top", "0", "--concept-map",
                                 str(maps)])  # fmt: skip
    assert r.exit_code == 0, r.output
    m = json.loads((out / "map.json").read_text(encoding="utf-8"))
    assert [e["itemid"] for e in m["entries"]] == [220046]  # only the rejected curated item
