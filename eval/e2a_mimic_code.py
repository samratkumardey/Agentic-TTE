"""E2 (a): compiled OMOP cohorts against MIT-LCP's validated mimic-code derived concepts, for
the first ICU stay of every MIMIC-IV patient (eval/SCORING.md, Phase 3 changelog item 17).

Each definition is written twice: as a target-trial criterion compiled to SQL on our OMOP CDM,
and as the mimic-code derived table it should reproduce. Both are evaluated per patient and
compared; only counts leave the data side, small cells suppressed.

Time zero is ICU admission for age65, and ICU admission + 24 h ("day 1") for the rest.

| Definition | Our compiled criterion | mimic-code reference |
|---|---|---|
| age65 | age >= 65 years | icustay_detail.admission_age >= 65 |
| creat2 | creatinine > 2.0 mg/dL, 30 h before day 1 | first_day_lab.creatinine_max > 2.0 |
| lact2 | lactate > 2.0 mmol/L, 30 h before day 1 | first_day_bg.lactate_max > 2.0 |
| ivent24 | invasive ventilation, 24 h before day 1 | ventilation: InvasiveVent, first 24 h |
| norepi24 | norepinephrine, 24 h before day 1 | norepinephrine: started, first 24 h |

The windows match mimic-code's first-day windows (from 6 hours before ICU admission for labs).
Expected sources of disagreement, to be adjudicated from the counts: mimic-code's age is
anchor-based at hospital admission, ours at ICU admission from year of birth; our ventilation
comes from procedureevents, mimic-code's from ventilator settings in chartevents; our
norepinephrine includes prescriptions, mimic-code's only infusions; patients who die in the
first 24 hours have no time zero in the 24-hour designs.

On Hellbender, after `hpc/export_mimic_derived.sbatch`:

    uv run python eval/e2a_mimic_code.py --db "$ATTE_OMOP_DIR/<cdm>.duckdb" \\
        --derived "$ATTE_MIMIC_DERIVED" -o runs/e2a
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from agentic_tte.omop.compiler import CompileSpec, EventDef, compile_protocol
from agentic_tte.omop.gate import suppress
from agentic_tte.protocol import (
    ConceptRef,
    Criterion,
    CriterionAtom,
    Duration,
    TargetTrialProtocol,
    TemporalConstraint,
    TimeZero,
    ValueConstraint,
)

# MIMIC-IV OMOP concepts (2026-10-27 CDM): ICU stay, creatinine (OHDSI map of 50912/52546),
# lactate (50813), invasive ventilation (reviewed procedureevents map), norepinephrine.
CONCEPTS = {"icu": 32037, "creatinine": 3016723, "lactate": 3047181,
            "invasive_ventilation": 37158404, "norepinephrine": 1321341}  # fmt: skip
DEFINITIONS = ("age65", "creat2", "lact2", "ivent24", "norepi24")


def _crit(cid, name, domain, *, value=None, hours=None):
    t = TemporalConstraint(relation="x_within_t_before_y",
                           window=Duration(amount=hours, unit="hour")) if hours else None
    return Criterion(id=cid, kind="inclusion", text=cid, atoms=[CriterionAtom(
        concepts=[ConceptRef(name=name, domain=domain)], value=value, temporal=t)])  # fmt: skip


def designs(concepts: dict = CONCEPTS) -> list[tuple[TargetTrialProtocol, CompileSpec]]:
    """Two designs: time zero at ICU admission (age), and 24 hours after it (the rest)."""
    gt2 = ValueConstraint(comparator=">", value=2.0)
    first = TargetTrialProtocol(
        protocol_id="E2A-ICU", time_zero=TimeZero(anchor_event="ICU admission"),
        eligibility=[Criterion(id="age65", kind="inclusion", text="age >= 65", atoms=[
            CriterionAtom(concepts=[ConceptRef(name="age", domain="Demographic")],
                          value=ValueConstraint(comparator=">=", value=65, unit="years"))])])
    day = TargetTrialProtocol(
        protocol_id="E2A-DAY1",
        time_zero=TimeZero(anchor_event="ICU admission", offset=Duration(amount=24, unit="hour")),
        eligibility=[_crit("creat2", "creatinine", "Measurement", value=gt2, hours=30),
                     _crit("lact2", "lactate", "Measurement", value=gt2, hours=30),
                     _crit("ivent24", "invasive ventilation", "Procedure", hours=24),
                     _crit("norepi24", "norepinephrine", "Drug", hours=24)])  # fmt: skip
    sets = {"creat2:creatinine": [concepts["creatinine"]],
            "lact2:lactate": [concepts["lactate"]],
            "ivent24:invasive ventilation": [concepts["invasive_ventilation"]],
            "norepi24:norepinephrine": [concepts["norepinephrine"]]}  # fmt: skip
    spec = CompileSpec(time_zero=EventDef(domain="Visit", concept_ids=[concepts["icu"]]),
                       concept_sets=sets)
    return [(first, spec), (day, spec)]


def omop_flags(con, concepts: dict = CONCEPTS) -> pd.DataFrame:
    """person_id and one boolean per definition, from the compiled criteria; a definition is
    NaN for patients outside its design's index population."""
    tables = {r[0].lower() for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}  # fmt: skip
    out = None
    for p, spec in designs(concepts):
        spec = spec.model_copy(update={"descendants": "concept_ancestor" in tables})
        cc = compile_protocol(p, spec)
        cols = {c.column: c.id for c in cc.criteria if c.column}
        df = con.execute(f"WITH {cc.ctes} SELECT person_id, {', '.join(cols)} FROM cohort").df()
        df = df.rename(columns=cols)
        out = df if out is None else out.merge(df, on="person_id", how="outer")
    return out


def reference_flags(derived: Path) -> pd.DataFrame:
    """person_id (subject_id) and one boolean per definition for each patient's first ICU
    stay, from the exported mimic-code tables."""
    icu = pd.read_csv(derived / "icustay_detail.csv.gz", parse_dates=["icu_intime"])
    icu = icu.sort_values(["subject_id", "icu_intime", "stay_id"]).groupby("subject_id").head(1)
    lab = pd.read_csv(derived / "first_day_lab.csv.gz")
    bg = pd.read_csv(derived / "first_day_bg.csv.gz")
    vent = pd.read_csv(derived / "ventilation.csv.gz", parse_dates=["starttime"])
    nor = pd.read_csv(derived / "norepinephrine.csv.gz", parse_dates=["starttime"])
    r = icu[["subject_id", "stay_id", "icu_intime", "admission_age"]].copy()
    r["age65"] = r["admission_age"] >= 65
    r = r.merge(lab.groupby("stay_id")["creatinine_max"].max().reset_index(), "left", "stay_id")
    r = r.merge(bg.groupby("stay_id")["lactate_max"].max().reset_index(), "left", "stay_id")
    r["creat2"] = r["creatinine_max"] > 2.0
    r["lact2"] = r["lactate_max"] > 2.0

    def started_day1(events: pd.DataFrame) -> set[int]:
        e = events.merge(r[["stay_id", "icu_intime"]], on="stay_id")
        lag = (e["starttime"] - e["icu_intime"]).dt.total_seconds()
        return set(e.loc[(lag >= 0) & (lag <= 86400), "stay_id"])

    r["ivent24"] = r["stay_id"].isin(started_day1(vent[vent["ventilation_status"] ==
                                                       "InvasiveVent"]))  # fmt: skip
    r["norepi24"] = r["stay_id"].isin(started_day1(nor))
    return r.rename(columns={"subject_id": "person_id"})[["person_id", *DEFINITIONS]]


def _kappa(a: int, b: int, c: int, d: int) -> float | None:
    """Cohen's kappa for both / ours only / reference only / neither."""
    n = a + b + c + d
    if not n:
        return None
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n**2
    return None if pe == 1 else (po - pe) / (1 - pe)


def compare(ours: pd.DataFrame, ref: pd.DataFrame) -> list[dict]:
    m = ours.merge(ref, on="person_id", how="outer", suffixes=("_omop", "_ref"))
    rows = []
    for k in DEFINITIONS:
        both_known = m[f"{k}_omop"].notna() & m[f"{k}_ref"].notna()
        x = m.loc[both_known, f"{k}_omop"].astype(bool)
        y = m.loc[both_known, f"{k}_ref"].astype(bool)
        a, b, c = int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum())
        d = int((~x & ~y).sum())
        rows.append({"definition": k, "compared": a + b + c + d, "both": a, "omop_only": b,
                     "reference_only": c, "neither": d, "kappa": _kappa(a, b, c, d),
                     "omop_without_reference": int((m[f"{k}_omop"].notna()
                                                    & m[f"{k}_ref"].isna()).sum()),
                     "reference_without_omop": int((m[f"{k}_ref"].notna()
                                                    & m[f"{k}_omop"].isna()).sum())})  # fmt: skip
    return rows


def public(rows: list[dict]) -> list[dict]:
    keys = ("compared", "both", "omop_only", "reference_only", "neither",
            "omop_without_reference", "reference_without_omop")  # fmt: skip
    return [{**r, **{k: suppress(r[k]) if r[k] else 0 for k in keys},
             "kappa": None if r["kappa"] is None else round(r["kappa"], 3)} for r in rows]


def to_markdown(rows: list[dict]) -> str:
    lines = ["## E2 (a): compiled OMOP criteria vs mimic-code derived concepts", "",
             "First ICU stay per patient. Counts below 11 suppressed.", "",
             "| Definition | Compared | Both | Ours only | mimic-code only | Neither | Kappa | "
             "Only in ours | Only in mimic-code |", "|---|---|---|---|---|---|---|---|---|"]
    for r in public(rows):
        lines.append(f"| {r['definition']} | {r['compared']} | {r['both']} | {r['omop_only']} | "
                     f"{r['reference_only']} | {r['neither']} | {r['kappa']} | "
                     f"{r['omop_without_reference']} | {r['reference_without_omop']} |")
    lines += ["", "*Only in ours / only in mimic-code*: patients with a time zero on one side "
              "only (e.g. death in the first 24 hours, or no first-day table)."]
    return "\n".join(lines) + "\n"

# --- by-source breakdown of the disagreements (changelog item 19) -----------------------------

CHART_ITEMS = 220000  # MIMIC-IV item IDs from 220000 are chartevents/icu items; 5xxxx are labs


def _source(itemid: str | None) -> str:
    """Where an OMOP row came from, by its source value (the MIMIC item ID, or a drug name)."""
    if itemid is None or not str(itemid).isdigit():
        return "prescriptions"
    i = int(itemid)
    return f"chartevents item {i}" if i >= CHART_ITEMS else f"labevents item {i}"


def breakdown(con, derived: Path, concepts: dict = CONCEPTS) -> list[dict]:
    """For each disagreeing group, how many patients fall in each source category. Patients are
    linked on the data side; only counts come out."""
    ours, ref = omop_flags(con, concepts), reference_flags(derived)
    m = ours.merge(ref, on="person_id", suffixes=("_omop", "_ref"))
    icu = pd.read_csv(derived / "icustay_detail.csv.gz", parse_dates=["icu_intime"])
    first = icu.sort_values(["subject_id", "icu_intime", "stay_id"]).groupby("subject_id").head(1)
    first = first.rename(columns={"subject_id": "person_id"})[["person_id", "stay_id",
                                                                "icu_intime"]]  # fmt: skip
    rows: list[dict] = []

    def group(k, ours_value, ref_value):
        sel = m[(m[f"{k}_omop"] == ours_value) & (m[f"{k}_ref"] == ref_value)]
        return first[first.person_id.isin(set(sel.person_id))]

    def records(table, cid_col, time_col, src_col, cid, people, extra=""):
        con.register("_p", people[["person_id", "icu_intime"]])
        try:
            return con.execute(f"""SELECT r.person_id, r.{time_col} AS t, r.{src_col} AS src,
                p.icu_intime FROM {table} r JOIN _p p USING (person_id)
                WHERE r.{cid_col} = {int(cid)} {extra}""").df()  # fmt: skip
        finally:
            con.unregister("_p")

    def add(definition, grp, category, n):
        rows.append({"definition": definition, "group": grp, "category": category,
                     "patients": int(n)})  # fmt: skip

    # lactate > 2, ours only: which sources gave our qualifying values?
    g = group("lact2", True, False)
    r = records("measurement", "measurement_concept_id", "measurement_datetime",
                "measurement_source_value", concepts["lactate"], g, "AND r.value_as_number > 2")
    lag = (r["t"] - r["icu_intime"]).dt.total_seconds() / 3600
    r = r[(lag >= -6) & (lag <= 24)].assign(source=lambda x: x["src"].map(_source))
    per = r.groupby("person_id")["source"].agg(lambda s: " + ".join(sorted(set(s))))
    for cat, n in per.value_counts().items():
        add("lact2", "ours only", f"qualifying values from: {cat}", n)
    add("lact2", "ours only", "patients in group", len(g))

    # invasive ventilation, ours only: what mimic-code's ventilation table says for that stay
    vent = pd.read_csv(derived / "ventilation.csv.gz", parse_dates=["starttime", "endtime"])
    g = group("ivent24", True, False)
    v = vent.merge(g[["stay_id", "icu_intime"]], on="stay_id")
    h0 = (v["starttime"] - v["icu_intime"]).dt.total_seconds() / 3600
    h1 = (v["endtime"] - v["icu_intime"]).dt.total_seconds() / 3600
    inv = v["ventilation_status"] == "InvasiveVent"
    day1 = (h0 >= 0) & (h0 <= 24)
    cats = {
        "mimic-code invasive episode already running at ICU admission": set(
            v.loc[inv & (h0 < 0) & (h1 > 0), "stay_id"]),
        "mimic-code invasive episode starting after 24 h": set(v.loc[inv & (h0 > 24), "stay_id"]),
        "mimic-code tracheostomy on day 1": set(
            v.loc[(v["ventilation_status"] == "Tracheostomy") & day1, "stay_id"]),
        "mimic-code other support only on day 1 (NIV, HFNC, oxygen)": set(
            v.loc[~inv & (v["ventilation_status"] != "Tracheostomy") & day1, "stay_id"]),
    }  # fmt: skip
    seen: set = set()
    for cat, stays in cats.items():
        add("ivent24", "ours only", cat, len(stays - seen))
        seen |= stays
    add("ivent24", "ours only", "no mimic-code ventilation row for the stay",
        len(set(g.stay_id) - set(v.stay_id)))
    add("ivent24", "ours only", "other", len(set(g.stay_id) - seen - (set(g.stay_id)
                                                                      - set(v.stay_id))))
    add("ivent24", "ours only", "patients in group", len(g))

    # invasive ventilation, mimic-code only: what our procedure records say
    g = group("ivent24", False, True)
    r = records("procedure_occurrence", "procedure_concept_id", "procedure_datetime",
                "procedure_source_value", concepts["invasive_ventilation"], g)
    hrs = (r["t"] - r["icu_intime"]).dt.total_seconds() / 3600
    before = set(r.loc[hrs < 0, "person_id"])
    after = set(r.loc[hrs > 24, "person_id"]) - before
    add("ivent24", "mimic-code only", "our invasive ventilation starts before ICU admission",
        len(before))
    add("ivent24", "mimic-code only", "our invasive ventilation starts after 24 h", len(after))
    add("ivent24", "mimic-code only", "no invasive ventilation record of ours",
        len(set(g.person_id) - set(r.person_id)))
    add("ivent24", "mimic-code only", "patients in group", len(g))

    # norepinephrine, ours only: which sources
    g = group("norepi24", True, False)
    r = records("drug_exposure", "drug_concept_id", "drug_exposure_start_datetime",
                "drug_source_value", concepts["norepinephrine"], g)
    hrs = (r["t"] - r["icu_intime"]).dt.total_seconds() / 3600
    r = r[(hrs >= 0) & (hrs <= 24)].assign(source=lambda x: x["src"].map(_source))
    per = r.groupby("person_id")["source"].agg(lambda s: " + ".join(sorted(set(s))))
    for cat, n in per.value_counts().items():
        add("norepi24", "ours only", f"day-1 records from: {cat}", n)
    add("norepi24", "ours only", "patients in group", len(g))
    return rows


def breakdown_markdown(rows: list[dict]) -> str:
    lines = ["## E2 (a): where the disagreements come from", "",
             "Patients per source category within each disagreeing group; counts below 11 "
             "suppressed.", "", "| Definition | Group | Category | Patients |",
             "|---|---|---|---|"]  # fmt: skip
    lines += [f"| {r['definition']} | {r['group']} | {r['category']} | "
              f"{suppress(r['patients']) if r['patients'] else 0} |" for r in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    import duckdb

    a = sys.argv[1:]
    db, derived = Path(a[a.index("--db") + 1]), Path(a[a.index("--derived") + 1])
    out = Path(a[a.index("-o") + 1])
    con = duckdb.connect(str(db), read_only=True)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if "--breakdown" in a:  # changelog item 19
            rows = breakdown(con, derived)
            md = breakdown_markdown(rows)
            (out / "e2a_breakdown.md").write_text(md, encoding="utf-8")
            pub = [{**r, "patients": suppress(r["patients"]) if r["patients"] else 0}
                   for r in rows]
            (out / "e2a_breakdown.json").write_text(json.dumps(pub, indent=2), "utf-8")
            print(md)
            return
        rows = compare(omop_flags(con), reference_flags(derived))
    finally:
        con.close()
    (out / "e2a_mimic_code.md").write_text(to_markdown(rows), encoding="utf-8")
    (out / "e2a_mimic_code.json").write_text(json.dumps(public(rows), indent=2), "utf-8")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
