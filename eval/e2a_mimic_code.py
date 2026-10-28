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


def main() -> None:
    import duckdb

    a = sys.argv[1:]
    db, derived = Path(a[a.index("--db") + 1]), Path(a[a.index("--derived") + 1])
    out = Path(a[a.index("-o") + 1])
    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = compare(omop_flags(con), reference_flags(derived))
    finally:
        con.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2a_mimic_code.md").write_text(to_markdown(rows), encoding="utf-8")
    (out / "e2a_mimic_code.json").write_text(json.dumps(public(rows), indent=2), "utf-8")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
