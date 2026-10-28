"""E2: checks that a compiled cohort is the cohort the design defines, without hand-built
reference cohorts (eval/SCORING.md, Phase 3 changelog items 11b and 16).

Two kinds, both run where the data are; only counts come out:

**Metamorphic relations.** A change to the design with a known effect on the eligible set:
- ``drop``: removing an inclusion or an exclusion criterion never removes anyone;
- ``order``: reversing the criteria leaves the cohort unchanged;
- ``duplicate``: repeating a criterion leaves it unchanged;
- ``window``: doubling a look-back window ("within 7 days before") never shrinks the set of
  patients meeting the criterion, so it never shrinks an inclusion's cohort or grows an
  exclusion's;
- ``threshold``: making a measurement threshold stricter (10% further) never grows an
  inclusion's cohort or shrinks an exclusion's;
- ``time_shift``: moving every date of the database by 400 years (a whole Gregorian cycle, so
  calendars and leap days repeat) leaves the cohort and the frame unchanged.
A violation is a patient who moves the wrong way.

**Differential testing.** ``reference_flags`` re-derives time zero and every compiled
criterion in Python (pandas) from the raw OMOP tables, written apart from the SQL compiler and
following its documented semantics; each criterion's flag is compared patient by patient. Atoms
it does not cover (anchors other than time zero, cumulative durations) are reported as not
compared. The pre-registered independent implementation also includes LLM-written SQL, as in
EmulatRx; that needs the GPU and is added on Hellbender.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from agentic_tte.omop.compiler import (
    _SECONDS,
    EVENT_TABLE,
    CompiledCohort,
    CompileSpec,
    compile_protocol,
)
from agentic_tte.omop.gate import concept_key, suppress
from agentic_tte.protocol.models import (
    Comparator,
    Criterion,
    Domain,
    TargetTrialProtocol,
    TemporalRelation,
)

SHIFT_YEARS = 400
STRICTER = 0.10  # threshold relation: 10% further


class Relation(BaseModel):
    relation: str
    detail: str
    expect: str  # superset, subset, equal
    n_before: int
    n_after: int
    violations: int  # patients who moved the wrong way

    @property
    def held(self) -> bool:
        return self.violations == 0


class CriterionDiff(BaseModel):
    id: str
    compared: int
    disagree: int
    note: str | None = None


class CohortCheckReport(BaseModel):
    protocol_id: str
    relations: list[Relation] = Field(default_factory=list)
    differential: list[CriterionDiff] = Field(default_factory=list)
    index_disagree: int | None = None  # time zero / base population vs the reference
    notes: list[str] = Field(default_factory=list)

    def public(self) -> dict:
        """Counts below the small-cell threshold suppressed, for anything leaving the data."""
        d = self.model_dump(mode="json")
        for r in d["relations"]:
            for k in ("n_before", "n_after", "violations"):
                r[k] = suppress(r[k]) if r[k] else 0
        for c in d["differential"]:
            c["compared"] = suppress(c["compared"])
            c["disagree"] = suppress(c["disagree"]) if c["disagree"] else 0
        return d

    def to_markdown(self) -> str:
        pub = self.public()
        ok = sum(r.held for r in self.relations)
        lines = [f"## Cohort checks (E2): {self.protocol_id}", "",
                 f"Metamorphic relations: {ok} of {len(self.relations)} held.", "",
                 "| Relation | Change | Expected | Eligible before | after | Violations |",
                 "|---|---|---|---|---|---|"]  # fmt: skip
        for r in pub["relations"]:
            lines.append(f"| {r['relation']} | {r['detail']} | {r['expect']} | {r['n_before']} | "
                         f"{r['n_after']} | {r['violations']} |")
        if self.differential:
            n = self.index_disagree
            idx = "" if n is None else f" (time zero: {suppress(n) if n else 0} patients differ)"
            lines += ["", f"Differential test against the Python reference evaluator{idx}:", "",
                      "| Criterion | Patients compared | Disagree | Note |", "|---|---|---|---|"]
            lines += [f"| {c['id']} | {c['compared']} | {c['disagree']} | {c['note'] or '–'} |"
                      for c in pub["differential"]]
        lines += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


# --- metamorphic relations ---------------------------------------------------------------------


def _eligible(con, cc: CompiledCohort) -> set[int]:
    return {r[0] for r in con.execute(f"WITH {cc.ctes} SELECT person_id FROM cohort "
                                      "WHERE eligible").fetchall()}  # fmt: skip


def _compare(rel, detail, expect, before: set[int], after: set[int]) -> Relation:
    wrong = {"superset": before - after, "subset": after - before,
             "equal": before ^ after}[expect]  # fmt: skip
    return Relation(relation=rel, detail=detail, expect=expect, n_before=len(before),
                    n_after=len(after), violations=len(wrong))  # fmt: skip


def _variant(p: TargetTrialProtocol, eligibility: list[Criterion]) -> TargetTrialProtocol:
    return p.model_copy(update={"eligibility": eligibility})


def _respec(spec: CompileSpec, old: Criterion, new: Criterion) -> CompileSpec:
    """Concept sets are keyed by criterion id: copy them to a renamed criterion."""
    sets = dict(spec.concept_sets)
    for a in old.atoms:
        for c in a.concepts:
            k = concept_key(old, c.name)
            if k in sets:
                sets[concept_key(new, c.name)] = sets[k]
    return spec.model_copy(update={"concept_sets": sets})


def metamorphic(con, protocol: TargetTrialProtocol, spec: CompileSpec, *,
                db_path: str | Path | None = None) -> list[Relation]:  # fmt: skip
    """Every applicable relation for this design. ``db_path`` (a DuckDB file) enables the
    time-shift relation, which reads the file through shifted views."""
    p = protocol
    cc = compile_protocol(p, spec)
    base = _eligible(con, cc)
    compiled = [c for c in cc.criteria if c.status == "compiled"]
    by_id = {c.id: c for c in p.eligibility}
    out: list[Relation] = []

    for c in compiled:
        rest = [x for x in p.eligibility if x.id != c.id]
        after = _eligible(con, compile_protocol(_variant(p, rest), spec))
        out.append(_compare("drop", f"without {c.id} ({c.kind})", "superset", base, after))

    after = _eligible(con, compile_protocol(_variant(p, list(reversed(p.eligibility))), spec))
    out.append(_compare("order", "criteria in reverse order", "equal", base, after))

    if compiled:
        orig = by_id[compiled[0].id]
        dup = orig.model_copy(update={"id": f"{orig.id}dup"}, deep=True)
        after = _eligible(con, compile_protocol(_variant(p, [*p.eligibility, dup]),
                                                _respec(spec, orig, dup)))  # fmt: skip
        out.append(_compare("duplicate", f"{orig.id} repeated", "equal", base, after))

    for c in compiled:
        crit = by_id[c.id]
        if len(crit.atoms) != 1 or crit.atoms[0].negated:
            continue
        a = crit.atoms[0]
        t, v = a.temporal, a.value
        if t is not None and t.window is not None and TemporalRelation(t.relation) == \
                TemporalRelation.X_WITHIN_T_BEFORE_Y:  # fmt: skip
            new = crit.model_copy(deep=True)
            new.atoms[0].temporal.window.amount *= 2
            after = _eligible(con, compile_protocol(
                _variant(p, [new if x.id == c.id else x for x in p.eligibility]), spec))
            out.append(_compare("window", f"{c.id} window doubled",
                                "superset" if crit.kind == "inclusion" else "subset", base, after))
        if v is not None and v.value is not None and Comparator(v.comparator) in (
                Comparator.GT, Comparator.GE, Comparator.LT, Comparator.LE):  # fmt: skip
            new = crit.model_copy(deep=True)
            up = Comparator(v.comparator) in (Comparator.GT, Comparator.GE)
            step = abs(v.value) * STRICTER or STRICTER
            new.atoms[0].value.value = v.value + step if up else v.value - step
            after = _eligible(con, compile_protocol(
                _variant(p, [new if x.id == c.id else x for x in p.eligibility]), spec))
            out.append(_compare("threshold", f"{c.id} threshold {100 * STRICTER:g}% stricter",
                                "subset" if crit.kind == "inclusion" else "superset", base, after))

    if db_path is not None:
        shifted = shifted_connection(db_path)
        try:
            after = _eligible(shifted, cc)
            out.append(_compare("time_shift", f"all dates +{SHIFT_YEARS} years", "equal",
                                base, after))  # fmt: skip
            if cc.frame_sql:
                q = ("SELECT count(*), sum(treated), sum(event), round(sum(time), 3) "
                     f"FROM ({cc.frame_sql})")  # fmt: skip
                a, b = con.execute(q).fetchone(), shifted.execute(q).fetchone()
                out.append(Relation(relation="time_shift", detail="analysis frame totals "
                                    "(patients, treated, events, follow-up)", expect="equal",
                                    n_before=int(a[0]), n_after=int(b[0]),
                                    violations=int(a != b)))  # fmt: skip
        finally:
            shifted.close()
    return out


def shifted_connection(db_path: str | Path, years: int = SHIFT_YEARS):
    """A new connection whose tables are views over ``db_path`` with every date and timestamp
    moved by ``years`` (and ``year_of_birth`` with them). The file is opened read-only."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"ATTACH '{Path(db_path).as_posix()}' AS src (READ_ONLY)")
    cols = con.execute("SELECT table_name, column_name, data_type FROM information_schema.columns "
                       "WHERE table_catalog = 'src' ORDER BY table_name, ordinal_position"
                       ).fetchall()  # fmt: skip
    tables: dict[str, list[tuple[str, str]]] = {}
    for t, c, typ in cols:
        tables.setdefault(t, []).append((c, typ.upper()))
    for t, cs in tables.items():
        rep = []
        for c, typ in cs:
            if typ.startswith("TIMESTAMP"):
                rep.append(f"{c} + INTERVAL {years} YEAR AS {c}")
            elif typ == "DATE":
                rep.append(f"CAST({c} + INTERVAL {years} YEAR AS DATE) AS {c}")
            elif t == "person" and c == "year_of_birth":
                rep.append(f"{c} + {years} AS {c}")
        sel = f"* REPLACE ({', '.join(rep)})" if rep else "*"
        con.execute(f'CREATE VIEW "{t}" AS SELECT {sel} FROM src."{t}"')
    return con


# --- differential testing: a Python reference evaluator -----------------------------------------


def _records(con, dom: Domain, ids: list[int], persons: list[int],
             descendants: bool) -> pd.DataFrame:  # fmt: skip
    table, col, dt, d = EVENT_TABLE[dom]
    ids_sql = ", ".join(str(int(i)) for i in ids)
    if descendants:
        try:
            more = con.execute(f"SELECT descendant_concept_id FROM concept_ancestor WHERE "
                               f"ancestor_concept_id IN ({ids_sql})").fetchall()  # fmt: skip
            ids = sorted(set(ids) | {r[0] for r in more})
            ids_sql = ", ".join(str(int(i)) for i in ids)
        except Exception:  # noqa: BLE001 - no concept_ancestor table
            pass
    val = "value_as_number" if dom == Domain.MEASUREMENT else "NULL"
    con.register("_persons", pd.DataFrame({"person_id": persons}))
    try:
        df = con.execute(f"SELECT r.person_id, r.{dt} AS dt, r.{d} AS d, {val} AS value "
                         f"FROM {table} r JOIN _persons USING (person_id) "
                         f"WHERE r.{col} IN ({ids_sql})").df()  # fmt: skip
    finally:
        con.unregister("_persons")
    df["time"] = pd.to_datetime(df["dt"]).fillna(pd.to_datetime(df["d"]))
    return df[["person_id", "time", "value"]]


def _seconds(d) -> float:
    return d.amount * _SECONDS[d.unit]


def reference_index(con, spec: CompileSpec, protocol: TargetTrialProtocol) -> pd.DataFrame:
    """person_id, t0, age, female: time zero and the index population, re-derived."""
    tz = spec.time_zero
    persons = [r[0] for r in con.execute("SELECT person_id FROM person").fetchall()]
    ev = _records(con, Domain(tz.domain), tz.concept_ids, persons, spec.descendants)
    g = ev.groupby("person_id")["time"]
    t0 = (g.min() if tz.occurrence == "first" else g.max()).rename("t0").reset_index()
    off = protocol.time_zero.offset if protocol.time_zero else None
    if off is not None and off.amount:
        t0["t0"] += pd.to_timedelta(_seconds(off), unit="s")
    person = con.execute("SELECT person_id, gender_concept_id, year_of_birth, month_of_birth, "
                         "day_of_birth, birth_datetime FROM person").df()  # fmt: skip
    op = con.execute("SELECT person_id, observation_period_start_date AS s, "
                     "observation_period_end_date AS e FROM observation_period").df()
    death = con.execute("SELECT person_id, death_date FROM death").df()
    b = t0.merge(person, on="person_id")
    birth = pd.to_datetime(b["birth_datetime"])
    fallback = pd.to_datetime(dict(year=b["year_of_birth"], month=b["month_of_birth"].fillna(7),
                                   day=b["day_of_birth"].fillna(1)), errors="coerce")  # fmt: skip
    b["age"] = (b["t0"] - birth.fillna(fallback)).dt.total_seconds() / 86400 / 365.25
    b["female"] = b["gender_concept_id"] == 8532
    day = b["t0"].dt.normalize()
    inside = b[["person_id"]].assign(day=day).merge(op, on="person_id")
    inside = inside[(inside["day"] >= pd.to_datetime(inside["s"]))
                    & (inside["day"] <= pd.to_datetime(inside["e"]))]["person_id"]  # fmt: skip
    dead = b[["person_id", "t0"]].merge(death, on="person_id")
    dead = dead[pd.to_datetime(dead["death_date"]) < dead["t0"].dt.normalize()]["person_id"]
    b = b[b.person_id.isin(set(inside)) & ~b.person_id.isin(set(dead))]
    return b[["person_id", "t0", "age", "female"]].reset_index(drop=True)


def _compare_value(x: pd.Series, v) -> pd.Series:
    c = Comparator(v.comparator)
    if c == Comparator.BETWEEN:
        return (x >= v.value) & (x <= v.upper)
    ref = v.value if v.value is not None else v.upper
    return {Comparator.LT: x < ref, Comparator.LE: x <= ref, Comparator.EQ: x == ref,
            Comparator.GE: x >= ref, Comparator.GT: x > ref}[c]  # fmt: skip


def _atom_flag(con, idx: pd.DataFrame, crit: Criterion, atom, spec: CompileSpec) -> pd.Series:
    t, v = atom.temporal, atom.value
    rel = TemporalRelation(t.relation) if t else None
    if t is not None and t.anchor.replace(" ", "_").lower() not in ("time_zero", "baseline",
                                                                     "randomization",
                                                                     "enrollment"):  # fmt: skip
        raise NotImplementedError(f"anchor '{t.anchor}'")
    flags = []
    for ref in atom.concepts:
        if ref.domain == Domain.DEMOGRAPHIC:
            n = ref.name.lower()
            if n == "age":
                flags.append(_compare_value(idx["age"], v))
            elif n in ("female", "women", "woman"):
                flags.append(idx["female"])
            elif n in ("male", "men", "man"):
                flags.append(~idx["female"])
            else:
                raise NotImplementedError(f"demographic '{ref.name}'")
            continue
        ids = spec.concept_sets.get(concept_key(crit, ref.name)) or (
            [ref.omop_concept_id] if ref.omop_concept_id else None)
        if not ids:
            continue  # unmappable in an "any" list: the compiler leaves it out too
        dom = Domain(ref.domain)
        r = _records(con, dom, ids, idx.person_id.tolist(), spec.descendants).merge(
            idx[["person_id", "t0"]], on="person_id")
        lag = (r["t0"] - r["time"]).dt.total_seconds()
        w = _seconds(t.window) if t and t.window else None
        if rel == TemporalRelation.X_BEFORE_Y:
            keep = lag > 0
        elif rel == TemporalRelation.X_WITHIN_T_BEFORE_Y and w is not None:
            keep = (lag >= 0) & (lag <= w)
        elif dom == Domain.MEASUREMENT and rel in (None, TemporalRelation.X_AT_Y) \
                or rel == TemporalRelation.X_WITHIN_T_BEFORE_Y:  # fmt: skip
            keep = (lag >= 0) & (lag <= (_seconds(spec.measurement_lookback)
                                         if dom == Domain.MEASUREMENT else np.inf))
        elif rel in (None, TemporalRelation.X_AT_Y):
            keep = lag >= 0
        else:
            raise NotImplementedError(f"relation {rel}")
        r = r[keep]
        latest = dom == Domain.MEASUREMENT and rel in (None, TemporalRelation.X_AT_Y) \
            and v is not None  # fmt: skip
        if latest:
            r = r[r["value"].notna()].sort_values("time").groupby("person_id").tail(1)
            r = r[_compare_value(r["value"], v)]
        elif v is not None:
            r = r[_compare_value(r["value"], v)]
        flags.append(idx.person_id.isin(set(r.person_id)))
    if not flags:
        raise NotImplementedError("no concept to evaluate")
    out = flags[0]
    for f in flags[1:]:
        out = (out & f) if atom.concept_logic == "all" else (out | f)
    return ~out if atom.negated else out


def reference_flags(con, protocol: TargetTrialProtocol, spec: CompileSpec,
                    compiled: CompiledCohort) -> tuple[pd.DataFrame, dict[str, str]]:  # fmt: skip
    """Per-patient flags for every compiled criterion, re-derived in Python; plus the reasons a
    criterion could not be evaluated."""
    idx = reference_index(con, spec, protocol)
    out = idx[["person_id"]].copy()
    skipped: dict[str, str] = {}
    by_id = {c.id: c for c in protocol.eligibility}
    for cc in compiled.criteria:
        if cc.status != "compiled":
            continue
        crit = by_id[cc.id]
        try:
            fs = [_atom_flag(con, idx, crit, a, spec) for a in crit.atoms]
        except NotImplementedError as e:
            skipped[cc.id] = f"not compared: {e}"
            continue
        f = fs[0]
        for x in fs[1:]:
            f = (f & x) if crit.atom_logic == "all" else (f | x)
        out[cc.column] = f.to_numpy()
    return out, skipped


def differential(con, protocol: TargetTrialProtocol,
                 spec: CompileSpec) -> tuple[list[CriterionDiff], int]:  # fmt: skip
    """Each compiled criterion's flag, SQL vs the reference evaluator, patient by patient; and
    the number of patients whose index (time zero, population) differs."""
    cc = compile_protocol(protocol, spec)
    sql = con.execute(f"WITH {cc.ctes} SELECT * FROM cohort").df()
    ref, skipped = reference_flags(con, protocol, spec, cc)
    index_disagree = len(set(sql.person_id) ^ set(ref.person_id))
    m = sql.merge(ref, on="person_id", suffixes=("", "_ref"))
    out = []
    for c in cc.criteria:
        if c.status != "compiled":
            continue
        if c.id in skipped:
            out.append(CriterionDiff(id=c.id, compared=0, disagree=0, note=skipped[c.id]))
            continue
        a = m[c.column].fillna(False).astype(bool).to_numpy()
        b = m[f"{c.column}_ref"].astype(bool).to_numpy()
        out.append(CriterionDiff(id=c.id, compared=len(m), disagree=int((a != b).sum())))
    return out, index_disagree


def check_cohort(con, protocol: TargetTrialProtocol, spec: CompileSpec, *,
                 db_path: str | Path | None = None) -> CohortCheckReport:  # fmt: skip
    rep = CohortCheckReport(protocol_id=protocol.protocol_id)
    rep.relations = metamorphic(con, protocol, spec, db_path=db_path)
    rep.differential, rep.index_disagree = differential(con, protocol, spec)
    if db_path is None:
        rep.notes.append("time-shift relation not run: needs the database file")
    return rep
