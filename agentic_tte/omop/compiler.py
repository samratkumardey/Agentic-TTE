"""Criteria-to-SQL compiler: a reviewed protocol -> OMOP CDM v5.4 SQL (PLAN.md §2, §15).

Deterministic: no LLM writes SQL. The inputs are the protocol (criteria atoms), the reviewed
concept sets (``ConceptSetBook.gate_sets()``) and a ``CompileSpec`` that a reviewer fills in
for what the protocol names but cannot itself ground: the time-zero event and other named
anchor events ("hospital admission"), each as a concept set.

Two queries come out:
- ``cohort_sql``: one row per person with an index event: ``t0``, one boolean column per
  compiled criterion (``c_I1``, ``c_E2``, ...) and ``eligible``. The attrition table is built
  from these columns.
- ``frame_sql``: the analysis frame for eligible persons, with the columns the statistics take:
  confounders, ``treated``, ``treat_start`` (days), ``time`` (days), ``event`` and
  ``end_reason``. On the synthetic cohort it reproduces ``synth.omop.ANALYSIS_SQL``.

Semantics (every one is listed in the compile report):
- **No look-ahead.** Eligibility reads only records at or before time zero. A criterion that
  needs records after time zero is not compiled: time zero is then wrong for that criterion.
- **Time zero / randomization / enrollment / baseline** all mean the index time ``t0``.
- A record "at the time of" an anchor means recorded at or before it. Without a temporal
  constraint, any record up to time zero counts (history).
- Measurements: "at the time of" and unconstrained use the latest value within
  ``measurement_lookback`` before the anchor; windowed relations accept any value in the window.
  A missing value means the criterion is not met. Units are not converted.
- Records with a date but no datetime are placed at midnight.
- A criterion compiles fully or not at all: if any atom cannot be compiled, the criterion is
  reported as not compiled with its reasons and left out of ``eligible`` (or raises in strict
  mode). Nothing is dropped silently.
- Observation periods are dates: follow-up ends at the end of the last observed day, so
  censoring is resolved to one day.

The SQL is DuckDB's dialect; other engines go through SQLGlot later (PLAN.md §5).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from agentic_tte.omop.gate import concept_key
from agentic_tte.protocol.models import (
    Comparator,
    ConceptRef,
    Criterion,
    CriterionAtom,
    Domain,
    Duration,
    MappingStatus,
    TargetTrialProtocol,
    TemporalRelation,
)

_SECONDS = {"minute": 60, "hour": 3600, "day": 86400, "week": 7 * 86400,
            "month": 30.44 * 86400, "year": 365.25 * 86400}  # fmt: skip

# table, concept column, datetime column, date column
EVENT_TABLE = {
    Domain.CONDITION: ("condition_occurrence", "condition_concept_id",
                       "condition_start_datetime", "condition_start_date"),
    Domain.DRUG: ("drug_exposure", "drug_concept_id",
                  "drug_exposure_start_datetime", "drug_exposure_start_date"),
    Domain.MEASUREMENT: ("measurement", "measurement_concept_id",
                         "measurement_datetime", "measurement_date"),
    Domain.PROCEDURE: ("procedure_occurrence", "procedure_concept_id",
                       "procedure_datetime", "procedure_date"),
    Domain.OBSERVATION: ("observation", "observation_concept_id",
                         "observation_datetime", "observation_date"),
    Domain.DEVICE: ("device_exposure", "device_concept_id",
                    "device_exposure_start_datetime", "device_exposure_start_date"),
    Domain.VISIT: ("visit_occurrence", "visit_concept_id",
                   "visit_start_datetime", "visit_start_date"),
}  # fmt: skip

T0_ALIASES = {"time zero", "time 0", "randomization", "randomisation", "enrollment",
              "enrolment", "baseline", "study entry", "inclusion", "index"}  # fmt: skip
FEMALE, MALE = 8532, 8507
_DEATH_WORDS = ("mortality", "death", "survival", "died")


class CompileError(ValueError):
    pass


class EventDef(BaseModel):
    """A named event as a concept set: time zero, or an anchor such as "hospital admission"."""

    domain: Domain
    concept_ids: list[int]
    occurrence: Literal["first", "last"] = "first"  # anchors: last occurrence up to t0
    note: str | None = None


class CompileSpec(BaseModel):
    """What a reviewer supplies besides the protocol and the concept sets."""

    time_zero: EventDef
    anchors: dict[str, EventDef] = Field(default_factory=dict)
    concept_sets: dict[str, list[int]] = Field(default_factory=dict)
    descendants: bool = True  # expand concept sets through concept_ancestor
    measurement_lookback: Duration = Field(
        default_factory=lambda: Duration(amount=24, unit="hour"))
    strict: bool = False  # raise if a clinical criterion cannot be compiled


class CompiledCriterion(BaseModel):
    id: str
    kind: Literal["inclusion", "exclusion"]
    text: str
    status: Literal["compiled", "not_compiled", "not_emulable"]
    column: str | None = None
    predicate: str | None = None
    description: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)  # why it was not compiled
    warnings: list[str] = Field(default_factory=list)


class CompiledCohort(BaseModel):
    protocol_id: str
    protocol_version: int
    protocol_hash: str
    dialect: str = "duckdb"
    spec: CompileSpec
    criteria: list[CompiledCriterion]
    frame_columns: list[str] = Field(default_factory=list)
    frame_notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    ctes: str  # idx, base, flags, cohort: shared by both queries and the attrition counts
    cohort_sql: str
    frame_sql: str | None = None
    horizon_days: float | None = None

    @property
    def not_compiled(self) -> list[CompiledCriterion]:
        return [c for c in self.criteria if c.status == "not_compiled"]

    def to_markdown(self) -> str:
        n = {s: sum(c.status == s for c in self.criteria)
             for s in ("compiled", "not_compiled", "not_emulable")}  # fmt: skip
        lines = [f"# Compiled cohort: {self.protocol_id} (protocol v{self.protocol_version})", "",
                 f"Protocol hash `{self.protocol_hash[:12]}`; dialect {self.dialect}. "
                 f"{n['compiled']} criteria compiled, {n['not_compiled']} not compiled, "
                 f"{n['not_emulable']} not emulable.", "",
                 "| Criterion | Status | Compiled as | Reasons and warnings |",
                 "|---|---|---|---|"]  # fmt: skip
        for c in self.criteria:
            how = "<br>".join(c.description) or "–"
            notes = "<br>".join([f"**{r}**" for r in c.reasons] + c.warnings) or "–"
            lines.append(f"| {c.id} ({c.kind}): {c.text[:70]} | {c.status} | {how} | {notes} |")
        if self.frame_notes:
            lines += ["", "## Analysis frame", ""] + [f"- {x}" for x in self.frame_notes]
        lines += ["", "## Assumptions", ""] + [f"- {a}" for a in self.assumptions]
        return "\n".join(lines) + "\n"


# --- SQL building blocks -----------------------------------------------------------------------


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower().replace("_", " ")).split())


def _interval(d: Duration | float) -> str:
    seconds = d if isinstance(d, (int, float)) else d.amount * _SECONDS[d.unit]
    return f"INTERVAL '{round(seconds)} seconds'"


def _ids(ids: list[int]) -> str:
    return ", ".join(str(int(i)) for i in sorted(set(ids)))


def _in_set(col: str, ids: list[int], descendants: bool) -> str:
    if not descendants:
        return f"{col} IN ({_ids(ids)})"
    return (f"({col} IN ({_ids(ids)}) OR {col} IN (SELECT descendant_concept_id "
            f"FROM concept_ancestor WHERE ancestor_concept_id IN ({_ids(ids)})))")


def _time(alias: str, domain: Domain) -> str:
    _, _, dt, d = EVENT_TABLE[domain]
    return f"coalesce({alias}.{dt}, CAST({alias}.{d} AS TIMESTAMP))"


def _days(a: str, b: str) -> str:
    return f"date_diff('second', {a}, {b}) / 86400.0"


def _column(name: str) -> str:
    c = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return c if c and not c[0].isdigit() else f"v_{c}"


def _value_sql(expr: str, v) -> str:
    if v.comparator == Comparator.BETWEEN:
        return f"{expr} BETWEEN {float(v.value)} AND {float(v.upper)}"
    return f"{expr} {v.comparator} {float(v.value if v.value is not None else v.upper)}"


class _Skip(Exception):
    pass


class _Ctx:
    def __init__(self, protocol: TargetTrialProtocol, spec: CompileSpec):
        self.protocol, self.spec = protocol, spec
        tz = protocol.time_zero.anchor_event if protocol.time_zero else None
        self.t0_names = T0_ALIASES | ({_norm(tz)} if tz else set())
        self.anchors = {_norm(k): v for k, v in spec.anchors.items()}

    def ids_for(self, key: str, ref: ConceptRef) -> list[int] | None:
        ids = self.spec.concept_sets.get(key)
        if ids:
            return ids
        return [ref.omop_concept_id] if ref.omop_concept_id else None

    def anchor(self, name: str) -> tuple[str, bool]:
        """SQL for the anchor time and whether it is time zero."""
        n = _norm(name)
        if n in self.t0_names or n.replace(" ", "_") == "time_zero":
            return "b.t0", True
        ev = self.anchors.get(n)
        if ev is None:
            raise _Skip(f"anchor '{name}' is not defined in the compile spec")
        table, col, _, _ = EVENT_TABLE[Domain(ev.domain)]
        agg = "min" if ev.occurrence == "first" else "max"
        t = _time("a", Domain(ev.domain))
        return (f"(SELECT {agg}({t}) FROM {table} a WHERE a.person_id = b.person_id AND "
                f"{_in_set(f'a.{col}', ev.concept_ids, self.spec.descendants)} "
                f"AND {t} <= b.t0)"), False  # fmt: skip


# --- atoms -------------------------------------------------------------------------------------


def _demographic(atom: CriterionAtom, ref: ConceptRef, warnings: list[str]) -> tuple[str, str]:
    name = _norm(ref.name)
    v = atom.value
    if name == "age":
        if v is None or v.comparator == Comparator.PRESENT:
            raise _Skip("age without a numeric bound")
        unit = (v.unit or "").lower().rstrip("s")
        if unit in ("", "year", "yr", "y"):
            if not unit:
                warnings.append("age bound has no unit: read as years")
            return _value_sql("b.age", v), f"age at time zero {v.comparator} {v.value:g} years"
        if unit == "month":
            return _value_sql("(b.age * 12)", v), f"age at time zero in months {v.comparator}"
        raise _Skip(f"age unit '{v.unit}' not supported")
    sex = None
    if name in ("female", "women", "woman"):
        sex = FEMALE
    elif name in ("male", "men", "man"):
        sex = MALE
    elif name in ("sex", "gender") and v is not None and v.text:
        t = _norm(v.text)
        sex = FEMALE if t in ("female", "women", "woman", "f") else \
            MALE if t in ("male", "men", "man", "m") else None  # fmt: skip
    if sex is None:
        raise _Skip(f"demographic '{ref.name}' has no person-table equivalent the compiler knows")
    return f"b.gender_concept_id = {sex}", f"person.gender_concept_id = {sex}"


def _atom(ctx: _Ctx, crit: Criterion, atom: CriterionAtom,
          warnings: list[str]) -> tuple[str, str]:  # fmt: skip
    """SQL predicate and a readable description for one atom."""
    if not atom.concepts:
        raise _Skip("atom has no concept")
    domains = {c.domain for c in atom.concepts}
    v = atom.value
    if v is not None and len(domains) > 1:
        raise _Skip("value constraint on concepts of several domains "
                    f"({', '.join(sorted(domains))}): split the atom")
    dom = Domain(next(iter(domains)))
    if v is not None and v.comparator == Comparator.PRESENT and dom != Domain.DEMOGRAPHIC:
        raise _Skip(f"qualitative value '{v.text or 'present'}' is not compilable")
    if v is not None and dom not in (Domain.MEASUREMENT, Domain.DEMOGRAPHIC):
        raise _Skip(f"value constraint '{v.text or v.comparator}' on a {dom} concept "
                    "(a count or a duration) is not supported yet")
    t = atom.temporal
    rel = TemporalRelation(t.relation) if t else None
    if rel == TemporalRelation.CUMULATIVE_DURATION:
        raise _Skip("cumulative duration is not supported yet")
    # Design problems first: they hold whatever the concept mapping.
    if t is not None:
        _, y_is_t0 = ctx.anchor(t.anchor)
        if y_is_t0 and rel in (TemporalRelation.X_AFTER_Y, TemporalRelation.X_WITHIN_T_AFTER_Y):
            raise _Skip("needs records after time zero (look-ahead); redefine time zero")

    parts, desc = [], []
    for ref in atom.concepts:
        if ref.domain == Domain.DEMOGRAPHIC:
            sql, d = _demographic(atom, ref, warnings)
            parts.append(sql)
            desc.append(d)
            continue
        ids = ctx.ids_for(concept_key(crit, ref.name), ref)
        if not ids:
            if ref.status == MappingStatus.UNMAPPABLE and atom.concept_logic == "any" \
                    and len(atom.concepts) > 1:  # fmt: skip
                warnings.append(f"'{ref.name}' is unmappable and left out of the 'any' list")
                continue
            unmappable = ref.status == MappingStatus.UNMAPPABLE
            raise _Skip(f"'{ref.name}': "
                        + ("unmappable" if unmappable else "no approved concept set"))
        sql, d = _record(ctx, Domain(ref.domain), ids, atom, rel, warnings)
        parts.append(sql)
        desc.append(f"{ref.name}: {d}")
    if not parts:
        raise _Skip("no concept left to compile")
    joiner = " AND " if atom.concept_logic == "all" else " OR "
    sql = "(" + joiner.join(f"coalesce({p}, FALSE)" for p in parts) + ")"
    text = f"{atom.concept_logic} of: " + "; ".join(desc) if len(desc) > 1 else desc[0]
    if atom.negated:
        return f"(NOT {sql})", f"NOT ({text})"
    return sql, text


def _record(ctx: _Ctx, dom: Domain, ids: list[int], atom: CriterionAtom,
            rel: TemporalRelation | None, warnings: list[str]) -> tuple[str, str]:  # fmt: skip
    table, col, _, _ = EVENT_TABLE[dom]
    x = _time("r", dom)
    t = atom.temporal
    y, y_is_t0 = ctx.anchor(t.anchor) if t else ("b.t0", True)
    where = ["r.person_id = b.person_id",_in_set(f"r.{col}", ids, ctx.spec.descendants)]
    w = t.window if t else None
    after = rel in (TemporalRelation.X_AFTER_Y, TemporalRelation.X_WITHIN_T_AFTER_Y)
    if rel in (TemporalRelation.X_WITHIN_T_BEFORE_Y, TemporalRelation.X_WITHIN_T_AFTER_Y) \
            and w is None:  # fmt: skip
        warnings.append("'within' without a window: read as any time up to / after the anchor")
        # "Onset within [missing] before randomization" includes onset at randomization.
        rel = TemporalRelation.X_AT_Y if rel == TemporalRelation.X_WITHIN_T_BEFORE_Y \
            else TemporalRelation.X_AFTER_Y  # fmt: skip
    anchor_txt = "time zero" if y_is_t0 else f"'{t.anchor}'"
    latest = dom == Domain.MEASUREMENT and rel in (None, TemporalRelation.X_AT_Y)
    if rel == TemporalRelation.X_BEFORE_Y:
        where.append(f"{x} < {y}")
        d = f"before {anchor_txt}"
    elif rel == TemporalRelation.X_WITHIN_T_BEFORE_Y:
        where.append(f"{x} BETWEEN {y} - {_interval(w)} AND {y}")
        d = f"within {w.amount:g} {w.unit}(s) before {anchor_txt}"
    elif rel == TemporalRelation.X_AFTER_Y:
        where.append(f"{x} > {y}")
        d = f"after {anchor_txt}"
    elif rel == TemporalRelation.X_WITHIN_T_AFTER_Y:
        where.append(f"{x} BETWEEN {y} AND {y} + {_interval(w)}")
        d = f"within {w.amount:g} {w.unit}(s) after {anchor_txt}"
    elif latest:
        lb = ctx.spec.measurement_lookback
        where.append(f"{x} BETWEEN {y} - {_interval(lb)} AND {y}")
        d = f"latest value within {lb.amount:g} {lb.unit}(s) before {anchor_txt}"
    else:
        where.append(f"{x} <= {y}")
        d = f"any record up to {anchor_txt}"
    if after:
        warnings.append("records after time zero are not used (no look-ahead)")
    if not y_is_t0:
        where.append(f"{x} <= b.t0")  # no look-ahead, whatever the anchor
    concept_txt = f"{dom} {{{_ids(ids)}}}" + ("+descendants" if ctx.spec.descendants else "")
    v = atom.value
    if v is not None and v.unit:
        warnings.append(f"units are not converted: values are read as {v.unit}")
    cond = " AND ".join(where)
    if latest and v is not None:
        sql = (f"(SELECT r.value_as_number FROM {table} r WHERE {cond} "
               f"AND r.value_as_number IS NOT NULL ORDER BY {x} DESC LIMIT 1)")
        return f"({_value_sql(sql, v)})", f"{concept_txt}, {d} {v.comparator} {v.value:g}"
    if v is not None:
        cond += f" AND {_value_sql('r.value_as_number', v)}"
        d += f", value {v.comparator} {v.value:g}"
    return f"EXISTS (SELECT 1 FROM {table} r WHERE {cond})", f"{concept_txt}, {d}"


def _criterion(ctx: _Ctx, crit: Criterion) -> CompiledCriterion:
    out = CompiledCriterion(id=crit.id, kind=crit.kind, text=crit.text, status="compiled")
    if not crit.emulable:
        out.status = "not_emulable"
        out.reasons.append("administrative criterion: not emulable, excluded with reason")
        return out
    if not crit.atoms:
        out.status = "not_compiled"
        out.reasons.append("no atoms")
        return out
    preds = []
    for i, atom in enumerate(crit.atoms, 1):
        try:
            sql, d = _atom(ctx, crit, atom, out.warnings)
        except _Skip as e:
            out.reasons.append(f"atom {i}: {e}" if len(crit.atoms) > 1 else str(e))
            continue
        preds.append(sql)
        out.description.append(d)
    if out.reasons:
        out.status, out.description = "not_compiled", []
        return out
    joiner = " AND " if crit.atom_logic == "all" else " OR "
    out.predicate = "(" + joiner.join(preds) + ")"
    out.column = "c_" + _column(crit.id)
    return out


# --- the queries -------------------------------------------------------------------------------


def _base_ctes(protocol: TargetTrialProtocol, spec: CompileSpec) -> str:
    tz = spec.time_zero
    table, col, _, _ = EVENT_TABLE[Domain(tz.domain)]
    agg = "min" if tz.occurrence == "first" else "max"
    offset = protocol.time_zero.offset if protocol.time_zero else None
    shift = f" + {_interval(offset)}" if offset and offset.amount else ""
    birth = ("coalesce(p.birth_datetime, CAST(make_date(p.year_of_birth, "
             "coalesce(p.month_of_birth, 7), coalesce(p.day_of_birth, 1)) AS TIMESTAMP))")
    return f"""idx AS (
  SELECT e.person_id, {agg}({_time('e', Domain(tz.domain))}){shift} AS t0
  FROM {table} e
  WHERE {_in_set(f'e.{col}', tz.concept_ids, spec.descendants)}
  GROUP BY e.person_id
),
base AS (
  -- index inside an observation period, and alive at time zero
  SELECT idx.person_id, idx.t0, p.gender_concept_id,
         date_diff('second', {birth}, idx.t0) / 86400.0 / 365.25 AS age,
         CAST(op.observation_period_end_date AS TIMESTAMP) + INTERVAL '1 day' AS obs_end
  FROM idx
  JOIN person p ON p.person_id = idx.person_id
  JOIN observation_period op ON op.person_id = idx.person_id
   AND CAST(idx.t0 AS DATE) BETWEEN op.observation_period_start_date
                                AND op.observation_period_end_date
  WHERE NOT EXISTS (SELECT 1 FROM death d WHERE d.person_id = idx.person_id
                    AND d.death_date < CAST(idx.t0 AS DATE))
)"""  # fmt: skip


def _flags_cte(criteria: list[CompiledCriterion]) -> str:
    cols = "".join(f",\n         {c.predicate} AS {c.column}" for c in criteria if c.column)
    inc = [c.column for c in criteria if c.column and c.kind == "inclusion"]
    exc = [c.column for c in criteria if c.column and c.kind == "exclusion"]
    elig = " AND ".join(inc + [f"NOT {c}" for c in exc]) or "TRUE"
    return f"""flags AS (
  SELECT b.person_id, b.t0, b.obs_end, b.age, b.gender_concept_id{cols}
  FROM base b
),
cohort AS (
  SELECT *, ({elig}) AS eligible FROM flags
)"""


def _days_of(d: Duration) -> float:
    return d.amount * _SECONDS[d.unit] / 86400


def _horizon(protocol: TargetTrialProtocol) -> Duration:
    primary = next((o for o in protocol.outcomes if o.primary), None)
    h = (primary.horizon if primary else None) or protocol.follow_up.horizon
    if h is None:
        raise CompileError("no follow-up horizon in the primary outcome or follow_up")
    return h


def _frame(ctx: _Ctx, notes: list[str]) -> tuple[str, list[str]] | None:
    p, spec = ctx.protocol, ctx.spec
    arms = [a for a in p.treatment_strategies if not a.is_comparator]
    comparators = [a for a in p.treatment_strategies if a.is_comparator]
    if len(arms) != 1 or len(comparators) != 1:
        notes.append("frame not compiled: needs exactly one treatment and one comparator arm")
        return None
    arm = arms[0]
    if comparators[0].components:
        notes.append("frame not compiled: active comparators are not supported yet")
        return None
    starts = []
    for comp in arm.components:
        ids = ctx.ids_for(f"T:{arm.label}:{comp.concept.name}", comp.concept)
        if not ids:
            notes.append(f"frame not compiled: treatment '{comp.concept.name}' has no approved "
                         "concept set")
            return None
        dom = Domain(comp.concept.domain)
        table, col, _, _ = EVENT_TABLE[dom]
        x = _time("r", dom)
        starts.append(f"(SELECT min({x}) FROM {table} r WHERE r.person_id = c.person_id AND "
                      f"{_in_set(f'r.{col}', ids, spec.descendants)} AND {x} >= c.t0)")
    if not starts:
        notes.append("frame not compiled: the treatment arm has no components")
        return None
    rx = starts[0] if len(starts) == 1 else f"greatest({', '.join(starts)})"
    if len(starts) > 1:
        notes.append(f"initiation of '{arm.label}' is when all {len(starts)} components have "
                     "started")
    notes.append(f"treated: '{arm.label}' initiated after time zero and before follow-up ends; "
                 f"comparator '{comparators[0].label}' is no initiation")

    h = _horizon(p)
    primary = next((o for o in p.outcomes if o.primary), None)
    death = ("(SELECT min(coalesce(d.death_datetime, CAST(d.death_date AS TIMESTAMP) "
             "+ INTERVAL '12 hours')) FROM death d WHERE d.person_id = c.person_id)")
    if primary is None:
        raise CompileError("no primary outcome")
    ids = ctx.ids_for(f"O:{primary.name}", primary.concept) if primary.concept else None
    if ids:
        dom = Domain(primary.concept.domain)
        table, col, _, _ = EVENT_TABLE[dom]
        x = _time("r", dom)
        outcome = (f"(SELECT min({x}) FROM {table} r WHERE r.person_id = c.person_id AND "
                   f"{_in_set(f'r.{col}', ids, spec.descendants)} AND {x} > c.t0)")
        notes.append(f"outcome '{primary.name}': first {dom} record after time zero; death "
                     "censors")
    elif any(w in _norm(f"{primary.name} {primary.description or ''}") for w in _DEATH_WORDS):
        outcome = None
        notes.append(f"outcome '{primary.name}': all-cause death (death table)")
    else:
        notes.append(f"frame not compiled: outcome '{primary.name}' has no approved concept set")
        return None
    end_event = primary.end_event or p.follow_up.end_event
    if end_event:
        notes.append(f"end event '{end_event}' is not compiled yet: follow-up runs to the "
                     "horizon or the end of observation")
    notes.append(f"follow-up: {h.amount:g} {h.unit}(s) from time zero, ended early by the end "
                 "of the observation period")

    conf_cols, conf_sql = [], []
    for ref in p.assignment.confounders:
        name = _column(ref.name)
        if ref.domain == Domain.DEMOGRAPHIC:
            n = _norm(ref.name)
            if n == "age":
                conf_sql.append("c.age")
                conf_cols.append("age")
            elif n in ("sex", "gender", "female"):
                conf_sql.append(f"CAST(c.gender_concept_id = {FEMALE} AS INTEGER) AS female")
                conf_cols.append("female")
            else:
                notes.append(f"confounder '{ref.name}' left out: no person-table field")
            continue
        cids = ctx.ids_for(f"C:{ref.name}", ref)
        if not cids:
            notes.append(f"confounder '{ref.name}' left out: no approved concept set")
            continue
        dom = Domain(ref.domain)
        table, col, _, _ = EVENT_TABLE[dom]
        x = _time("r", dom)
        cond = (f"r.person_id = c.person_id AND {_in_set(f'r.{col}', cids, spec.descendants)} "
                f"AND {x} <= c.t0")  # fmt: skip
        if dom == Domain.MEASUREMENT:
            lb = spec.measurement_lookback
            conf_sql.append(f"(SELECT r.value_as_number FROM {table} r WHERE {cond} AND {x} >= "
                            f"c.t0 - {_interval(lb)} AND r.value_as_number IS NOT NULL "
                            f"ORDER BY {x} DESC LIMIT 1) AS {name}")  # fmt: skip
            notes.append(f"confounder {name}: latest value within {lb.amount:g} {lb.unit}(s) "
                         "before time zero (missing stays missing)")
        else:
            conf_sql.append(f"CAST(EXISTS (SELECT 1 FROM {table} r WHERE {cond}) AS INTEGER) "
                            f"AS {name}")
            notes.append(f"confounder {name}: any record up to time zero")
        conf_cols.append(name)

    ev = "c.death_time" if outcome is None else "c.outcome_time"
    label = "death" if outcome is None else "outcome"
    ends = "c.death_time, c.admin_end" if outcome is None else \
        "c.death_time, c.outcome_time, c.admin_end"  # fmt: skip
    select_conf = "".join(f",\n       {s}" for s in conf_sql)
    sql = f""",
fu AS (
  SELECT c.*, {death} AS death_time{f', {outcome} AS outcome_time' if outcome else ''},
         least(c.t0 + {_interval(h)}, c.obs_end) AS admin_end,
         {rx} AS rx_start
  FROM cohort c WHERE c.eligible
),
o AS (
  SELECT c.*, least({ends}) AS end_time,
         coalesce({ev} = least({ends}), FALSE) AS hit
  FROM fu c
)
SELECT c.person_id{select_conf},
       CAST(coalesce(c.rx_start < c.end_time, FALSE) AS INTEGER) AS treated,
       CASE WHEN c.rx_start < c.end_time THEN {_days('c.t0', 'c.rx_start')} END AS treat_start,
       {_days('c.t0', 'c.end_time')} AS time,
       CAST(c.hit AS INTEGER) AS event,
       CASE WHEN c.hit THEN '{label}'
            WHEN coalesce(c.death_time = c.end_time, FALSE) THEN 'death'
            WHEN c.end_time = c.t0 + {_interval(h)} THEN 'end_of_follow_up'
            ELSE 'end_of_observation' END AS end_reason
FROM o c
ORDER BY c.person_id"""  # fmt: skip
    cols = ["person_id", *conf_cols, "treated", "treat_start", "time", "event", "end_reason"]
    return sql, cols


def compile_protocol(protocol: TargetTrialProtocol, spec: CompileSpec) -> CompiledCohort:
    """Compile the protocol's eligibility (and, where it can, the analysis frame) to SQL."""
    ctx = _Ctx(protocol, spec)
    criteria = [_criterion(ctx, c) for c in protocol.eligibility]
    bad = [c for c in criteria if c.status == "not_compiled"]
    if spec.strict and bad:
        raise CompileError("not compiled: " + "; ".join(f"{c.id}: {', '.join(c.reasons)}"
                                                        for c in bad))
    seen: dict[str, int] = {}
    for c in criteria:  # unique column names
        if c.column:
            seen[c.column] = seen.get(c.column, 0) + 1
            if seen[c.column] > 1:
                c.column += f"_{seen[c.column]}"
    ctes = _base_ctes(protocol, spec) + ",\n" + _flags_cte(criteria)
    cohort_sql = (f"WITH {ctes}\nSELECT * FROM cohort ORDER BY person_id")
    notes: list[str] = []
    frame = _frame(ctx, notes)
    lb = spec.measurement_lookback
    assumptions = [
        f"time zero: {spec.time_zero.occurrence} {spec.time_zero.domain} record of "
        f"{{{_ids(spec.time_zero.concept_ids)}}}"
        + (f" ({spec.time_zero.note})" if spec.time_zero.note else "")
        + "; it must fall inside an observation period, with the person alive",
        "time zero, randomization, enrollment and baseline all mean the index time",
        "eligibility reads only records at or before time zero (no look-ahead)",
        "'at the time of' an anchor means recorded at or before it; no temporal constraint "
        "means any record up to time zero",
        f"measurements: the latest value within {lb.amount:g} {lb.unit}(s) before the anchor, "
        "unless the criterion gives a window; a missing value means not met; units are not "
        "converted",
        "records with a date and no time are placed at midnight",
        "concept sets include descendants (concept_ancestor)" if spec.descendants
        else "concept sets are used as given (no descendants)",
        "criteria that are not compiled are left out of 'eligible' and listed above",
    ]  # fmt: skip
    for name, ev in spec.anchors.items():
        assumptions.append(f"anchor '{name}': {ev.occurrence} {ev.domain} record of "
                           f"{{{_ids(ev.concept_ids)}}} up to time zero")
    return CompiledCohort(
        protocol_id=protocol.protocol_id, protocol_version=protocol.version,
        protocol_hash=protocol.content_hash(), spec=spec, criteria=criteria,
        ctes=ctes, cohort_sql=cohort_sql, assumptions=assumptions, frame_notes=notes,
        horizon_days=_days_of(_horizon(protocol)) if frame else None,
        frame_sql=f"WITH {ctes}{frame[0]}" if frame else None,
        frame_columns=frame[1] if frame else [])  # fmt: skip


def spec_template(protocol: TargetTrialProtocol) -> dict:
    """A compile spec to fill in: time zero and every named anchor the criteria use."""
    tz = protocol.time_zero.anchor_event if protocol.time_zero else "time zero"
    t0_names = T0_ALIASES | {_norm(tz)}
    anchors = sorted({a.temporal.anchor for c in protocol.eligibility for a in c.atoms
                      if a.temporal and _norm(a.temporal.anchor) not in t0_names
                      and _norm(a.temporal.anchor).replace(" ", "_") != "time_zero"})  # fmt: skip
    blank = {"domain": "Visit", "concept_ids": [], "occurrence": "first", "note": ""}
    return {"time_zero": {**blank, "note": f"time zero: {tz}"},
            "anchors": {a: {**blank, "occurrence": "last", "note": ""} for a in anchors},
            "measurement_lookback": {"amount": 24, "unit": "hour"}, "strict": False}
