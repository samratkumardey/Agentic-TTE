"""Data Support Gate: can this OMOP database support this protocol? (PLAN.md §14)

Runs before any cohort is built. Every check returns green, amber or red; a red check blocks
the run, an amber one is a limitation the report must state (a criterion needing a surrogate,
coarse timestamps, ...). The gate reads only aggregates. Counts below 11 are shown as "<11"
in every report, the usual small-cell rule for MIMIC and claims data.

Checks:
  structure      required CDM tables and columns
  cdm_metadata   cdm_source: CDM version
  vocabulary     concept table, standard concepts, concept_ancestor
  volume         persons and observation periods
  time           datetime coverage and time-of-day, against the protocol's finest window
  death          death records and their consistency with observation periods
  follow_up      observation time after the index visit, against the follow-up horizon
  quality        plausibility (end before start, death before birth, impossible ages, ...)
  concepts       patients with a record for each protocol concept (needs concept IDs)
  governance     data classification and approval; no patient rows to any LLM
  feasibility    eligible N, events per arm, power: from the compiled cohort's attrition
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_tte.protocol.models import Criterion, Domain, TargetTrialProtocol

SMALL_CELL = 11


class GateStatus(StrEnum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    NOT_RUN = "not_run"  # needs a later pipeline step; does not affect the overall status


class GateCheck(BaseModel):
    id: str
    title: str
    status: GateStatus
    detail: str
    values: dict[str, Any] = Field(default_factory=dict)


class DataContext(BaseModel):
    """What the data owner declares about the source. The gate does not trust a source to
    classify itself: these values come from the person running it, and go into the report."""

    data_classification: Literal["synthetic", "deidentified", "limited", "identified",
                                 "unknown"] = "unknown"  # fmt: skip
    approved_by_rcss: bool = False  # storage/compute approved for this classification
    notes: str | None = None


class GateReport(BaseModel):
    source_name: str | None = None
    cdm_version: str | None = None
    vocabulary_version: str | None = None
    protocol_id: str | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: DataContext = Field(default_factory=DataContext)
    checks: list[GateCheck] = Field(default_factory=list)

    @property
    def overall(self) -> GateStatus:
        statuses = {c.status for c in self.checks}
        for s in (GateStatus.RED, GateStatus.AMBER):
            if s in statuses:
                return s
        return GateStatus.GREEN

    @property
    def blocks_run(self) -> bool:
        return self.overall == GateStatus.RED

    def check(self, check_id: str) -> GateCheck:
        return next(c for c in self.checks if c.id == check_id)

    def to_markdown(self) -> str:
        icon = {"green": "🟢", "amber": "🟠", "red": "🔴", "not_run": "⚪"}
        lines = [
            f"# Data Support Gate: {self.overall.upper()}"
            + (" (run blocked)" if self.blocks_run else ""),
            "",
            f"Source: {self.source_name or 'unknown'}; CDM {self.cdm_version or 'unknown'}; "
            f"vocabulary {self.vocabulary_version or 'unknown'}; protocol "
            f"{self.protocol_id or 'none'}; data classification "
            f"{self.context.data_classification}; checked {self.checked_at:%Y-%m-%d %H:%M} UTC.",
            "",
            "| Check | Status | Detail |",
            "|---|---|---|",
        ]
        for c in self.checks:
            lines.append(f"| {c.title} | {icon[c.status]} {c.status} | {c.detail} |")
        return "\n".join(lines) + "\n"


def suppress(n: int) -> int | str:
    """Small-cell suppression for anything shown to people or sent to an LLM."""
    return f"<{SMALL_CELL}" if 0 < n < SMALL_CELL else n


# --- database helpers (DuckDB now; other dialects through SQLGlot later) -----------------------

REQUIRED_TABLES = {
    "person": ["person_id", "gender_concept_id", "year_of_birth"],
    "observation_period": ["person_id", "observation_period_start_date",
                           "observation_period_end_date"],
    "visit_occurrence": ["person_id", "visit_concept_id", "visit_start_date"],
    # ICU and ward stays; the compiler reads Visit concepts from both visit tables.
    "visit_detail": ["person_id", "visit_detail_concept_id", "visit_detail_start_date"],
    "condition_occurrence": ["person_id", "condition_concept_id", "condition_start_date"],
    "drug_exposure": ["person_id", "drug_concept_id", "drug_exposure_start_date"],
    "measurement": ["person_id", "measurement_concept_id", "measurement_date"],
    "death": ["person_id", "death_date"],
    "concept": ["concept_id", "concept_name", "domain_id", "vocabulary_id"],
}  # fmt: skip

DOMAIN_TABLE = {
    Domain.CONDITION: ("condition_occurrence", "condition_concept_id"),
    Domain.DRUG: ("drug_exposure", "drug_concept_id"),
    Domain.MEASUREMENT: ("measurement", "measurement_concept_id"),
    Domain.PROCEDURE: ("procedure_occurrence", "procedure_concept_id"),
    Domain.OBSERVATION: ("observation", "observation_concept_id"),
    Domain.DEVICE: ("device_exposure", "device_concept_id"),
    Domain.VISIT: ("visit_occurrence", "visit_concept_id"),
}

DATETIME_COLUMNS = {
    "visit_occurrence": "visit_start_datetime",
    "condition_occurrence": "condition_start_datetime",
    "drug_exposure": "drug_exposure_start_datetime",
    "measurement": "measurement_datetime",
}


def _tables(con) -> set[str]:
    rows = con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    return {r[0].lower() for r in rows}


def _columns(con, table: str) -> set[str]:
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns WHERE lower(table_name) = ?", [table]
    ).fetchall()
    return {r[0].lower() for r in rows}


def _scalar(con, sql: str, params: list | None = None) -> Any:
    return con.execute(sql, params or []).fetchone()[0]


# --- checks ------------------------------------------------------------------------------------


def _structure(con, tables: set[str]) -> GateCheck:
    missing_tables = sorted(t for t in REQUIRED_TABLES if t not in tables)
    missing_cols = {t: sorted(set(c) - _columns(con, t))
                    for t, c in REQUIRED_TABLES.items() if t in tables}  # fmt: skip
    missing_cols = {t: c for t, c in missing_cols.items() if c}
    if missing_tables or missing_cols:
        parts = []
        if missing_tables:
            parts.append("missing tables: " + ", ".join(missing_tables))
        if missing_cols:
            parts.append("missing columns: " + "; ".join(f"{t}.{', '.join(c)}"
                                                         for t, c in missing_cols.items()))
        return GateCheck(id="structure", title="CDM structure", status=GateStatus.RED,
                         detail="; ".join(parts),
                         values={"missing_tables": missing_tables,
                                 "missing_columns": missing_cols})  # fmt: skip
    return GateCheck(id="structure", title="CDM structure", status=GateStatus.GREEN,
                     detail=f"all {len(REQUIRED_TABLES)} required tables and columns present")


def _cdm_metadata(con, tables: set[str]) -> tuple[GateCheck, dict[str, str | None]]:
    meta: dict[str, str | None] = {"source_name": None, "cdm_version": None,
                                   "vocabulary_version": None}  # fmt: skip
    if "cdm_source" not in tables:
        return GateCheck(id="cdm_metadata", title="CDM version", status=GateStatus.AMBER,
                         detail="no cdm_source table: CDM version unknown"), meta
    row = con.execute("SELECT * FROM cdm_source LIMIT 1").df()
    if not row.empty:
        r = row.iloc[0]
        meta = {"source_name": r.get("cdm_source_name"), "cdm_version": r.get("cdm_version"),
                "vocabulary_version": r.get("vocabulary_version")}  # fmt: skip
    version = str(meta["cdm_version"] or "")
    if version.startswith("5.4"):
        status, detail = GateStatus.GREEN, f"OMOP CDM {version}"
    elif version.startswith("5."):
        status, detail = GateStatus.AMBER, f"OMOP CDM {version}; the compiler targets 5.4"
    else:
        status, detail = GateStatus.RED, f"unsupported CDM version {version or '(empty)'}"
    return GateCheck(id="cdm_metadata", title="CDM version", status=status, detail=detail,
                     values=meta), meta  # fmt: skip


def _vocabulary(con, tables: set[str]) -> GateCheck:
    if "concept" not in tables:
        return GateCheck(id="vocabulary", title="Vocabulary", status=GateStatus.RED,
                         detail="no concept table")
    n = _scalar(con, "SELECT count(*) FROM concept")
    standard = _scalar(con, "SELECT count(*) FROM concept WHERE standard_concept = 'S'") \
        if "standard_concept" in _columns(con, "concept") else 0  # fmt: skip
    vocabs = [r[0] for r in con.execute(
        "SELECT DISTINCT vocabulary_id FROM concept ORDER BY 1").fetchall()]  # fmt: skip
    values = {"concepts": n, "standard_concepts": standard, "vocabularies": vocabs,
              "concept_ancestor": "concept_ancestor" in tables}  # fmt: skip
    if n == 0:
        return GateCheck(id="vocabulary", title="Vocabulary", status=GateStatus.RED,
                         detail="concept table is empty", values=values)
    # A full Athena download holds millions of standard concepts.
    if standard < 10_000 or "concept_ancestor" not in tables:
        return GateCheck(
            id="vocabulary", title="Vocabulary", status=GateStatus.AMBER,
            detail=f"{n:,} concepts, {standard:,} standard, vocabularies {', '.join(vocabs)}; "
                   "no full standard vocabulary or concept_ancestor, so concept mapping is "
                   "limited to the concepts present and descendants are not expanded",
            values=values)  # fmt: skip
    return GateCheck(id="vocabulary", title="Vocabulary", status=GateStatus.GREEN,
                     detail=f"{standard:,} standard concepts; concept_ancestor present",
                     values=values)  # fmt: skip


def _volume(con) -> GateCheck:
    persons = _scalar(con, "SELECT count(*) FROM person")
    without = _scalar(con, "SELECT count(*) FROM person p WHERE NOT EXISTS "
                           "(SELECT 1 FROM observation_period o WHERE o.person_id = p.person_id)")
    values = {"persons": persons, "persons_without_observation_period": without}
    if persons == 0:
        return GateCheck(id="volume", title="Volume", status=GateStatus.RED,
                         detail="no persons", values=values)
    status = GateStatus.AMBER if without else GateStatus.GREEN
    detail = f"{suppress(persons):,} persons" if isinstance(suppress(persons), int) \
        else f"{suppress(persons)} persons"  # fmt: skip
    if without:
        detail += f"; {suppress(without)} without an observation period (cannot be followed)"
    return GateCheck(id="volume", title="Volume", status=status, detail=detail, values=values)


def _finest_window_hours(protocol: TargetTrialProtocol | None) -> float | None:
    if protocol is None:
        return None
    hours = {"minute": 1 / 60, "hour": 1, "day": 24, "week": 168, "month": 730, "year": 8766}
    found = []
    for c in protocol.eligibility:
        for a in c.atoms:
            if a.temporal and a.temporal.window:
                found.append(a.temporal.window.amount * hours[a.temporal.window.unit])
    if protocol.time_zero and protocol.time_zero.offset:
        o = protocol.time_zero.offset
        found.append(o.amount * hours[o.unit])
    found = [f for f in found if f > 0]
    return min(found) if found else None


def _time(con, tables: set[str], protocol: TargetTrialProtocol | None) -> GateCheck:
    coverage = {}
    for table, col in DATETIME_COLUMNS.items():
        if table not in tables or col not in _columns(con, table):
            coverage[table] = {"datetime": 0.0, "time_of_day": 0.0}
            continue
        n, with_dt, with_time = con.execute(
            f"SELECT count(*), count({col}), "
            f"count(*) FILTER (WHERE {col} IS NOT NULL AND CAST({col} AS TIME) <> TIME '00:00:00') "
            f"FROM {table}").fetchone()  # fmt: skip
        if n == 0:  # an empty table says nothing about time resolution
            continue
        coverage[table] = {"datetime": with_dt / n if n else 0.0,
                           "time_of_day": with_time / n if n else 0.0}  # fmt: skip
    worst = min((v["time_of_day"] for v in coverage.values()), default=0.0)
    finest = _finest_window_hours(protocol)
    values = {"coverage": coverage, "finest_window_hours": finest}
    summary = ", ".join(f"{t} {100 * v['time_of_day']:.0f}%" for t, v in coverage.items())
    if finest is not None and finest < 24:
        if worst == 0:
            return GateCheck(id="time", title="Time resolution", status=GateStatus.RED,
                             detail=f"protocol needs {finest:g}-hour windows but some tables "
                                    f"have dates only (time of day: {summary})",
                             values=values)  # fmt: skip
        if worst < 0.95:
            return GateCheck(id="time", title="Time resolution", status=GateStatus.AMBER,
                             detail=f"protocol needs {finest:g}-hour windows; time of day "
                                    f"missing for some records ({summary})",
                             values=values)  # fmt: skip
    return GateCheck(id="time", title="Time resolution", status=GateStatus.GREEN,
                     detail=f"time of day recorded: {summary}"
                            + (f"; finest protocol window {finest:g} h" if finest else ""),
                     values=values)  # fmt: skip


def _death(con) -> GateCheck:
    deaths = _scalar(con, "SELECT count(*) FROM death")
    outside = _scalar(con, """
        SELECT count(*) FROM death d JOIN observation_period o USING (person_id)
        WHERE d.death_date > o.observation_period_end_date + INTERVAL 1 DAY
           OR d.death_date < o.observation_period_start_date""")
    values = {"deaths": deaths, "deaths_outside_observation": outside}
    if deaths == 0:
        return GateCheck(id="death", title="Death data", status=GateStatus.AMBER,
                         detail="no death records: mortality outcomes cannot be ascertained "
                                "unless deaths are linked from another source", values=values)
    if outside / deaths > 0.01:
        return GateCheck(id="death", title="Death data", status=GateStatus.AMBER,
                         detail=f"{suppress(deaths)} deaths; {suppress(outside)} fall outside "
                                "the person's observation period", values=values)
    return GateCheck(id="death", title="Death data", status=GateStatus.GREEN,
                     detail=f"{suppress(deaths)} deaths, consistent with observation periods",
                     values=values)  # fmt: skip


def _follow_up(con, protocol: TargetTrialProtocol | None) -> GateCheck:
    fu = protocol.follow_up if protocol else None
    if fu is None or fu.horizon is None:
        return GateCheck(id="follow_up", title="Follow-up coverage", status=GateStatus.NOT_RUN,
                         detail="no follow-up horizon in the protocol")
    days = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30.44,
            "year": 365.25}[fu.horizon.unit] * fu.horizon.amount  # fmt: skip
    # Index proxy until the compiler supplies time zero: each person's first visit start.
    share = _scalar(con, f"""
        WITH idx AS (SELECT person_id, min(visit_start_date) AS t0
                     FROM visit_occurrence GROUP BY person_id)
        SELECT avg(CASE WHEN date_diff('day', idx.t0, o.observation_period_end_date) >= {days}
                         OR d.person_id IS NOT NULL THEN 1.0 ELSE 0.0 END)
        FROM idx JOIN observation_period o USING (person_id)
        LEFT JOIN death d USING (person_id)""")  # fmt: skip
    share = float(share or 0.0)
    values = {"horizon_days": days, "share_followed_or_dead": share}
    status = GateStatus.GREEN if share >= 0.5 else GateStatus.AMBER
    return GateCheck(id="follow_up", title="Follow-up coverage", status=status,
                     detail=f"{100 * share:.0f}% of persons are observed for {days:g} days after "
                            "their first visit, or die within it (index proxy: first visit)",
                     values=values)  # fmt: skip


def _quality(con, tables: set[str]) -> GateCheck:
    checks = {
        "visit ends before it starts": "SELECT count(*) FROM visit_occurrence "
                                       "WHERE visit_end_date < visit_start_date",
        "death before birth": "SELECT count(*) FROM death d JOIN person p USING (person_id) "
                              "WHERE year(d.death_date) < p.year_of_birth",
        "age over 120 at first visit": """SELECT count(*) FROM person p JOIN
            (SELECT person_id, min(visit_start_date) AS t0 FROM visit_occurrence GROUP BY 1) v
            USING (person_id) WHERE year(v.t0) - p.year_of_birth > 120""",
        "observation period ends before it starts":
            "SELECT count(*) FROM observation_period "
            "WHERE observation_period_end_date < observation_period_start_date",
        "measurement without a value": "SELECT count(*) FROM measurement "
                                       "WHERE value_as_number IS NULL",
    }  # fmt: skip
    persons = max(_scalar(con, "SELECT count(*) FROM person"), 1)
    rows = max(_scalar(con, "SELECT count(*) FROM measurement"), 1)
    found = {}
    for label, sql in checks.items():
        n = _scalar(con, sql)
        if n:
            found[label] = n
    rates = {k: v / (rows if k.startswith("measurement") else persons) for k, v in found.items()}
    values = {"violations": found, "rates": rates}
    if not found:
        return GateCheck(id="quality", title="Data quality", status=GateStatus.GREEN,
                         detail="no plausibility violations found", values=values)
    worst = max(rates.values())
    detail = "; ".join(f"{k}: {suppress(v)} ({100 * rates[k]:.2f}%)" for k, v in found.items())
    status = GateStatus.RED if worst > 0.05 else GateStatus.AMBER
    return GateCheck(id="quality", title="Data quality", status=status, detail=detail,
                     values=values)  # fmt: skip


def concept_key(criterion: Criterion, concept_name: str) -> str:
    return f"{criterion.id}:{concept_name}"


def _patients_with(con, tables: set[str], domain: str, ids: list[int]) -> int | None:
    table, col = DOMAIN_TABLE.get(Domain(domain), (None, None))
    if table is None or table not in tables:
        return None
    id_list = ", ".join(str(int(i)) for i in ids)
    if "concept_ancestor" in tables:
        where = (f"{col} IN (SELECT descendant_concept_id FROM concept_ancestor "
                 f"WHERE ancestor_concept_id IN ({id_list})) OR {col} IN ({id_list})")
    else:
        where = f"{col} IN ({id_list})"
    source = table
    if Domain(domain) == Domain.VISIT and "visit_detail" in tables:
        # ICU stays live in visit_detail (32037 Intensive Care), admissions in visit_occurrence.
        source = ("(SELECT person_id, visit_concept_id FROM visit_occurrence UNION ALL "
                  "SELECT person_id, visit_detail_concept_id FROM visit_detail)")
    return int(_scalar(con, f"SELECT count(DISTINCT person_id) FROM {source} v WHERE {where}"))


def _concepts(con, tables: set[str], protocol: TargetTrialProtocol | None,
              concept_sets: dict[str, list[int]] | None) -> GateCheck:  # fmt: skip
    if protocol is None:
        return GateCheck(id="concepts", title="Protocol concepts", status=GateStatus.NOT_RUN,
                         detail="no protocol given")
    concept_sets = concept_sets or {}
    rows, status = [], GateStatus.GREEN

    def worse(s: GateStatus) -> None:
        nonlocal status
        order = [GateStatus.GREEN, GateStatus.AMBER, GateStatus.RED]
        if order.index(s) > order.index(status):
            status = s

    person_cols = _columns(con, "person") if "person" in tables else set()
    for crit in protocol.eligibility:
        if not crit.emulable:
            rows.append({"criterion": crit.id, "concept": None, "result": "administrative: "
                         "not emulable, excluded with reason"})  # fmt: skip
            continue
        for atom in crit.atoms:
            for c in atom.concepts:
                entry = {"criterion": crit.id, "kind": crit.kind, "concept": c.name,
                         "domain": c.domain}  # fmt: skip
                if c.domain == Domain.DEMOGRAPHIC:
                    name = c.name.lower()
                    need = "year_of_birth" if "age" in name else \
                        "gender_concept_id" if name in ("sex", "gender") else None  # fmt: skip
                    ok = need is None or need in person_cols
                    entry["result"] = "available in person" if ok else f"missing person.{need}"
                    if need is None:
                        entry["result"] = "demographic without a person field: review"
                        worse(GateStatus.AMBER)
                    elif not ok:
                        worse(GateStatus.RED)
                    rows.append(entry)
                    continue
                ids = concept_sets.get(concept_key(crit, c.name)) or (
                    [c.omop_concept_id] if c.omop_concept_id else [])
                if not ids:
                    entry["result"] = "unmapped"
                    worse(GateStatus.AMBER)
                    rows.append(entry)
                    continue
                n = _patients_with(con, tables, c.domain, ids)
                if n is None:
                    entry["result"] = f"no {DOMAIN_TABLE[Domain(c.domain)][0]} table"
                    worse(GateStatus.RED if crit.kind == "inclusion" else GateStatus.AMBER)
                elif n == 0:
                    entry["result"] = "0 patients"
                    # An inclusion concept nobody has empties the cohort; an exclusion concept
                    # nobody has removes no one (absent from the data, or truly rare).
                    worse(GateStatus.RED if crit.kind == "inclusion" else GateStatus.AMBER)
                else:
                    entry["result"] = f"{suppress(n)} patients"
                    if n < SMALL_CELL:
                        worse(GateStatus.AMBER)
                entry["patients"] = suppress(n) if n is not None else None
                rows.append(entry)
    unmapped = sum(r.get("result") == "unmapped" for r in rows)
    def blocking(r: dict) -> bool:
        result = str(r.get("result", ""))
        empty = result == "0 patients" or result.startswith("no ")
        return (empty and r.get("kind") == "inclusion") or result.startswith("missing")

    red = [f"{r['criterion']} {r['concept']} ({r['result']})" for r in rows if blocking(r)]
    detail = f"{len(rows)} concept(s) checked; {unmapped} unmapped"
    if red:
        detail += "; blocking: " + ", ".join(red)
    return GateCheck(id="concepts", title="Protocol concepts", status=status, detail=detail,
                     values={"concepts": rows})  # fmt: skip


def _governance(ctx: DataContext) -> GateCheck:
    rule = "LLMs receive schema, protocol and suppressed aggregate counts only, never patient rows"
    c = ctx.data_classification
    if c == "synthetic":
        status, why = GateStatus.GREEN, "synthetic data"
    elif c == "unknown":
        status, why = GateStatus.RED, "data classification not declared"
    elif ctx.approved_by_rcss:
        status, why = GateStatus.GREEN, f"{c} data, storage and compute approved"
    elif c == "deidentified":
        status, why = GateStatus.AMBER, "de-identified data; confirm the classification with RCSS"
    else:
        status, why = GateStatus.RED, f"{c} data without a recorded approval"
    return GateCheck(id="governance", title="Governance", status=status, detail=f"{why}. {rule}.",
                     values=ctx.model_dump())  # fmt: skip


def run_gate(con, protocol: TargetTrialProtocol | None = None, *,
             concept_sets: dict[str, list[int]] | None = None,
             context: DataContext | None = None, attrition=None,
             design_hr: float | None = None) -> GateReport:  # fmt: skip
    """Run every check on an open DuckDB connection to an OMOP CDM database.

    ``concept_sets`` maps ``concept_key(criterion, concept_name)`` to OMOP concept IDs, as the
    Phase 2 vocabulary step produces them; concepts already carrying ``omop_concept_id`` in the
    protocol are used as they are. ``attrition`` (an ``AttritionReport`` from the compiled
    cohort) runs the feasibility check, with power for ``design_hr`` when given."""
    context = context or DataContext()
    tables = _tables(con)
    report = GateReport(protocol_id=protocol.protocol_id if protocol else None, context=context)
    structure = _structure(con, tables)
    report.checks.append(structure)
    meta_check, meta = _cdm_metadata(con, tables)
    report.checks.append(meta_check)
    report.source_name = meta["source_name"]
    report.cdm_version = meta["cdm_version"]
    report.vocabulary_version = meta["vocabulary_version"]
    report.checks.append(_vocabulary(con, tables))
    report.checks.append(_governance(context))
    if structure.status == GateStatus.RED:
        for cid, title in (("volume", "Volume"), ("time", "Time resolution"),
                           ("death", "Death data"), ("follow_up", "Follow-up coverage"),
                           ("quality", "Data quality"), ("concepts", "Protocol concepts")):
            report.checks.append(GateCheck(id=cid, title=title, status=GateStatus.NOT_RUN,
                                           detail="skipped: CDM structure incomplete"))  # fmt: skip
    else:
        report.checks += [
            _volume(con),
            _time(con, tables, protocol),
            _death(con),
            _follow_up(con, protocol),
            _quality(con, tables),
            _concepts(con, tables, protocol, concept_sets),
        ]
    if attrition is not None:
        from agentic_tte.omop.attrition import feasibility

        report.checks.append(feasibility(attrition, protocol, design_hr=design_hr))
    else:
        report.checks.append(GateCheck(
            id="feasibility", title="Feasibility", status=GateStatus.NOT_RUN,
            detail="needs the compiled cohort's attrition (agentic-tte compile --db)"))
    return report
