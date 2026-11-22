"""Concept counts for a concept review: for each set the reviewer would keep (approve, replace,
surrogate), the vocabulary name of every ID and how many records and persons the CDM has for
the set, counted in the table the compiler reads for the set's domain and with descendants as
the compiler expands them. A set with no records makes an inclusion criterion exclude everyone,
and a wrong name (a pasted ID that lost a digit, a proposal that matched the wrong concept) is
caught before the run.

Whole-database aggregates only: nothing by arm or outcome, and counts are small-cell
suppressed, so the output can be shown and pasted (SCORING.md S8)."""

from __future__ import annotations

from pydantic import BaseModel

from agentic_tte.omop.compiler import EVENT_TABLE
from agentic_tte.omop.concept_sets import ConceptSetBook
from agentic_tte.omop.gate import suppress
from agentic_tte.protocol.models import Domain

KEPT = ("approve", "replace", "surrogate")


class SetCount(BaseModel):
    key: str
    action: str
    domain: str | None  # the set's domain (the table the compiler reads)
    concepts: list[str]  # "id name [vocabulary domain, standard]"
    records: int | str
    persons: int | str
    units: list[str] = []  # measurements: "unit: records, median value", most records first
    warnings: list[str]


def _no_records(key: str) -> str:
    """What a set with no records does, by the kind of set its key names."""
    kind = key.split(":", 1)[0]
    if kind.startswith(("I", "S")):
        return "NO RECORDS: this inclusion criterion would exclude everyone"
    if kind.startswith("E"):
        return "no records: this exclusion criterion removes no one"
    if kind == "C":
        return "no records: this confounder is missing for everyone (constant)"
    if kind == "T":
        return "NO RECORDS: no one would start this treatment"
    if kind == "O":
        return "NO RECORDS: the outcome would have no events"
    return "no records"


def _table(domain: str) -> tuple[str, str] | None:
    try:
        d = Domain(domain)
    except ValueError:
        return None
    if d not in EVENT_TABLE:
        return None
    table, col, _, _ = EVENT_TABLE[d]  # Visit: a subquery over both visit tables
    return table, col


def concept_counts(con, decisions: dict, book: ConceptSetBook | None = None) -> list[SetCount]:
    """Counts for every kept set in a decisions file. ``book`` gives each set's domain and
    whether descendants are included; without it, each concept's vocabulary domain is used."""
    sets = {s.key: s for s in (book.sets if book else [])}
    kept = [(k, d) for k, d in (decisions.get("decisions") or {}).items()
            if d.get("action") in KEPT]  # fmt: skip
    ids = sorted({int(i) for _, d in kept for i in d.get("concept_ids") or []})
    vocab = {}
    if ids:
        vocab = {r[0]: r[1:] for r in con.execute(
            "SELECT concept_id, concept_name, domain_id, standard_concept FROM concept "
            f"WHERE concept_id IN ({', '.join(map(str, ids))})").fetchall()}  # fmt: skip
    con.execute("CREATE OR REPLACE TEMP TABLE _review_sets "
                "(key VARCHAR, domain VARCHAR, cid BIGINT, descendants BOOLEAN)")
    rows, out = [], {}
    for k, d in kept:
        s = sets.get(k)
        cids = [int(i) for i in d.get("concept_ids") or []]
        warn, names = [], []
        for c in cids:
            v = vocab.get(c)
            if v is None:
                names.append(f"{c} NOT IN VOCABULARY")
                warn.append(f"{c} is not in the vocabulary")
                continue
            name, vdom, std = v
            names.append(f"{c} {name} [{vdom}, {'standard' if std == 'S' else 'non-standard'}]")
            if std != "S":
                warn.append(f"{c} is not a standard concept")
            if s is not None and vdom != s.domain:
                warn.append(f"{c} is a {vdom} concept; the set is read from the {s.domain} table")
        domain = s.domain if s is not None else (vocab[cids[0]][1] if cids and cids[0] in vocab
                                                 else None)  # fmt: skip
        if not cids:
            warn.append(f"action {d.get('action')!r} with no concept IDs")
        elif domain is None or _table(domain) is None:
            warn.append(f"no event table for the domain {domain!r}")
        else:
            desc = s.include_descendants if s is not None else True
            rows += [(k, domain, c, desc) for c in cids]
        out[k] = SetCount(key=k, action=d["action"], domain=domain, concepts=names, records=0,
                          persons=0, warnings=warn)  # fmt: skip
    if rows:
        con.executemany("INSERT INTO _review_sets VALUES (?, ?, ?, ?)", rows)
        con.execute("""CREATE OR REPLACE TEMP TABLE _review_expanded AS
            SELECT DISTINCT key, domain, cid AS desc_id FROM _review_sets
            UNION SELECT DISTINCT s.key, s.domain, a.descendant_concept_id FROM _review_sets s
            JOIN concept_ancestor a ON a.ancestor_concept_id = s.cid WHERE s.descendants""")
        for domain in sorted({r[1] for r in rows}):
            table, col = _table(domain)
            for k, r, p in con.execute(
                    f"SELECT e.key, count(*), count(DISTINCT t.person_id) FROM {table} t "
                    f"JOIN _review_expanded e ON t.{col} = e.desc_id AND e.domain = ? "
                    "GROUP BY e.key", [domain]).fetchall():  # fmt: skip
                out[k].records, out[k].persons = suppress(r), suppress(p)
        if "Measurement" in {r[1] for r in rows}:  # units are not converted: show the data's
            for k, unit, r, med in con.execute(
                    "SELECT e.key, coalesce(u.concept_name, 'no unit'), count(*), "
                    "median(m.value_as_number) FROM measurement m "
                    "JOIN _review_expanded e ON m.measurement_concept_id = e.desc_id "
                    "AND e.domain = 'Measurement' "
                    "LEFT JOIN concept u ON u.concept_id = m.unit_concept_id "
                    "GROUP BY ALL ORDER BY e.key, count(*) DESC").fetchall():  # fmt: skip
                if len(out[k].units) < 3:
                    m = ("no value" if med is None else "median suppressed" if r < 11
                         else f"median {med:g}")  # a median of a few values is near a row
                    out[k].units.append(f"{unit}: {suppress(r)} records, {m}")
    for c in out.values():
        if c.records == 0 and c.domain and not any("no event table" in w for w in c.warnings):
            c.warnings.append(_no_records(c.key))
        if c.units:
            c.warnings.append("units are not converted: a threshold must be in the data's unit")
    return list(out.values())


def search(con, word: str, domain: str, limit: int = 12) -> list[tuple[int, str, str, int | str]]:
    """The concepts in use in the domain's table whose name contains ``word``, most records
    first: what the data actually holds, to choose a replacement from."""
    t = _table(domain)
    if t is None:
        raise ValueError(f"no event table for the domain {domain!r}")
    table, col = t
    return [(cid, name, "standard" if std == "S" else "non-standard", suppress(r))
            for cid, name, std, r in con.execute(
                f"SELECT c.concept_id, c.concept_name, c.standard_concept, count(*) AS r "
                f"FROM {table} t JOIN concept c ON c.concept_id = t.{col} "
                "WHERE lower(c.concept_name) LIKE ? GROUP BY ALL ORDER BY r DESC LIMIT ?",
                [f"%{word.lower()}%", limit]).fetchall()]  # fmt: skip


def to_markdown(counts: list[SetCount], searches: dict[str, list] | None = None) -> str:
    lines = ["| Key | Action | Concepts | Records | Persons | Units in the data | Check |",
             "|---|---|---|---|---|---|---|"]
    for c in counts:
        lines.append(f"| `{c.key}` | {c.action} | {'; '.join(c.concepts) or '–'} | {c.records} "
                     f"| {c.persons} | {'; '.join(c.units) or '–'} "
                     f"| {'; '.join(c.warnings) or 'ok'} |")
    for term, found in (searches or {}).items():
        lines += ["", f"**In use, name contains {term}:**", "",
                  "| ID | Name | Standard | Records |", "|---|---|---|---|"]
        lines += [f"| {i} | {n} | {s} | {r} |" for i, n, s, r in found] or ["| – | none | | |"]
    return "\n".join(lines) + "\n"
