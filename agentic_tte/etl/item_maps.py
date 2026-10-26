"""Reviewed item maps for MIMIC-IV items: labevents and inputevents (no curated map), and
reviewed overrides for chartevents and procedureevents (curated by MIT-LCP).

For each item this module proposes a standard concept from the vocabulary search on the item's
label, and a person decides (approve, replace, reject, unmappable), as for protocol concept
sets. For chartevents and procedureevents a reviewed decision overrides the curated map for
that item: it fixes curated targets the vocabulary no longer has as standard (e.g. Invasive
Ventilation, 225792, in the 2026 vocabulary), adds items the curated map lacks, or turns a
curated mapping off. ``propose_items`` lists curated items only when their target was
rejected, with the reason (``curated``).

    propose_items(...)   -> ItemMap (proposals, most frequent items first)
    review_sheet(map)    -> markdown for the reviewer
    decisions_template() -> a decisions file to fill in
    apply_decisions(...) -> the reviewed ItemMap the ETL takes (``EtlConfig.item_maps``)

Labs are searched in the Measurement domain, with the specimen ("Blood", "Urine") used to
rank LOINC candidates; inputs in the Drug domain, preferring the ingredient; chart items in
Measurement and Observation; procedure items in Procedure, Observation and Device. Only
approved items are converted. The files hold item labels and per-item row counts from the
credentialed data, so they stay where the data are (Hellbender) and never go into git; counts
below 11 are suppressed in them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agentic_tte.omop.gate import SMALL_CELL, suppress
from agentic_tte.omop.vocab import Candidate, norm, search

Kind = Literal["labevents", "inputevents", "chartevents", "procedureevents"]
Status = Literal["proposed", "no_match", "approved", "rejected", "unmappable"]
DOMAINS = {"labevents": ("Measurement",), "inputevents": ("Drug",),
           "chartevents": ("Measurement", "Observation"),
           "procedureevents": ("Procedure", "Observation", "Device")}  # fmt: skip
CURATED = ("chartevents", "procedureevents")  # kinds with an MIT-LCP curated map

# Specimen words in MIMIC's d_labitems.fluid, and the words LOINC names use for them.
_SPECIMEN = {"blood": ("blood", "serum", "plasma"), "urine": ("urine",),
             "cerebrospinal fluid": ("cerebrospinal",), "ascites": ("peritoneal", "ascites"),
             "pleural": ("pleural",), "joint fluid": ("synovial",),
             "stool": ("stool", "feces")}  # fmt: skip


class ItemEntry(BaseModel):
    itemid: int
    label: str
    specimen: str | None = None  # labs: d_labitems.fluid
    category: str | None = None
    unit: str | None = None  # inputs: d_items.unitname
    rows: int | str  # events with this item (suppressed below 11)
    curated: str | None = None  # the curated map's target and why it was rejected
    candidates: list[Candidate] = Field(default_factory=list)
    concept_id: int | None = None
    status: Status = "proposed"
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    note: str | None = None


class ItemMap(BaseModel):
    kind: Kind
    vocabulary: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_rows: int  # events of this kind (all items)
    listed_rows: int  # events of the listed items
    entries: list[ItemEntry] = Field(default_factory=list)

    def approved(self) -> dict[int, int]:
        return {e.itemid: e.concept_id for e in self.entries
                if e.status == "approved" and e.concept_id}  # fmt: skip

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.status] = out.get(e.status, 0) + 1
        return out

    @property
    def fully_reviewed(self) -> bool:
        return all(e.status not in ("proposed", "no_match") for e in self.entries)


def _specimen_rank(c: Candidate, specimen: str | None) -> int:
    if not specimen:
        return 0
    words = _SPECIMEN.get(norm(specimen), (norm(specimen),))
    return 0 if any(w in norm(c.concept_name) for w in words) else 1


def propose_items(con, kind: Kind, items: list[dict], *, total_rows: int,
                  n_candidates: int = 5) -> ItemMap:  # fmt: skip
    """Proposals for ``items`` (dicts with itemid, label, rows and, for labs, specimen and
    category; for inputs, category and unit), most frequent first."""
    entries = []
    for it in sorted(items, key=lambda x: (-x["rows"], x["itemid"])):
        cands = []
        for domain in DOMAINS[kind]:
            cands += search(con, it["label"], domain, limit=n_candidates * 3)
        cands.sort(key=lambda c: c.rank)
        if kind == "labevents":
            cands.sort(key=lambda c: (c.rank[0], _specimen_rank(c, it.get("specimen")),
                                      c.rank[1:]))  # fmt: skip
        cands = cands[:n_candidates]
        e = ItemEntry(itemid=it["itemid"], label=it["label"], specimen=it.get("specimen"),
                      category=it.get("category"), unit=it.get("unit"),
                      rows=suppress(int(it["rows"])), candidates=cands,
                      curated=it.get("curated"))  # fmt: skip
        if cands:
            e.concept_id = cands[0].concept_id
        else:
            e.status = "no_match"
        entries.append(e)
    vocab = None
    try:
        row = con.execute("SELECT vocabulary_version FROM vocabulary "
                          "WHERE vocabulary_id = 'None'").fetchone()  # fmt: skip
        vocab = row[0] if row else None
    except Exception:  # noqa: BLE001 - VOCABULARY.csv is optional
        pass
    return ItemMap(kind=kind, vocabulary=vocab, total_rows=total_rows,
                   listed_rows=sum(int(it["rows"]) for it in items), entries=entries)


def review_sheet(m: ItemMap) -> str:
    share = 100 * m.listed_rows / m.total_rows if m.total_rows else 0.0
    lines = [f"# Item map for review: {m.kind}", "",
             f"Vocabulary {m.vocabulary or 'unknown'}. {len(m.entries)} items, covering "
             f"{share:.1f}% of {m.kind} rows, most frequent first. Only approved items are "
             "converted. In the decisions file, give each item you review an explicit action "
             "(approve, replace, reject, unmappable); nothing is pre-filled, and items left "
             "blank are not converted. Proposals come from a word search: check each one.", "",
             "| Item | Label | Specimen / category | Rows | Proposed | Alternatives | Status |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip

    def fmt(c: Candidate) -> str:
        return f"{c.concept_id} {c.concept_name} [{c.vocabulary_id}, {c.match}]"

    for e in m.entries:
        ctx = " / ".join(x for x in (e.specimen, e.category, e.unit) if x) or "–"
        if e.curated:
            ctx += f" / **curated map rejected: {e.curated}**"
        prop = fmt(e.candidates[0]) if e.candidates else "–"
        alts = "; ".join(fmt(c) for c in e.candidates[1:4]) or "–"
        lines.append(f"| {e.itemid} | {e.label} | {ctx} | {e.rows} | {prop} | {alts} | "
                     f"{e.status} |")
    return "\n".join(lines) + "\n"


ACTIONS_HELP = ("Fill in 'action' for each item you decide: approve (the proposed concept), "
                "replace (with 'concept_id'), reject (the proposal is wrong), unmappable (no "
                "standard equivalent). Items left blank stay undecided and are not converted.")


def decisions_template(m: ItemMap) -> dict:
    """A decisions file with NOTHING pre-filled: every item needs an explicit decision.
    (Pre-filling "approve" let a whole file of proposals be approved unread; the first real
    proposals mapped "NaCl 0.9%" to linaclotide.) The proposal is shown for reference."""
    return {"reviewer": "", "_actions": ACTIONS_HELP, "decisions": {
        str(e.itemid): {"label": e.label, "action": "",
                        "proposed_concept_id": e.concept_id,
                        "proposed_name": e.candidates[0].concept_name if e.candidates else None,
                        "concept_id": None, "note": ""}
        for e in m.entries if e.status in ("proposed", "no_match")}}  # fmt: skip


class DecisionError(ValueError):
    pass


def apply_decisions(m: ItemMap, decisions: dict) -> ItemMap:
    """approve (the proposal, or the concept_id given), replace (needs concept_id), reject
    (the proposal is wrong), unmappable (no standard equivalent). A blank action leaves the
    item undecided (not converted). Returns a new map."""
    reviewer = (decisions.get("reviewer") or "").strip()
    if not reviewer:
        raise DecisionError("decisions need a reviewer")
    out = m.model_copy(deep=True)
    by_id = {e.itemid: e for e in out.entries}
    now = datetime.now(UTC)
    for key, d in decisions.get("decisions", {}).items():
        e = by_id.get(int(key))
        if e is None:
            raise DecisionError(f"unknown item {key}")
        action, cid = (d.get("action") or "").strip(), d.get("concept_id")
        if not action:
            continue  # undecided: stays proposed / no_match, not converted
        if action == "approve":
            if not (cid or e.concept_id):
                raise DecisionError(f"{key}: nothing to approve")
            e.concept_id, e.status = int(cid or e.concept_id), "approved"
        elif action == "replace":
            if not cid:
                raise DecisionError(f"{key}: replace needs concept_id")
            e.concept_id, e.status = int(cid), "approved"
        elif action in ("reject", "unmappable"):
            e.concept_id, e.status = None, "rejected" if action == "reject" else "unmappable"
        else:
            raise DecisionError(f"{key}: unknown action {action!r}")
        e.reviewer, e.reviewed_at, e.note = reviewer, now, d.get("note") or None
    return out


def item_counts(con, kind: Kind, source_sql: str, dictionary_sql: str,
                *, min_rows: int = SMALL_CELL) -> tuple[list[dict], int]:  # fmt: skip
    """Per-item row counts joined to the item dictionary (one scan of the events table).
    Returns the items with at least ``min_rows`` rows and the total row count."""
    counts = con.execute(f"""SELECT TRY_CAST(itemid AS BIGINT) AS itemid, count(*) AS n
        FROM {source_sql} GROUP BY 1""").fetchall()  # fmt: skip
    total = sum(n for _, n in counts)
    if kind == "labevents":
        meta = {r[0]: r[1:] for r in con.execute(f"""SELECT TRY_CAST(itemid AS BIGINT), label,
            fluid, category FROM {dictionary_sql}""").fetchall()}  # fmt: skip
        items = [{"itemid": i, "rows": n, "label": meta.get(i, ("?",))[0],
                  "specimen": meta.get(i, (None, None, None))[1],
                  "category": meta.get(i, (None, None, None))[2]}
                 for i, n in counts if n >= min_rows and i is not None]  # fmt: skip
    else:
        meta = {r[0]: r[1:] for r in con.execute(f"""SELECT TRY_CAST(itemid AS BIGINT), label,
            category, unitname FROM {dictionary_sql}""").fetchall()}  # fmt: skip
        items = [{"itemid": i, "rows": n, "label": meta.get(i, ("?",))[0],
                  "category": meta.get(i, (None, None, None))[1],
                  "unit": meta.get(i, (None, None, None))[2]}
                 for i, n in counts if n >= min_rows and i is not None]  # fmt: skip
    return items, total


def curated_status(con, concept_map: Path, kind: str) -> dict[int, tuple[str, int, str]]:
    """The curated map's targets checked against the loaded vocabulary, as the ETL checks them:
    {itemid: (label, curated concept_id, status)}, status accepted, remapped or a rejection."""
    from agentic_tte.etl.mimic_omop import ITEM_MAPS, _checked_map

    path = (Path(concept_map) / ITEM_MAPS[kind]).as_posix()
    _checked_map(con, f"curated_{kind}", f"""
        SELECT TRY_CAST(replace(subject_id, 'mimic:', '') AS BIGINT) AS itemid,
               subject_label AS label, predicate_id AS predicate,
               TRY_CAST(replace(object_id, 'omop:', '') AS INTEGER) AS curated_id,
               coalesce(reviewer_id, '') AS reviewer
        FROM read_csv('{path}', header = true, all_varchar = true)
        WHERE object_id LIKE 'omop:%'""")  # fmt: skip
    rows = con.execute(f"SELECT itemid, label, curated_id, status FROM map_curated_{kind}"
                       ).fetchall()  # fmt: skip
    con.execute(f"DROP TABLE map_curated_{kind}")
    return {i: (label, cid, status) for i, label, cid, status in rows}


def with_curated(items: list[dict], all_counts: dict[int, int], labels: dict[int, str],
                 curated: dict[int, tuple[str, int, str]]) -> list[dict]:  # fmt: skip
    """For chartevents/procedureevents: drop items whose curated map works (nothing to review)
    and always list items whose curated target was rejected, whatever their count."""
    out = [it for it in items if curated.get(it["itemid"], ("", 0, "none"))[2]
           not in ("accepted", "remapped")]  # fmt: skip
    listed = {it["itemid"] for it in out}
    for i, (label, cid, status) in curated.items():
        if status in ("accepted", "remapped"):
            continue
        note = f"{cid} ({status})"
        if i in listed:
            next(it for it in out if it["itemid"] == i)["curated"] = note
        elif all_counts.get(i):
            out.append({"itemid": i, "label": labels.get(i, label), "rows": all_counts[i],
                        "curated": note})  # fmt: skip
    return out
