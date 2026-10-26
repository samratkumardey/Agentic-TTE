"""Reviewed item maps for MIMIC-IV items without a curated map: labevents and inputevents.

MIT-LCP's mimic-code has curated maps for chartevents and procedureevents only. For labs
(d_labitems) and ICU inputs (d_items, linksto = inputevents) this module proposes a standard
concept per item from the vocabulary search on the item's label, and a person approves each
one (approve, replace, reject, unmappable), as for protocol concept sets:

    propose_items(...)   -> ItemMap (proposals, most frequent items first)
    review_sheet(map)    -> markdown for the reviewer
    decisions_template() -> a decisions file to fill in
    apply_decisions(...) -> the reviewed ItemMap the ETL takes (``EtlConfig.item_maps``)

Labs are searched in the Measurement domain, with the specimen ("Blood", "Urine") used to
rank LOINC candidates; inputs in the Drug domain, preferring the ingredient. Only approved
items are converted. The files hold item labels and per-item row counts from the credentialed
data, so they stay where the data are (Hellbender) and never go into git; counts below 11
are suppressed in them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from agentic_tte.omop.gate import SMALL_CELL, suppress
from agentic_tte.omop.vocab import Candidate, norm, search

Kind = Literal["labevents", "inputevents"]
Status = Literal["proposed", "no_match", "approved", "rejected", "unmappable"]
DOMAIN = {"labevents": "Measurement", "inputevents": "Drug"}

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
        cands = search(con, it["label"], DOMAIN[kind], limit=n_candidates * 3)
        if kind == "labevents":
            cands.sort(key=lambda c: (c.rank[0], _specimen_rank(c, it.get("specimen")),
                                      c.rank[1:]))  # fmt: skip
        cands = cands[:n_candidates]
        e = ItemEntry(itemid=it["itemid"], label=it["label"], specimen=it.get("specimen"),
                      category=it.get("category"), unit=it.get("unit"),
                      rows=suppress(int(it["rows"])), candidates=cands)  # fmt: skip
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
             "converted. Fill in the decisions file.", "",
             "| Item | Label | Specimen / category | Rows | Proposed | Alternatives | Status |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip

    def fmt(c: Candidate) -> str:
        return f"{c.concept_id} {c.concept_name} [{c.vocabulary_id}, {c.match}]"

    for e in m.entries:
        ctx = " / ".join(x for x in (e.specimen, e.category, e.unit) if x) or "–"
        prop = fmt(e.candidates[0]) if e.candidates else "–"
        alts = "; ".join(fmt(c) for c in e.candidates[1:4]) or "–"
        lines.append(f"| {e.itemid} | {e.label} | {ctx} | {e.rows} | {prop} | {alts} | "
                     f"{e.status} |")
    return "\n".join(lines) + "\n"


def decisions_template(m: ItemMap) -> dict:
    return {"reviewer": "", "decisions": {
        str(e.itemid): {"action": "approve" if e.status == "proposed" else "unmappable",
                        "concept_id": e.concept_id, "note": ""}
        for e in m.entries if e.status in ("proposed", "no_match")}}  # fmt: skip


class DecisionError(ValueError):
    pass


def apply_decisions(m: ItemMap, decisions: dict) -> ItemMap:
    """approve (the proposal, or the concept_id given), replace (needs concept_id), reject
    (the proposal is wrong), unmappable (no standard equivalent). Returns a new map."""
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
        action, cid = d.get("action"), d.get("concept_id")
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
