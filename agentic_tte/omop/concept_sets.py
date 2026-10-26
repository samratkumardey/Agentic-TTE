"""Concept sets for a protocol, proposed from the vocabulary and approved by a clinician.

Workflow (Informatician + clinician review gate, PLAN.md §3):
1. ``propose(con, protocol)``: one entry per protocol concept (eligibility, treatment
   components, confounders and outcome concepts), with vocabulary candidates
   (``vocab.search``) and a proposed concept set.
2. ``review_sheet(book)``: a table the clinician reviews; ``decisions_template(book)``: a JSON
   file they fill in, one decision per entry: approve, replace, reject, unmappable, surrogate.
3. ``apply_decisions(book, decisions)``: records each decision with reviewer and note.
4. ``book.gate_sets()``: approved sets only, as ``run_gate(concept_sets=...)`` takes them.
   ``apply_to_protocol(protocol, book)``: writes the mappings into a new protocol version.

Nothing is used unreviewed unless a caller asks for it explicitly (``include_proposed``), which
benchmark runs may do and must report.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from agentic_tte.omop.gate import concept_key
from agentic_tte.omop.vocab import Candidate, search
from agentic_tte.protocol.models import Domain, MappingStatus, TargetTrialProtocol
from agentic_tte.protocol.versioning import new_version

SetStatus = Literal["proposed", "no_match", "approved", "rejected", "unmappable", "surrogate",
                    "not_needed"]  # fmt: skip
Action = Literal["approve", "replace", "reject", "unmappable", "surrogate"]


class ConceptSet(BaseModel):
    key: str  # "I1:septic shock", "T:<arm>:<component>", "C:<confounder>", "O:<outcome>"
    scope: Literal["eligibility", "treatment", "confounder", "outcome"]
    criterion_text: str
    concept_name: str
    domain: str
    candidates: list[Candidate] = Field(default_factory=list)
    concept_ids: list[int] = Field(default_factory=list)  # the set (descendants added later)
    include_descendants: bool = True
    status: SetStatus = "proposed"
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    note: str | None = None


class ConceptSetBook(BaseModel):
    protocol_id: str
    protocol_version: int
    vocabulary: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sets: list[ConceptSet] = Field(default_factory=list)

    def get(self, key: str) -> ConceptSet:
        return next(s for s in self.sets if s.key == key)

    def gate_sets(self, *, include_proposed: bool = False) -> dict[str, list[int]]:
        ok = {"approved", "surrogate"} | ({"proposed"} if include_proposed else set())
        return {s.key: s.concept_ids for s in self.sets if s.status in ok and s.concept_ids}

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.sets:
            out[s.status] = out.get(s.status, 0) + 1
        return out

    @property
    def fully_reviewed(self) -> bool:
        return all(s.status not in ("proposed", "no_match") for s in self.sets)


def _entries(protocol: TargetTrialProtocol):
    for crit in protocol.eligibility:
        for atom in crit.atoms:
            for c in atom.concepts:
                yield concept_key(crit, c.name), "eligibility", crit, c
    for arm in protocol.treatment_strategies:
        for comp in arm.components:
            yield f"T:{arm.label}:{comp.concept.name}", "treatment", arm, comp.concept
    for c in protocol.assignment.confounders:
        yield f"C:{c.name}", "confounder", None, c
    for o in protocol.outcomes:
        if o.concept is not None:
            yield f"O:{o.name}", "outcome", o, o.concept


def propose(con, protocol: TargetTrialProtocol, *, n_candidates: int = 5) -> ConceptSetBook:
    """Candidates and a proposed set for every protocol concept. The proposal is the best
    candidate; it is only a suggestion until a reviewer approves it."""
    vocab = None
    try:
        vocab = con.execute("SELECT vocabulary_version FROM cdm_source LIMIT 1").fetchone()[0]
    except Exception:  # noqa: BLE001 - cdm_source is optional
        pass
    book = ConceptSetBook(protocol_id=protocol.protocol_id, protocol_version=protocol.version,
                          vocabulary=vocab)  # fmt: skip
    seen = set()
    for key, scope, owner, concept in _entries(protocol):
        if key in seen:
            continue
        seen.add(key)
        if scope == "eligibility":
            text = owner.text
        elif scope == "treatment":
            text = owner.description or owner.label
        else:
            text = owner.name if scope == "outcome" else "confounder (assignment)"
        entry = ConceptSet(key=key, scope=scope, criterion_text=text,
                           concept_name=concept.name, domain=concept.domain)  # fmt: skip
        if scope == "eligibility" and not owner.emulable:
            entry.status, entry.note = "not_needed", "administrative criterion: not emulated"
        elif concept.domain == Domain.DEMOGRAPHIC:
            entry.status, entry.note = "not_needed", "demographic: read from the person table"
        else:
            entry.candidates = search(con, concept.name, concept.domain, limit=n_candidates)
            if entry.candidates:
                entry.concept_ids = [entry.candidates[0].concept_id]
                entry.status = "proposed"
            else:
                entry.status = "no_match"
        book.sets.append(entry)
    return book


def review_sheet(book: ConceptSetBook) -> str:
    """Markdown for the clinician: every entry with its proposal and alternatives."""
    lines = [f"# Concept sets for review: {book.protocol_id} (protocol v{book.protocol_version})",
             "", f"Vocabulary: {book.vocabulary or 'unknown'}. Fill in the decisions file; "
             "nothing is used until approved.", "",
             "| Key | Criterion | Concept (domain) | Proposed | Alternatives | Status |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for s in book.sets:
        def fmt(c: Candidate) -> str:
            via = f" via {c.via_name} ({c.via})" if c.via else ""
            return f"{c.concept_id} {c.concept_name} [{c.vocabulary_id}, {c.match}{via}]"

        proposed = fmt(s.candidates[0]) if s.candidates else "–"
        alts = "; ".join(fmt(c) for c in s.candidates[1:4]) or "–"
        lines.append(f"| `{s.key}` | {s.criterion_text[:80]} | {s.concept_name} ({s.domain}) | "
                     f"{proposed} | {alts} | {s.status} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def decisions_template(book: ConceptSetBook) -> dict:
    """A decisions file to fill in: one entry per set that needs a decision."""
    return {
        "reviewer": "",
        "decisions": {
            s.key: {"action": "approve" if s.status == "proposed" else "unmappable",
                    "concept_ids": s.concept_ids, "note": ""}  # fmt: skip
            for s in book.sets if s.status in ("proposed", "no_match")
        },
    }


class DecisionError(ValueError):
    pass


def apply_decisions(book: ConceptSetBook, decisions: dict) -> ConceptSetBook:
    """Apply a filled-in decisions file and return the updated book (the input is unchanged).
    approve: keep the proposed IDs (or the IDs given); replace: use the IDs given; reject: the
    proposal is wrong and nothing replaces it; unmappable: no RWD equivalent, reported as such;
    surrogate: IDs of a clinically justified proxy, with a note saying why."""
    reviewer = (decisions.get("reviewer") or "").strip()
    if not reviewer:
        raise DecisionError("decisions need a reviewer")
    out = book.model_copy(deep=True)
    now = datetime.now(UTC)
    for key, d in decisions.get("decisions", {}).items():
        try:
            s = out.get(key)
        except StopIteration as exc:
            raise DecisionError(f"unknown key {key!r}") from exc
        action: Action = d.get("action")
        ids = [int(i) for i in d.get("concept_ids") or []]
        note = d.get("note") or None
        if action == "approve":
            if not (ids or s.concept_ids):
                raise DecisionError(f"{key}: nothing to approve")
            s.concept_ids, s.status = ids or s.concept_ids, "approved"
        elif action == "replace":
            if not ids:
                raise DecisionError(f"{key}: replace needs concept_ids")
            s.concept_ids, s.status = ids, "approved"
        elif action == "surrogate":
            if not ids or not note:
                raise DecisionError(f"{key}: a surrogate needs concept_ids and a note")
            s.concept_ids, s.status = ids, "surrogate"
        elif action in ("reject", "unmappable"):
            s.concept_ids, s.status = [], "rejected" if action == "reject" else "unmappable"
        else:
            raise DecisionError(f"{key}: unknown action {action!r}")
        s.reviewer, s.reviewed_at, s.note = reviewer, now, note
    return out


def apply_to_protocol(protocol: TargetTrialProtocol, book: ConceptSetBook,
                      created_by: str = "informatician") -> TargetTrialProtocol:  # fmt: skip
    """New protocol version with reviewed mappings on each ConceptRef: approved -> mapped
    (first concept ID and its name), surrogate -> surrogate, unmappable -> unmappable."""
    by_key = {s.key: s for s in book.sets}
    eligibility = [c.model_copy(deep=True) for c in protocol.eligibility]
    arms = [a.model_copy(deep=True) for a in protocol.treatment_strategies]
    assignment = protocol.assignment.model_copy(deep=True)
    outcomes = [o.model_copy(deep=True) for o in protocol.outcomes]

    def update(ref, s: ConceptSet | None) -> None:
        if s is None:
            return
        if s.status in ("approved", "surrogate") and s.concept_ids:
            ref.omop_concept_id = s.concept_ids[0]
            match = next((c for c in s.candidates if c.concept_id == s.concept_ids[0]), None)
            ref.standard_name = match.concept_name if match else None
            ref.status = (MappingStatus.MAPPED if s.status == "approved"
                          else MappingStatus.SURROGATE).value
        elif s.status == "unmappable":
            ref.status = MappingStatus.UNMAPPABLE.value

    for crit in eligibility:
        for atom in crit.atoms:
            for c in atom.concepts:
                update(c, by_key.get(concept_key(crit, c.name)))
    for arm in arms:
        for comp in arm.components:
            update(comp.concept, by_key.get(f"T:{arm.label}:{comp.concept.name}"))
    for c in assignment.confounders:
        update(c, by_key.get(f"C:{c.name}"))
    for o in outcomes:
        if o.concept is not None:
            update(o.concept, by_key.get(f"O:{o.name}"))
    reviewers = sorted({s.reviewer for s in book.sets if s.reviewer})
    return new_version(protocol, created_by=created_by,
                       notes=f"concept sets reviewed by {', '.join(reviewers) or 'nobody'}; "
                             f"{json.dumps(book.summary())}",
                       eligibility=[c.model_dump() for c in eligibility],
                       treatment_strategies=[a.model_dump() for a in arms],
                       assignment=assignment.model_dump(),
                       outcomes=[o.model_dump() for o in outcomes])  # fmt: skip
