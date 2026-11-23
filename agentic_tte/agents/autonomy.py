"""Rules for the autonomous arm (SCORING.md Phase 3 item 34): the choices a reviewer makes at
concept review, made instead by fixed rules, recorded and labelled.

- ``choose_in_data`` (rules ``data-v1``): a proposed concept with no records in the data is
  replaced by the vocabulary candidate with the most records; with none, the set is unmappable
  (its criterion is reported as not compiled). A treatment given intravenously keeps only the
  injectable forms of the drug that have records. Counts are whole-database aggregates, never
  split by arm or outcome; concept IDs come only from the vocabulary search.
- ``time_zero_spec`` (rules ``tz-v1``): time zero is the first ICU admission (the first
  inpatient visit for a trial that never mentions intensive care), then waits for the
  time-stamped inclusion events: each positive drug, procedure or device atom, and each
  measurement atom with a numeric threshold, of an inclusion criterion whose atoms must all
  hold; for a criterion met by any of its atoms, the first of its drug, procedure or device
  atoms (its measurement alternatives are then conditions at time zero). Named anchors map to
  one of these events, or to hospital or ICU admission, or are left out.

No LLM. The rules read the protocol, the vocabulary and record counts.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from agentic_tte.omop.compiler import EVENT_TABLE, T0_ALIASES, _in_set, spec_template
from agentic_tte.omop.compiler import _norm as cnorm
from agentic_tte.omop.concept_sets import ConceptSetBook
from agentic_tte.omop.gate import concept_key, suppress
from agentic_tte.protocol.models import Domain, TargetTrialProtocol

DATA_RULES = "data-v1"
TZ_RULES = "tz-v1"
# Standard OMOP visit concepts, checked by name against the vocabulary (``check_visits``).
ICU_VISIT, INPATIENT_VISIT = 32037, 9201
VISIT_NAMES = {ICU_VISIT: "intensive care", INPATIENT_VISIT: "inpatient visit"}
_ICU_WORDS = re.compile(r"\b(icu|intensive care|critical care|critically ill)\b", re.I)
_IV = re.compile(r"\b(iv|i\.v\.|intravenous(ly)?|infusion|bolus|injection)\b", re.I)
_INJECTABLE = ("%inject%", "%intravenous%", "%infusion%")
_LOOK_AHEAD = {"x_after_y", "x_within_t_after_y"}
_EVENT_DOMAINS = {Domain.DRUG, Domain.PROCEDURE, Domain.DEVICE, Domain.MEASUREMENT}


class AutonomyNotes(BaseModel):
    rules: str
    notes: list[str]

    def to_markdown(self, title: str) -> str:
        return "\n".join([f"## {title} ({self.rules})", "", "Fixed rules, no person.", ""]
                         + [f"- {n}" for n in self.notes or ["nothing to change"]]) + "\n"


def check_visits(con) -> None:
    """The hand-chosen visit concepts must name what the rules mean (CLAUDE.md)."""
    for cid, want in VISIT_NAMES.items():
        row = con.execute("SELECT concept_name FROM concept WHERE concept_id = ?", [cid]).fetchone()
        if row is None or want not in row[0].lower():
            raise ValueError(f"concept {cid} is {row[0] if row else 'missing'!r}, not {want!r}")


# --- concept choice from record counts ----------------------------------------------------------


def _records(con, wanted: dict[str, list[int]], domain: Domain) -> dict[int, int]:
    """Records per candidate concept (with its descendants) in the domain's table."""
    ids = sorted({i for v in wanted.values() for i in v})
    if not ids or domain not in EVENT_TABLE:
        return {}
    table, col, _, _ = EVENT_TABLE[domain]
    values = ", ".join(f"({i})" for i in ids)
    sql = f"""
WITH cand(cid) AS (VALUES {values}),
x AS (SELECT cid, cid AS did FROM cand
      UNION SELECT a.ancestor_concept_id, a.descendant_concept_id FROM concept_ancestor a
      JOIN cand ON a.ancestor_concept_id = cand.cid)
SELECT x.cid, count(*) FROM {table} r JOIN x ON r.{col} = x.did GROUP BY x.cid"""
    return {int(c): int(n) for c, n in con.execute(sql).fetchall()}


def _injectable(con, ids: list[int]) -> list[int]:
    """Drug concepts under ``ids`` whose name is an injectable form and that have records."""
    like = " OR ".join(f"lower(c.concept_name) LIKE '{p}'" for p in _INJECTABLE)
    sql = f"""
SELECT c.concept_id FROM concept c
WHERE ({_in_set('c.concept_id', ids, True)}) AND ({like})
  AND EXISTS (SELECT 1 FROM drug_exposure d WHERE d.drug_concept_id = c.concept_id)
ORDER BY c.concept_id"""
    return [int(r[0]) for r in con.execute(sql).fetchall()]


def _route(protocol: TargetTrialProtocol, key: str) -> str | None:
    for arm in protocol.treatment_strategies:
        for comp in arm.components:
            if key == f"T:{arm.label}:{comp.concept.name}":
                return " ".join(x for x in (comp.route, comp.dose) if x) or None
    return None


def choose_in_data(con, book: ConceptSetBook, protocol: TargetTrialProtocol
                   ) -> tuple[ConceptSetBook, AutonomyNotes]:  # fmt: skip
    """The book with every open proposal moved to a concept that has records (rules data-v1)."""
    out = book.model_copy(deep=True)
    notes: list[str] = []
    open_sets = [s for s in out.sets if s.status in ("proposed", "no_match") and s.candidates
                 and Domain(s.domain) in EVENT_TABLE]  # fmt: skip
    counts: dict[int, int] = {}
    for dom in {Domain(s.domain) for s in open_sets}:
        counts |= _records(con, {s.key: [c.concept_id for c in s.candidates]
                                 for s in open_sets if Domain(s.domain) == dom}, dom)  # fmt: skip
    for s in open_sets:
        top = s.candidates[0]
        have = [c for c in s.candidates if counts.get(c.concept_id, 0) > 0]
        if counts.get(top.concept_id, 0) > 0:
            s.concept_ids, s.status = [top.concept_id], "proposed"
        elif have:
            best = max(have, key=lambda c: (counts[c.concept_id], -s.candidates.index(c)))
            s.concept_ids, s.status = [best.concept_id], "proposed"
            s.note = (f"data rule: {top.concept_id} {top.concept_name} has no records; "
                      f"{best.concept_id} {best.concept_name} has "
                      f"{suppress(counts[best.concept_id])}")  # fmt: skip
            notes.append(f"`{s.key}`: {s.note}")
        else:
            s.concept_ids, s.status = [], "no_match"
            s.note = "data rule: no vocabulary candidate has records in the data"
            notes.append(f"`{s.key}`: {s.note}; criteria that use it are not compiled")
            continue
        route = _route(protocol, s.key) if s.scope == "treatment" else None
        if route and _IV.search(route) and Domain(s.domain) == Domain.DRUG:
            inj = _injectable(con, s.concept_ids)
            if inj:
                s.concept_ids, s.include_descendants = inj, True
                msg = f"route '{route}': only the injectable forms with records ({len(inj)})"
            else:
                msg = f"route '{route}': no injectable form has records; route not restricted"
            s.note = "; ".join(x for x in (s.note, f"data rule: {msg}") if x)
            notes.append(f"`{s.key}`: {msg}")
    return out, AutonomyNotes(rules=DATA_RULES, notes=notes)


# --- time zero ----------------------------------------------------------------------------------


def time_zero_spec(protocol: TargetTrialProtocol,
                   sets: dict[str, list[int]]) -> tuple[dict, AutonomyNotes]:  # fmt: skip
    """A compile spec (time zero, the events it waits for, anchors) from the protocol and the
    approved concept sets (key -> IDs), by rules tz-v1."""
    icu = bool(_ICU_WORDS.search(protocol.model_dump_json()))
    base = ICU_VISIT if icu else INPATIENT_VISIT
    notes = [f"time zero starts at the first {'ICU admission' if icu else 'inpatient visit'} "
             f"(Visit {base})"]  # fmt: skip
    treatment = {i for k, v in sets.items() if k.startswith("T:") for i in v}
    after: list[dict] = []
    for crit in protocol.eligibility:
        if crit.kind != "inclusion" or not crit.emulable:
            continue
        either = len(crit.atoms) > 1 and crit.atom_logic == "any"
        found = []  # (domain, concepts, value)
        for atom in crit.atoms:
            t = atom.temporal
            if atom.negated or (t is not None and (str(t.relation) in _LOOK_AHEAD or (
                    cnorm(t.anchor) not in T0_ALIASES and cnorm(t.anchor) != "time zero"))):
                continue  # fmt: skip
            doms = {Domain(c.domain) for c in atom.concepts}
            if len(doms) != 1 or (dom := doms.pop()) not in _EVENT_DOMAINS:
                continue
            numeric = atom.value is not None and atom.value.value is not None
            if (dom == Domain.MEASUREMENT) != numeric or (atom.value and not numeric):
                continue  # measurements need a threshold; other events no value
            if either and dom == Domain.MEASUREMENT:
                continue  # an alternative condition, read at time zero
            if either or atom.concept_logic == "any" or len(atom.concepts) == 1:
                found.append((dom, list(atom.concepts), atom.value if numeric else None))
            else:
                found += [(dom, [c], None) for c in atom.concepts]
        if either:  # the first of the alternatives, one event per domain
            merged: dict[Domain, list] = {}
            for dom, concepts, _ in found:
                merged.setdefault(dom, []).extend(concepts)
            if len(merged) > 1:
                notes.append(f"{crit.id}: alternatives in several domains, not a time-zero event")
                continue
            found = [(dom, concepts, None) for dom, concepts in merged.items()]
        for dom, group, v in found:
            ids = sorted({i for c in group for i in sets.get(concept_key(crit, c.name), [])})
            names = ", ".join(c.name for c in group)
            if not ids:
                notes.append(f"{crit.id} ({names}): no approved concept set, not used")
                continue
            if set(ids) & treatment:
                notes.append(f"{crit.id} ({names}): a treatment drug, not used")
                continue
            after.append({"domain": dom.value, "concept_ids": ids, "occurrence": "first",
                          "note": f"{crit.id}: {names}",
                          "value": v.model_dump(mode="json") if v else None})  # fmt: skip
            notes.append(f"time zero waits for {crit.id} ({names}"
                         + (f" {v.comparator} {v.value:g}{' ' + v.unit if v.unit else ''}"
                            if v else "") + ")")  # fmt: skip
    if not after:
        notes.append("no time-stamped inclusion event: time zero is the admission itself")
    anchors = {}
    for name in spec_template(protocol)["anchors"]:
        n = cnorm(name)
        event = next((e for e in after if e["value"] is None and set(n.split()) & {
            w for w in cnorm(e["note"].split(": ", 1)[1]).split() if len(w) >= 4}), None)
        if event is not None:
            anchors[name] = {**{k: event[k] for k in ("domain", "concept_ids")},
                             "occurrence": "last", "note": f"tz-v1: {event['note']}"}
            notes.append(f"anchor '{name}': the last {event['note']} up to time zero")
            continue
        cid = (ICU_VISIT if re.search(r"\b(icu|intensive care)\b", n) else INPATIENT_VISIT
               if re.search(r"\b(admission|admitted|hospitali[sz]ation|hospital)\b", n)
               else None)  # fmt: skip
        if cid is None:
            notes.append(f"anchor '{name}': no rule; criteria that use it are not compiled")
            continue
        anchors[name] = {"domain": "Visit", "concept_ids": [cid], "occurrence": "last",
                         "note": f"tz-v1: {'ICU' if cid == ICU_VISIT else 'hospital'} admission"}
        notes.append(f"anchor '{name}': the last {'ICU' if cid == ICU_VISIT else 'hospital'} "
                     "admission up to time zero")
    spec = {"time_zero": {"domain": "Visit", "concept_ids": [base], "occurrence": "first",
                          "note": f"{TZ_RULES}: {'ICU admission' if icu else 'inpatient visit'}"},
            "time_zero_after": after, "time_zero_window": {"amount": 72, "unit": "hour"},
            "anchors": anchors, "measurement_lookback": {"amount": 24, "unit": "hour"},
            "strict": False}  # fmt: skip
    return spec, AutonomyNotes(rules=TZ_RULES, notes=notes)
