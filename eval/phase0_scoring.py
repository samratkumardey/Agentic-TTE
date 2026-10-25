"""Scoring primitives for the Phase 0 exit run, as pre-registered in eval/SCORING.md.

Concept matching (part B): within a criterion, predicted and reference concepts are matched
one-to-one. Match keys are exact after normalization, so maximum bipartite matching reduces to
multiset intersection.
  relaxed: (normalized name, domain)
  strict:  relaxed + the concept's atom-level value and time constraint
The same functions compare a run with the gold standard, with another replicate (part D) or
with another model.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_tte.protocol.models import (
    CriterionAtom,
    TargetTrialProtocol,
    TemporalConstraint,
    ValueConstraint,
)
from agentic_tte.protocol.normalize import normalize_anchor

SYNONYMS_FILE = Path(__file__).parent / "gold" / "synonyms.json"


def load_synonyms(path: Path = SYNONYMS_FILE) -> dict[str, str]:
    """Adjudicated synonym table {variant: canonical}, normalized; empty until gold exists."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {norm_text(k): norm_text(v) for k, v in raw.items()}


def norm_text(s: str | None) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def norm_name(name: str, synonyms: dict[str, str] | None = None) -> str:
    n = norm_text(name)
    return (synonyms or {}).get(n, n)


def _unit(u: str | None) -> str | None:
    if not u:
        return None
    return norm_text(u).rstrip("s") or None


def value_key(v: ValueConstraint | None) -> tuple | None:
    if v is None:
        return None
    if v.comparator == "present":
        return ("present", norm_text(v.text))
    return (v.comparator, v.value, v.upper, _unit(v.unit))


def temporal_key(t: TemporalConstraint | None) -> tuple | None:
    if t is None:
        return None
    window = (t.window.amount, t.window.unit) if t.window else None
    return (t.relation, window, norm_text(normalize_anchor(t.anchor)))


def atom_concept_keys(atom: CriterionAtom, strict: bool, synonyms: dict[str, str] | None) -> list:
    keys = []
    for c in atom.concepts:
        key: tuple = (norm_name(c.name, synonyms), c.domain)
        if strict:
            key += (value_key(atom.value), temporal_key(atom.temporal))
        keys.append(key)
    return keys


def criterion_keys(criterion, strict: bool, synonyms: dict[str, str] | None = None) -> Counter:
    return Counter(k for a in criterion.atoms for k in atom_concept_keys(a, strict, synonyms))


@dataclass
class PRF:
    tp: int = 0
    pred: int = 0
    ref: int = 0

    def add(self, pred: Counter, ref: Counter) -> None:
        self.tp += sum((pred & ref).values())
        self.pred += sum(pred.values())
        self.ref += sum(ref.values())

    def __iadd__(self, other: PRF) -> PRF:
        self.tp += other.tp
        self.pred += other.pred
        self.ref += other.ref
        return self

    @property
    def precision(self) -> float | None:
        return self.tp / self.pred if self.pred else None

    @property
    def recall(self) -> float | None:
        return self.tp / self.ref if self.ref else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None if p is None or r is None else 0.0
        return 2 * p * r / (p + r)


def concept_prf(pred: TargetTrialProtocol, ref: TargetTrialProtocol, *, strict: bool,
                synonyms: dict[str, str] | None = None) -> PRF:  # fmt: skip
    """Micro P/R/F1 of ``pred`` against ``ref`` over criteria aligned by id. Both come from the
    same deterministic split; a text mismatch means they do not, and is an error."""
    ref_by_id = {c.id: c for c in ref.eligibility}
    out = PRF()
    for c in pred.eligibility:
        r = ref_by_id.get(c.id)
        if r is None or r.text != c.text:
            raise ValueError(f"criterion {c.id} does not align between protocols")
        out.add(criterion_keys(c, strict, synonyms), criterion_keys(r, strict, synonyms))
    missing = set(ref_by_id) - {c.id for c in pred.eligibility}
    for cid in missing:  # criteria absent from pred count as false negatives
        out.add(Counter(), criterion_keys(ref_by_id[cid], strict, synonyms))
    return out


# --- substantive identity (part D) -------------------------------------------------------------

# Free-text and bookkeeping fields: they can differ without changing who is eligible, what is
# compared or how follow-up runs.
_FREE_TEXT = {"rationale", "evidence", "notes", "source_text", "causal_contrast_evidence",
              "grace_period_evidence", "description", "text", "complexity"}  # fmt: skip


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {k: _strip(v) for k, v in obj.items() if k not in _FREE_TEXT}
        # "any" vs "all" over a single atom or concept selects the same patients.
        if "atoms" in out and len(out["atoms"]) < 2:
            out.pop("atom_logic", None)
        if "concepts" in out and len(out["concepts"]) < 2:
            out.pop("concept_logic", None)
        return out
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


ELEMENTS = ("eligibility", "treatment_strategies", "assignment", "time_zero", "follow_up",
            "outcomes", "causal_contrast", "analysis_plan")  # fmt: skip


def element_projection(p: TargetTrialProtocol, element: str) -> Any:
    data = p.model_dump(mode="json")
    if element == "causal_contrast":
        return [data["causal_contrast"], data["causal_contrast_origin"]]
    if element == "time_zero" and data["time_zero"]:
        tz = data["time_zero"]
        return {"anchor_event": norm_text(tz["anchor_event"]), "offset": tz["offset"],
                "origin": tz["origin"]}  # fmt: skip
    return _strip(data[element])


def element_identity(a: TargetTrialProtocol, b: TargetTrialProtocol) -> dict[str, bool]:
    return {e: element_projection(a, e) == element_projection(b, e) for e in ELEMENTS}


def criterion_identity(a: TargetTrialProtocol, b: TargetTrialProtocol) -> tuple[int, int, list]:
    """(substantively identical criteria, criteria compared, list of atom_logic flips)."""
    b_by_id = {c.id: c for c in b.eligibility}
    same, flips = 0, []
    for c in a.eligibility:
        d = b_by_id.get(c.id)
        if d is None:
            continue
        same += _strip(c.model_dump(mode="json")) == _strip(d.model_dump(mode="json"))
        if c.atom_logic != d.atom_logic and len(c.atoms) > 1 and len(d.atoms) > 1:
            flips.append((c.id, c.atom_logic, d.atom_logic))
    return same, len(a.eligibility), flips


def load_protocols(run_dir: Path) -> dict[str, TargetTrialProtocol]:
    out = {}
    for f in sorted(run_dir.glob("NCT*.protocol.json")):
        p = TargetTrialProtocol.model_validate_json(f.read_text(encoding="utf-8"))
        out[p.source.nct_id] = p
    return out


def mean(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None
