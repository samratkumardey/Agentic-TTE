"""Abbreviations in protocol concepts: "diagnosis of ADHF", "MAP", "AKI".

The vocabulary search (vocab.py) matches words, so an abbreviation finds nothing unless the
vocabulary lists it as a synonym. Expansions come from two sources, and only these:
- the protocol's own text: "acute decompensated heart failure (ADHF)" defines ADHF for that
  trial. Definitions are found with the Schwartz-Hearst algorithm (Schwartz & Hearst 2003),
  are grounded in the registry text, and are used without further approval;
- a curated list (``AbbreviationList``): only entries a named reviewer has approved are used.
  The list shipped with the package (``resources/abbreviations.json``) is a seed: every entry is
  unapproved until someone copies it (``agentic-tte abbreviations init``) and approves it.

Every candidate found through an expansion records the abbreviation, the expansion and its
source, and the review sheet shows them, so the clinician approving a concept set sees that
the match went through an abbreviation. The LLM plays no part here.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, Field

from agentic_tte.protocol.models import Domain, TargetTrialProtocol

PROTOCOL_TEXT = "protocol text"


class Abbreviation(BaseModel):
    abbr: str
    expansions: list[str]
    domain: Domain | None = None  # only for concepts of this domain (disambiguation)
    source: str = "list"  # "list", or "protocol text: <where>"
    approved_by: str | None = None
    approved_at: datetime | None = None
    note: str | None = None

    @property
    def usable(self) -> bool:
        return bool(self.approved_by)


class AbbreviationList(BaseModel):
    description: str | None = None
    entries: list[Abbreviation] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> AbbreviationList:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def seed(cls) -> AbbreviationList:
        text = resources.files("agentic_tte.omop").joinpath("resources/abbreviations.json")
        return cls.model_validate_json(text.read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2, exclude_none=True) + "\n",
                              encoding="utf-8")  # fmt: skip

    def approved(self) -> list[Abbreviation]:
        return [e for e in self.entries if e.usable]

    def approve(self, reviewer: str, only: set[str] | None = None) -> int:
        """Record a reviewer's approval of all entries, or of the abbreviations in ``only``.
        Returns how many entries were newly approved."""
        if not reviewer.strip():
            raise ValueError("approval needs a reviewer")
        now, n = datetime.now(UTC), 0
        keys = {key(a) for a in only} if only else None
        for e in self.entries:
            if (keys is None or key(e.abbr) in keys) and not e.usable:
                e.approved_by, e.approved_at, n = reviewer.strip(), now, n + 1
        return n


def key(abbr: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", abbr.upper())


# --- definitions in text (Schwartz & Hearst 2003) --------------------------------------------

_PAREN = re.compile(r"\(([^()\[\]]{1,60})\)")
_STOP_START = ("i.e", "e.g", "eg ", "ie ", "see ", "or ", "and ")


def _is_short_form(s: str) -> bool:
    s = s.strip()
    return (2 <= len(s) <= 12 and len(s.split()) <= 2 and s[0].isalnum()
            and any(c.isupper() for c in s) and any(c.isalpha() for c in s)
            and not s.lower().startswith(_STOP_START))  # fmt: skip


def _best_long_form(sf: str, lf: str) -> str | None:
    """The shortest end of ``lf`` that contains the characters of ``sf`` in order, the first
    one at the start of a word (Schwartz & Hearst's findBestLongForm)."""
    s, ln = len(sf) - 1, len(lf) - 1
    while s >= 0:
        c = sf[s].lower()
        if not c.isalnum():
            s -= 1
            continue
        while ln >= 0 and (lf[ln].lower() != c or (s == 0 and ln > 0 and lf[ln - 1].isalnum())):
            ln -= 1
        if ln < 0:
            return None
        ln, s = ln - 1, s - 1
    start = lf.rfind(" ", 0, ln + 1) + 1
    out = lf[start:].strip(" ,;:-")
    return out or None


def defined_in_text(text: str) -> list[tuple[str, str]]:
    """(abbreviation, long form) pairs defined in the text: "long form (SF)" and
    "SF (long form)"."""
    found = []
    for m in _PAREN.finditer(text):
        inner = m.group(1).strip()
        before = text[: m.start()].rstrip()
        if _is_short_form(inner):
            words = before.split()
            n = min(len(inner) + 5, 2 * len(inner))
            window = " ".join(words[-n:]) if words else ""
            lf = _best_long_form(inner, window)
            if lf and key(inner) not in key(lf) and len(lf.split()) >= 1 and len(lf) > len(inner):
                found.append((inner, lf))
        elif len(inner.split()) >= 2 and before:
            sf = before.split()[-1].strip(",;:")
            if _is_short_form(sf) and sf.upper() == sf:
                lf = _best_long_form(sf, inner)
                if lf and len(lf.split()) >= 2:
                    found.append((sf, lf))
    return found


def _protocol_texts(protocol: TargetTrialProtocol) -> list[tuple[str, str]]:
    out = [(f"{c.id}", c.text) for c in protocol.eligibility]
    out += [(f"arm {a.label}", a.description or "") for a in protocol.treatment_strategies]
    out += [(f"outcome {o.name}", f"{o.name}. {o.description or ''}") for o in protocol.outcomes]
    # Not the title: its acronym is the trial's name ("STASIS"), never a clinical concept.
    return out


def from_protocol(protocol: TargetTrialProtocol) -> list[Abbreviation]:
    """Abbreviations the protocol's registry text defines, one entry per abbreviation (the
    first definition wins; a conflicting later one is noted)."""
    seen: dict[str, Abbreviation] = {}
    for where, text in _protocol_texts(protocol):
        for sf, lf in defined_in_text(text):
            k = key(sf)
            if k not in seen:
                seen[k] = Abbreviation(abbr=sf, expansions=[lf], approved_by=PROTOCOL_TEXT,
                                       source=f"{PROTOCOL_TEXT}: {where}")  # fmt: skip
            elif lf.lower() not in (x.lower() for x in seen[k].expansions):
                seen[k].note = f"also defined as '{lf}' ({where}); first definition used"
    return list(seen.values())


def expansions_for(name: str, domain: str, abbreviations: list[Abbreviation]
                   ) -> list[tuple[Abbreviation, str, str]]:  # fmt: skip
    """(entry, expansion, name with the abbreviation replaced) for each usable abbreviation
    written in ``name`` as a word, in capitals as defined (so "map" the word is left alone)."""
    out = []
    for a in abbreviations:
        if not a.usable or (a.domain is not None and a.domain != domain):
            continue
        forms = {a.abbr, key(a.abbr)} if key(a.abbr) else {a.abbr}
        for form in forms:
            pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(form)}(?![A-Za-z0-9])")
            if pat.search(name):
                for exp in a.expansions:
                    out.append((a, exp, pat.sub(exp, name)))
                break
    return out


def merged(protocol: TargetTrialProtocol, curated: AbbreviationList | None) -> list[Abbreviation]:
    """Protocol-text definitions first (they are this trial's), then approved list entries
    for abbreviations the protocol does not define."""
    own = from_protocol(protocol)
    keys = {key(a.abbr) for a in own}
    rest = [a for a in (curated.approved() if curated else []) if key(a.abbr) not in keys]
    return own + rest


def summary(abbreviations: list[Abbreviation]) -> list[dict]:
    return [json.loads(a.model_dump_json(exclude_none=True)) for a in abbreviations]
