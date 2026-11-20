"""Clinician surrogate answers: a stand-in variable for each protocol concept the vocabulary
could not map (PLAN.md §3, §17.4; the Clinician answering the Informatician).

When concept search finds nothing for an eligibility concept or a confounder ("plateau
pressure", "SAPS II"), the run would report it as unmappable and leave it out. The Clinician
proposes a baseline variable that is routinely recorded and carries the same clinical
information, with a rationale and the limitation of the substitution, or says there is none.

What is deterministic and what uses the LLM:
- LLM (one schema-constrained call for all unmapped concepts): the substitute's name, domain,
  rationale and limitation. It never gives concept IDs.
- deterministic: which concepts are asked about (no vocabulary match, eligibility or
  confounder scope; treatment and outcome concepts change the estimand and stay with the
  reviewer); the checks (the substitute is not the treatment, the primary outcome, the
  unmapped concept itself, or a variable measured after time zero); and the concept IDs, which
  come only from the vocabulary search for the substitute's name.

Each surviving proposal becomes a ``surrogate`` decision in the concept-review template, with
the rationale and limitation in its note. Nothing is used until the reviewer keeps it;
``AutoApprove`` turns it into ``unmappable`` unless its policy accepts surrogates, and says so.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentic_tte.agents.clinician import _POST, _norm, _summary
from agentic_tte.llm.base import LLMClient, Message
from agentic_tte.omop.concept_sets import ConceptSet, ConceptSetBook
from agentic_tte.omop.vocab import Candidate, search
from agentic_tte.protocol.models import Domain, TargetTrialProtocol

PROMPT_VERSION = "surrogates-v1"
SCOPES = ("eligibility", "confounder")


class SurrogateProposal(BaseModel):
    key: str = Field(max_length=200)
    substitute: str | None = Field(default=None, max_length=80)  # None: no surrogate
    domain: Domain | None = None
    rationale: str = Field(default="", max_length=300)
    limitation: str = Field(default="", max_length=300)


class SurrogateAnswer(BaseModel):
    proposals: list[SurrogateProposal] = Field(default_factory=list, max_length=40)


class SurrogateItem(BaseModel):
    key: str
    concept_name: str
    scope: str
    substitute: str | None = None
    domain: str | None = None
    rationale: str = ""
    limitation: str = ""
    candidates: list[Candidate] = Field(default_factory=list)
    status: Literal["suggested", "no_vocabulary_match", "none", "dropped"]
    note: str = ""


class SurrogateSheet(BaseModel):
    nct_id: str | None = None
    model: str
    model_digest: str | None = None
    prompt_version: str = PROMPT_VERSION
    items: list[SurrogateItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def decisions(self) -> dict[str, dict]:
        """Concept-review decisions for the suggested surrogates (the vocabulary's best match
        for the substitute's name), to be kept, changed or replaced by the reviewer."""
        return {i.key: {"action": "surrogate", "concept_ids": [i.candidates[0].concept_id],
                        "note": f"Clinician ({self.model}) proposes '{i.substitute}' "
                                f"({i.candidates[0].concept_name}): {i.rationale} "
                                f"Limitation: {i.limitation or 'not stated'}"}
                for i in self.items if i.status == "suggested"}  # fmt: skip

    def to_markdown(self) -> str:
        lines = [f"## Clinician surrogates ({self.model}, {self.prompt_version})", "",
                 "Proposals for concepts the vocabulary could not map. Concept IDs come from the "
                 "vocabulary search for the substitute's name; nothing is used until the "
                 "reviewer keeps it.", "",
                 "| Key | Unmapped concept | Substitute | Vocabulary match | Status | Rationale "
                 "| Limitation |", "|---|---|---|---|---|---|---|"]  # fmt: skip
        for i in self.items:
            match = (f"{i.candidates[0].concept_id} {i.candidates[0].concept_name}"
                     if i.candidates else "–")  # fmt: skip
            status = i.status + (f" ({i.note})" if i.note else "")
            lines.append(f"| `{i.key}` | {i.concept_name} | {i.substitute or '–'} | {match} | "
                         f"{status} | {i.rationale or '–'} | {i.limitation or '–'} |")
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


SYSTEM = """You are the Clinician agent in a target trial emulation system. Some concepts of \
the trial protocol have no match in the health-record vocabulary. For each one, propose a \
surrogate: a baseline variable that is routinely recorded in intensive care and hospital \
records and carries the same clinical information, or say there is none.

Rules:
1. Answer every key you are given, once, with the key exactly as given.
2. substitute: the name of one variable as a clinician would look it up in the record \
("positive end-expiratory pressure", "SOFA score", "norepinephrine"), with exactly one \
domain: Condition, Drug, Procedure, Device, Measurement, Observation or Visit. Use null when \
no recorded variable is a fair stand-in.
3. The surrogate must be measured at or before time zero. Never the trial's treatment or its \
primary outcome, and never the unmapped concept's own name.
4. rationale: one sentence on why it carries the same information. limitation: one sentence \
on what is lost or how the substitution could bias the emulation.
Return only the JSON object."""


def unmapped(book: ConceptSetBook) -> list[ConceptSet]:
    """Concepts the vocabulary found nothing for, in the scopes a surrogate may stand in for."""
    return [s for s in book.sets if s.status == "no_match" and s.scope in SCOPES]


def request(p: TargetTrialProtocol, sets: list[ConceptSet]) -> list[Message]:
    asks = "\n".join(f"- key {s.key}: {s.concept_name} ({s.domain}; {s.scope}; "
                     f"{s.criterion_text[:160]})" for s in sets)  # fmt: skip
    return [Message(role="system", content=SYSTEM),
            Message(role="user", content=f"Protocol:\n{_summary(p)}\n\nUnmapped concepts:\n"
                                         f"{asks}")]  # fmt: skip


def excluded(p: TargetTrialProtocol, name: str, own: str) -> str | None:
    """Why a stand-in variable ``name`` for ``own`` is refused, or None: the trial's treatment,
    its primary outcome, the concept itself, or a variable measured after time zero."""
    treat = {_norm(c.concept.name) for a in p.treatment_strategies if not a.is_comparator
             for c in a.components} | {_norm(a.label) for a in p.treatment_strategies
                                       if not a.is_comparator}  # fmt: skip
    primary = next((o for o in p.outcomes if o.primary), None)
    outcome = set()
    if primary:
        outcome.add(_norm(primary.name))
        if primary.concept:
            outcome.add(_norm(primary.concept.name))
    n = _norm(name)
    return ("the trial's treatment" if n in treat or any(
                t and f" {t} " in f" {n} " for t in treat) else
            "the primary outcome" if n in outcome else
            "the unmapped concept itself" if n == _norm(own) else
            "measured after time zero" if _POST.search(name) else None)  # fmt: skip


def check(answer: SurrogateAnswer, p: TargetTrialProtocol, sets: list[ConceptSet], con,
          model: str, abbreviations=None) -> SurrogateSheet:  # fmt: skip
    """The deterministic checks (module docstring) and the vocabulary search."""
    sheet = SurrogateSheet(nct_id=p.source.nct_id, model=model)
    asked = {s.key: s for s in sets}
    got: dict[str, SurrogateProposal] = {}
    for x in answer.proposals:
        if x.key not in asked:
            sheet.notes.append(f"answer for unknown key {x.key!r} ignored")
        elif x.key in got:
            sheet.notes.append(f"second answer for {x.key!r} ignored")
        else:
            got[x.key] = x
    for key, s in asked.items():
        x = got.get(key)
        item = SurrogateItem(key=key, concept_name=s.concept_name, scope=s.scope,
                             status="none")  # fmt: skip
        if x is None:
            item.note = "no answer"
        elif not x.substitute or x.domain is None:
            item.rationale, item.limitation = x.rationale, x.limitation
            item.note = "no surrogate proposed"
        else:
            item.substitute, item.domain = x.substitute, x.domain.value
            item.rationale, item.limitation = x.rationale, x.limitation
            why = excluded(p, x.substitute, s.concept_name)
            if why:
                item.status, item.note = "dropped", why
            else:
                item.candidates = search(con, x.substitute, x.domain.value, limit=5,
                                         abbreviations=abbreviations)  # fmt: skip
                item.status = "suggested" if item.candidates else "no_vocabulary_match"
        sheet.items.append(item)
    return sheet


class Surrogates:
    """The Clinician's surrogate answers to the Informatician's unmapped concepts."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose(self, p: TargetTrialProtocol, book: ConceptSetBook, con,
                abbreviations=None) -> SurrogateSheet:  # fmt: skip
        """``abbreviations``: an ``AbbreviationList``, expanded as in concept search."""
        from agentic_tte.omop.abbreviations import merged

        sets = unmapped(book)
        if not sets:
            return SurrogateSheet(nct_id=p.source.nct_id, model=self.llm.model,
                                  notes=["every concept had a vocabulary match: nothing asked"])
        answer = self.llm.structured(request(p, sets), SurrogateAnswer)
        sheet = check(answer, p, sets, con, self.llm.model, merged(p, abbreviations))
        sheet.model_digest = self.llm.model_digest()
        return sheet
