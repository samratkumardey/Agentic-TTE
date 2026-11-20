"""The Clinician's answer to the Statistician: what to do about confounders that stay
imbalanced after weighting or are often missing (PLAN.md §3, §17.2; the Statistician <->
Clinician loop).

Result-blind by construction. The Statistician's balance check
(``agentic_tte.agents.statistician.balance``) reads treatment and confounders only, before any
outcome model is fitted, and the Clinician sees only those aggregates (SMDs, missing shares,
positivity): never an outcome, an event count or an effect estimate. So changing the
confounder set cannot be steered by the result.

What is deterministic and what uses the LLM:
- LLM (one schema-constrained call for the flagged confounders): keep, drop or replace each,
  with a rationale. It never gives concept IDs.
- deterministic: which confounders are asked about (the balance flags); the checks: age and
  sex are never dropped; a replacement is refused when it is the treatment, the primary
  outcome, the confounder itself or measured after time zero
  (``agentic_tte.agents.surrogates.excluded``); a replacement already in the set counts as a
  drop. Replacement concepts are proposed from the vocabulary at concept review, as for any new
  confounder.

Imbalance alone is not a reason to drop a confounder: a true common cause left out biases the
estimate more than residual imbalance does. The prompt says so, and the reviewer decides: the
advice goes to design review as a proposed confounder list, which the reviewer applies,
changes, or declines (the analysis then runs with the imbalance reported).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentic_tte.agents.clinician import _summary
from agentic_tte.agents.statistician import BalanceReport
from agentic_tte.agents.surrogates import excluded
from agentic_tte.llm.base import LLMClient, Message
from agentic_tte.omop.compiler import _column
from agentic_tte.protocol.models import ConceptRef, Domain, TargetTrialProtocol

PROMPT_VERSION = "balance-v1"
KEEP_ALWAYS = {"age", "female"}  # the standard confounders' frame columns


def column(ref: ConceptRef) -> str:
    """The analysis-frame column of a confounder, named as the compiler names it."""
    if ref.domain == Domain.DEMOGRAPHIC:
        n = " ".join(ref.name.lower().split())
        return "female" if n in ("sex", "gender", "female") else _column(ref.name)
    return _column(ref.name)


class BalanceAdvice(BaseModel):
    column: str = Field(max_length=200)
    action: Literal["keep", "drop", "replace"]
    substitute: str | None = Field(default=None, max_length=80)
    domain: Domain | None = None
    rationale: str = Field(default="", max_length=300)


class BalanceAnswer(BaseModel):
    advice: list[BalanceAdvice] = Field(default_factory=list, max_length=40)


class BalanceItem(BaseModel):
    column: str
    confounder: str
    reasons: list[str]
    action: Literal["keep", "drop", "replace"] = "keep"
    substitute: str | None = None
    domain: str | None = None
    rationale: str = ""
    note: str = ""  # why the checks changed the advice, or "no answer"


class BalanceSheet(BaseModel):
    nct_id: str | None = None
    model: str
    model_digest: str | None = None
    prompt_version: str = PROMPT_VERSION
    items: list[BalanceItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def changes(self) -> bool:
        return any(i.action != "keep" for i in self.items)

    def confounders(self, p: TargetTrialProtocol) -> list[dict]:
        """The confounder list with the advice applied, in order (JSON for design review)."""
        by = {i.column: i for i in self.items}
        out, seen = [], set()
        for ref in p.assignment.confounders:
            i = by.get(column(ref))
            if i and i.action == "drop":
                continue
            if i and i.action == "replace":
                ref = ConceptRef(name=i.substitute, domain=i.domain)
            if ref.name.lower() not in seen:
                seen.add(ref.name.lower())
                out.append(ref.model_dump(mode="json"))
        return out

    def to_markdown(self) -> str:
        lines = [f"## Clinician balance advice ({self.model}, {self.prompt_version})", "",
                 "Asked about the confounders the balance check flagged; it saw the balance "
                 "aggregates only, no outcome or estimate. Nothing changes until the design "
                 "reviewer applies it.", "",
                 "| Confounder | Flag | Advice | Rationale | Check |", "|---|---|---|---|---|"]
        for i in self.items:
            adv = i.action + (f": {i.substitute} ({i.domain})" if i.action == "replace" else "")
            lines.append(f"| {i.confounder} | {'; '.join(i.reasons)} | {adv} | "
                         f"{i.rationale or '–'} | {i.note or '–'} |")
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


SYSTEM = """You are the Clinician agent in a target trial emulation system. The Statistician \
weighted the emulated trial's patients by the propensity of treatment and found baseline \
confounders that stay imbalanced or are often missing. For each one, advise keep, drop or \
replace.

Rules:
1. Answer every column you are given, once, with the column exactly as given.
2. Imbalance alone is never a reason to drop a confounder: a common cause of treatment and \
outcome must stay, and residual imbalance is reported. Advise drop only when the variable is \
not a common cause (it affects the treatment choice but not the outcome), or it is measured \
after time zero.
3. Advise replace when the variable is poorly recorded (often missing) and another routinely \
recorded baseline variable carries the same information. substitute: its name as a clinician \
would look it up, with exactly one domain: Condition, Drug, Procedure, Device, Measurement, \
Observation or Visit. Never the trial's treatment or primary outcome, never a variable \
measured after time zero.
4. Otherwise advise keep.
5. rationale: one sentence.
Return only the JSON object."""


def request(p: TargetTrialProtocol, rep: BalanceReport, cols: dict[str, str]) -> list[Message]:
    rows = []
    for c, reasons in rep.flagged.items():
        before, after = rep.smd_before.get(c), rep.smd_after.get(c)
        smds = (f"SMD {before:.2f} before and {after:.2f} after weighting"
                if before is not None and after is not None else "SMD not computed")
        rows.append(f"- column {c} (confounder '{cols.get(c, c)}'): {'; '.join(reasons)}; "
                    f"{smds}; missing {100 * rep.missing.get(c, 0):.0f}%")
    pos = (f"\nPropensity scores near 0 or 1: {100 * rep.ps_extreme:.1f}% of patients."
           if rep.ps_extreme is not None else "")
    return [Message(role="system", content=SYSTEM),
            Message(role="user", content=f"Protocol:\n{_summary(p)}\n\nAll confounders: "
                                         f"{', '.join(cols.values())}.{pos}\n\nFlagged:\n"
                                         + "\n".join(rows))]  # fmt: skip


def check(answer: BalanceAnswer, p: TargetTrialProtocol, rep: BalanceReport,
          model: str) -> BalanceSheet:  # fmt: skip
    """The deterministic checks (module docstring)."""
    cols = {column(r): r.name for r in p.assignment.confounders}
    sheet = BalanceSheet(nct_id=p.source.nct_id, model=model)
    got: dict[str, BalanceAdvice] = {}
    for x in answer.advice:
        if x.column not in rep.flagged:
            sheet.notes.append(f"advice for unflagged column {x.column!r} ignored")
        elif x.column in got:
            sheet.notes.append(f"second advice for {x.column!r} ignored")
        else:
            got[x.column] = x
    have = {r.name.lower() for r in p.assignment.confounders}
    for c, reasons in rep.flagged.items():
        item = BalanceItem(column=c, confounder=cols.get(c, c), reasons=reasons)
        x = got.get(c)
        if x is None:
            item.note = "no answer: kept"
            sheet.items.append(item)
            continue
        item.action, item.rationale = x.action, x.rationale
        if x.action != "keep" and c in KEEP_ALWAYS:
            item.action, item.note = "keep", "age and sex are always adjusted for"
        elif x.action == "replace":
            why = ("no substitute or domain given" if not x.substitute or x.domain is None
                   else excluded(p, x.substitute, item.confounder))  # fmt: skip
            if why:
                item.action, item.note = "keep", f"replacement refused: {why}"
            else:
                item.substitute, item.domain = x.substitute, x.domain.value
                if x.substitute.lower() in have:
                    item.note = "the substitute is already a confounder: this drops the column"
        sheet.items.append(item)
    return sheet


class BalanceAdvisor:
    """The Clinician answering the Statistician's balance flags."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def advise(self, p: TargetTrialProtocol, rep: BalanceReport) -> BalanceSheet:
        if not rep.flagged:
            return BalanceSheet(nct_id=p.source.nct_id, model=self.llm.model,
                                notes=["no confounder flagged: nothing asked"])
        cols = {column(r): r.name for r in p.assignment.confounders}
        answer = self.llm.structured(request(p, rep, cols), BalanceAnswer)
        sheet = check(answer, p, rep, self.llm.model)
        sheet.model_digest = self.llm.model_digest()
        return sheet
