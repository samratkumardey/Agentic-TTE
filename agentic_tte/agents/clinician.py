"""Clinician agent: protocol + literature -> baseline confounders with cited passages
(PLAN.md §3, §17.2 point 1).

What is deterministic and what uses the LLM:
- deterministic: the literature (PubMed searches built from the protocol, cached), retrieval
  (BM25 over abstract passages), and every check on the answer:
  - each citation's quote must occur verbatim in the cited abstract, or it is dropped;
  - the treatment itself and the primary outcome are never confounders (fault F-h);
  - variables named as measured after time zero are dropped (post-baseline adjustment);
  - duplicates are merged; age and sex are always included.
  Nothing is dropped silently: the sheet lists every dropped proposal with its reason.
- LLM (one schema-constrained call): which variables are common causes of treatment choice and
  the outcome, with a rationale and quotes from the retrieved passages.

**Result-blind literature.** For a Replay, the agent must not see the trial's answer. The
literature leaves out the trial's own publications (registry RESULT and DERIVED references),
every article that names the trial's NCT ID or acronym, and everything published after the
trial's primary completion date. What was left out is listed with the reason.

The proposal is a draft for review: it goes to design review, where the reviewer approves or
replaces the list, and then through concept review like every other concept.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from agentic_tte.llm.base import LLMClient, Message
from agentic_tte.protocol.models import ConceptRef, Domain, TargetTrialProtocol
from agentic_tte.sources.pubmed.client import Article, PubMedClient

# clinician-v2: the answer schema bounds every length (at most 20 confounders, 2 citations each,
# quotes under 300 characters), a guard against runaway output. Added after timeouts on the core
# trials (2026-11-01) that turned out to be a shared GPU (SCORING.md item 29), not the schema.
PROMPT_VERSION = "clinician-v2"
TOP_K = 15  # passages shown to the model
PER_ARTICLE = 3  # at most this many passages from one abstract


# --- literature -----------------------------------------------------------------------------


class Excluded(BaseModel):
    pmid: str
    reason: str


class Literature(BaseModel):
    """The articles the Clinician may cite for one trial, and what was left out and why."""

    nct_id: str | None = None
    cutoff: str | None = None  # primary completion date; nothing later is used
    queries: list[str] = Field(default_factory=list)
    articles: list[Article] = Field(default_factory=list)
    excluded: list[Excluded] = Field(default_factory=list)
    built_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def get(self, pmid: str) -> Article | None:
        return next((a for a in self.articles if a.pmid == pmid), None)


def _or(terms: list[str]) -> str:
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(t.strip() for t in terms if t.strip()))


# Human studies only, and no case reports or opinion pieces; adults only for an adult trial.
_FILTER = ("NOT (animals[mh] NOT humans[mh]) NOT (case reports[pt] OR comment[pt] OR "
           "editorial[pt] OR letter[pt])")  # fmt: skip
_ADULT = "NOT (pediatric[tiab] OR paediatric[tiab] OR neonatal[tiab] OR children[tiab])"


def literature_queries(p: TargetTrialProtocol, *, adult: bool = True) -> list[str]:
    """Prognostic factors of the main condition (the first listed), and observational studies
    of the same treatment.
    Registry "conditions" sometimes name the intervention or a setting ("Vitamin C"); a
    condition that is a treatment component is left out."""
    names = [c.concept.name for a in p.treatment_strategies if not a.is_comparator
             for c in a.components]  # fmt: skip
    treat_keys = {_norm(n) for n in names} | {_norm(a.label) for a in p.treatment_strategies}
    cond = _or([c for c in p.source.conditions if _norm(c) not in treat_keys][:1])
    treat = _or(names[:3])
    if not cond:
        return []
    tail = f" {_FILTER}" + (f" {_ADULT}" if adult else "")
    qs = [f"({cond}) AND (\"risk factors\"[tiab] OR prognostic[tiab] OR predictors[tiab]) "
          f"AND (mortality[tiab] OR outcome[tiab]){tail}"]  # fmt: skip
    if treat:
        qs.append(f"({cond}) AND ({treat}) AND (observational[tiab] OR cohort[tiab] OR "
                  f"propensity[tiab] OR confounding[tiab]){tail}")  # fmt: skip
    return qs


def _adult(ps: dict) -> bool:
    m = re.match(r"(\d+)\s*Year", (ps.get("eligibilityModule") or {}).get("minimumAge") or "")
    return bool(m) and int(m.group(1)) >= 18


def acronym_pattern(acronym: str) -> re.Pattern | None:
    """Registry acronyms come as "(BICAR-ICU)" or "CAPE_COD": match the words with a space,
    hyphen, underscore or nothing between them, case-sensitively (lower-case "chest" is not
    the CHEST trial). None under 4 letters, which would match too much."""
    words = re.findall(r"[A-Za-z0-9]+", acronym)
    if sum(map(len, words)) < 4:
        return None
    return re.compile(r"\b" + r"[-_ ]?".join(map(re.escape, words)) + r"\b")


def build_literature(study: dict, p: TargetTrialProtocol, client: PubMedClient, *,
                     per_query: int = 40) -> Literature:  # fmt: skip
    """The result-blind literature for the trial in ``study`` (a CT.gov v2 record)."""
    ps = study.get("protocolSection", study)
    nct = ps.get("identificationModule", {}).get("nctId") or p.source.nct_id
    acronym = (ps.get("identificationModule", {}).get("acronym") or "").strip()
    date = (ps.get("statusModule", {}).get("primaryCompletionDateStruct") or {}).get("date")
    refs = ps.get("referencesModule", {}).get("references", [])
    own = {r["pmid"] for r in refs if r.get("pmid") and r.get("type") in ("RESULT", "DERIVED")}
    background = [r["pmid"] for r in refs if r.get("pmid") and r.get("type") == "BACKGROUND"]
    maxdate = date.replace("-", "/") + ("/28" if len(date) == 7 else "") if date else None
    queries = literature_queries(p, adult=_adult(ps))
    pmids = list(background)
    for q in queries:
        pmids += client.search(q, retmax=per_query, maxdate=maxdate)
    acro = acronym_pattern(acronym)
    keep, excluded = [], []
    for a in client.fetch(list(dict.fromkeys(pmids))):
        why = ("a publication of the trial (registry reference)" if a.pmid in own else
               "names the trial's NCT ID" if nct and nct in a.nct_ids else
               f"names the trial's acronym ({acronym})" if acro and acro.search(a.text) else
               f"published after primary completion ({date})"
               if date and a.year and a.year > int(date[:4]) else None)  # fmt: skip
        if why:
            excluded.append(Excluded(pmid=a.pmid, reason=why))
        else:
            keep.append(a)
    excluded += [Excluded(pmid=x, reason="a publication of the trial (registry reference)")
                 for x in sorted(own - {e.pmid for e in excluded})]  # fmt: skip
    return Literature(nct_id=nct, cutoff=date, queries=queries, articles=keep, excluded=excluded)


# --- retrieval ------------------------------------------------------------------------------


class Passage(BaseModel):
    id: str  # "<pmid>#<n>"
    pmid: str
    text: str


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("the of and in to a with for or on by at as is was were be are an from that this "
            "than which we our these those its it not".split())  # fmt: skip


def passages(lit: Literature, size: int = 3, stride: int = 2) -> list[Passage]:
    out = []
    for a in lit.articles:
        sents = [s for s in _SENT.split(a.abstract) if s.strip()]
        for n, i in enumerate(range(0, max(len(sents) - size + stride, 1), stride)):
            body = " ".join(sents[i:i + size])
            out.append(Passage(id=f"{a.pmid}#{n}", pmid=a.pmid,
                               text=(f"{a.title} " if n == 0 else "") + body))  # fmt: skip
    return out


def _tokens(s: str) -> list[str]:
    return [t for t in _TOK.findall(s.lower()) if t not in _STOP]


def bm25(query: str, docs: list[Passage], k1: float = 1.5, b: float = 0.75) -> list[float]:
    toks = [_tokens(d.text) for d in docs]
    if not toks:
        return []
    avg = sum(map(len, toks)) / len(toks)
    df = Counter(t for d in toks for t in set(d))
    n = len(toks)
    q = _tokens(query)
    scores = []
    for d in toks:
        tf = Counter(d)
        scores.append(sum(
            math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
            for t in q if t in tf))  # fmt: skip
    return scores


def retrieval_query(p: TargetTrialProtocol) -> str:
    primary = next((o for o in p.outcomes if o.primary), None)
    parts = [*p.source.conditions, p.time_zero.anchor_event if p.time_zero else "",
             primary.name if primary else "",
             *[c.concept.name for a in p.treatment_strategies for c in a.components],
             "risk factors prognostic predictors mortality severity baseline confounding "
             "comorbidity treatment allocation"]  # fmt: skip
    return " ".join(x for x in parts if x)


def retrieve(p: TargetTrialProtocol, lit: Literature, k: int = TOP_K) -> list[Passage]:
    """The ``k`` best passages for the protocol, at most ``PER_ARTICLE`` per abstract; ties
    keep corpus order, so retrieval is deterministic."""
    docs = passages(lit)
    ranked = sorted(zip(bm25(retrieval_query(p), docs), range(len(docs)), strict=True),
                    key=lambda x: (-x[0], x[1]))  # fmt: skip
    out, per = [], Counter()
    for score, i in ranked:
        if score <= 0 or len(out) == k:
            break
        if per[docs[i].pmid] < PER_ARTICLE:
            out.append(docs[i])
            per[docs[i].pmid] += 1
    return out


# --- the agent ------------------------------------------------------------------------------


# The bounds go into the JSON schema, so a grammar-constrained decoder cannot run on.
class Citation(BaseModel):
    pmid: str = Field(max_length=12)
    quote: str = Field(max_length=300)


class ProposedConfounder(BaseModel):
    name: str = Field(max_length=80)
    domain: Domain
    rationale: str = Field(max_length=300)
    citations: list[Citation] = Field(default_factory=list, max_length=2)


class ClinicianAnswer(BaseModel):
    confounders: list[ProposedConfounder] = Field(max_length=20)


class ConfounderItem(BaseModel):
    name: str
    domain: Domain
    rationale: str
    citations: list[Citation] = Field(default_factory=list)  # verified quotes only
    status: Literal["cited", "uncited", "standard"]
    notes: list[str] = Field(default_factory=list)


class Dropped(BaseModel):
    name: str
    reason: str


class ConfounderSheet(BaseModel):
    nct_id: str | None = None
    model: str
    model_digest: str | None = None
    prompt_version: str = PROMPT_VERSION
    n_articles: int = 0
    n_excluded: int = 0
    cutoff: str | None = None
    passages: list[str] = Field(default_factory=list)  # ids shown to the model
    items: list[ConfounderItem] = Field(default_factory=list)
    dropped: list[Dropped] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def refs(self) -> list[ConceptRef]:
        return [ConceptRef(name=i.name, domain=i.domain) for i in self.items]

    def to_markdown(self) -> str:
        lines = ["## Confounders proposed by the Clinician", "",
                 f"Model {self.model} ({self.prompt_version}); literature: {self.n_articles} "
                 f"abstracts, {self.n_excluded} left out as result-revealing"
                 + (f", nothing after {self.cutoff}" if self.cutoff else "")
                 + f"; {len(self.passages)} passages retrieved. A proposal for review, not a "
                 "decision.", "",
                 "| Confounder | Domain | Status | Why | Citations |", "|---|---|---|---|---|"]
        for i in self.items:
            cites = "; ".join(f"PMID {c.pmid}: \"{c.quote}\"" for c in i.citations) or "–"
            why = i.rationale + ("" if not i.notes else " (" + "; ".join(i.notes) + ")")
            lines.append(f"| {i.name} | {i.domain} | {i.status} | {why} | {cites} |")
        if self.dropped:
            lines += ["", "Dropped by the checks:", ""]
            lines += [f"- {d.name}: {d.reason}" for d in self.dropped]
        if self.notes:
            lines += ["", *[f"- {n}" for n in self.notes]]
        return "\n".join(lines) + "\n"


SYSTEM = """You are the Clinician agent in a target trial emulation system. Given a trial \
protocol and passages from the medical literature, list the baseline confounders that an \
observational emulation of this trial must adjust for.

Rules:
1. A confounder is a common cause of which treatment strategy a patient receives and of the \
primary outcome. Prefer variables that drive clinicians' treatment decisions and strongly \
predict the outcome: illness severity, organ support, key laboratory values, comorbidities.
2. Only baseline variables: measured at or before time zero. Never a variable measured after \
time zero, such as response to treatment, time to treatment or events during follow-up.
3. Never the treatment itself, its components, or the outcome.
4. Do not list a variable that every eligible patient has by definition (the trial's \
inclusion condition itself).
5. Use names a clinician would map to health-record data ("serum lactate", "SOFA score", \
"invasive mechanical ventilation", "chronic kidney disease"), one variable per entry, with \
exactly one domain: Demographic, Condition, Drug, Procedure, Device, Measurement, Observation \
or Visit.
6. rationale: one sentence on why the variable affects both treatment choice and outcome.
7. citations: one or two quotes copied word for word from the passages (at least four words, \
under 300 characters each), with the passage's PMID. Cite only what a passage actually says; \
leave citations empty rather than invent one.
8. List the 8 to 20 most important variables.
Return only the JSON object."""


def _summary(p: TargetTrialProtocol) -> str:
    primary = next((o for o in p.outcomes if o.primary), None)
    crit = [f"- {c.kind} {c.id}: {c.text}" for c in p.eligibility if c.category == "clinical"]
    arms = [f"- {a.label}{' (comparator)' if a.is_comparator else ''}: "
            + (", ".join(c.concept.name for c in a.components) or a.description or "")
            for a in p.treatment_strategies]  # fmt: skip
    horizon = f" at {primary.horizon.amount:g} {primary.horizon.unit}" \
        if primary and primary.horizon else ""  # fmt: skip
    return "\n".join([
        f"Trial: {p.source.title or p.source.nct_id}",
        f"Conditions: {', '.join(p.source.conditions) or 'not stated'}",
        "Eligibility:", *crit[:40], "Treatment strategies:", *arms,
        f"Time zero: {p.time_zero.anchor_event if p.time_zero else 'not set'}",
        f"Primary outcome: {primary.name if primary else 'not set'}{horizon}"])


def _norm(s: str) -> str:
    return " ".join(_TOK.findall(s.lower()))


def _ws(s: str) -> str:
    return " ".join(s.split()).lower()


_POST = re.compile(
    r"\b(after|following|post)[- ](randomi[sz]ation|time zero|enrol?ment|inclusion|treatment"
    r"|study drug)\b|\bduring (the )?(follow[- ]?up|study|treatment)\b"
    r"|\btime (from|to|until) [^,;]*\b(treatment|study drug|randomi[sz]ation|initiation)\b",
    re.IGNORECASE)
STANDARD = [("age", Domain.DEMOGRAPHIC), ("sex", Domain.DEMOGRAPHIC)]


def check(answer: ClinicianAnswer, p: TargetTrialProtocol, lit: Literature,
          model: str, shown: list[Passage]) -> ConfounderSheet:  # fmt: skip
    """The deterministic checks on the model's answer (module docstring)."""
    treat = {_norm(c.concept.name) for a in p.treatment_strategies if not a.is_comparator
             for c in a.components} | {_norm(a.label) for a in p.treatment_strategies
                                       if not a.is_comparator}  # fmt: skip
    primary = next((o for o in p.outcomes if o.primary), None)
    outcome = set()
    if primary:
        outcome.add(_norm(primary.name))
        if primary.concept:
            outcome.add(_norm(primary.concept.name))
    sheet = ConfounderSheet(nct_id=p.source.nct_id, model=model, n_articles=len(lit.articles),
                            n_excluded=len(lit.excluded), cutoff=lit.cutoff,
                            passages=[x.id for x in shown])  # fmt: skip
    seen: dict[str, ConfounderItem] = {}
    for c in answer.confounders:
        key = _norm(c.name)
        if not key:
            continue
        if key in treat:
            sheet.dropped.append(Dropped(name=c.name, reason="the treatment itself (F-h)"))
            continue
        if key in outcome:
            sheet.dropped.append(Dropped(name=c.name, reason="the primary outcome"))
            continue
        if _POST.search(c.name):
            sheet.dropped.append(Dropped(name=c.name, reason="measured after time zero "
                                                             "(post-baseline adjustment)"))
            continue
        good, notes = [], []
        for cit in c.citations:
            art = lit.get(cit.pmid.strip())
            if art is None:
                notes.append(f"citation to PMID {cit.pmid} dropped: not in the literature")
            elif len(cit.quote.split()) < 4 or _ws(cit.quote) not in _ws(art.text):
                notes.append(f"citation to PMID {cit.pmid} dropped: quote not found verbatim")
            else:
                good.append(Citation(pmid=art.pmid, quote=" ".join(cit.quote.split())))
        if any(t and f" {t} " in f" {key} " for t in treat):
            notes.append("names the treatment: baseline use only; check the new-user exclusion")
        if key in seen:
            seen[key].citations += good
            continue
        seen[key] = ConfounderItem(name=c.name, domain=c.domain, rationale=c.rationale,
                                   citations=good, status="cited" if good else "uncited",
                                   notes=notes)  # fmt: skip
    have = set(seen) | ({"sex"} if "gender" in seen else set())
    for name, dom in STANDARD:
        if name not in have:
            seen[name] = ConfounderItem(name=name, domain=dom, status="standard",
                                        rationale="standard demographic adjustment")
    sheet.items = list(seen.values())
    if not lit.articles:
        sheet.notes.append("No literature: every proposal is uncited.")
    return sheet


def request(p: TargetTrialProtocol, lit: Literature) -> tuple[list[Message], list[Passage]]:
    """The messages the Clinician sends, and the passages they show."""
    shown = retrieve(p, lit)
    ctx = "\n\n".join(f"[PMID {x.pmid}] {x.text}" for x in shown) or "(no passages)"
    msgs = [Message(role="system", content=SYSTEM),
            Message(role="user", content=f"Protocol:\n{_summary(p)}\n\nPassages:\n{ctx}")]
    return msgs, shown


class Clinician:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose(self, p: TargetTrialProtocol, lit: Literature) -> ConfounderSheet:
        msgs, shown = request(p, lit)
        answer = self.llm.structured(msgs, ClinicianAnswer)
        sheet = check(answer, p, lit, self.llm.model, shown)
        sheet.model_digest = self.llm.model_digest()
        return sheet
