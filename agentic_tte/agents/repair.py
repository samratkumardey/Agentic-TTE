"""The Repair agent: fixed rules that make a drafted design emulable before design review
(SCORING.md Phase 3 item 33). It is what the autonomous arm does where a person fixed the
draft by hand; every repair is recorded with its rule and reason, and what the rules cannot
decide is flagged, never guessed.

No LLM and no data: the rules read the protocol and, for the arms only, the trial's published
contrast ("A vs B": the comparison being emulated, an input like the NCT ID; never an estimate).
They were developed on the 12 core-trial drafts (a development set, disclosed); the breadth set
is the held-out test of the rules.

Rules:
- R-arm-null: an arm the registry marks PLACEBO_COMPARATOR or NO_INTERVENTION, or whose
  components are all placebo, is a null arm (comparator, no initiation); its non-placebo
  components (a vehicle such as D5W or saline) are removed.
- R-arm-shared: a component every arm has (intubation in both arms) is background, removed;
  so is a treatment component that the comparator also gets (its non-placebo components or
  description: "PCI plus medical treatment" in both arms of ECLS vs usual care).
- R-arm-one: with one non-null arm and a null arm, the non-null arm is the treatment.
- R-arm-contrast: with several non-null arms (a factorial, or two active arms) and no valid
  treatment/comparator pair, the contrast picks them: components that match neither side of
  the contrast are removed from arms that match it (the other factor of a factorial); an arm
  whose description says "placebo of" a treatment word is not the treatment; a tie goes to
  the arm whose removed components its description calls placebo. Other arms are dropped.
- R-amend: a criterion that says criteria were removed ("... was removed", "... were removed:")
  is amendment text, dropped; a list after "were removed:" is dropped up to the next dated
  amendment. A removal that may name a kept criterion is flagged.
- R-value: an inclusion atom the compiler cannot compile as drafted (compiler.design_problem)
  loses what it cannot compile: a value on a non-measurement concept (a duration or a dose) or
  a qualitative value is removed; a cumulative duration becomes a window before time zero.
  Exclusions are left as drafted: one that does not compile removes no one and is reported.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from agentic_tte.agents.concordance import _null, _words
from agentic_tte.protocol import TargetTrialProtocol

RULES_VERSION = "repair-v1"
_NULL_TYPES = {"PLACEBO_COMPARATOR", "NO_INTERVENTION", "SHAM_COMPARATOR"}
_PLACEBO_OF = re.compile(r"placebos? (?:of |for )?(?:the )?([a-z0-9 /-]+?)(?=,|;| and | plus |"
                         r" with |\.|$)")
_REMOVED = re.compile(r"\b(?:was|were|is|are|has been|have been) (?:removed|deleted|withdrawn|"
                      r"suppressed)\b", re.I)
_LIST_FOLLOWS = re.compile(r"(?:removed|deleted|withdrawn|suppressed)\s*:\s*$", re.I)
_DATED = re.compile(r"\b(?:on )?\d{1,2}/\d{1,2}/\d{2,4}\b", re.I)


class Repair(BaseModel):
    rule: str
    element: str  # treatment_strategies, eligibility
    target: str  # arm label, criterion ID or atom
    what: str


class RepairReport(BaseModel):
    rules_version: str = RULES_VERSION
    contrast: str | None = None
    repairs: list[Repair] = Field(default_factory=list)
    flagged: list[str] = Field(default_factory=list)  # for a person; nothing was changed

    def to_markdown(self) -> str:
        lines = [f"## Repair agent ({self.rules_version})", "",
                 "Fixed rules, no LLM; the contrast is "
                 + (f"'{self.contrast}'." if self.contrast else "not given.")]  # fmt: skip
        if self.repairs:
            lines += ["", "| Rule | Element | Target | Repair |", "|---|---|---|---|"]
            lines += [f"| {r.rule} | {r.element} | {r.target} | {r.what} |" for r in self.repairs]
        else:
            lines += ["", "No repair needed."]
        if self.flagged:
            lines += ["", "**Flagged (not changed):**", ""] + [f"- {f}" for f in self.flagged]
        return "\n".join(lines) + "\n"


def _placebo(name: str) -> bool:
    from agentic_tte.omop.compiler import _is_placebo

    return _is_placebo(name)


def _norm(name: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", name.lower()))


def _placebo_words(description: str | None) -> set[str]:
    """Words the description calls placebo: 'a placebo of activated protein C' -> {activated,
    protein}."""
    out: set[str] = set()
    for m in _PLACEBO_OF.finditer((description or "").lower()):
        out |= _words(m.group(1))
    return out


def _is_null(arm: dict) -> bool:
    comps = [c["concept"]["name"] for c in arm["components"]]
    return (str(arm.get("arm_type") or "").upper() in _NULL_TYPES or _null(arm["label"])
            or not comps or all(_placebo(n) for n in comps))  # fmt: skip


def _valid(arms: list[dict]) -> bool:
    t = [a for a in arms if not a["is_comparator"]]
    return len(t) == 1 and len(arms) - len(t) == 1


_SHORT_STOP = {"and", "the", "for", "per", "via", "not", "any", "all", "use", "day", "dose"}


def _terms(s: str) -> set[str]:
    """Like ``_words`` but keeps three-letter terms (PCI, CRP, ICU): background components are
    often abbreviations."""
    return _words(s) | {w for w in re.findall(r"[a-z0-9]+", s.lower())
                        if len(w) == 3 and w not in _SHORT_STOP}  # fmt: skip


def _given(arm: dict) -> set[str]:
    """Words of what an arm actually gets: its non-placebo components and its description,
    without what the description calls placebo."""
    comps = [c["concept"]["name"] for c in arm["components"] if not _placebo(c["concept"]["name"])]
    desc = _terms(arm.get("description") or "") - _placebo_words(arm.get("description"))
    return set().union(desc, *(_terms(n) for n in comps))


def _background(t: dict, c: dict, given_c: set[str], rep: RepairReport) -> None:
    gone = [x["concept"]["name"] for x in t["components"]
            if _terms(x["concept"]["name"]) and _terms(x["concept"]["name"]) <= given_c]
    if gone and len(gone) < len(t["components"]):
        t["components"] = [x for x in t["components"] if x["concept"]["name"] not in gone]
        rep.repairs.append(Repair(rule="R-arm-shared", element="treatment_strategies",
                                  target=t["label"], what=f"components the comparator "
                                  f"'{c['label']}' also gets, {gone}: background, removed"))


def _arms(arms: list[dict], contrast: str | None, rep: RepairReport) -> list[dict]:
    arms = _pairs(arms, contrast, rep)
    t = [a for a in arms if not a["is_comparator"]]
    c = [a for a in arms if a["is_comparator"]]
    if len(t) == 1 and len(c) == 1:
        if not _is_null(c[0]):
            rep.flagged.append(f"the comparator '{c[0]['label']}' is an active treatment: the "
                               "compiler needs an active-comparator frame")
    return arms


def _pairs(arms: list[dict], contrast: str | None, rep: RepairReport) -> list[dict]:
    add = rep.repairs.append
    given = {a["label"]: _given(a) for a in arms}  # before any vehicle is removed
    if _valid(arms):
        t = next(a for a in arms if not a["is_comparator"])
        c = next(a for a in arms if a["is_comparator"])
        _background(t, c, given[c["label"]], rep)
    for a in arms:  # R-arm-null
        if not _is_null(a):
            continue
        why = (f"the registry marks it {a['arm_type']}"
               if str(a.get("arm_type") or "").upper() in _NULL_TYPES
               else f"its label '{a['label']}' names no treatment" if _null(a["label"])
               else "its components are all placebo")  # fmt: skip
        if not a["is_comparator"]:
            a["is_comparator"] = True
            add(Repair(rule="R-arm-null", element="treatment_strategies", target=a["label"],
                       what=f"a null arm ({why}): made the comparator"))
        vehicle = [c for c in a["components"] if not _placebo(c["concept"]["name"])]
        if vehicle:
            a["components"] = [c for c in a["components"] if _placebo(c["concept"]["name"])]
            add(Repair(rule="R-arm-null", element="treatment_strategies", target=a["label"],
                       what=f"a null arm ({why}): its components "
                            f"{[c['concept']['name'] for c in vehicle]} are vehicle or usual "
                            "care, removed (the comparator is no initiation)"))  # fmt: skip
    real = [a for a in arms if not _is_null(a)]
    if len(real) >= 2:  # R-arm-shared
        shared = set.intersection(*({_norm(c["concept"]["name"]) for c in a["components"]}
                                    for a in real))  # fmt: skip
        for a in real:
            gone = [c["concept"]["name"] for c in a["components"]
                    if _norm(c["concept"]["name"]) in shared]  # fmt: skip
            if gone and len(gone) < len(a["components"]):
                a["components"] = [c for c in a["components"]
                                   if _norm(c["concept"]["name"]) not in shared]  # fmt: skip
                add(Repair(rule="R-arm-shared", element="treatment_strategies",
                           target=a["label"], what=f"components every arm has, {gone}: "
                                                   "background, removed"))  # fmt: skip
    if _valid(arms):
        return arms
    nulls = [a for a in arms if _is_null(a)]
    if len(real) == 1 and nulls:  # R-arm-one
        keep = [real[0], nulls[0]]
        real[0]["is_comparator"] = False
        add(Repair(rule="R-arm-one", element="treatment_strategies", target=real[0]["label"],
                   what="the one non-null arm: made the treatment"))
        for a in nulls[1:]:
            add(Repair(rule="R-arm-one", element="treatment_strategies", target=a["label"],
                       what="a second null arm: dropped"))
        return keep
    if not contrast:
        rep.flagged.append("the arms are not one treatment and one comparator, and no "
                           "contrast is given: the arms were left as drafted")
        return arms
    parts = re.split(r"\s+vs\.?\s+", re.sub(r"\(.*?\)", "", contrast).strip(), maxsplit=1)
    if len(parts) != 2:
        rep.flagged.append(f"the contrast '{contrast}' is not 'A vs B': the arms were left")
        return arms
    wa, wb = _words(parts[0]), _words(parts[1])
    b_null = _null(parts[1].strip().lower()) or "placebo" in wb
    dropped: dict[str, set[str]] = {}
    for a in real:  # R-arm-contrast: the other factor of a factorial
        inside = [c for c in a["components"] if _words(c["concept"]["name"]) & (wa | wb)]
        outside = [c for c in a["components"] if c not in inside
                   and not _placebo(c["concept"]["name"])]  # fmt: skip
        if inside and outside:
            a["components"] = [c for c in a["components"] if c not in outside]
            dropped[a["label"]] = set().union(*(_words(c["concept"]["name"]) for c in outside))
            add(Repair(rule="R-arm-contrast", element="treatment_strategies", target=a["label"],
                       what=f"components outside the contrast, "
                            f"{[c['concept']['name'] for c in outside]}: removed"))  # fmt: skip

    def words(a):
        return set().union(set(), *(_words(c["concept"]["name"]) for c in a["components"]))

    def side(a, w, other):
        return bool(words(a) & w) and not (words(a) & other) and not (_placebo_words(
            a.get("description")) & w)  # fmt: skip

    cand_t = [a for a in real if side(a, wa, wb)]
    if len(cand_t) > 1:  # the tie-break: its removed components are called placebo
        best = [a for a in cand_t if dropped.get(a["label"])
                and dropped[a["label"]] <= _placebo_words(a.get("description"))]  # fmt: skip
        cand_t = best if len(best) == 1 else cand_t
    cand_c = nulls[:1] if b_null else [a for a in real if side(a, wb, wa)]
    if len(cand_t) != 1 or len(cand_c) != 1:
        rep.flagged.append(f"the contrast '{contrast}' matches {len(cand_t)} treatment and "
                           f"{len(cand_c)} comparator arm(s): the arms were left as drafted")
        return arms
    t, c = cand_t[0], cand_c[0]
    t["is_comparator"], c["is_comparator"] = False, True
    add(Repair(rule="R-arm-contrast", element="treatment_strategies", target=t["label"],
               what=f"matches '{parts[0].strip()}': the treatment"))
    add(Repair(rule="R-arm-contrast", element="treatment_strategies", target=c["label"],
               what=f"matches '{parts[1].strip()}': the comparator"))
    for a in arms:
        if a is not t and a is not c:
            add(Repair(rule="R-arm-contrast", element="treatment_strategies", target=a["label"],
                       what="not part of the contrast: dropped"))
    _background(t, c, given[c["label"]], rep)
    return [t, c]


def _amendments(criteria: list[dict], rep: RepairReport) -> list[dict]:
    out, dropping = [], False
    for c in criteria:
        text = c.get("text") or ""
        if dropping and _DATED.search(text):
            dropping = False
        if dropping:
            rep.repairs.append(Repair(rule="R-amend", element="eligibility", target=c["id"],
                                      what="listed as removed by an amendment: dropped"))
            continue
        if _REMOVED.search(text):
            rep.repairs.append(Repair(rule="R-amend", element="eligibility", target=c["id"],
                                      what="amendment text (criteria removed): dropped"))
            if _LIST_FOLLOWS.search(text.strip()):
                dropping = True
            else:
                rep.flagged.append(f"{c['id']} says a criterion was removed ('{text[:80]}'); "
                                   "a kept criterion may be the one it removed")
            continue
        out.append(c)
    return out


def _values(criteria: list[dict], rep: RepairReport) -> list[dict]:
    from agentic_tte.omop.compiler import T0_ALIASES, design_problem
    from agentic_tte.omop.compiler import _norm as cnorm
    from agentic_tte.protocol.models import Criterion

    out = []
    for raw in criteria:
        c = json.loads(json.dumps(raw))
        crit = Criterion.model_validate(c)
        if crit.kind != "inclusion" or not crit.emulable:
            out.append(c)
            continue
        for i, (a, atom) in enumerate(zip(c["atoms"], crit.atoms, strict=True), 1):
            t0 = atom.temporal is None or cnorm(atom.temporal.anchor) in T0_ALIASES
            why = design_problem(crit, atom, anchor_is_t0=t0)
            if not why:
                continue
            target = f"{c['id']}.{i}" if len(c["atoms"]) > 1 else c["id"]
            if "cumulative duration" in why and a["temporal"].get("window"):
                w = a["temporal"]["window"]
                a["temporal"] = {**a["temporal"], "relation": "x_within_t_before_y"}
                what = (f"cumulative duration read as a window: within {w['amount']:g} "
                        f"{w['unit']}(s) before time zero")
            elif "value constraint" in why and "several domains" not in why or \
                    "qualitative value" in why:  # fmt: skip
                what = f"value '{(a['value'] or {}).get('text') or 'value'}' removed ({why})"
                a["value"] = None
            else:
                rep.flagged.append(f"{target}: {why} (not repaired)")
                continue
            rep.repairs.append(Repair(rule="R-value", element="eligibility", target=target,
                                      what=what))  # fmt: skip
            again = design_problem(Criterion.model_validate(c), Criterion.model_validate(
                c).atoms[i - 1], anchor_is_t0=t0)  # fmt: skip
            if again:
                rep.flagged.append(f"{target}: still not compilable after the repair ({again})")
        out.append(c)
    return out


def repair_design(protocol: TargetTrialProtocol,
                  contrast: str | None = None) -> tuple[dict, RepairReport]:  # fmt: skip
    """The changes the rules make (protocol fields -> new values, for ``new_version``) and the
    report. No change: an empty dict."""
    rep = RepairReport(contrast=contrast)
    p = protocol.model_dump(mode="json")
    arms = _arms(json.loads(json.dumps(p["treatment_strategies"])), contrast, rep)
    elig = _values(_amendments(p["eligibility"], rep), rep)
    changes = {}
    if arms != p["treatment_strategies"]:
        changes["treatment_strategies"] = arms
    if elig != p["eligibility"]:
        changes["eligibility"] = elig
    if changes:
        TargetTrialProtocol.model_validate({**p, **changes})
    return changes, rep
