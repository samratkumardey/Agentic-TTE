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

Added in repair-v2 (SCORING.md item 34, before any core run):
- R-arm-standard: a comparator whose components are all standard fluids (saline, sodium
  chloride, crystalloids, Ringer's, balanced solutions) is no initiation of the study fluid:
  nearly every ICU patient gets standard fluids, so starting them defines no strategy.
- R-arm-timing: a comparator that is the treatment's own intervention at another time
  ("standard" vs "accelerated" RRT initiation) is no initiation within the grace period.
- R-grace: a treatment arm with no grace period gets ``DEFAULT_GRACE`` (24 hours), origin
  default, so initiation after time zero is handled (clone-censor-weight or the active-
  comparator frame) rather than counted from the future.
- R-anchor: an atom whose anchor is the name of a temporal relation ("x_at_time_of_y"), a
  drafting slip, is anchored at time zero.

Added in repair-v3 (SCORING.md item 34, after the counts-only MIMIC check and before any core
run; repair-v2 never ran a trial):
- R-amend also drops a criterion that says one was changed or clarified ('"X" was changed for
  "Y"'; the criterion as listed is kept, flagged), and an amendment heading (a dated line, or
  one starting "Amendments", that ends with a colon). A criterion an amendment added is kept.
- R-subgroup: a criterion that names a sub-group and ends with a colon ("On the sub-group of
  patients included with COVID19:") is a heading, dropped, with the criteria right after it of
  the same kind that name its condition (they define the sub-group, not the trial).
- R-name: a treatment component named with timing or action words ("Accelerated RRT
  initiation", "ECLS insertion") is named without them; when what is left is one
  abbreviation, it is expanded from the trial's registered title when the title spells it
  out (Schwartz-Hearst: "Renal Replacement Therapy" for RRT), else kept.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from agentic_tte.agents.concordance import _null, _words
from agentic_tte.protocol import TargetTrialProtocol

RULES_VERSION = "repair-v3"
DEFAULT_GRACE = {"amount": 24, "unit": "hour"}
_STANDARD_FLUID = re.compile(r"^(?:(?:0\.9 ?%|normal|isotonic|physiological) )?(?:saline|sodium "
                             r"chloride(?: 0\.9 ?%)?|crystalloids?(?: solutions?)?|(?:lactated )?"
                             r"ringer'?s?(?: lactate| solution)?|balanced (?:crystalloids?|"
                             r"solutions?|salt solutions?)|plasma-?lyte)$")
_TIMING = {"early", "accelerated", "immediate", "delayed", "standard", "late", "deferred",
           "conventional", "initiation", "strategy", "start", "timing"}
_RELATIONS = {"x_within_t_before_y", "x_within_t_after_y", "x_before_y", "x_after_y",
              "x_at_time_of_y", "cumulative_duration"}
_NULL_TYPES = {"PLACEBO_COMPARATOR", "NO_INTERVENTION", "SHAM_COMPARATOR"}
_PLACEBO_OF = re.compile(r"placebos? (?:of |for )?(?:the )?([a-z0-9 /-]+?)(?=,|;| and | plus |"
                         r" with |\.|$)")
_REMOVED = re.compile(r"\b(?:was|were|is|are|has been|have been) (?:removed|deleted|withdrawn|"
                      r"suppressed)\b", re.I)
_LIST_FOLLOWS = re.compile(r"(?:removed|deleted|withdrawn|suppressed)\s*:\s*$", re.I)
_DATED = re.compile(r"\b(?:on )?\d{1,2}/\d{1,2}/\d{2,4}\b", re.I)
_CHANGED = re.compile(r"\b(?:was|were|has been|have been) (?:changed|clarified|modified|"
                      r"replaced|amended|reworded)\b", re.I)
_AMEND_HEAD = re.compile(r"^\s*amendments?\b", re.I)
_SUBGROUP = re.compile(r"\bsub-? ?(?:groups?|stud(?:y|ies)|populations?)\b", re.I)
_GENERIC = {"group", "groups", "subgroup", "subgroups", "patients", "patient", "included",
            "inclusion", "study", "studies", "substudy", "population", "sub"}
_NAME_DROP = {"early", "accelerated", "immediate", "delayed", "late", "deferred", "initiation",
              "insertion", "placement", "implantation", "start", "timing", "strategy"}
_ABBR = re.compile(r"^[A-Z][A-Z0-9-]{1,7}$")


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


def _standard(name: str) -> bool:
    return bool(_STANDARD_FLUID.match(" ".join(name.lower().split())))


def _timing_words(arm: dict) -> set[str]:
    return set().union(set(), *(_terms(c["concept"]["name"]) for c in arm["components"]))


def _comparator_rules(t: dict, c: dict, rep: RepairReport) -> None:
    names = [x["concept"]["name"] for x in c["components"]]
    if not names or _is_null(c):
        return
    why = None
    if all(_standard(n) for n in names):  # R-arm-standard
        rule, why = "R-arm-standard", (f"standard fluids {names}: nearly every patient gets "
                                       "them, so the comparator is no initiation of the "
                                       "study fluid")  # fmt: skip
    else:  # R-arm-timing
        wt, wc = _timing_words(t) - _TIMING, _timing_words(c) - _TIMING
        if wt and wt == wc and _timing_words(t) != _timing_words(c):
            rule, why = "R-arm-timing", (f"the same intervention at another time ({names}): "
                                         "the comparator is no initiation within the grace "
                                         "period")  # fmt: skip
    if why:
        c["components"] = []
        rep.repairs.append(Repair(rule=rule, element="treatment_strategies", target=c["label"],
                                  what=why))


def _expand(abbr: str, texts: list[str]) -> str | None:
    """The phrase of the registered title that spells out ``abbr`` (all its words used)."""
    from agentic_tte.omop.abbreviations import _best_long_form

    letters = re.sub(r"[^a-z0-9]", "", abbr.lower())
    spans = []
    for text in texts:
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text or "")
        spans += [words[i:i + n] for n in range(2, len(abbr) + 2)
                  for i in range(len(words) - n + 1)]  # fmt: skip
    for span in spans:  # first: the words' initials are the abbreviation
        if "".join(w[0].lower() for w in span) == letters:
            return " ".join(span)
    for span in spans:  # then Schwartz-Hearst (ExtraCorporeal Life Support)
        phrase = " ".join(span)
        if _best_long_form(abbr, phrase) == phrase and phrase[0].lower() == letters[:1]:
            return phrase
    return None


def _names(arms: list[dict], texts: list[str], rep: RepairReport) -> None:
    """R-name: treatment components without timing or action words."""
    for a in arms:
        if a["is_comparator"]:
            continue
        for comp in a["components"]:
            name = comp["concept"]["name"]
            tokens = name.split()
            rest = [w for w in tokens if w.lower().strip(",.;:") not in _NAME_DROP]
            if not rest or rest == tokens:
                continue
            new, how = " ".join(rest), "timing or action words removed"
            if len(rest) == 1 and _ABBR.match(rest[0]):
                full = _expand(rest[0], texts)
                if full:
                    new, how = full.lower(), (f"timing or action words removed; {rest[0]} as "
                                              "spelled out in the registered title")  # fmt: skip
            comp["concept"] = {**comp["concept"], "name": new}
            rep.repairs.append(Repair(rule="R-name", element="treatment_strategies",
                                      target=a["label"], what=f"'{name}' -> '{new}' ({how})"))


def _arms(arms: list[dict], contrast: str | None, rep: RepairReport,
          texts: list[str] | None = None) -> list[dict]:  # fmt: skip
    arms = _pairs(arms, contrast, rep)
    t = [a for a in arms if not a["is_comparator"]]
    c = [a for a in arms if a["is_comparator"]]
    if len(t) == 1 and len(c) == 1:
        _comparator_rules(t[0], c[0], rep)
        if not t[0].get("grace_period"):  # R-grace
            t[0]["grace_period"] = dict(DEFAULT_GRACE)
            t[0]["grace_period_origin"] = "default"
            t[0]["grace_period_evidence"] = (f"{RULES_VERSION} R-grace: no grace period "
                                             "drafted; the default is used")  # fmt: skip
            rep.repairs.append(Repair(rule="R-grace", element="treatment_strategies",
                                      target=t[0]["label"],
                                      what=f"no grace period: {DEFAULT_GRACE['amount']} "
                                           f"{DEFAULT_GRACE['unit']}s (default)"))  # fmt: skip
        if not _is_null(c[0]) and c[0]["components"]:
            rep.flagged.append(f"the comparator '{c[0]['label']}' is an active treatment: "
                               "the active-comparator frame is used (patients who start "
                               "either within the grace period, by the one started first)")
    _names(arms, texts or [], rep)
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
        if _CHANGED.search(text):
            rep.repairs.append(Repair(rule="R-amend", element="eligibility", target=c["id"],
                                      what="amendment text (a criterion changed or clarified): "
                                           "dropped"))  # fmt: skip
            rep.flagged.append(f"{c['id']} says a criterion was changed ('{text[:80]}'); the "
                               "criterion as listed is kept: check it is the current version")
            continue
        if text.strip().endswith(":") and (_DATED.search(text) or _AMEND_HEAD.match(text)):
            rep.repairs.append(Repair(rule="R-amend", element="eligibility", target=c["id"],
                                      what="an amendment heading: dropped"))
            continue
        out.append(c)
    return out


def _subgroups(criteria: list[dict], rep: RepairReport) -> list[dict]:
    """R-subgroup: a sub-group heading and the criteria after it that name its condition."""
    out, words, kind = [], set(), None
    for c in criteria:
        text = (c.get("text") or "").strip()
        if words and c["kind"] == kind and _terms(text) & words:
            rep.repairs.append(Repair(rule="R-subgroup", element="eligibility", target=c["id"],
                                      what=f"defines the sub-group ({', '.join(sorted(words))})"
                                           ", not the trial: dropped"))  # fmt: skip
            continue
        words = set()
        if _SUBGROUP.search(text) and text.endswith(":"):
            words, kind = _terms(text) - _GENERIC, c["kind"]
            rep.repairs.append(Repair(rule="R-subgroup", element="eligibility", target=c["id"],
                                      what="a sub-group heading: dropped"))
            continue
        out.append(c)
    return out


def _anchors(criteria: list[dict], rep: RepairReport) -> list[dict]:
    out = json.loads(json.dumps(criteria))
    for c in out:
        for i, a in enumerate(c["atoms"], 1):
            t = a.get("temporal")
            if t and str(t.get("anchor") or "").strip().lower() in _RELATIONS:
                rep.repairs.append(Repair(rule="R-anchor", element="eligibility",
                                          target=f"{c['id']}.{i}" if len(c["atoms"]) > 1
                                          else c["id"], what=f"anchor '{t['anchor']}' is a "
                                          "relation's name: anchored at time zero"))  # fmt: skip
                t["anchor"] = "time zero"
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
    texts = [str(p.get("source", {}).get("title") or "")]
    arms = _arms(json.loads(json.dumps(p["treatment_strategies"])), contrast, rep, texts)
    elig = _values(_anchors(_subgroups(_amendments(p["eligibility"], rep), rep), rep), rep)
    changes = {}
    if arms != p["treatment_strategies"]:
        changes["treatment_strategies"] = arms
    if elig != p["eligibility"]:
        changes["eligibility"] = elig
    if changes:
        TargetTrialProtocol.model_validate({**p, **changes})
    return changes, rep
