"""Helpers that write a reviewer's answer file from the reviewer's own decisions, so that a
review can be answered with a command instead of hand-edited JSON.

Nothing here decides anything: every change (which arm is the treatment, which arms,
components or criteria go, the grace period) is an argument the reviewer gives, and the result
is checked against the protocol schema before it is written."""

from __future__ import annotations

import json
import re

from agentic_tte.protocol import TargetTrialProtocol
from agentic_tte.protocol.models import Duration, Origin


def parse_duration(text: str) -> Duration:
    """'24 hour', '3 day' -> Duration."""
    amount, _, unit = text.strip().partition(" ")
    return Duration(amount=float(amount), unit=unit.strip().rstrip("s") or unit)


_VALUE = re.compile(r"^\s*(<=|>=|<|>|=)\s*(-?[\d.]+)\s*(.*?)\s*$")
_WINDOW = re.compile(r"^\s*within\s+([\d.]+)\s+(\w+?)s?\s+before\s*$")
EDITS = ("no-value", "value", "window", "drop-atom", "atom-logic")


def _target(text: str) -> tuple[str, int | None]:
    """'I2' -> ('I2', None): every atom; 'I5.3' -> ('I5', 3): the third atom."""
    cid, dot, n = text.strip().partition(".")
    if dot and not n.isdigit():
        raise ValueError(f"{text!r}: give a criterion ID, or ID.n for its n-th atom")
    return cid, int(n) if dot else None


def edit_criteria(criteria: list[dict],
                  edits: list[tuple[str, str]]) -> tuple[list[dict], list[str]]:  # fmt: skip
    """Apply the reviewer's edits to eligibility criteria (copies; the input is unchanged).

    Each edit is (operation, "TARGET" or "TARGET=ARGUMENT"); TARGET is a criterion ID (every
    atom) or ID.n (its n-th atom, numbered as in the draft):
    - ``no-value I2.1``: remove the atom's value constraint (e.g. a duration drafted as a value);
    - ``value E11.1=< 30 K/uL``: set the threshold, in the unit the data records (units are not
      converted);
    - ``window I2.1=before`` (any record up to time zero), ``within 6 hour before``, or ``none``
      (no temporal constraint: any record up to time zero; a measurement's latest value in the
      lookback). Windows after time zero are refused (look-ahead);
    - ``drop-atom I5.3``: remove the atom;
    - ``atom-logic I5=any``: whether all atoms or any one must hold.
    Every edited criterion gets a note saying what the reviewer changed."""
    out = {c["id"]: json.loads(json.dumps(c)) for c in criteria}
    drop: dict[str, set[int]] = {}
    done: dict[str, list[str]] = {}
    for op, spec in edits:
        if op not in EDITS:
            raise ValueError(f"unknown edit {op!r}; the edits are {EDITS}")
        target, eq, arg = spec.partition("=")
        cid, n = _target(target)
        if cid not in out:
            raise ValueError(f"{op} {spec!r}: no criterion {cid!r}")
        crit = out[cid]
        if op == "atom-logic":
            if arg.strip() not in ("all", "any"):
                raise ValueError(f"atom-logic {spec!r}: give {cid}=all or {cid}=any")
            crit["atom_logic"] = arg.strip()
            done.setdefault(cid, []).append(f"atom logic {arg.strip()}")
            continue
        atoms = crit["atoms"]
        if n is not None and not 1 <= n <= len(atoms):
            raise ValueError(f"{op} {spec!r}: {cid} has {len(atoms)} atom(s)")
        idx = [n] if n is not None else list(range(1, len(atoms) + 1))
        if op in ("value", "window") and not eq:
            raise ValueError(f"{op} {spec!r}: give {target}=...")
        for i in idx:
            a = atoms[i - 1]
            names = "/".join(c["name"] for c in a["concepts"]) or "atom"
            if op == "no-value":
                a["value"] = None
                done.setdefault(cid, []).append(f"{names}: value removed")
            elif op == "value":
                m = _VALUE.match(arg)
                if not m:
                    raise ValueError(f"value {spec!r}: give e.g. {target}=< 30 K/uL")
                a["value"] = {"comparator": m[1], "value": float(m[2]), "upper": None,
                              "unit": m[3] or None, "text": arg.strip()}  # fmt: skip
                done.setdefault(cid, []).append(f"{names}: value {arg.strip()}")
            elif op == "window":
                w = arg.strip().lower()
                if w == "none":
                    a["temporal"] = None
                elif w == "before":
                    a["temporal"] = {"relation": "x_before_y", "window": None,
                                     "anchor": "time_zero", "text": "before time zero"}
                elif (m := _WINDOW.match(w)):
                    d = Duration(amount=float(m[1]), unit=m[2])
                    a["temporal"] = {"relation": "x_within_t_before_y",
                                     "window": d.model_dump(mode="json"), "anchor": "time_zero",
                                     "text": f"{w} time zero"}  # fmt: skip
                elif "after" in w:
                    raise ValueError(f"window {spec!r}: a window after time zero is look-ahead")
                else:
                    raise ValueError(f"window {spec!r}: give before, none, or "
                                     "'within 6 hour before'")
                done.setdefault(cid, []).append(f"{names}: window {w}")
            elif op == "drop-atom":
                if n is None:
                    raise ValueError(f"drop-atom {spec!r}: give {cid}.n (drop-criterion removes "
                                     "a whole criterion)")
                drop.setdefault(cid, set()).add(i)
                done.setdefault(cid, []).append(f"{names}: atom removed")
    for cid, gone in drop.items():
        atoms = out[cid]["atoms"]
        out[cid]["atoms"] = [a for i, a in enumerate(atoms, 1) if i not in gone]
        if not out[cid]["atoms"]:
            raise ValueError(f"{cid}: every atom removed; use --drop-criterion {cid}")
    notes = []
    for cid, what in done.items():
        text = "edited at design review: " + "; ".join(what)
        out[cid]["notes"] = f"{out[cid]['notes']}; {text}" if out[cid].get("notes") else text
        notes.append(f"{cid}: " + "; ".join(what))
    return [out[c["id"]] for c in criteria], notes


def compile_problems(criteria: list[dict]) -> list[tuple[str, str, str]]:
    """(kind, criterion ID, reason) for each atom the compiler will skip whatever its concepts:
    the design problems a reviewer can still fix at design review."""
    from agentic_tte.omop.compiler import T0_ALIASES, _norm, design_problem
    from agentic_tte.protocol.models import Criterion

    out = []
    for raw in criteria:
        c = Criterion.model_validate(raw)
        if not c.emulable:
            continue
        for i, a in enumerate(c.atoms, 1):
            t0 = a.temporal is None or _norm(a.temporal.anchor) in T0_ALIASES
            why = design_problem(c, a, anchor_is_t0=t0)
            if why:
                out.append((c.kind, f"{c.id}.{i}" if len(c.atoms) > 1 else c.id, why))
    return out


def design_answer(protocol: dict, template: dict, reviewer: str, *, treatment: str | None = None,
                  drop_arms: tuple[str, ...] = (), drop_components: tuple[str, ...] = (),
                  drop_criteria: tuple[str, ...] = (), grace: Duration | None = None,
                  criterion_edits: tuple[tuple[str, str], ...] = ()
                  ) -> tuple[dict, list[str]]:  # fmt: skip
    """The design answer for ``protocol`` (the paused review's protocol.json), starting from
    the review's template (which keeps "confounders": null, the Clinician's proposal, unless
    the reviewer replaces it). Returns the answer and a summary to show the reviewer."""
    if not reviewer.strip():
        raise ValueError("the design answer needs a reviewer")
    arms = [dict(a) for a in protocol["treatment_strategies"]]
    labels = {a["label"] for a in arms}
    for label in (*drop_arms, *([treatment] if treatment else [])):
        if label not in labels:
            raise ValueError(f"no arm labelled {label!r}; the arms are {sorted(labels)}")
    if treatment in drop_arms:
        raise ValueError(f"arm {treatment!r} is both the treatment and dropped")
    arms = [a for a in arms if a["label"] not in drop_arms]
    unused = set(drop_components)
    for a in arms:
        kept = []
        for c in a["components"]:
            hit = [t for t in drop_components if t.lower() in c["concept"]["name"].lower()]
            unused -= set(hit)
            if not hit:
                kept.append(c)
        a["components"] = kept
        if treatment is not None:
            a["is_comparator"] = a["label"] != treatment
    if unused:
        raise ValueError(f"no component of the kept arms matches {sorted(unused)}")
    treated = [a for a in arms if not a["is_comparator"]]
    if grace is not None:
        if len(treated) != 1:
            raise ValueError("a grace period needs exactly one treatment arm")
        treated[0].update(grace_period=grace.model_dump(mode="json"),
                          grace_period_origin=Origin.USER.value,
                          grace_period_evidence=None)  # fmt: skip
    ids = [c["id"] for c in protocol["eligibility"]]
    missing = set(drop_criteria) - set(ids)
    if missing:
        raise ValueError(f"no criteria {sorted(missing)}; the criteria are {ids}")
    changes = {}
    if drop_arms or drop_components or treatment is not None or grace is not None:
        changes["treatment_strategies"] = arms
    edited = []
    if drop_criteria or criterion_edits:
        kept_criteria = [c for c in protocol["eligibility"] if c["id"] not in drop_criteria]
        kept_criteria, edited = edit_criteria(kept_criteria, list(criterion_edits))
        changes["eligibility"] = kept_criteria
    TargetTrialProtocol.model_validate({**protocol, **changes})  # the shape new_version needs
    answer = {**template, "reviewer": reviewer, "changes": changes}

    out = []
    for a in arms:
        g = a.get("grace_period")
        gtext = "{:g} {}".format(g["amount"], g["unit"]) if g else "none"
        role = "comparator" if a["is_comparator"] else "TREATMENT"
        names = [c["concept"]["name"] for c in a["components"]]
        out.append(f"arm {a['label']}: {role}, components {names}, grace {gtext}")
    comparators = [a for a in arms if a["is_comparator"]]
    if len(treated) != 1 or len(comparators) != 1:
        out.append(f"WARNING: {len(treated)} treatment and {len(comparators)} comparator arms; "
                   "the compiler needs exactly one of each")
    if len(treated) == 1 and not treated[0].get("grace_period"):
        out.append("WARNING: the treatment arm has no grace period; if treatment starts after "
                   "time zero, the Protocol Auditor (A-a) sends the design back")
    kept = [i for i in ids if i not in drop_criteria]
    out.append(f"eligibility: {len(kept)} kept {kept}" +
               (f"; dropped {[i for i in ids if i in drop_criteria]}" if drop_criteria else ""))
    out += [f"edited {e}" for e in edited]
    for kind, cid, why in compile_problems(changes.get("eligibility", protocol["eligibility"])):
        flag = "WARNING" if kind == "inclusion" else "note"
        out.append(f"{flag}: {kind} {cid} will not compile ({why}); fix it here, or it is "
                   "reported and left out" + (" of the cohort definition" if flag == "WARNING"
                                              else ""))  # fmt: skip
    conf = answer.get("confounders")
    out.append("confounders: " + ("the Clinician's proposal, kept" if conf is None
                                  else f"{len(conf)} given by the reviewer"))  # fmt: skip
    return answer, out


ACTIONS = ("approve", "replace", "surrogate", "unmappable", "reject")


def _key(decisions: dict, given: str) -> str:
    """A decisions key, given exactly or by a unique case-insensitive prefix ('I4:SOFA')."""
    if given in decisions:
        return given
    hits = [k for k in decisions if k.lower().startswith(given.lower())]
    if len(hits) != 1:
        raise ValueError(f"{given!r} matches {len(hits)} keys" + (f": {hits}" if hits else
                         "; the keys are in decisions.json"))  # fmt: skip
    return hits[0]


def _ids(text: str) -> list[int]:
    try:
        return [int(i) for i in text.replace(" ", "").split(",") if i]
    except ValueError as e:
        raise ValueError(f"{text!r}: give concept IDs as 123,456") from e


def _event(text: str) -> dict:
    """'Drug=1321341,1343916' -> an EventDef dict."""
    domain, eq, ids = text.partition("=")
    if not eq or not _ids(ids):
        raise ValueError(f"{text!r}: give Domain=ID,ID (e.g. Drug=1321341,1343916)")
    return {"domain": domain.strip(), "concept_ids": _ids(ids)}


def concepts_answer(decisions: dict, spec: dict, reviewer: str, *,
                    actions: list[tuple[str, str]] = (), notes: list[str] = (),
                    time_zero: str | None = None, t0_occurrence: str | None = None,
                    t0_note: str | None = None, anchors: list[str] = (),
                    lookback: Duration | None = None,
                    book=None) -> tuple[dict, dict, list[str]]:  # fmt: skip
    """The concept-review answer (decisions and compile spec) from the reviewer's decisions.

    ``actions``: (action, "KEY" or "KEY=ID,ID"): approve keeps the proposed IDs (or the IDs
    given), replace and surrogate need IDs, unmappable and reject need none. ``notes``:
    "KEY=text" (a surrogate needs one). ``time_zero``: "Domain=ID,ID"; ``anchors``:
    "name=Domain=ID,ID". Keys not named keep their pre-filled decision. With ``book`` (the
    review's book.json), the result is checked as the Replay will apply it."""
    from agentic_tte.omop.compiler import CompileSpec

    if not reviewer.strip():
        raise ValueError("the concept answer needs a reviewer")
    dec = json.loads(json.dumps(decisions))
    entries = dec["decisions"]
    changed = []
    for action, given in actions:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; the actions are {ACTIONS}")
        k, eq, ids = given.partition("=")
        key = _key(entries, k.strip())
        e = entries[key]
        if action in ("replace", "surrogate") and not eq:
            raise ValueError(f"{action} {key!r} needs concept IDs: {k}=ID,ID")
        if action in ("unmappable", "reject"):
            e["concept_ids"] = []
        elif eq:
            e["concept_ids"] = _ids(ids)
        elif not e.get("concept_ids"):
            raise ValueError(f"approve {key!r}: nothing proposed; give {k}=ID,ID")
        if action != "surrogate" and e.get("action") == "surrogate":
            e["note"] = ""  # the Clinician's surrogate note no longer applies
        e["action"] = action
        changed.append(key)
    for given in notes:
        k, eq, text = given.partition("=")
        if not eq or not text.strip():
            raise ValueError(f"--note {given!r}: give KEY=text")
        entries[_key(entries, k.strip())]["note"] = text.strip()
    for key, e in entries.items():
        if e["action"] == "surrogate" and not (e.get("note") or "").strip():
            raise ValueError(f"{key}: a surrogate needs a note saying why (--note)")
    dec["reviewer"] = reviewer

    sp = json.loads(json.dumps(spec))
    if time_zero:
        sp["time_zero"].update(_event(time_zero))
    if t0_occurrence:
        sp["time_zero"]["occurrence"] = t0_occurrence
    if t0_note:
        sp["time_zero"]["note"] = t0_note
    for given in anchors:
        name, eq, event = given.partition("=")
        if not eq or name.strip() not in sp.get("anchors", {}):
            raise ValueError(f"--anchor {given!r}: give name=Domain=ID,ID; the anchors are "
                             f"{sorted(sp.get('anchors', {}))}")
        sp["anchors"][name.strip()].update(_event(event))
    if lookback is not None:
        sp["measurement_lookback"] = lookback.model_dump(mode="json")
    if not sp["time_zero"].get("concept_ids"):
        raise ValueError("time zero has no concept IDs: give --time-zero Domain=ID,ID")
    empty = [a for a, ev in sp.get("anchors", {}).items() if not ev.get("concept_ids")]
    sp["anchors"] = {a: ev for a, ev in sp.get("anchors", {}).items() if ev.get("concept_ids")}
    CompileSpec.model_validate({**sp, "concept_sets": {}})
    if book is not None:
        from agentic_tte.omop.concept_sets import apply_decisions

        apply_decisions(book, dec)  # raises on an unknown key or a bad decision

    out = [f"{k}: {entries[k]['action']} {entries[k]['concept_ids'] or ''}".rstrip()
           for k in changed]
    counts = {a: sum(e["action"] == a for e in entries.values()) for a in ACTIONS}
    out.append("decisions: " + ", ".join(f"{n} {a}" for a, n in counts.items() if n))
    t0 = sp["time_zero"]
    out.append(f"time zero: {t0['occurrence']} {t0['domain']} record of {t0['concept_ids']}")
    if empty:
        out.append(f"anchors left empty (criteria using them are reported, not compiled): "
                   f"{empty}")
    return dec, sp, out
