"""Helpers that write a reviewer's answer file from the reviewer's own decisions, so that a
review can be answered with a command instead of hand-edited JSON.

Nothing here decides anything: every change (which arm is the treatment, which arms,
components or criteria go, the grace period) is an argument the reviewer gives, and the result
is checked against the protocol schema before it is written."""

from __future__ import annotations

from agentic_tte.protocol import TargetTrialProtocol
from agentic_tte.protocol.models import Duration, Origin


def parse_duration(text: str) -> Duration:
    """'24 hour', '3 day' -> Duration."""
    amount, _, unit = text.strip().partition(" ")
    return Duration(amount=float(amount), unit=unit.strip().rstrip("s") or unit)


def design_answer(protocol: dict, template: dict, reviewer: str, *, treatment: str | None = None,
                  drop_arms: tuple[str, ...] = (), drop_components: tuple[str, ...] = (),
                  drop_criteria: tuple[str, ...] = (),
                  grace: Duration | None = None) -> tuple[dict, list[str]]:  # fmt: skip
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
    if drop_criteria:
        changes["eligibility"] = [c for c in protocol["eligibility"]
                                  if c["id"] not in drop_criteria]  # fmt: skip
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
    conf = answer.get("confounders")
    out.append("confounders: " + ("the Clinician's proposal, kept" if conf is None
                                  else f"{len(conf)} given by the reviewer"))  # fmt: skip
    return answer, out
