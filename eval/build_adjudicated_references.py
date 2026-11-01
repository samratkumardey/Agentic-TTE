"""Adjudicated E3 references (eval/SCORING.md, Phase 3 changelog items 24-27).

    uv run python eval/build_adjudicated_references.py

Each ``<NCT>.adjudicated.json`` starts from the investigator's reference (``<NCT>.SKD.json``)
and applies the decisions recorded in ``adjudication.json``: ``"SKD"`` keeps the
investigator's version, ``"AI"`` takes the second annotator's, and a dict sets a value. The
two annotated files are never changed, so every difference between them and the adjudicated
reference is traceable to one recorded decision. Refuses to build while any decision is open.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

from make_reference_templates import check

EVAL = Path(__file__).parent
REFS = EVAL / "reference_protocols"
_ARM = re.compile(r"arm '(.+)' ")


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def _clean(path: Path) -> dict:
    return {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items()
            if not k.startswith("_")}  # fmt: skip


def _arm(p: dict, label: str) -> dict:
    return next(a for a in p["treatment_strategies"] if a["label"] == label)


def apply(skd: dict, ai: dict, decisions: list[dict]) -> dict:
    """The investigator's reference with every decision applied."""
    out = copy.deepcopy(skd)
    crit = {c["id"]: c for c in out["eligibility"]}
    ai_crit = {c["id"]: c for c in ai["eligibility"]}
    primary = next(o for o in out["outcomes"] if o.get("primary"))
    ai_primary = next(o for o in ai["outcomes"] if o.get("primary"))
    for d in decisions:
        dec, el, item = d["decision"], d["element"], d["item"]
        if dec is None:
            raise ValueError(f"decision {d['id']} ({el}: {item}) is still open")
        if dec == "SKD":
            continue
        if el == "eligibility":
            cid, what = item.split(" ", 1)
            c, a = crit[cid], ai_crit[cid]
            if what == "concepts" and dec == "AI":
                c["atoms"], c["atom_logic"] = copy.deepcopy(a["atoms"]), a["atom_logic"]
                c["notes"] = "; ".join(x for x in (c.get("notes"), a.get("notes")) if x) or None
            elif what == "category" and dec == "AI":
                c["category"] = a["category"]
                c["atoms"] = copy.deepcopy(a["atoms"]) if a["category"] == "clinical" else []
            else:
                raise ValueError(f"decision {d['id']}: cannot apply {dec!r} to {item}")
        elif el == "treatment":
            label = _ARM.match(item).group(1)
            arm, ai_arm = _arm(out, label), _arm(ai, label)
            if "components" in item and dec == "AI":
                arm["components"] = copy.deepcopy(ai_arm["components"])
            elif "grace" in item and dec == "AI":
                for k in ("grace_period", "grace_period_origin", "grace_period_evidence"):
                    arm[k] = ai_arm.get(k)
            elif "grace" in item and "value_hours" in dec:
                arm["grace_period"] = {"amount": dec["value_hours"], "unit": "hour"}
                arm["grace_period_origin"] = "user"
            elif "grace" in item and isinstance(dec, dict):
                arm.update(dec)
            else:
                raise ValueError(f"decision {d['id']}: cannot apply {dec!r} to {item}")
        elif el == "outcome" and dec == "AI":
            primary[item] = ai_primary.get(item)
            out["follow_up"][item] = ai["follow_up"].get(item)
        elif el == "confounders" and isinstance(dec, dict):
            drop = {_norm(n) for n in dec.get("remove", [])}
            conf = [c for c in out["assignment"]["confounders"] if _norm(c["name"]) not in drop]
            if len(conf) != len(out["assignment"]["confounders"]) - len(drop):
                raise ValueError(f"decision {d['id']}: a confounder to remove is not listed")
            out["assignment"]["confounders"] = conf + copy.deepcopy(dec.get("add", []))
        else:
            raise ValueError(f"decision {d['id']}: cannot apply {dec!r} to {el}: {item}")
    out["protocol_id"] = f"{skd['source']['nct_id']}-reference-adjudicated"
    out["provenance"] = {
        "created_by": "adjudication (eval/build_adjudicated_references.py)",
        "notes": "Investigator's reference (SKD) with the decisions in adjudication.json "
                 "applied; decisions marked audit=true go to the clinician audit."}  # fmt: skip
    return out


def main(argv: list[str]) -> int:
    decisions = json.loads((REFS / "adjudication.json").read_text(encoding="utf-8"))
    open_ = [d["id"] for d in decisions if d["decision"] is None]
    if open_:
        print(f"open decisions: {open_}")
        return 1
    bad = 0
    for f in sorted(REFS.glob("NCT*.SKD.json")):
        nct = f.name.split(".")[0]
        ref = apply(_clean(f), _clean(REFS / f"{nct}.AI.json"),
                    [d for d in decisions if d["nct_id"] == nct])  # fmt: skip
        path = REFS / f"{nct}.adjudicated.json"
        path.write_text(json.dumps(ref, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        problems = check(path)
        bad += bool(problems)
        print(f"{path.name}: " + ("complete" if not problems else "; ".join(problems)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
