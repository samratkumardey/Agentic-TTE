"""E3 reference agreement: investigator vs LLM second annotator (eval/SCORING.md, Phase 3
changelog item 24).

    uv run python eval/compare_references.py [--a SKD] [--b AI]

Compares ``eval/reference_protocols/<NCT>.<a>.json`` with ``<NCT>.<b>.json`` element by element,
before adjudication, and writes:
- ``eval/results/e3_reference_agreement.md``: agreement per element and per trial;
- ``eval/reference_protocols/adjudication.json``: every disagreement, side by side, with an empty
  ``decision`` for the investigator (then the clinician) to fill;
- ``eval/reference_protocols/adjudication.md``: the same, to read.

Concept matching follows Phase 0 part B (``phase0_scoring``): names normalized, matched
one-to-one within a criterion; *relaxed* = name + domain, *strict* = relaxed + the atom's value
and time constraint. Annotators word concepts differently ("PaO2/FiO2 ratio" vs "P/F ratio"),
so a *lenient* name match (exploratory, not pre-registered) is reported next to them. Time zero
is free text: it is never scored automatically; both anchors go to adjudication with a rating.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from agentic_tte.protocol import TargetTrialProtocol
from agentic_tte.protocol.models import TemporalConstraint

sys.path.insert(0, str(Path(__file__).parent))
from phase0_scoring import (  # noqa: E402
    PRF,
    concept_prf,
    load_synonyms,
    norm_name,
    norm_text,
    temporal_key,
)

EVAL = Path(__file__).parent
REFS = EVAL / "reference_protocols"
RESULTS = EVAL / "results"
HOURS = {"minute": 1 / 60, "hour": 1, "day": 24, "week": 168, "month": 730.5, "year": 8766}
NAMES = {"NCT00299650": "ACURASYS", "NCT00318942": "CRISTAL", "NCT00625209": "APROCCHSS",
         "NCT00935168": "CHEST", "NCT01448109": "ADRENAL", "NCT01636089": "HYDRAREA",
         "NCT02476253": "BICAR-ICU", "NCT02517489": "CAPE COD", "NCT02568722": "STARRT-AKI",
         "NCT02643381": "EvK", "NCT03637205": "ECLS-SHOCK", "NCT03680274": "LOVIT"}  # fmt: skip


def hours(d) -> float | None:
    return None if d is None else round(d.amount * HOURS[d.unit], 3)


def grace_hours(d) -> float | None:
    """A 0-hour grace period is no grace period: the Statistician reads both the same way."""
    return hours(d) or None


AT_T0 = TemporalConstraint(relation="x_at_time_of_y")


def compiled_keys(c) -> Counter:
    """Strict keys as the compiler reads them: no time constraint = at time zero (both compile
    to the latest value / any record up to time zero), and units are not used (the compiler
    does not convert them)."""
    out = Counter()
    for a in c.atoms:
        v = a.value
        vk = None if v is None else (("present", norm_text(v.text)) if v.comparator == "present"
                                     else (v.comparator, v.value, v.upper))  # fmt: skip
        tk = temporal_key(a.temporal or AT_T0)
        for x in a.concepts:
            out[(norm_name(x.name), x.domain, vk, tk)] += 1
    return out


def lenient(a: str, b: str) -> bool:
    """Same concept despite wording: equal, one name's words contain the other's (2+ words),
    or near-identical spelling."""
    a, b = norm_text(a), norm_text(b)
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(small) >= 2 and small <= big:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.85


def match_count(xs: list[str], ys: list[str]) -> int:
    """Greedy one-to-one lenient matches between two name lists."""
    left, n = list(ys), 0
    for x in xs:
        for i, y in enumerate(left):
            if lenient(x, y):
                n += 1
                del left[i]
                break
    return n


def names_prf(a: list[str], b: list[str]) -> PRF:
    """``a`` scored against ``b`` with lenient names; F1 is symmetric (Dice)."""
    out = PRF()
    out.tp, out.pred, out.ref = match_count(a, b), len(a), len(b)
    return out


def kappa(pairs: list[tuple]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / n**2
    return None if pe == 1 else (po - pe) / (1 - pe)


def concepts_of(c) -> list[str]:
    return [x.name for a in c.atoms for x in a.concepts]


def arm_view(p: TargetTrialProtocol) -> dict:
    return {a.label: {"is_comparator": a.is_comparator,
                      "components": [x.concept.name for x in a.components],
                      "grace_hours": grace_hours(a.grace_period)}
            for a in p.treatment_strategies}  # fmt: skip


def outcome_view(p: TargetTrialProtocol) -> dict:
    o = next((o for o in p.outcomes if o.primary), None)
    h = (o.horizon if o and o.horizon else None) or p.follow_up.horizon
    end = (o.end_event if o else None) or p.follow_up.end_event
    rule = (o.end_rule if o else None) or p.follow_up.end_rule
    return {"horizon_hours": hours(h), "end_event": norm_text(end) or None, "end_rule": rule}


def compare(a: TargetTrialProtocol, b: TargetTrialProtocol, syn: dict) -> dict:
    """Agreement of one trial's two references, plus the list of disagreements."""
    nct = a.source.nct_id
    out: dict = {"nct_id": nct, "name": NAMES.get(nct, nct), "disagreements": []}
    dis = out["disagreements"]

    def flag(element, item, va, vb, **extra):
        dis.append({"nct_id": nct, "element": element, "item": item, "a": va, "b": vb,
                    "decision": None, "note": None, **extra})  # fmt: skip

    # Eligibility.
    b_by_id = {c.id: c for c in b.eligibility}
    cats, n_same_atoms, lenient_prf, compiled_prf = [], 0, PRF(), PRF()
    for c in a.eligibility:
        d = b_by_id[c.id]
        cats.append((c.category, d.category))
        compiled_prf.add(compiled_keys(d), compiled_keys(c))
        if c.category != d.category:
            flag("eligibility", f"{c.id} category", c.category, d.category, text=c.text)
        lenient_prf += names_prf(concepts_of(d), concepts_of(c))
        both_clinical = c.category == d.category == "clinical"
        if both_clinical:
            n_same_atoms += len(c.atoms) == len(d.atoms)
            if match_count(concepts_of(d), concepts_of(c)) < max(len(concepts_of(c)),
                                                                  len(concepts_of(d))):
                flag("eligibility", f"{c.id} concepts", concepts_of(c), concepts_of(d),
                     text=c.text)
    n_clin = sum(x == y == "clinical" for x, y in cats)
    out["eligibility"] = {
        "criteria": len(cats),
        "category_agree": sum(x == y for x, y in cats),
        "category_pairs": cats,
        "both_clinical": n_clin,
        "same_atom_count": n_same_atoms,
        "relaxed": concept_prf(b, a, strict=False, synonyms=syn),
        "strict": concept_prf(b, a, strict=True, synonyms=syn),
        "compiled": compiled_prf,
        "lenient": lenient_prf,
    }

    # Treatment strategies.
    va, vb = arm_view(a), arm_view(b)
    common = sorted(set(va) & set(vb))
    same_arms = set(va) == set(vb)
    if not same_arms:
        flag("treatment", "arms kept", sorted(va), sorted(vb))
    comp_agree = all(va[k]["is_comparator"] == vb[k]["is_comparator"] for k in common)
    if not comp_agree:
        flag("treatment", "comparator", {k: va[k]["is_comparator"] for k in common},
             {k: vb[k]["is_comparator"] for k in common})
    treat = [k for k in common if not va[k]["is_comparator"] and not vb[k]["is_comparator"]]
    comp_prf, grace_same = PRF(), True
    for k in treat:
        comp_prf += names_prf(vb[k]["components"], va[k]["components"])
        if match_count(vb[k]["components"], va[k]["components"]) < max(
                len(va[k]["components"]), len(vb[k]["components"])):  # fmt: skip
            flag("treatment", f"arm '{k}' components", va[k]["components"], vb[k]["components"])
        if va[k]["grace_hours"] != vb[k]["grace_hours"]:
            grace_same = False
            flag("treatment", f"arm '{k}' grace period (hours)", va[k]["grace_hours"],
                 vb[k]["grace_hours"])
    out["treatment"] = {"same_arms": same_arms, "comparator_agree": comp_agree and bool(common),
                        "components": comp_prf, "grace_agree": grace_same}  # fmt: skip

    # Time zero: never scored automatically.
    flag("time_zero", "anchor event", a.time_zero.anchor_event, b.time_zero.anchor_event,
         rating=None, rating_scale="same | acceptable alternative | different")

    # Outcome and follow-up.
    oa, ob = outcome_view(a), outcome_view(b)
    out["outcome"] = {k: oa[k] == ob[k] for k in oa}
    for k in oa:
        if oa[k] != ob[k]:
            flag("outcome", k, oa[k], ob[k])

    # Causal contrast, estimand.
    out["contrast_agree"] = a.causal_contrast == b.causal_contrast
    out["estimand_agree"] = a.analysis_plan.estimand == b.analysis_plan.estimand
    if not out["contrast_agree"]:
        flag("causal_contrast", "contrast", a.causal_contrast, b.causal_contrast)
    if not out["estimand_agree"]:
        flag("analysis_plan", "estimand", a.analysis_plan.estimand, b.analysis_plan.estimand)

    # Confounders: names only (domains are a vocabulary question).
    ca = [norm_name(c.name, syn) for c in a.assignment.confounders]
    cb = [norm_name(c.name, syn) for c in b.assignment.confounders]
    out["confounders"] = names_prf(cb, ca)
    unmatched_a = [x for x in ca if match_count([x], cb) == 0]
    unmatched_b = [x for x in cb if match_count([x], ca) == 0]
    if unmatched_a or unmatched_b:
        flag("confounders", "only one annotator listed", unmatched_a, unmatched_b)
    return out


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def render(rows: list[dict], la: str, lb: str) -> str:
    tot = {k: PRF() for k in ("relaxed", "strict", "compiled", "lenient")}
    conf, comp = PRF(), PRF()
    cats = []
    for r in rows:
        for k in tot:
            tot[k] += r["eligibility"][k]
        conf += r["confounders"]
        comp += r["treatment"]["components"]
        cats += r["eligibility"]["category_pairs"]
    n = len(rows)

    def share(f):
        return f"{sum(1 for r in rows if f(r))}/{n}"

    lines = [
        f"# E3 reference agreement: {la} vs {lb}",
        "",
        f"Before adjudication (eval/SCORING.md, Phase 3 changelog item 24). {la}: investigator; "
        f"{lb}: LLM second annotator. F1 is symmetric; precision and recall read {lb} against "
        f"{la}. *Lenient* concept matching is exploratory, not pre-registered.",
        "",
        "## Summary over the 12 core trials",
        "",
        "| Element | Agreement |",
        "|---|---|",
        f"| Eligibility: category (clinical vs administrative) | "
        f"{sum(x == y for x, y in cats)}/{len(cats)} criteria, kappa "
        f"{kappa(cats) if kappa(cats) is None else round(kappa(cats), 2)} |",
        f"| Eligibility: concepts, strict F1 | {pct(tot['strict'].f1)} |",
        f"| Eligibility: concepts, strict as compiled F1 (no time constraint = at time zero; "
        f"units ignored) | {pct(tot['compiled'].f1)} |",
        f"| Eligibility: concepts, relaxed F1 (name + domain) | {pct(tot['relaxed'].f1)} |",
        f"| Eligibility: concepts, lenient names F1 | {pct(tot['lenient'].f1)} |",
        f"| Treatment: same arms kept | {share(lambda r: r['treatment']['same_arms'])} trials |",
        f"| Treatment: same comparator | "
        f"{share(lambda r: r['treatment']['comparator_agree'])} trials |",
        f"| Treatment: components, lenient F1 | {pct(comp.f1)} |",
        f"| Treatment: same grace period | "
        f"{share(lambda r: r['treatment']['grace_agree'])} trials |",
        "| Time zero | rated at adjudication (free text) |",
        f"| Outcome: same horizon | {share(lambda r: r['outcome']['horizon_hours'])} trials |",
        f"| Outcome: same end event | {share(lambda r: r['outcome']['end_event'])} trials |",
        f"| Causal contrast | {share(lambda r: r['contrast_agree'])} trials |",
        f"| Estimand | {share(lambda r: r['estimand_agree'])} trials |",
        f"| Confounders, lenient F1 | {pct(conf.f1)} |",
        "",
        "## Per trial",
        "",
        "| Trial | Criteria (category agree) | Concepts strict / as compiled / relaxed / lenient "
        "F1 | Comparator | "
        "Components F1 | Grace | Horizon | Contrast | Estimand | Confounders F1 | "
        "Disagreements |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    yn = {True: "same", False: "differs"}
    for r in rows:
        e, t = r["eligibility"], r["treatment"]
        lines.append(
            f"| {r['name']} | {e['criteria']} ({e['category_agree']}) | "
            f"{pct(e['strict'].f1)} / {pct(e['compiled'].f1)} / {pct(e['relaxed'].f1)} / "
            f"{pct(e['lenient'].f1)} | "
            f"{yn[t['comparator_agree']]} | {pct(t['components'].f1)} | {yn[t['grace_agree']]} | "
            f"{yn[r['outcome']['horizon_hours']]} | {yn[r['contrast_agree']]} | "
            f"{yn[r['estimand_agree']]} | {pct(r['confounders'].f1)} | "
            f"{len(r['disagreements'])} |")  # fmt: skip
    lines += ["", f"Disagreements to adjudicate: {sum(len(r['disagreements']) for r in rows)} "
              "(including one time-zero rating per trial), in "
              "`eval/reference_protocols/adjudication.md`.", ""]  # fmt: skip
    return "\n".join(lines)


def render_adjudication(items: list[dict], la: str, lb: str) -> str:
    lines = [f"# Adjudication: {la} vs {lb}", "",
             f"Record each decision in `adjudication.json` (`decision`: `{la}`, `{lb}` or your own "
             "value; `note`: why). Time zero also takes a `rating`: same, acceptable "
             "alternative, or different. Items you cannot settle go to the clinician.",
             ""]  # fmt: skip
    cur = None
    for i, x in enumerate(items):
        if x["nct_id"] != cur:
            cur = x["nct_id"]
            lines += [f"## {NAMES.get(cur, cur)} ({cur})", ""]
        lines.append(f"**{i}. {x['element']}: {x['item']}**")
        if x.get("text"):
            lines.append(f"- Registry text: {x['text'][:300]}")
        lines += [f"- {la}: {json.dumps(x['a'], ensure_ascii=False)}",
                  f"- {lb}: {json.dumps(x['b'], ensure_ascii=False)}", ""]  # fmt: skip
    return "\n".join(lines)


def load(path: Path) -> TargetTrialProtocol:
    """A filled reference; underscore keys (registry hints, the annotator's own notes) are not
    part of the protocol."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return TargetTrialProtocol.model_validate({k: v for k, v in data.items()
                                               if not k.startswith("_")})  # fmt: skip


def main(argv: list[str]) -> int:
    la = argv[argv.index("--a") + 1] if "--a" in argv else "SKD"
    lb = argv[argv.index("--b") + 1] if "--b" in argv else "AI"
    syn = load_synonyms()
    rows = []
    for fa in sorted(REFS.glob(f"NCT*.{la}.json")):
        fb = fa.with_name(fa.name.replace(f".{la}.", f".{lb}."))
        if not fb.exists():
            print(f"missing {fb.name}")
            return 1
        pa, pb = (load(f) for f in (fa, fb))
        rows.append(compare(pa, pb, syn))
    items = [d for r in rows for d in r["disagreements"]]
    for i, x in enumerate(items):
        x["id"] = i
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e3_reference_agreement.md").write_text(render(rows, la, lb), encoding="utf-8")
    (REFS / "adjudication.md").write_text(render_adjudication(items, la, lb), encoding="utf-8")
    adj = REFS / "adjudication.json"
    if adj.exists() and any(x.get("decision") for x in json.loads(adj.read_text("utf-8"))):
        print(f"{adj.name} already holds decisions: not overwritten")
    else:
        adj.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(render(rows, la, lb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
