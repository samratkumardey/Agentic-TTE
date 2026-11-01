"""E3: the Trialist's protocols scored against the adjudicated references (eval/SCORING.md,
Phase 3 changelog items 11b and 28).

    uv run python eval/score_e3.py runs/exit-H100-core [--model qwen3_32b]

Replicate 1 is the analysis set; replicate 2 is checked for identity only (part D). Each draft
is compared with ``eval/reference_protocols/<NCT>.adjudicated.json`` by the same functions that
compared the two annotators (``compare_references``), so agent and annotators are measured
alike. Criteria the extractor adds from structured registry fields (``S*``, e.g. minimum age)
have no counterpart in the references, whose templates hold the free-text criteria only; they
are listed, not scored. Time zero is free text: the draft and reference anchors go to a rating
sheet (``eval/results/e3_time_zero_ratings.json``), never scored automatically.

Writes ``eval/results/e3_<model>.md``. Uses protocols only (public registry text), no data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from compare_references import PRF, compare, compiled_keys, load, pct
from phase0_scoring import criterion_keys, element_identity

from agentic_tte.protocol import TargetTrialProtocol

EVAL = Path(__file__).parent
REFS = EVAL / "reference_protocols"
RESULTS = EVAL / "results"


def _draft(path: Path) -> TargetTrialProtocol:
    return TargetTrialProtocol.model_validate_json(path.read_text(encoding="utf-8"))


def eligibility_grade(ref: TargetTrialProtocol, draft: TargetTrialProtocol) -> dict:
    """Part C: share of the reference's clinical criteria whose concepts all match strictly and
    whose atom count is right; also as the compiler reads them."""
    d = {c.id: c for c in draft.eligibility}
    rows = [(c, d[c.id]) for c in ref.eligibility if c.category == "clinical"]
    strict = sum(criterion_keys(c, True) == criterion_keys(x, True) and
                 len(c.atoms) == len(x.atoms) for c, x in rows)  # fmt: skip
    comp = sum(compiled_keys(c) == compiled_keys(x) and len(c.atoms) == len(x.atoms)
               for c, x in rows)  # fmt: skip
    n = len(rows)

    def grade(k):
        s = k / n if n else None
        return None if s is None else ("correct" if s >= 0.9 else "partly" if s >= 0.5
                                       else "wrong")  # fmt: skip
    return {"n": n, "strict": strict, "compiled": comp, "grade_strict": grade(strict),
            "grade_compiled": grade(comp)}  # fmt: skip


def score(run: Path, model: str) -> dict:
    rows, sheet = [], []
    for ref_path in sorted(REFS.glob("NCT*.adjudicated.json")):
        nct = ref_path.name.split(".")[0]
        ref = load(ref_path)
        d1 = _draft(run / model / "rep1" / f"{nct}.protocol.json")
        d2 = _draft(run / model / "rep2" / f"{nct}.protocol.json")
        structured = [c for c in d1.eligibility if c.source == "registry_structured"]
        d1s = d1.model_copy(update={"eligibility": [c for c in d1.eligibility
                                                    if c.source != "registry_structured"]})
        r = compare(ref, d1s, {})
        r["structured"] = [c.text for c in structured]
        r["grade"] = eligibility_grade(ref, d1s)
        r["rep_identical"] = all(element_identity(d1, d2).values())
        r["draft_contrast_origin"] = d1.causal_contrast_origin
        r["n_conf_draft"] = len(d1.assignment.confounders)
        rows.append(r)
        sheet.append({"nct_id": nct, "reference": ref.time_zero.anchor_event,
                      "draft": d1.time_zero.anchor_event if d1.time_zero else None,
                      "rating": None,
                      "rating_scale": "same | acceptable alternative | different"})
    return {"rows": rows, "sheet": sheet}


def render(s: dict, run: Path, model: str) -> str:
    rows = s["rows"]
    tot = {k: PRF() for k in ("strict", "compiled", "relaxed", "lenient")}
    comp, conf = PRF(), PRF()
    for r in rows:
        for k in tot:
            tot[k] += r["eligibility"][k]
        comp += r["treatment"]["components"]
        conf += r["confounders"]
    n = len(rows)

    def share(f):
        return f"{sum(1 for r in rows if f(r))}/{n}"

    def prf(x: PRF) -> str:
        return f"P {pct(x.precision)}, R {pct(x.recall)}, F1 {pct(x.f1)}"

    lines = [
        f"# E3: {model} protocols vs the adjudicated references",
        "",
        f"Run `{run.as_posix()}`, replicate 1 (analysis set); replicate 2 identical in "
        f"{share(lambda r: r['rep_identical'])} trials. References: "
        "`eval/reference_protocols/<NCT>.adjudicated.json` (investigator-written, LLM second "
        "annotation, adjudicated; clinician audit pending). Precision: share of the draft's "
        "concepts in the reference; recall: share of the reference's concepts in the draft.",
        "",
        "## Summary over the 12 core trials",
        "",
        "| Element | Result |",
        "|---|---|",
        f"| Eligibility concepts, strict (pre-registered) | {prf(tot['strict'])} |",
        f"| Eligibility concepts, as compiled | {prf(tot['compiled'])} |",
        f"| Eligibility concepts, relaxed (name + domain) | {prf(tot['relaxed'])} |",
        f"| Eligibility concepts, lenient names (exploratory) | {prf(tot['lenient'])} |",
        f"| Eligibility element correct (>= 90% of criteria exact), strict / as compiled | "
        f"{share(lambda r: r['grade']['grade_strict'] == 'correct')} / "
        f"{share(lambda r: r['grade']['grade_compiled'] == 'correct')} trials |",
        f"| Criterion category (clinical vs administrative) | "
        f"{sum(r['eligibility']['category_agree'] for r in rows)}/"
        f"{sum(r['eligibility']['criteria'] for r in rows)} criteria |",
        f"| Treatment: same arms | {share(lambda r: r['treatment']['same_arms'])} trials |",
        f"| Treatment: right comparator | "
        f"{share(lambda r: r['treatment']['comparator_agree'])} trials |",
        f"| Treatment components (lenient names) | {prf(comp)} |",
        f"| Treatment: right grace period | {share(lambda r: r['treatment']['grace_agree'])} "
        "trials |",
        "| Time zero | rating sheet `eval/results/e3_time_zero_ratings.json` |",
        f"| Outcome horizon | {share(lambda r: r['outcome']['horizon_hours'])} trials |",
        f"| Outcome end event | {share(lambda r: r['outcome']['end_event'])} trials |",
        f"| Causal contrast | {share(lambda r: r['contrast_agree'])} trials |",
        f"| Estimand | {share(lambda r: r['estimand_agree'])} trials |",
        f"| Confounders (lenient names) | {prf(conf)} |",
        "",
        "## Per trial",
        "",
        "| Trial | Clinical criteria exact, strict / compiled | Concepts F1 strict / compiled / "
        "lenient | Comparator | Components F1 | Grace | Horizon | End event | Contrast | "
        "Estimand | Confounders (draft n, F1) | Registry-structured criteria (not scored) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    yn = {True: "right", False: "wrong", None: "n/a"}
    for r in rows:
        e, t, g = r["eligibility"], r["treatment"], r["grade"]
        lines.append(
            f"| {r['name']} | {g['strict']}/{g['n']} / {g['compiled']}/{g['n']} | "
            f"{pct(e['strict'].f1)} / {pct(e['compiled'].f1)} / {pct(e['lenient'].f1)} | "
            f"{yn[t['comparator_agree']]} | {pct(t['components'].f1)} | {yn[t['grace_agree']]} | "
            f"{yn[r['outcome']['horizon_hours']]} | {yn[r['outcome']['end_event']]} | "
            f"{yn[r['contrast_agree']]} | {yn[r['estimand_agree']]} | "
            f"{r['n_conf_draft']}, {pct(r['confounders'].f1)} | "
            f"{'; '.join(r['structured']) or '-'} |")  # fmt: skip
    lines += ["", "## Element differences (draft vs reference)", ""]
    for r in rows:
        diffs = [d for d in r["disagreements"] if d["element"] not in ("time_zero",
                                                                       "eligibility")]
        if diffs:
            lines.append(f"- **{r['name']}:** " + "; ".join(
                f"{d['element']} {d['item']}: reference {json.dumps(d['a'], ensure_ascii=False)}"
                f", draft {json.dumps(d['b'], ensure_ascii=False)}" for d in diffs))
    lines += ["", "## Reading these results", "", *SCOPE]
    return "\n".join(lines) + "\n"


# What the scored drafts could not contain (eval/SCORING.md, Phase 3 changelog item 28).
SCOPE = [
    "- These are the Trialist's drafts alone: no concept review, Clinician or Supervisor agent "
    "has touched them. The references are not gold: the clinician audit is still pending.",
    "- **Confounders:** the Trialist does not propose confounders (the Clinician agent is not "
    "built), so recall is 0 by construction. This measures a missing agent, not a wrong one.",
    "- **Estimand:** the Trialist defaults to a hazard ratio, and the schema has no risk ratio; "
    "the references chose risk differences for fixed-horizon mortality. A default, not a "
    "per-trial judgement.",
    "- **Comparator:** wrong where the registry labels both arms ACTIVE_COMPARATOR (ACURASYS, "
    "APROCCHSS, "
    "ADRENAL) or neither (EvK), and the Trialist copied the label.",
    "- **Horizon:** ACURASYS takes the registry's 36-month time frame (the paper's primary "
    "outcome is 90 days). BICAR-ICU's 0 days came from the time-frame parser reading 'Day 0 to "
    "Day 28' as day 0; the parser is fixed, the scored draft is kept as generated.",
    "- **Grace period:** where the draft gives none and the reference gives one, the draft "
    "would be analysed by IPTW with an immortal-time flag rather than clone-censor-weight.",
]


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    model = argv[argv.index("--model") + 1] if "--model" in argv else "qwen3_32b"
    if model in args:
        args.remove(model)
    if len(args) != 1:
        print(__doc__)
        return 2
    run = Path(args[0])
    s = score(run, model)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"e3_{model}.md").write_text(render(s, run, model), encoding="utf-8")
    sheet = RESULTS / "e3_time_zero_ratings.json"
    if sheet.exists() and any(x.get("rating") for x in json.loads(sheet.read_text("utf-8"))):
        print(f"{sheet.name} already holds ratings: not overwritten")
    else:
        sheet.write_text(json.dumps(s["sheet"], indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(render(s, run, model))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(EVAL))
    raise SystemExit(main(sys.argv[1:]))
