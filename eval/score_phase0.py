"""Score a Phase 0 exit run (eval/SCORING.md) and write the results tables.

    uv run python eval/score_phase0.py runs/exit-H100 [--out eval/results]

Writes <out>/phase0_exit_<GPU>.md and .json. Parts that need the human gold standard
(B: concepts vs gold, C: element accuracy vs gold) are reported as pending, never estimated.
What is scored here needs no gold: completion and failures, operations, element origins,
run-to-run stability (part D), agreement between models, and a silver comparison with
EmulatRx's own cached parse of NCT03872011.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import check_exit_run  # noqa: E402
import phase0_scoring as sc  # noqa: E402

from agentic_tte.protocol.emulatrx_import import load_emulatrx_cache  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EMULATRX_NCT03872011 = (
    ROOT / "third_party" / "emulatrx" / "trialistUtils" / "results" / "trial_info" / "NCT03872011"
)


def fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.2f}"


def hours(m: dict) -> float | None:
    try:
        start = datetime.fromisoformat(m["started"])
        end = datetime.fromisoformat(m["finished"])
    except (KeyError, ValueError):
        return None
    return (end - start).total_seconds() / 3600


def score(root: Path, ctgov: Path | None) -> dict:
    rows, problems = check_exit_run.check(root, ctgov)
    runs: dict[str, dict[str, dict]] = {}
    for r in rows:
        run_dir = Path(r["run"])
        # Key by arm (directory), not model: one model can run at two reasoning levels.
        model, rep = r["arm"], f"rep{r['replicate']}"
        manifest = check_exit_run.read_manifest(run_dir / "manifest.txt")
        runs.setdefault(model, {})[rep] = {
            "row": r,
            "manifest": manifest,
            "protocols": sc.load_protocols(run_dir),
            "hours": hours(manifest),
        }
    trials = sorted({t for m in runs.values() for rep in m.values()
                     for t in rep["manifest"].get("trials", "").split()})  # fmt: skip
    first = next(iter(next(iter(runs.values())).values()))["manifest"]
    out: dict = {
        "run_root": str(root),
        "gpu": first.get("gpu"),
        "git_commit": first.get("git_commit"),
        "ollama_version": first.get("ollama_version"),
        "trials": trials,
        "checker_problems": problems,
        "models": {},
    }
    for model, reps in sorted(runs.items()):
        res: dict = {"completion": {}, "operations": {}, "origins": {}, "stability": {}}
        for rep, d in sorted(reps.items()):
            r = d["row"]
            res["completion"][rep] = {
                "protocols": r["protocols"],
                "trials": r["trials"],
                "trial_failures": r["missing"],
                "criterion_parse_failed": r["parse_failed"],
                "dropped_events": r["dropped"],
            }
            res["operations"][rep] = {
                "calls": r["calls"],
                "cache_hits": r["cache_hits"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "completion_tokens_per_call": r["completion_tokens"] / r["calls"]
                if r["calls"] else None,
                "wall_hours": d["hours"],
            }
        rep1 = reps.get("rep1", {}).get("protocols", {})
        origins = Counter((e, v) for p in rep1.values() for e, v in p.element_status().items())
        res["origins"] = {f"{e}:{v}": n for (e, v), n in sorted(origins.items())}
        rep2 = reps.get("rep2", {}).get("protocols", {})
        both = sorted(set(rep1) & set(rep2))
        elem_same = Counter()
        crit_same = crit_n = 0
        flips, full_same, all_same = [], 0, 0
        relaxed, strict = sc.PRF(), sc.PRF()
        for t in both:
            a, b = rep1[t], rep2[t]
            full_same += a.content_hash() == b.content_hash()
            ident = sc.element_identity(a, b)
            all_same += all(ident.values())
            elem_same.update(e for e, ok in ident.items() if ok)
            s, n, f = sc.criterion_identity(a, b)
            crit_same, crit_n = crit_same + s, crit_n + n
            flips += [(t, *x) for x in f]
            relaxed += sc.concept_prf(b, a, strict=False)
            strict += sc.concept_prf(b, a, strict=True)
        res["stability"] = {
            "trials_compared": len(both),
            "identical_full": full_same,
            "identical_substantive": all_same,
            "element_identical": {e: elem_same[e] for e in sc.ELEMENTS},
            "criteria_identical": crit_same,
            "criteria_compared": crit_n,
            "atom_logic_flips": flips,
            "concept_f1_relaxed": relaxed.f1,
            "concept_f1_strict": strict.f1,
        }
        if "NCT03872011" in rep1 and EMULATRX_NCT03872011.exists():
            silver = load_emulatrx_cache(EMULATRX_NCT03872011)
            prf = sc.concept_prf(rep1["NCT03872011"], silver, strict=False)
            res["silver_emulatrx_NCT03872011"] = {
                "precision": prf.precision, "recall": prf.recall, "f1": prf.f1,
                "tp": prf.tp, "pred": prf.pred, "ref": prf.ref,
            }  # fmt: skip
        out["models"][model] = res
    agreement = {}
    for m1, m2 in combinations(sorted(runs), 2):
        p1 = runs[m1].get("rep1", {}).get("protocols", {})
        p2 = runs[m2].get("rep1", {}).get("protocols", {})
        common = sorted(set(p1) & set(p2))
        rel, st = sc.PRF(), sc.PRF()
        for t in common:
            rel += sc.concept_prf(p1[t], p2[t], strict=False)
            st += sc.concept_prf(p1[t], p2[t], strict=True)
        agreement[f"{m1} vs {m2}"] = {"trials": len(common), "f1_relaxed": rel.f1,
                                      "f1_strict": st.f1}  # fmt: skip
    out["inter_model_agreement"] = agreement
    return out


def render(s: dict) -> str:
    models = list(s["models"])
    L = []
    w = L.append
    w(f"# Phase 0 results: `{Path(s['run_root']).as_posix()}` ({s['gpu'].split(',')[0]})\n")
    w(f"Commit `{s['git_commit']}`; Ollama "
      f"{s['ollama_version']}. Method: [eval/SCORING.md](../SCORING.md). Generated by "
      "`eval/score_phase0.py`.\n")  # fmt: skip
    w("**Not yet scored: the pre-registered accuracy endpoints.** Part B (concept P/R/F1 "
      "against gold) and part C (element accuracy against the reference protocols) need the "
      "human gold annotation in `eval/gold/`, which does not exist yet. Nothing below is an "
      "accuracy estimate.\n")  # fmt: skip

    w("## Completion and failures\n")
    w("| Arm | Rep | Protocols | Trial-level failures | Criterion parse failures | "
      "Dropped (grounding) |")  # fmt: skip
    w("|---|---|---|---|---|---|")
    for m in models:
        for rep, c in s["models"][m]["completion"].items():
            fails = ", ".join(f"{t} ({why})" for t, why in c["trial_failures"].items()) or "none"
            w(f"| {m} | {rep} | {c['protocols']}/{c['trials']} | {fails} | "
              f"{c['criterion_parse_failed']} | {c['dropped_events']} |")  # fmt: skip
    w("")

    w("## Operations\n")
    w("| Arm | Rep | LLM calls | Cache hits | Prompt tokens | Completion tokens | "
      "Completion tokens / call | Wall time (h) |")  # fmt: skip
    w("|---|---|---|---|---|---|---|---|")
    for m in models:
        for rep, o in s["models"][m]["operations"].items():
            w(f"| {m} | {rep} | {o['calls']} | {o['cache_hits']} | {o['prompt_tokens']:,} | "
              f"{o['completion_tokens']:,} | {fmt(o['completion_tokens_per_call'], False)} | "
              f"{fmt(o['wall_hours'], False)} |")  # fmt: skip
    w("\nCache hits are repeated identical requests within a run (checked by "
      "`eval/check_exit_run.py`), not reuse across runs.\n")  # fmt: skip

    w("## Element origins (replicate 1)\n")
    keys = sorted({k for m in models for k in s["models"][m]["origins"]})
    w("| Element: origin | " + " | ".join(models) + " |")
    w("|---|" + "---|" * len(models))
    for k in keys:
        w(f"| {k} | " + " | ".join(str(s["models"][m]["origins"].get(k, 0)) for m in models)
          + " |")  # fmt: skip
    w("")

    w("## Run-to-run stability (part D: replicate 2 vs replicate 1)\n")
    w("Trials compared are those both replicates completed. *Substantive* ignores free text "
      "(time-zero rationale and evidence, notes, source spans, complexity labels).\n")
    w("| Measure | " + " | ".join(models) + " |")
    w("|---|" + "---|" * len(models))
    st = {m: s["models"][m]["stability"] for m in models}

    def row(label, f):
        w(f"| {label} | " + " | ".join(f(st[m]) for m in models) + " |")

    row("Trials compared", lambda x: str(x["trials_compared"]))
    row("Identical, all fields", lambda x: f"{x['identical_full']}/{x['trials_compared']}")
    row("Identical, substantive", lambda x: f"{x['identical_substantive']}/{x['trials_compared']}")
    for e in sc.ELEMENTS:
        row(f"… {e} identical",
            lambda x, e=e: f"{x['element_identical'][e]}/{x['trials_compared']}")  # fmt: skip
    row("Criteria identical (substantive)",
        lambda x: f"{x['criteria_identical']}/{x['criteria_compared']}")  # fmt: skip
    row("Concept F1, relaxed (%)", lambda x: fmt(x["concept_f1_relaxed"]))
    row("Concept F1, strict (%)", lambda x: fmt(x["concept_f1_strict"]))
    row("atom_logic flips", lambda x: "; ".join(f"{t} {c}: {a}→{b}"
                                                 for t, c, a, b in x["atom_logic_flips"])
        or "none")  # fmt: skip
    w("")

    w("## Agreement between models (replicate 1, descriptive)\n")
    w("Symmetric concept F1 between two models on the trials both completed. Agreement is "
      "not accuracy: two models can agree on the same mistake.\n")
    w("| Pair | Trials | F1 relaxed (%) | F1 strict (%) |")
    w("|---|---|---|---|")
    for pair, a in s["inter_model_agreement"].items():
        w(f"| {pair} | {a['trials']} | {fmt(a['f1_relaxed'])} | {fmt(a['f1_strict'])} |")
    w("")

    w("## Silver comparison: NCT03872011 vs EmulatRx's cached GPT-4o parse\n")
    w("One trial, relaxed matching only (upstream records one annotation per criterion, "
      "without atom structure). Upstream concept names are tagged spans (\"Systemic "
      "corticosteroid therapy\"), so exact-name matching understates agreement. Indicative "
      "only.\n")  # fmt: skip
    w("| Arm | Precision (%) | Recall (%) | F1 (%) | Matched / predicted / upstream |")
    w("|---|---|---|---|---|")
    for m in models:
        v = s["models"][m].get("silver_emulatrx_NCT03872011")
        if v:
            w(f"| {m} | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['f1'])} | "
              f"{v['tp']} / {v['pred']} / {v['ref']} |")
        else:
            w(f"| {m} | – | – | – | trial not completed |")
    w("")
    if s["checker_problems"]:
        w("## Checker findings\n")
        for p in s["checker_problems"]:
            w(f"- {p}")
        w("")
    return "\n".join(L)


def main(argv: list[str]) -> int:
    args = list(argv)
    out_dir = Path("eval/results")
    if "--out" in args:
        i = args.index("--out")
        out_dir = Path(args[i + 1])
        del args[i : i + 2]
    if len(args) != 1:
        print(__doc__)
        return 2
    root = Path(args[0])
    s = score(root, check_exit_run.default_ctgov_dir())
    name = f"phase0_{root.resolve().name}"  # phase0_exit-H100, phase0_np1_fa1, ...
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    (out_dir / f"{name}.md").write_text(render(s), encoding="utf-8")
    print(f"wrote {out_dir / name}.md and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
