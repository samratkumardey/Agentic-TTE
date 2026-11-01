"""Trialist + Clinician: the Clinician's confounder lists scored against the adjudicated
references (eval/SCORING.md, Phase 3 changelog item 29).

    uv run python eval/score_clinician.py runs/exit-H100-core-clinician [--model qwen3_32b]

Replicate 1 is the analysis set; replicate 2 is checked for identity only. Names are matched
leniently, one to one, as the annotators' lists were (``compare_references.names_prf``). The
frozen E3 result (item 28) is not changed by this score. Writes
``eval/results/e3_clinician_<model>.md``. Uses protocols and public abstracts only, no data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from compare_references import NAMES, PRF, load, names_prf, pct

EVAL = Path(__file__).parent
REFS = EVAL / "reference_protocols"
RESULTS = EVAL / "results"

DISCLOSURE = (
    "**Disclosure (changelog item 29):** the Clinician was built by the LLM that wrote the "
    "second annotation and adjudicated these references, after E3 showed the confounders "
    "missing. Its prompt holds no trial-specific content, but its developer had seen the "
    "reference lists, so this score is not blind to development. The references await the "
    "clinician audit.")  # fmt: skip


def _sheet(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score(run: Path, model: str) -> list[dict]:
    rows = []
    for ref_path in sorted(REFS.glob("NCT*.adjudicated.json")):
        nct = ref_path.name.split(".")[0]
        ref = [c.name for c in load(ref_path).assignment.confounders]
        s1 = _sheet(run / model / "rep1" / f"{nct}.clinician.json")
        rep2 = run / model / "rep2" / f"{nct}.clinician.json"
        names = [i["name"] for i in s1["items"]]
        rows.append({
            "nct_id": nct, "name": NAMES.get(nct, nct), "prf": names_prf(names, ref),
            "n": len(names), "cited": sum(i["status"] == "cited" for i in s1["items"]),
            "dropped": [f"{d['name']} ({d['reason']})" for d in s1["dropped"]],
            "rep_identical": rep2.exists() and [i["name"] for i in _sheet(rep2)["items"]] == names,
            "model": s1["model"], "prompt_version": s1["prompt_version"]})  # fmt: skip
    return rows


def render(rows: list[dict], run: Path, model: str) -> str:
    tot = PRF()
    for r in rows:
        tot += r["prf"]
    n = len(rows)
    lines = [
        f"# Trialist + Clinician: {model} confounders vs the adjudicated references", "",
        f"Run `{run.as_posix()}`, replicate 1; replicate 2 identical in "
        f"{sum(r['rep_identical'] for r in rows)}/{n} trials. Prompt "
        f"{rows[0]['prompt_version'] if rows else '?'}. Lenient one-to-one name matching. The "
        "frozen E3 result (`e3_qwen3_32b.md`, confounder recall 0) is unchanged.", "",
        DISCLOSURE, "",
        f"**Over the {n} core trials:** precision {pct(tot.precision)}, recall "
        f"{pct(tot.recall)}, F1 {pct(tot.f1)} ({tot.tp} matched; {tot.pred} proposed, "
        f"{tot.ref} in the references).", "",
        "| Trial | Proposed (cited) | Reference | Matched | Precision | Recall | Dropped by the "
        "checks |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        p = r["prf"]
        lines.append(f"| {r['name']} | {r['n']} ({r['cited']}) | {p.ref} | {p.tp} | "
                     f"{pct(p.precision)} | {pct(p.recall)} | "
                     f"{'; '.join(r['dropped']) or '–'} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    model = argv[argv.index("--model") + 1] if "--model" in argv else "qwen3_32b"
    if model in args:
        args.remove(model)
    if len(args) != 1:
        print(__doc__)
        return 2
    run = Path(args[0])
    report = render(score(run, model), run, model)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"e3_clinician_{model}.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(EVAL))
    raise SystemExit(main(sys.argv[1:]))
