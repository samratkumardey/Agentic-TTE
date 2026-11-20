"""Trialist + Clinician: the Clinician's confounder lists scored against the adjudicated
references (eval/SCORING.md, Phase 3 changelog item 29).

    uv run python eval/score_clinician.py runs/exit-H100-core-clinician [--model qwen3_32b]
        [--review]

Replicate 1 is the analysis set; replicate 2 is checked for identity only. Names are matched
leniently, one to one, as the annotators' lists were (``compare_references.names_prf``). The
frozen E3 result (item 28) is not changed by this score. Writes
``eval/results/e3_clinician_<model>_<prompt version>.md`` (e.g. ``..._v3.md``), so a new prompt
version never overwrites an earlier result. Uses protocols and public abstracts only, no data.

**Match review.** The lenient matching misses synonyms ("gender" for "sex"). ``--review``
writes ``eval/results/clinician_match_review_<model>_<version>.json``: one entry per reference
confounder left unmatched, with the proposals left unmatched in that trial. A person sets
``match`` to one of those proposal names, or to "none", with a note and their name in ``rater``.
Once every entry is rated, the report adds a reviewed score beside the automatic one. A rated
sheet is never overwritten, and one made for a different run is refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from compare_references import NAMES, PRF, lenient, load, names_prf, pct

EVAL = Path(__file__).parent
REFS = EVAL / "reference_protocols"
RESULTS = EVAL / "results"

DISCLOSURE = (
    "**Disclosure (changelog item 29):** the Clinician was built by the LLM that wrote the "
    "second annotation and adjudicated these references, after E3 showed the confounders "
    "missing. Its prompt holds no trial-specific content, but its developer had seen the "
    "reference lists, so this score is not blind to development. The references await the "
    "clinician audit.")  # fmt: skip

RULE = ("match: the name of the proposal that is the same baseline variable as the reference "
        "(a synonym, or a narrower or broader name a clinician would map to the same data), "
        "or \"none\". Each proposal can match one reference only.")  # fmt: skip


def _path(rep: Path, nct: str) -> Path:
    """``<NCT>.clinician.json``; runs before the CLI fix of 2026-11-01 wrote ``<NCT>.json``."""
    p = rep / f"{nct}.clinician.json"
    return p if p.exists() else rep / f"{nct}.json"


def _sheet(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def unmatched(names: list[str], ref: list[str]) -> tuple[list[str], list[str]]:
    """The reference names and proposals left over by the greedy one-to-one lenient matching,
    pairing in the same order as ``names_prf``."""
    left_ref, left_prop = list(ref), []
    for x in names:
        i = next((k for k, y in enumerate(left_ref) if lenient(x, y)), None)
        if i is None:
            left_prop.append(x)
        else:
            del left_ref[i]
    return left_ref, left_prop


def score(run: Path, model: str) -> list[dict]:
    rows = []
    for ref_path in sorted(REFS.glob("NCT*.adjudicated.json")):
        nct = ref_path.name.split(".")[0]
        ref = [c.name for c in load(ref_path).assignment.confounders]
        s1 = _sheet(_path(run / model / "rep1", nct))
        rep2 = _path(run / model / "rep2", nct)
        names = [i["name"] for i in s1["items"]]
        missed, left = unmatched(names, ref)
        why = {i["name"]: i.get("rationale") or i["status"] for i in s1["items"]}
        rows.append({
            "nct_id": nct, "name": NAMES.get(nct, nct), "prf": names_prf(names, ref),
            "n": len(names), "cited": sum(i["status"] == "cited" for i in s1["items"]),
            "dropped": [f"{d['name']} ({d['reason']})" for d in s1["dropped"]],
            "rep_identical": ([i["name"] for i in _sheet(rep2)["items"]] == names
                              if rep2.exists() else None),
            "missed": missed, "left": [{"name": x, "rationale": why[x]} for x in left],
            "model": s1["model"], "prompt_version": s1["prompt_version"]})  # fmt: skip
    return rows


def review_template(rows: list[dict]) -> list[dict]:
    return [{"nct_id": r["nct_id"], "trial": r["name"], "reference": ref,
             "candidates": r["left"], "match": None, "rule": RULE, "note": "", "rater": ""}
            for r in rows for ref in r["missed"]]  # fmt: skip


def _key(sheet: list[dict]) -> list[tuple]:
    return [(x["nct_id"], x["reference"], [c["name"] for c in x["candidates"]]) for x in sheet]


def reviewed(rows: list[dict], sheet: list[dict]) -> dict[str, int] | str:
    """Matches the review adds per trial, or why there is no reviewed score."""
    if _key(sheet) != _key(review_template(rows)):
        return "the review sheet was made for a different run: not used"
    done = [x for x in sheet if x.get("match")]
    if len(done) < len(sheet):
        return f"{len(done)}/{len(sheet)} rated: not reported"
    added: dict[str, int] = {}
    used = set()
    for x in sheet:
        m = x["match"]
        if m == "none":
            continue
        if m not in {c["name"] for c in x["candidates"]}:
            return f"{m!r} is not a candidate for {x['reference']!r}: not reported"
        if (x["nct_id"], m) in used:
            return f"{m!r} is matched twice in {x['trial']}: not reported"
        used.add((x["nct_id"], m))
        added[x["nct_id"]] = added.get(x["nct_id"], 0) + 1
    return added


def render(rows: list[dict], run: Path, model: str, review: list[dict] | None = None) -> str:
    tot = PRF()
    for r in rows:
        tot += r["prf"]
    n = len(rows)
    have = [r["rep_identical"] for r in rows if r["rep_identical"] is not None]
    rep2 = (f"replicate 2 identical in {sum(have)}/{len(have)} trials" if have
            else "replicate 2 not run yet")
    lines = [
        f"# Trialist + Clinician: {model} confounders vs the adjudicated references", "",
        f"Run `{run.as_posix()}`, replicate 1; {rep2}. Prompt "
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
    if review is not None:
        lines += ["", "## Match review (rated, not automatic)", ""]
        added = reviewed(rows, review)
        if isinstance(added, str):
            lines.append(f"Match review: {added}.")
        else:
            rev = PRF()
            for r in rows:
                rev.tp += r["prf"].tp + added.get(r["nct_id"], 0)
                rev.pred += r["prf"].pred
                rev.ref += r["prf"].ref
            raters = sorted({x["rater"] for x in review if x.get("rater")}) or ["not named"]
            lines += [
                f"The {len(review)} reference confounders the lenient matching left unmatched "
                f"were rated against each trial's unmatched proposals ({sum(added.values())} "
                f"matched). Rater: {'; '.join(raters)}.", "",
                f"**Reviewed, over the {n} core trials:** precision {pct(rev.precision)}, "
                f"recall {pct(rev.recall)}, F1 {pct(rev.f1)} ({rev.tp} matched).", "",
                "| Trial | Added by the review | Reviewed recall |", "|---|---|---|"]
            for r in rows:
                k = r["prf"].tp + added.get(r["nct_id"], 0)
                lines.append(f"| {r['name']} | {added.get(r['nct_id'], 0)} | "
                             f"{pct(k / r['prf'].ref if r['prf'].ref else None)} |")
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
    rows = score(run, model)
    version = rows[0]["prompt_version"].removeprefix("clinician-") if rows else "unknown"
    RESULTS.mkdir(exist_ok=True)
    sheet = RESULTS / f"clinician_match_review_{model}_{version}.json"
    review = None
    if sheet.exists():
        review = json.loads(sheet.read_text(encoding="utf-8"))
    elif "--review" in argv:
        review = review_template(rows)
        sheet.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")  # fmt: skip
        print(f"wrote {sheet.as_posix()}: {len(review)} entries to rate")
    report = render(rows, run, model, review)
    (RESULTS / f"e3_clinician_{model}_{version}.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(EVAL))
    raise SystemExit(main(sys.argv[1:]))
