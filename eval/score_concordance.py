"""E5: agreement of the Replay with the trials' published results (eval/SCORING.md R4), by the
Concordance agent (``agentic_tte.agents.concordance``).

    uv run python eval/score_concordance.py runs/<label> [--confirm <orientation.json>]

``runs/<label>/<NCT>/result.json`` is written by ``agentic-tte replay`` when a run finishes.
Every trial of a set is scored, whether or not it has a run: a missing or failed run counts
as no agreement (R4). Sets come from ``eval/reference/replay_selection.json`` and are never
pooled; reference values are the user-verified ones in ``replay_reference_values.json``.

``--confirm``: orientations a person confirmed for trials whose contrast the agent could not
match to the protocol's arms, as ``{"<NCT>": {"orientation": "same" | "reversed",
"by": "<name>"}}``.

Writes ``eval/results/e5_agreement_<label>.md`` and ``.json``. Aggregates only, no data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentic_tte.agents.concordance import (  # noqa: E402
    Emulation,
    Reference,
    agree,
    summarize,
    to_markdown,
)

EVAL = ROOT / "eval"
REF = EVAL / "reference"


def references() -> dict[str, Reference]:
    vals = json.loads((REF / "replay_reference_values.json").read_text(encoding="utf-8"))
    return {v["nct_id"]: Reference.model_validate(v) for v in vals["values"]
            if v.get("s2") == "pass" and v.get("value") is not None}  # fmt: skip


def sets(refs: dict[str, Reference]) -> dict[str, list[str]]:
    """Set name -> NCT IDs with a verified reference, in file order."""
    cand = json.loads((REF / "replay_selection.json").read_text(encoding="utf-8"))["candidates"]
    rules = [("Core set (primary agreement set)", lambda c: c.get("core")),
             ("Breadth set (secondary, exploratory)", lambda c: c.get("breadth_agreement")),
             ("Secondary agreement set (S7)", lambda c: c["status"] == "secondary"),
             ("Dynamic strategies (S5, exploratory)",
              lambda c: c["status"] == "exploratory" and c.get("failed") == "S5")]
    out = {}
    for name, rule in rules:
        ids = [c["nct_id"] for c in cand if rule(c) and c["nct_id"] in refs]
        if ids:
            out[name] = ids
    return out


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    confirm = {}
    if "--confirm" in argv:
        path = Path(argv[argv.index("--confirm") + 1])
        args.remove(str(argv[argv.index("--confirm") + 1]))
        confirm = json.loads(path.read_text(encoding="utf-8"))
    if len(args) != 1:
        print(__doc__)
        return 2
    run = Path(args[0])
    refs = references()
    scored = {}
    for name, ids in sets(refs).items():
        rows = []
        for nct in ids:
            path = run / nct / "result.json"
            emu = (Emulation.model_validate_json(path.read_text(encoding="utf-8"))
                   if path.exists() else None)  # fmt: skip
            rows.append(agree(refs[nct], emu, confirmed=(confirm.get(nct) or {}).get(
                "orientation")))  # fmt: skip
        scored[name] = rows
    label = run.name
    out = EVAL / "results"
    out.mkdir(exist_ok=True)
    md = to_markdown(scored, run=run.as_posix())
    (out / f"e5_agreement_{label}.md").write_text(md, encoding="utf-8")
    (out / f"e5_agreement_{label}.json").write_text(json.dumps(
        {name: {"summary": summarize(name, rows).model_dump(mode="json"),
                "trials": [r.model_dump(mode="json") for r in rows]}
         for name, rows in scored.items()}, indent=2), encoding="utf-8")  # fmt: skip
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
