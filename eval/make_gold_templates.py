"""Write blank gold-annotation templates for the Replay trials (eval/SCORING.md, Gold standards).

    uv run python eval/make_gold_templates.py [--ctgov DIR]

One file per trial in eval/gold/templates/<NCT>.json, built only from the registry record and
the deterministic extractor: the same criterion units the models saw, with every field a
model fills left empty. Annotators copy a template to eval/gold/<NCT>.<annotator>.json and fill
it without looking at any model output (eval/gold/README.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from check_exit_run import default_ctgov_dir  # noqa: E402

from agentic_tte.protocol.normalize import parse_time_frame  # noqa: E402
from agentic_tte.sources.ctgov import extract_sections  # noqa: E402

EVAL = Path(__file__).parent
REFERENCE = EVAL / "reference" / "replay_trials.json"


def template(study: dict) -> dict:
    s = extract_sections(study)
    criteria = [
        {"id": f"{prefix}{i}", "kind": kind, "text": text, "atoms": [], "atom_logic": None,
         "category": None, "complexity": None, "registry_error": None}  # fmt: skip
        for kind, items, prefix in (("inclusion", s.inclusion, "I"),
                                    ("exclusion", s.exclusion, "E"))  # fmt: skip
        for i, text in enumerate(items, 1)
    ]
    arms = [
        {"label": a.label, "arm_type": a.type, "description": a.description,
         "intervention_text": [f"{iv.name}: {iv.description}" for iv in a.interventions],
         "is_comparator": None, "components": []}  # fmt: skip
        for a in s.arms
    ]
    primary = next((o for o in s.outcomes if o.primary), None)
    return {
        "nct_id": s.source.nct_id,
        "record_sha256": s.source.record_sha256,
        "annotator": "",
        "notes": "",
        "criteria": criteria,
        "treatment_strategies": arms,
        "time_zero": {"anchor_event": None, "acceptable_alternatives": [], "offset": None},
        "follow_up": {"horizon": None, "end_event": None, "end_rule": None},
        "primary_outcome": {
            "name": primary.measure if primary else None,
            "time_frame": primary.time_frame if primary else None,
            "horizon": None,
            "end_event": None,
        },
        "causal_contrast": {"value": None, "origin": None, "evidence": None},
        "detailed_description": s.detailed_description,
        "_parser_hint_not_gold": {
            "primary_time_frame": [str(x) for x in parse_time_frame(
                primary.time_frame if primary else None)],
        },
    }  # fmt: skip


def main(argv: list[str]) -> int:
    ctgov = default_ctgov_dir()
    if "--ctgov" in argv:
        ctgov = Path(argv[argv.index("--ctgov") + 1])
    trials = [t["nct_id"] for t in json.loads(REFERENCE.read_text(encoding="utf-8"))["trials"]]
    out = EVAL / "gold" / "templates"
    out.mkdir(parents=True, exist_ok=True)
    for t in trials:
        study = json.loads((ctgov / f"{t}.json").read_text(encoding="utf-8"))
        (out / f"{t}.json").write_text(json.dumps(template(study), indent=2, ensure_ascii=False),
                                       encoding="utf-8")  # fmt: skip
    print(f"wrote {len(trials)} templates to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
