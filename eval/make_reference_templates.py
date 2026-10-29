"""E3 reference protocols (eval/SCORING.md, Phase 3 changelog item 11b): templates and a check.

    uv run python eval/make_reference_templates.py            # write the 12 core templates
    uv run python eval/make_reference_templates.py --check eval/reference_protocols/NCT....SKD.json

Each template is a target-trial protocol in our schema (``TargetTrialProtocol``), filled only
with registry facts: criteria texts (no atoms), arm labels and types, the primary outcome's
name and time frame. Everything a design decides is left empty for the investigator, who
writes it from the trial paper without looking at any model draft. The filled file is both the
E3 reference and the protocol substituted for the agent's in the attribution analysis (item
11d), so it must validate and compile; ``--check`` says what is still missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agentic_tte.protocol import TargetTrialProtocol
from agentic_tte.protocol.normalize import parse_time_frame
from agentic_tte.sources.ctgov import CTGovClient, extract_sections

EVAL = Path(__file__).parent
OUT = EVAL / "reference_protocols"
SELECTION = EVAL / "reference" / "replay_selection.json"


def core_trials() -> list[str]:
    d = json.loads(SELECTION.read_text(encoding="utf-8"))
    return [r["nct_id"] for r in d["candidates"] if r.get("core")]


def template(study: dict) -> dict:
    """A protocol skeleton from the registry record only."""
    s = extract_sections(study)
    primary = next((o for o in s.outcomes if o.primary), None)
    horizon, end_event, end_rule = parse_time_frame(primary.time_frame if primary else None)
    return {
        "protocol_id": f"{s.source.nct_id}-reference",
        "mode": "replay",
        "source": {"nct_id": s.source.nct_id, "title": s.source.title,
                   "record_sha256": s.source.record_sha256},
        "eligibility": [
            {"id": f"{p}{i}", "kind": kind, "text": text, "atoms": [], "atom_logic": "all",
             "category": "clinical", "notes": None}
            for kind, items, p in (("inclusion", s.inclusion, "I"), ("exclusion", s.exclusion, "E"))
            for i, text in enumerate(items, 1)],
        "treatment_strategies": [
            {"label": a.label, "arm_type": a.type, "description": a.description,
             "components": [], "is_comparator": None, "grace_period": None,
             "grace_period_origin": None, "grace_period_evidence": None}
            for a in s.arms],
        "assignment": {"method": "auto", "confounders": [], "origin": "user"},
        "time_zero": {"anchor_event": None, "offset": None, "rationale": None, "origin": "user",
                      "evidence": None},
        "follow_up": {"start": "time_zero", "horizon": None, "end_event": None, "end_rule": None,
                      "censoring_events": [], "competing_events": [], "origin": "user"},
        "outcomes": [{"name": primary.measure if primary else None, "primary": True,
                      "time_frame": primary.time_frame if primary else None,
                      "concept": None, "horizon": None, "end_event": None, "end_rule": None}],
        "causal_contrast": "intention_to_treat",
        "causal_contrast_origin": "user",
        "causal_contrast_evidence": None,
        "analysis_plan": {"estimand": "hazard_ratio", "outcome_model": "auto",
                          "immortal_time_handling": "auto", "origin": "user"},
        "_registry": {
            "arms": [{"label": a.label, "interventions": [f"{iv.name}: {iv.description}"
                                                          for iv in a.interventions]}
                     for a in s.arms],
            "primary_outcome_parser_hint_not_reference": [str(horizon), end_event, end_rule],
            "detailed_description": s.detailed_description,
        },
    }  # fmt: skip


def check(path: Path) -> list[str]:
    """What a filled reference still lacks, or [] when it is complete and valid."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    # Plain messages for the fields a template leaves empty. They are filled with stand-ins so
    # that validation can go on and list everything else still missing in the same pass.
    problems = []
    for a in data.get("treatment_strategies", []):
        if a.get("is_comparator") is None:
            problems.append(f"arm '{a.get('label')}': set is_comparator (true or false)")
            a["is_comparator"] = False
    unset_arms = bool(problems)
    tz = data.get("time_zero") or {}
    if not tz.get("anchor_event"):
        problems.append("time zero: no anchor event")
        data["time_zero"] = {**tz, "anchor_event": "unset"}
    try:
        p = TargetTrialProtocol.model_validate(data)
    except Exception as e:  # noqa: BLE001 - reported to the person filling it in
        return problems + [f"not a valid protocol: {e}"]
    for c in p.eligibility:
        if c.emulable and not c.atoms:
            problems.append(f"{c.id}: no atoms (or set category 'administrative')")
    arms = [a for a in p.treatment_strategies if not a.is_comparator]
    comps = [a for a in p.treatment_strategies if a.is_comparator]
    if not unset_arms and (len(arms) != 1 or len(comps) != 1):
        problems.append("treatment strategies: set is_comparator so that exactly one arm is the "
                        "treatment and one the comparator")
    for a in arms if not unset_arms else []:
        if not a.components:
            problems.append(f"arm '{a.label}': no components (the treatment's concepts)")
    if unset_arms and not any(a.components for a in p.treatment_strategies):
        problems.append("treatment arm: no components (the treatment's concepts)")
    primary = next((o for o in p.outcomes if o.primary), None)
    if primary is None or (primary.horizon or p.follow_up.horizon) is None:
        problems.append("primary outcome: no horizon")
    if not p.assignment.confounders:
        problems.append("assignment: no confounders")
    return problems


def main(argv: list[str]) -> int:
    if "--check" in argv:
        path = Path(argv[argv.index("--check") + 1])
        problems = check(path)
        print(f"{path.name}: " + ("complete" if not problems else f"{len(problems)} to do"))
        for x in problems:
            print(f"  - {x}")
        return 1 if problems else 0
    client = CTGovClient(cache_dir=Path(".cache/agentic_tte"))
    tdir = OUT / "templates"
    tdir.mkdir(parents=True, exist_ok=True)
    trials = core_trials()
    for t in trials:
        (tdir / f"{t}.json").write_text(json.dumps(template(client.get_study(t)), indent=2,
                                                   ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(trials)} templates to {tdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
