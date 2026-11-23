"""Check the autonomous arm's rules on the data before any run (SCORING.md Phase 3 item 34).

For each trial: the Repair agent's changes, the concept choices from record counts (data-v1),
time zero by rule (tz-v1), what compiles, and pooled counts: persons with the time-zero
admission, persons with a time zero, eligible persons, and how many meet each criterion. No
analysis frame is run, nothing is split by arm, and no outcome is read: the counts are what the
gate's attrition table shows, small cells suppressed. The point is to see whether the rules
behave on MIMIC (sets that lose all records, time-zero events nobody has), not to choose
trials: S8 is still applied at each run's first gate.

The rules are frozen (item 34): if this check shows one is wrong, the fix is a new rules
version in a new SCORING item, and the core set stays the disclosed development set.

    uv run python eval/autonomy_check.py --db $ATTE_CDM --drafts DIR --out runs/autonomy-check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_tte.agents.autonomy import ICU_VISIT, check_visits, choose_in_data, time_zero_spec
from agentic_tte.agents.repair import repair_design
from agentic_tte.graph import AutoApprove
from agentic_tte.omop.compiler import VISITS, CompileSpec, compile_protocol
from agentic_tte.omop.concept_sets import (
    apply_decisions,
    apply_to_protocol,
    decisions_template,
    propose,
)
from agentic_tte.omop.gate import suppress
from agentic_tte.protocol import TargetTrialProtocol

REF = Path(__file__).parent / "reference"


def _core() -> list[str]:
    sel = json.loads((REF / "replay_selection.json").read_text(encoding="utf-8"))
    return [r["nct_id"] for r in sel["candidates"] if r.get("core")]


def _contrasts() -> dict[str, str]:
    vals = json.loads((REF / "replay_reference_values.json").read_text(encoding="utf-8"))
    return {v["nct_id"]: v.get("contrast") for v in vals["values"]}


def check(con, protocol: TargetTrialProtocol, contrast: str | None) -> tuple[dict, str]:
    """One trial: the rules' choices and pooled counts (JSON and markdown)."""
    nct = protocol.source.nct_id or protocol.protocol_id
    changes, rep = repair_design(protocol, contrast)
    p = TargetTrialProtocol.model_validate({**protocol.model_dump(mode="json"), **changes})
    book, data = choose_in_data(con, propose(con, p), p)
    auto = AutoApprove(autonomous=True)
    answer = auto.answer("concepts", {"decisions_template": decisions_template(book)})
    reviewed = apply_decisions(book, answer["decisions"])
    spec_d, tz = time_zero_spec(p, reviewed.gate_sets())
    spec = CompileSpec.model_validate({**spec_d, "concept_sets": reviewed.gate_sets()})
    p = apply_to_protocol(p, reviewed)
    cc = compile_protocol(p, spec)

    base = con.execute(f"SELECT count(DISTINCT person_id) FROM {VISITS} v "
                       f"WHERE visit_concept_id = {spec.time_zero.concept_ids[0]}"
                       ).fetchone()[0]  # fmt: skip
    cols = [c.column for c in cc.criteria if c.column]
    sums = ", ".join(f"count(*) FILTER (WHERE {c})" for c in [*cols, "eligible"])
    row = con.execute(f"WITH {cc.ctes} SELECT count(*), {sums} FROM cohort").fetchone()
    with_t0, met, eligible = row[0], dict(zip(cols, row[1:-1], strict=True)), row[-1]
    crit = [{"id": c.id, "kind": c.kind, "status": c.status, "text": c.text[:90],
             "persons_meeting": suppress(met[c.column]) if c.column else None,
             "reasons": c.reasons} for c in cc.criteria]  # fmt: skip
    out = {"nct_id": nct, "contrast": contrast, "repair": rep.model_dump(mode="json"),
           "data_rules": data.model_dump(mode="json"), "time_zero": tz.model_dump(mode="json"),
           "concepts": reviewed.summary(), "frame_compiles": cc.frame_sql is not None,
           "frame_design": cc.design, "frame_notes": cc.frame_notes,
           "persons_with_admission": suppress(base), "persons_with_time_zero": suppress(with_t0),
           "eligible": suppress(eligible), "criteria": crit}  # fmt: skip
    md = [f"# {nct}: {contrast or 'no contrast'}", "",
          f"Pooled counts only (no arms, no outcomes; small cells suppressed). Persons with the "
          f"time-zero admission (Visit {spec.time_zero.concept_ids[0]}): "
          f"**{suppress(base)}**; with a time zero: **{suppress(with_t0)}**; eligible: "
          f"**{suppress(eligible)}**. Frame: "
          f"{'compiles (' + str(cc.design) + ')' if cc.frame_sql else 'not compiled'}.", "",
          rep.to_markdown().replace("## ", "## 1. "),
          data.to_markdown("2. Concepts chosen from the data"),
          f"Concept sets after the rules: {reviewed.summary()}", "",
          tz.to_markdown("3. Time zero by rule"),
          "## 4. Criteria", "", "| Criterion | Status | Persons meeting it | Why not compiled |",
          "|---|---|---|---|"]  # fmt: skip
    md += [f"| {c['id']} ({c['kind']}): {c['text']} | {c['status']} | "
           f"{c['persons_meeting'] if c['persons_meeting'] is not None else '–'} | "
           f"{'; '.join(c['reasons'])[:160] or '–'} |" for c in crit]
    md += ["", "## 5. Frame (not run)", ""] + [f"- {n}" for n in cc.frame_notes]
    return out, "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--drafts", required=True, type=Path, help="<NCT>.protocol.json drafts")
    ap.add_argument("--out", required=True, type=Path, help="a directory under runs/")
    ap.add_argument("--trials", nargs="*", help="default: the 12 core trials")
    a = ap.parse_args()
    import duckdb

    con = duckdb.connect(str(a.db), read_only=True)
    check_visits(con)
    a.out.mkdir(parents=True, exist_ok=True)
    contrasts, summary = _contrasts(), []
    for nct in a.trials or _core():
        p = TargetTrialProtocol.model_validate_json(
            (a.drafts / f"{nct}.protocol.json").read_text(encoding="utf-8"))
        try:
            res, md = check(con, p, contrasts.get(nct))
        except Exception as e:  # noqa: BLE001 - one trial's failure is a finding, not a stop
            res, md = {"nct_id": nct, "error": f"{type(e).__name__}: {e}"}, \
                f"# {nct}\n\nFAILED: {type(e).__name__}: {e}\n"  # fmt: skip
        (a.out / f"{nct}.json").write_text(json.dumps(res, indent=2, default=str), "utf-8")
        (a.out / f"{nct}.md").write_text(md, encoding="utf-8")
        summary.append(res)
        print(f"{nct}: " + (res.get("error") or f"time zero {res['persons_with_time_zero']}, "
                            f"eligible {res['eligible']}, frame {res['frame_design']}"),
              flush=True)  # fmt: skip
    lines = ["# Autonomy check (pooled counts only)", "",
             f"ICU admission concept {ICU_VISIT}. One file per trial.", "",
             "| Trial | With time zero | Eligible | Inclusions not compiled | Sets without "
             "records | Frame |", "|---|---|---|---|---|---|"]  # fmt: skip
    for r in summary:
        if "error" in r:
            lines.append(f"| {r['nct_id']} | FAILED: {r['error'][:80]} | | | | |")
            continue
        nc = [c["id"] for c in r["criteria"]
              if c["kind"] == "inclusion" and c["status"] == "not_compiled"]  # fmt: skip
        gone = sum("no vocabulary candidate has records" in n for n in r["data_rules"]["notes"])
        lines.append(f"| {r['nct_id']} | {r['persons_with_time_zero']} | {r['eligible']} | "
                     f"{', '.join(nc) or '–'} | {gone} | {r['frame_design'] or 'none'} |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {a.out}/summary.md")


if __name__ == "__main__":
    main()
