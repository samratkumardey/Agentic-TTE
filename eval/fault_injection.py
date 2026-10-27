"""E1 on synthetic OMOP: plant each fault of the catalogue in a correct design, run the
Protocol Auditor and the pipeline, and report what was caught and what the fault does to the
estimate (eval/SCORING.md, Phase 3 changelog items 11 and 14).

    uv run python eval/fault_injection.py [--n 3000] [--seed 20261027]

Writes eval/results/fault_injection_synthetic.md. Synthetic data only: the truth is the
marginal HR of "initiate within the grace period vs never" from potential outcomes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from agentic_tte.agents.auditor import CHECKS, audit, data_summary
from agentic_tte.agents.statistician import analyze
from agentic_tte.faults import FAULTS, NotApplicable, inject, new_user_exclusion
from agentic_tte.omop.compiler import CompileSpec, EventDef, compile_protocol
from agentic_tte.protocol import (
    AssignmentProcedure,
    ConceptRef,
    Criterion,
    CriterionAtom,
    Duration,
    Outcome,
    TargetTrialProtocol,
    TimeZero,
    TreatmentComponent,
    TreatmentStrategy,
)
from agentic_tte.synth import omop as so
from agentic_tte.synth.dgp import Scenario, generate

GRACE_DAYS = 3.0
WRONG_CONCEPT = so.NOISE_BASE + 1  # F-j: a concept that is not the treatment


def synthetic_design() -> tuple[TargetTrialProtocol, CompileSpec]:
    """A correct design for the synthetic ICU cohort: septic shock at ICU admission, a study
    drug started within 3 days vs never, new users only, 28-day all-cause mortality."""
    p = TargetTrialProtocol(
        protocol_id="SYNTH-E1",
        eligibility=[Criterion(id="I1", kind="inclusion", text="Septic shock",
                               atoms=[CriterionAtom(concepts=[ConceptRef(
                                   name="septic shock", domain="Condition")])])],
        treatment_strategies=[
            TreatmentStrategy(label="Study drug", arm_type="EXPERIMENTAL",
                              components=[TreatmentComponent(concept=ConceptRef(
                                  name="study drug", domain="Drug"))],
                              grace_period=Duration(amount=GRACE_DAYS, unit="day"),
                              grace_period_origin="registry",
                              grace_period_evidence="started within 3 days of ICU admission"),
            TreatmentStrategy(label="No study drug", arm_type="PLACEBO_COMPARATOR",
                              is_comparator=True)],
        assignment=AssignmentProcedure(confounders=[
            ConceptRef(name="age", domain="Demographic"),
            ConceptRef(name="sex", domain="Demographic"),
            ConceptRef(name="sofa", domain="Measurement"),
            ConceptRef(name="lactate", domain="Measurement")]),
        time_zero=TimeZero(anchor_event="ICU admission", origin="registry"),
        outcomes=[Outcome(name="28-day all-cause mortality", primary=True, time_frame="28 days",
                          horizon=Duration(amount=28, unit="day"))])  # fmt: skip
    p.eligibility.append(new_user_exclusion(p))
    spec = CompileSpec(
        time_zero=EventDef(domain="Visit", concept_ids=[so.ICU_VISIT]),
        concept_sets={"T:Study drug:study drug": [so.STUDY_DRUG], "C:sofa": [so.SOFA],
                      "C:lactate": [so.LACTATE], "I1:septic shock": [so.SEPTIC_SHOCK],
                      "E-new-user:study drug": [so.STUDY_DRUG]},
        descendants=False)  # fmt: skip
    return p, spec


def run_design(con, p: TargetTrialProtocol, spec: CompileSpec, approved: dict) -> dict:
    """Audit, compile and analyse one design; aggregates only."""
    cc = compile_protocol(p, spec)
    summary = data_summary(con, p, spec, cc)
    rep = audit(p, spec, compiled=cc, summary=summary, approved_sets=approved)
    out = {"failed": rep.failed, "summary": summary, "estimate": None, "method": None,
           "notes": []}  # fmt: skip
    if not cc.frame_sql:
        out["notes"] = [n for n in cc.frame_notes if "not compiled" in n]
        return out
    frame = con.execute(cc.frame_sql).df()
    conf = cc.frame_columns[1:cc.frame_columns.index("treated")]
    try:
        res = analyze(frame, p, confounders=conf, horizon_days=cc.horizon_days)
    except Exception as e:  # noqa: BLE001 - a fault may break the analysis; that is a result
        out["notes"] = [f"analysis failed: {type(e).__name__}: {str(e)[:80]}"]
        return out
    if res.primary:
        out.update(estimate=(res.primary.hr, res.primary.lower, res.primary.upper),
                   method=res.primary_method)
    out["notes"] = [n for n in res.notes if "IMMORTAL" in n or "fewer than" in n]
    return out


def run(n: int = 3000, seed: int = 20261027) -> dict:
    cohort = generate(Scenario(n=n, seed=seed, timing="delayed", grace_days=GRACE_DAYS))
    con = so.write_duckdb(so.to_omop(cohort))
    truth = cohort.truth["marginal_hr_grace_deadline"]
    p, spec = synthetic_design()
    approved = dict(spec.concept_sets)
    rows = {"baseline": {"description": "correct design", **run_design(con, p, spec, approved)}}
    for f in FAULTS:
        try:
            fp, fs, desc = inject(f, p, spec, wrong_concept=WRONG_CONCEPT)
        except NotApplicable as e:
            rows[f] = {"description": f"not applicable: {e}", "failed": [], "estimate": None}
            continue
        rows[f] = {"description": desc, **run_design(con, fp, fs, approved)}
    return {"truth": truth, "n": n, "seed": seed, "rows": rows}


def to_markdown(r: dict) -> str:
    base = r["rows"]["baseline"]
    lines = ["# E1 fault injection on synthetic OMOP", "",
             f"Synthetic ICU cohort, n = {r['n']} (seed {r['seed']}), treatment started within "
             f"{GRACE_DAYS:g} days of time zero. Truth: marginal HR {r['truth']:.3f} (initiate "
             "within the grace period vs never, potential outcomes).", "",
             f"**Correct design:** checks failed: {', '.join(base['failed']) or 'none'}.", "",
             "| Fault | Planted error | Target check | Caught | Checks failed | Primary estimate "
             "(HR, 95% CI) | Bias (log HR) |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    caught = applicable = 0
    for f, (name, check) in FAULTS.items():
        row = r["rows"][f]
        if row["description"].startswith("not applicable"):
            lines.append(f"| {f} {name} | {row['description']} | {check} | – | – | – | – |")
            continue
        applicable += 1
        hit = check in row["failed"]
        caught += hit
        est = row.get("estimate")
        est_txt = f"{est[0]:.3f} ({est[1]:.3f}–{est[2]:.3f}) {row['method']}" if est else \
            "none: " + "; ".join(row.get("notes", []))[:80]  # fmt: skip
        bias = f"{math.log(est[0]) - math.log(r['truth']):+.3f}" if est else "–"
        lines.append(f"| {f} {name} | {row['description']} | {check} {CHECKS[check]} | "
                     f"{'yes' if hit else '**no**'} | {', '.join(row['failed']) or 'none'} | "
                     f"{est_txt} | {bias} |")  # fmt: skip
    est = base.get("estimate")
    if est:
        lines += ["", f"Correct design: {est[0]:.3f} ({est[1]:.3f}–{est[2]:.3f}), "
                      f"{base['method']}; bias {math.log(est[0]) - math.log(r['truth']):+.3f}."]
    lines += ["", f"**Caught {caught} of {applicable} applicable faults** by their target check; "
                  f"false alarms on the correct design: {len(base['failed'])}.", ""]
    return "\n".join(lines)


def main() -> None:
    args = sys.argv[1:]
    n = int(args[args.index("--n") + 1]) if "--n" in args else 3000
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 20261027
    md = to_markdown(run(n, seed))
    out = Path(__file__).parent / "results" / "fault_injection_synthetic.md"
    out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
