"""Phase 2 exit check: Replay three trials end to end on synthetic OMOP (PLAN.md §9, §15).

The drafts are the real Phase 0 benchmark outputs (qwen3:32b, final H100 run, replicate 1;
replicate 2 has the same content hash), copied to tests/fixtures/protocols/*.final-H100.*.
Each trial gets a synthetic OMOP database whose concepts look like its population
(``synth.omop.Profile``) and whose effect is planted, so the estimate can be checked against
the potential-outcome truth in the population the run actually selects.

The reviews are AUTO-APPROVED (``graph.AutoApprove``), not clinical: approve a vocabulary
match, mark everything without one unmappable, plus the per-trial decisions below, each with
its reason. Each run's report says so at the top and ends with status ``auto_approved``. They
exercise the workflow; they are not a clinician's judgement and must not be reported as one.

    uv run python eval/replay_phase2.py            # writes eval/results/phase2_replay.md
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_tte.graph import AUTO_REVIEWER, AutoApprove, build_replay  # noqa: E402
from agentic_tte.omop import DataContext  # noqa: E402
from agentic_tte.omop.compiler import CompiledCohort  # noqa: E402
from agentic_tte.omop.concept_sets import ConceptSetBook, review_sheet  # noqa: E402
from agentic_tte.synth import omop as so  # noqa: E402
from agentic_tte.synth.dgp import Scenario, generate  # noqa: E402

DRAFTS = ROOT / "tests" / "fixtures" / "protocols"
REVIEWER = AUTO_REVIEWER
INPATIENT = "Inpatient hospitalization (synthetic)"


def conf(*names_domains):
    return [{"name": n, "domain": d} for n, d in names_domains]


@dataclass
class TrialReplay:
    nct: str
    profile: so.Profile
    scenario: Scenario
    truth_strategy: str  # potential outcome the primary method targets (dgp.Cohort.potential)
    design: dict  # the design review's answer
    spec: dict  # compile spec (time zero, anchors, outcome)
    overrides: dict = field(default_factory=dict)  # concept decisions by key, with notes
    why: str = ""  # what the design review changed and why

    @property
    def draft(self) -> Path:
        return DRAFTS / f"{self.nct}.final-H100.qwen3-32b.json"


_CONFOUNDERS = conf(("age", "Demographic"), ("sex", "Demographic"))

TRIALS = [
    TrialReplay(
        nct="NCT04134403",  # STASIS: hydrocortisone + thiamine + ascorbic acid in septic shock
        profile=so.Profile(name="stasis-septic-shock",
                           lab="Lactate [Moles/volume] in Serum or Plasma (synthetic)",
                           treatment=("Hydrocortisone (synthetic)", "Thiamine (synthetic)",
                                      "Ascorbic acid (synthetic)"),
                           standard_local=True),  # fmt: skip
        scenario=Scenario(n=4000, true_hr=0.7, horizon_days=28.0, seed=4134403),
        truth_strategy="t1",
        design={"reviewer": REVIEWER, "confounders": _CONFOUNDERS + conf(
            ("SOFA score", "Measurement"), ("serum lactate", "Measurement"))},
        spec={"time_zero": {"domain": "Visit", "concept_ids": [so.ICU_VISIT],
                            "note": "ICU admission with septic shock"},
              "anchors": {"primary admission": {"domain": "Visit",
                                                "concept_ids": [so.ICU_VISIT],
                                                "occurrence": "last"}}},
        why="confounders named (the draft had none)",
    ),
    TrialReplay(
        nct="NCT00475852",  # ASCEND-HF: nesiritide in acute decompensated heart failure
        profile=so.Profile(name="ascend-hf", index_visit=INPATIENT,
                           index_condition=(0, "Acute decompensated heart failure (synthetic)",
                                            "ATTE-SYNTH"),
                           severity="Heart failure severity score (synthetic)",
                           lab="Natriuretic peptide B [Mass/volume] in Serum or Plasma "
                               "(synthetic)",
                           treatment=("Nesiritide (synthetic)",), standard_local=True),
        scenario=Scenario(n=4000, true_hr=0.8, horizon_days=30.0, timing="delayed",
                          grace_days=2.0, baseline_hazard=0.01, seed=475852),  # fmt: skip
        truth_strategy="t1_deadline",
        design={"reviewer": REVIEWER, "confounders": _CONFOUNDERS + conf(
            ("heart failure severity score", "Measurement"),
            ("natriuretic peptide B", "Measurement"))},
        spec={"time_zero": {"domain": "Visit", "concept_ids": [so.ICU_VISIT],
                            "note": "hospital admission for ADHF"},
              "anchors": {"hospitalization for another reason": {
                  "domain": "Visit", "concept_ids": [so.ICU_VISIT], "occurrence": "last"}},
              # The primary outcome is a composite; a reviewer defines it. The synthetic data
              # has no readmissions, so here the composite reduces to death.
              "outcome": {"events": [{"domain": "Visit", "concept_ids": [so.ICU_VISIT]}],
                          "death": True,
                          "note": "rehospitalization (inpatient visit after time zero) or "
                                  "death"}},
        why="confounders named",
    ),
    TrialReplay(
        nct="NCT06091982",  # RRT vs none in cardiac-surgery-associated AKI (observational)
        profile=so.Profile(name="csa-aki", index_visit=INPATIENT,
                           index_condition=(0, "Acute kidney injury (synthetic)", "ATTE-SYNTH"),
                           severity="Kidney injury severity score (synthetic)",
                           lab="Creatinine [Mass/volume] in Serum or Plasma (synthetic)",
                           treatment=("Renal replacement therapy (synthetic)",),
                           treatment_domain="Procedure",
                           prior_procedure="Cardiac surgery (synthetic)", standard_local=True),
        scenario=Scenario(n=4000, true_hr=0.75, horizon_days=91.32, timing="delayed",
                          grace_days=3.0, seed=6091982),  # fmt: skip
        truth_strategy="t1_deadline",
        design={"reviewer": REVIEWER, "confounders": _CONFOUNDERS + conf(
            ("kidney injury severity score", "Measurement"), ("serum creatinine", "Measurement")),
            "changes": {"treatment_strategies": [
                {"label": "AKI-RRT", "components": [{"concept": {
                    "name": "renal replacement therapy", "domain": "Procedure"}}],
                 "grace_period": {"amount": 3, "unit": "day"}, "grace_period_origin": "user",
                 "grace_period_evidence": "RRT is started within days of AKI; no grace period "
                                          "in the draft"},
                {"label": "AKI non RRT", "is_comparator": True}]}},
        spec={"time_zero": {"domain": "Condition", "concept_ids": [so.INDEX_CONDITION_LOCAL],
                            "note": "first post-operative AKI record"},
              "anchors": {"Procedure": {"domain": "Procedure",
                                        "concept_ids": [so.PRIOR_PROCEDURE],
                                        "occurrence": "last"}}},
        why="the four RRT modalities were drafted as a combination (all four); replaced by one "
            "'renal replacement therapy' component, and a 3-day grace period added (RRT starts "
            "after AKI, so the draft classified patients by future treatment)",
    ),
]


AS_DRAFTED_RRT = {  # NCT06091982 without the design fix: each modality mapped to RRT
    f"T:AKI-RRT:{m}": {"action": "surrogate", "concept_ids": [so.STUDY_DRUG],
                       "note": "auto: the synthetic data does not distinguish RRT modalities"}
    for m in ("intermittent hemodialysis", "continuous veno-venous hemofiltration",
              "continuous veno-venous hemodialysis", "continuous veno-venous hemodiafiltration")
}  # fmt: skip


def replay(t: TrialReplay, design: dict | None = None, overrides: dict | None = None) -> dict:
    import json

    cohort = generate(t.scenario, with_truth=False)
    con = so.write_duckdb(so.to_omop(cohort, t.profile))
    policy = AutoApprove(design=design or t.design, spec=t.spec,
                         overrides=overrides or t.overrides)  # fmt: skip
    graph = build_replay(con, context=DataContext(data_classification="synthetic"), auto=policy)
    state = graph.invoke({"protocol": json.loads(t.draft.read_text(encoding="utf-8"))})
    sheet = review_sheet(ConceptSetBook.model_validate(state["book"])) if state.get("book") \
        else ""  # fmt: skip
    out = {"state": state, "review_sheet": sheet, "truth": None, "eligible": None}
    if state.get("compiled"):
        cc = CompiledCohort.model_validate(state["compiled"])
        ids = con.execute(f"SELECT person_id FROM ({cc.cohort_sql}) WHERE eligible").df()
        out["eligible"] = len(ids)
        if len(ids):  # evaluation only: the truth never reaches the workflow
            out["truth"] = cohort.truth_in(ids.person_id, t.truth_strategy)
    return out


def _estimates(res: dict) -> list[dict]:
    a = res["state"].get("analysis") or {}
    return a.get("estimates", [])


def summarize(t: TrialReplay, res: dict, label: str) -> list[str]:
    st = res["state"]
    a = st.get("analysis") or {}
    cc = CompiledCohort.model_validate(st["compiled"]) if st.get("compiled") else None
    gate = {c["id"]: c["status"] for c in st.get("gate", {}).get("checks", [])}
    lines = [f"### {t.nct}: {label}", ""]
    lines.append(f"- Status **{st.get('status')}**; gate "
                 f"{', '.join(f'{k} {v}' for k, v in gate.items() if v != 'green')}.")
    if cc is not None:
        n = {s: sum(c.status == s for c in cc.criteria)
             for s in ("compiled", "not_compiled", "not_emulable")}  # fmt: skip
        lines.append(f"- Criteria: {n['compiled']} compiled, {n['not_compiled']} not compiled, "
                     f"{n['not_emulable']} not emulable. Eligible {res['eligible']} of "
                     f"{t.scenario.n}.")
        for c in cc.criteria:
            if c.status == "not_compiled":
                lines.append(f"  - {c.id} not compiled: {'; '.join(c.reasons)}")
        for note in cc.frame_notes:
            lines.append(f"  - frame: {note}")
    if a.get("estimates"):
        truth = res["truth"]
        lines.append(f"- True marginal HR in the selected population ({t.truth_strategy}): "
                     f"{truth:.3f}. Planted conditional HR {t.scenario.true_hr}.")
        lines += ["", "| Method | HR (95% CI) | log error | CI covers truth |",
                  "|---|---|---|---|"]
        for e in a["estimates"]:
            star = " (primary)" if e["method"] == a["primary_method"] else ""
            err = abs(np.log(e["hr"]) - np.log(truth))
            cover = "yes" if e["lower"] <= truth <= e["upper"] else "no"
            lines.append(f"| {e['method']}{star} | {e['hr']:.3f} ({e['lower']:.3f}–"
                         f"{e['upper']:.3f}) | {err:.3f} | {cover} |")
        lines.append("")
        lines.append(f"- Max SMD after weighting {max(a['smd_after'].values()):.3f}.")
        for note in a.get("notes", []):
            lines.append(f"- Statistician: {note}")
    else:
        lines.append("- No estimate: " + "; ".join((st.get("log") or [])[-2:]))
    return lines + [""]


def main() -> None:
    out = ROOT / "eval" / "results" / "phase2_replay.md"
    runs = ROOT / "runs" / "phase2-replay"
    lines = ["# Phase 2: Replay of three trials on synthetic OMOP", "",
             "Generated by `eval/replay_phase2.py`. Drafts: the Phase 0 final benchmark "
             "(qwen3:32b, H100, replicate 1). Data: synthetic OMOP, one profile per trial, with "
             "a planted effect. **Reviews are auto-approved, not clinical** (`AutoApprove`; "
             "see the script). The "
             "truth is the marginal HR from potential outcomes in the population each run "
             "selects. Replay agreement with the trials' published results is not assessed "
             "here: the data are synthetic.", ""]  # fmt: skip
    for t in TRIALS:
        print(f"replaying {t.nct} ...", file=sys.stderr)
        res = replay(t)
        lines += [f"## {t.nct}", "", f"Design review: {t.why}.", ""]
        if t.nct == "NCT06091982":
            raw = replay(t, design={"reviewer": REVIEWER, "confounders": t.design["confounders"]},
                         overrides=AS_DRAFTED_RRT)
            lines += summarize(t, raw, "as drafted (modalities as a combination, no grace "
                                       "period)")
        lines += summarize(t, res, "after design review")
        d = runs / t.nct
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.md").write_text(res["state"].get("report", ""), encoding="utf-8")
        (d / "review.md").write_text(res["review_sheet"], encoding="utf-8")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}; full reports in {runs}", file=sys.stderr)


if __name__ == "__main__":
    main()
