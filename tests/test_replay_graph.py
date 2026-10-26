"""End-to-end Replay on synthetic OMOP through the LangGraph workflow: Trialist (scripted LLM on
the real NCT03872011 registry record) -> design review -> concept review -> compile -> gate
-> gate review -> statistics -> report, checked against potential-outcome truth."""

import json

import numpy as np
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agentic_tte.graph import build_replay, pending
from agentic_tte.llm import ScriptedClient
from agentic_tte.omop import DataContext
from agentic_tte.synth import omop as so
from agentic_tte.synth.dgp import Scenario, generate
from tests.test_trialist import _respond

CONFOUNDERS = [{"name": "age", "domain": "Demographic"}, {"name": "sex", "domain": "Demographic"},
               {"name": "sofa", "domain": "Measurement"},
               {"name": "lactate", "domain": "Measurement"}]  # fmt: skip
SPEC = {"time_zero": {"domain": "Visit", "concept_ids": [so.ICU_VISIT],
                      "note": "ICU admission with septic shock (synthetic index visit)"}}


def clinician_decisions(template: dict) -> dict:
    """What a reviewer would answer for the synthetic database: the study drug stands in for
    the trial's hydrocortisone (a surrogate, with a note), local concepts for SOFA and lactate,
    and the made-up criteria are unmappable."""
    out = {"reviewer": "Dr Test", "decisions": {}}
    for key, d in template["decisions"].items():
        if key.startswith("T:"):
            out["decisions"][key] = {"action": "surrogate", "concept_ids": [so.STUDY_DRUG],
                                     "note": "synthetic study drug stands in for the trial arm"}
        elif key == "C:sofa":
            out["decisions"][key] = {"action": "replace", "concept_ids": [so.SOFA]}
        elif key == "C:lactate":
            out["decisions"][key] = {"action": "replace", "concept_ids": [so.LACTATE]}
        else:
            out["decisions"][key] = {"action": d["action"], "concept_ids": d["concept_ids"]}
    return out


@pytest.fixture(scope="module")
def synthetic():
    cohort = generate(Scenario(n=3000, horizon_days=90.0), with_truth=False)
    return cohort, so.write_duckdb(so.to_omop(cohort))


def _run(con, study, context, **kw):
    graph = build_replay(con, llm=ScriptedClient(_respond, max_retries=1), context=context,
                         checkpointer=InMemorySaver(), **kw)  # fmt: skip
    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"study": study}, cfg)
    return graph, cfg


def test_replay_end_to_end(synthetic, nct03872011_study):
    cohort, con = synthetic
    graph, cfg = _run(con, nct03872011_study, DataContext(data_classification="synthetic"),
                      design_hr=0.6)

    ask = pending(graph, cfg)
    assert ask["stage"] == "design" and ask["protocol"]["source"]["nct_id"] == "NCT03872011"
    graph.invoke(Command(resume={"confounders": CONFOUNDERS, "reviewer": "Dr Test"}), cfg)

    ask = pending(graph, cfg)
    assert ask["stage"] == "concepts" and "septic shock" in ask["review_sheet"]
    assert any(k.startswith("C:sofa") for k in ask["decisions_template"]["decisions"])
    graph.invoke(Command(resume={"decisions": clinician_decisions(ask["decisions_template"]),
                                 "spec": SPEC}), cfg)  # fmt: skip

    ask = pending(graph, cfg)  # amber: made-up criteria unmappable, local vocabulary only
    assert ask["stage"] == "gate" and "AMBER" in ask["gate"]
    state = graph.invoke(Command(resume={"proceed": True, "reviewer": "Dr Test",
                                         "note": "synthetic run"}), cfg)  # fmt: skip

    assert state["status"] == "done" and pending(graph, cfg) is None
    versions = [v["version"] for v in state["versions"]]
    assert versions == [1, 2, 3]  # draft, design review (confounders), concept review
    a = state["analysis"]
    assert a["primary_method"] == "clone-censor-weight"  # the arm has a 12-hour grace period
    assert a["confounders"] == ["age", "female", "sofa", "lactate"]
    est = {e["method"]: e for e in a["estimates"]}
    truth = cohort.truth_in(cohort.frame.person_id, "t1")  # everyone is eligible here
    for m in ("clone-censor-weight", "iptw"):
        assert abs(np.log(est[m]["hr"]) - np.log(truth)) < 0.15, (m, est[m]["hr"], truth)
        assert est[m]["lower"] <= truth <= est[m]["upper"]
    # Confounding by indication: the crude estimate is further from the truth.
    assert abs(np.log(est["crude"]["hr"]) - np.log(truth)) > abs(np.log(est["iptw"]["hr"])
                                                                - np.log(truth))  # fmt: skip
    assert max(a["smd_after"].values()) < 0.1
    rep = state["report"]
    for text in ("# Replay: NCT03872011", "Data Support Gate", "Attrition", "Effect estimates",
                 "clone-censor-weight (primary)", "gate limitations accepted by Dr Test"):
        assert text in rep, text
    # Only aggregates in the state: no frame, and the run is serializable as it is.
    assert not {"frame", "rows"} & set(state)
    assert len(json.dumps(state, default=str)) < 400_000


def test_red_gate_stops_the_run(synthetic, nct03872011_study):
    _, con = synthetic
    graph, cfg = _run(con, nct03872011_study, DataContext())  # classification not declared
    graph.invoke(Command(resume={"reviewer": "Dr Test"}), cfg)  # approve the design as drafted
    ask = pending(graph, cfg)
    graph.invoke(Command(resume={"decisions": clinician_decisions(ask["decisions_template"]),
                                 "spec": SPEC}), cfg)  # fmt: skip
    state = graph.get_state(cfg).values
    assert pending(graph, cfg) is None  # no gate review: red blocks
    assert state["status"] == "blocked" and "analysis" not in state
    assert "run blocked" in state["report"]


def test_reviewer_can_stop_at_an_amber_gate(synthetic, nct03872011_study):
    _, con = synthetic
    graph, cfg = _run(con, nct03872011_study, DataContext(data_classification="synthetic"))
    graph.invoke(Command(resume={"confounders": CONFOUNDERS}), cfg)
    ask = pending(graph, cfg)
    graph.invoke(Command(resume={"decisions": clinician_decisions(ask["decisions_template"]),
                                 "spec": SPEC}), cfg)  # fmt: skip
    state = graph.invoke(Command(resume={"proceed": False, "reviewer": "Dr Test"}), cfg)
    assert state["status"] == "stopped" and "analysis" not in state


def test_cli_replay_resumes_from_checkpoints(tmp_path, nct03872011_study):
    """Each command builds a new graph: the run carries over in checkpoints.sqlite, as it
    would between a clinician's sessions."""
    from typer.testing import CliRunner

    from agentic_tte.agents import Trialist
    from agentic_tte.cli import app

    db = tmp_path / "synth.duckdb"
    so.write_duckdb(so.to_omop(generate(Scenario(n=800, horizon_days=90.0),
                                        with_truth=False)), db).close()  # fmt: skip
    draft = Trialist(ScriptedClient(_respond, max_retries=1)).draft_from_study(nct03872011_study)
    proto = tmp_path / "draft.json"
    proto.write_text(draft.model_dump_json(), encoding="utf-8")
    out, run = tmp_path / "run", CliRunner().invoke

    r = run(app, ["replay", "start", "-o", str(out), "--db", str(db), "--protocol", str(proto),
                  "--classification", "synthetic"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "waiting for the design review" in r.output
    r = run(app, ["replay", "resume", "-o", str(out)])  # design.json has no reviewer yet
    assert r.exit_code == 2 and "needs a reviewer" in r.output
    (out / "design.json").write_text(json.dumps({"reviewer": "Dr Test",
                                                 "confounders": CONFOUNDERS}), "utf-8")
    r = run(app, ["replay", "resume", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert "waiting for the concepts review" in r.output and (out / "review.md").exists()

    tmpl = json.loads((out / "decisions.json").read_text(encoding="utf-8"))
    (out / "decisions.json").write_text(json.dumps(clinician_decisions(tmpl)), "utf-8")
    spec = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    assert spec["time_zero"]["concept_ids"] == []  # the reviewer fills this in
    (out / "spec.json").write_text(json.dumps({**spec, **SPEC}), "utf-8")
    r = run(app, ["replay", "resume", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert "waiting for the gate review" in r.output and "AMBER" in (out / "gate.md").read_text(
        encoding="utf-8")  # fmt: skip

    (out / "gate.json").write_text(json.dumps({"proceed": True, "reviewer": "Dr Test"}), "utf-8")
    r = run(app, ["replay", "resume", "-o", str(out)])
    assert r.exit_code == 0, r.output
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Status: **done**" in report and "clone-censor-weight (primary)" in report
    r = run(app, ["replay", "start", "-o", str(out), "--db", str(db), "--protocol", str(proto),
                  "--classification", "synthetic"])  # fmt: skip
    assert r.exit_code == 2  # a run directory is never reused
