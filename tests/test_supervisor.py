"""The Supervisor (agentic_tte/agents/supervisor.py): the fixed routing of the Replay graph,
its trace in the state and the report, and the loop caps and LLM budget that stop a run."""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agentic_tte.agents.supervisor import Budget, Supervisor
from agentic_tte.graph import AutoApprove, build_replay, pending
from agentic_tte.llm import ScriptedClient
from agentic_tte.omop import DataContext
from tests.test_replay_graph import _auto, synthetic  # noqa: F401
from tests.test_trialist import _respond

CTX = DataContext(data_classification="synthetic")


def test_routes_follow_the_replay_order():
    sup = Supervisor()
    r, up = sup.route({})
    assert (r.after, r.to, r.step) == ("start", "draft", 1) and up["visits"] == {"draft": 1}
    assert sup.route({"last": "draft", "steps": 1})[0].to == "design_review"
    assert Supervisor(clinician=True).route({"last": "draft"})[0].to == "clinician"
    gate = {"last": "gate", "gate": {"checks": [{"status": "green"}, {"status": "amber"}]}}
    assert sup.route(gate)[0].to == "gate_review"
    gate["gate"]["checks"].append({"status": "red"})
    assert sup.route(gate)[0].to == "report"
    audit = {"last": "audit", "audit": {"failed": ["A-f"]}}
    assert sup.route(audit)[0].to == "design_review"
    audit["audit_override"] = {"checks": ["A-f"]}
    r = sup.route(audit)[0]
    assert r.to == "gate" and "overridden" in r.reason
    assert sup.route({"last": "report", "status": "done"})[0].to == "signoff"
    assert sup.route({"last": "report", "status": "blocked"})[0].to == "end"
    with pytest.raises(ValueError, match="no route"):
        sup.route({"last": "nowhere"})


def test_caps_stop_the_run():
    sup = Supervisor(Budget(max_audit_rounds=2, max_steps=5, max_llm_calls=1),
                     llm_nodes={"draft", "clinician"})  # fmt: skip
    r, up = sup.route({"last": "audit", "audit": {"failed": ["A-f"]}, "audit_rounds": 2,
                       "protocol": {"x": 1}})  # fmt: skip
    assert r.to == "report" and up["status"] == "stopped"
    assert up["stop_reason"] == "the design failed the audit 2 times"
    r, up = sup.route({"last": "compile", "steps": 5, "protocol": {"x": 1}})
    assert r.to == "report" and "5 routing steps" in up["stop_reason"]
    r, _ = sup.route({"last": "report", "status": "done", "steps": 9})
    assert r.to == "signoff"  # a finished run is never cut off before its sign-off
    # The LLM budget: drafting from the registry is a model call, a given protocol is not.
    r, up = sup.route({"llm_calls": 1})
    assert r.to == "end" and "LLM budget" in up["stop_reason"]  # no protocol: no report
    assert sup.route({"llm_calls": 1, "protocol": {"x": 1}})[0].to == "draft"


def test_auto_run_records_every_route(synthetic, nct03872011_study):  # noqa: F811
    _, con = synthetic
    graph = build_replay(con, llm=ScriptedClient(_respond, max_retries=1), context=CTX,
                         auto=_auto())  # fmt: skip
    state = graph.invoke({"study": nct03872011_study})
    assert state["status"] == "auto_approved"
    path = [r["to"] for r in state["routes"]]
    assert path[:7] == ["draft", "design_review", "propose_concepts", "concept_review",
                        "compile", "audit", "gate"]  # fmt: skip
    assert path[-3:] == ["report", "signoff", "end"]
    assert [r["step"] for r in state["routes"]] == list(range(1, len(path) + 1))
    assert state["llm_calls"] >= 1 and state["visits"]["draft"] == 1
    assert "## Supervisor" in state["report"] and "| 1 | start | draft |" in state["report"]
    assert "Fixed routing policy" in state["report"]
    json.dumps(state, default=str)  # the trace is plain JSON


def test_a_tighter_audit_cap_stops_earlier(synthetic, nct03872011_study):  # noqa: F811
    _, con = synthetic
    graph = build_replay(con, llm=ScriptedClient(_respond, max_retries=1), context=CTX,
                         checkpointer=InMemorySaver(), budget=Budget(max_audit_rounds=1),
                         auto=AutoApprove(stages={"concepts"}, spec=_auto().spec,
                                          overrides=_auto().overrides))  # fmt: skip
    cfg = {"configurable": {"thread_id": "cap"}}
    graph.invoke({"study": nct03872011_study}, cfg)
    graph.invoke(Command(resume={"reviewer": "Dr Test"}), cfg)  # design approved as drafted
    state = graph.get_state(cfg).values
    assert pending(graph, cfg) is None and state["status"] == "stopped"
    assert state["stop_reason"] == "the design failed the audit 1 times"
    assert "Stopped by the Supervisor" in state["report"] and "analysis" not in state


def test_the_llm_budget_stops_before_the_model_is_called(synthetic, nct03872011_study):  # noqa: F811
    _, con = synthetic
    draft = build_replay(con, llm=ScriptedClient(_respond, max_retries=1), context=CTX,
                         auto=_auto()).invoke({"study": nct03872011_study})["protocol"]
    llm = ScriptedClient([])  # any request would fail the test
    graph = build_replay(con, llm=llm, context=CTX, surrogates=True, auto=_auto(),
                         budget=Budget(max_llm_calls=0))  # fmt: skip
    state = graph.invoke({"protocol": draft})
    assert not llm.requests and state["status"] == "stopped"
    assert "LLM budget is spent (0 of 0 requests)" in state["stop_reason"]
    assert state["routes"][-2]["after"] == "design_review"
