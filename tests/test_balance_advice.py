"""The Statistician <-> Clinician loop: the outcome-blind balance check
(agentic_tte/agents/statistician.py ``balance``), the Clinician's advice and its checks
(agentic_tte/agents/balance_advice.py), and the route through design review in the Replay
graph, on synthetic data."""

import json

import numpy as np
import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agentic_tte.agents.balance_advice import BalanceAnswer, BalanceSheet, check, column
from agentic_tte.agents.statistician import BalanceReport, balance
from agentic_tte.agents.supervisor import Budget
from agentic_tte.graph import AutoApprove, build_replay, pending
from agentic_tte.llm import ScriptedClient
from agentic_tte.omop import DataContext
from agentic_tte.protocol import ConceptRef
from agentic_tte.synth import omop as so
from tests.test_clinician import drafted  # noqa: F401
from tests.test_replay_graph import CONFOUNDERS, _auto, synthetic  # noqa: F401
from tests.test_trialist import _respond

CTX = DataContext(data_classification="synthetic")


def test_balance_flags_from_treatment_and_confounders_only():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    y[rng.random(n) < 0.4] = np.nan
    frame = pd.DataFrame({"treated": ((x + rng.normal(scale=0.3, size=n)) > 0).astype(int),
                          "x": x, "y": y, "c": 1.0})  # fmt: skip
    rep = balance(frame, ["x", "y", "c"])  # no time, event or outcome column exists
    assert rep.flagged["y"] == ["missing 40%"] or rep.flagged["y"][0].startswith("missing")
    assert rep.flagged["c"] == ["constant in the complete cases"]
    assert rep.flagged["x"][0].startswith("SMD") and rep.smd_after["x"] >= 0.1
    assert rep.ps_extreme > 0 and "before any outcome analysis" in rep.to_markdown()
    assert not balance(frame, []).flagged


def _rep(**flagged) -> BalanceReport:
    return BalanceReport(n_frame=100, n_complete=100, confounders=list(flagged), flagged=flagged,
                         smd_before={k: 0.5 for k in flagged},
                         smd_after={k: 0.2 for k in flagged})  # fmt: skip


def test_checks_keep_the_rules(drafted):  # noqa: F811
    arm = next(a for a in drafted.treatment_strategies if not a.is_comparator)
    p = drafted.model_copy(update={"assignment": drafted.assignment.model_copy(update={
        "confounders": [ConceptRef(name="age", domain="Demographic"),
                        ConceptRef(name="sex", domain="Demographic"),
                        ConceptRef(name="SAPS II", domain="Measurement"),
                        ConceptRef(name="lactate", domain="Measurement"),
                        ConceptRef(name="frailty", domain="Observation"),
                        ConceptRef(name="SOFA score", domain="Measurement")]})})
    assert [column(r) for r in p.assignment.confounders][:3] == ["age", "female", "saps_ii"]
    rep = _rep(age=["SMD 0.20 after weighting"], female=["SMD 0.20 after weighting"],
               saps_ii=["missing 60%"], lactate=["missing 50%"], frailty=["missing 90%"])
    answer = BalanceAnswer.model_validate({"advice": [
        {"column": "age", "action": "drop", "rationale": "planted: imbalanced"},
        {"column": "female", "action": "replace", "substitute": "pregnancy",
         "domain": "Condition"},
        {"column": "saps_ii", "action": "replace", "substitute": "SOFA score",
         "domain": "Measurement", "rationale": "better recorded severity score"},
        {"column": "lactate", "action": "replace",
         "substitute": arm.components[0].concept.name, "domain": "Drug"},
        {"column": "frailty", "action": "drop", "rationale": "rarely recorded"},
        {"column": "heart_rate", "action": "drop"}]})
    sheet = check(answer, p, rep, "scripted")
    by = {i.column: i for i in sheet.items}
    assert by["age"].action == "keep" and "always adjusted" in by["age"].note
    assert by["female"].action == "keep"
    assert by["saps_ii"].action == "replace" and "already a confounder" in by["saps_ii"].note
    assert by["lactate"].action == "keep" and "treatment" in by["lactate"].note
    assert by["frailty"].action == "drop"
    assert any("unflagged" in n for n in sheet.notes) and sheet.changes
    names = [c["name"] for c in sheet.confounders(p)]
    assert names == ["age", "sex", "SOFA score", "lactate"]  # frailty dropped, no duplicate
    assert "Clinician balance advice" in sheet.to_markdown()


ADVICE = {"advice": [{"column": "septic_shock", "action": "drop",
                      "rationale": "Every patient has it at time zero: it carries no "
                                   "information about the treatment choice."}]}  # fmt: skip


def _llm(seen: list):
    def respond(messages, schema):
        if "Statistician weighted" in messages[0].content:
            seen.append(messages[1].content)
            return json.dumps(ADVICE)
        return _respond(messages, schema)

    return ScriptedClient(respond, max_retries=1)


def _auto_with_shock(**kw) -> AutoApprove:
    base = _auto()
    return AutoApprove(
        design={"confounders": [*CONFOUNDERS, {"name": "septic shock", "domain": "Condition"}]},
        spec=base.spec, overrides={**base.overrides, "C:septic shock": {
            "action": "replace", "concept_ids": [so.SEPTIC_SHOCK]}}, **kw)


def test_advice_applied_reruns_the_design_before_any_estimate(synthetic, nct03872011_study):  # noqa: F811
    _, con = synthetic
    seen: list = []
    graph = build_replay(con, llm=_llm(seen), context=CTX, balance_advice=True,
                         auto=_auto_with_shock(accept_balance=True))  # fmt: skip
    state = graph.invoke({"study": nct03872011_study})
    path = [r["to"] for r in state["routes"]]
    first, second = path.index("balance_advice"), path.index("analyze")
    assert first < second  # the advice comes before the outcome analysis
    assert path[first:second + 1] == ["balance_advice", "design_review", "compile", "audit",
                                      "gate", "gate_review", "balance", "analyze"]
    assert len(seen) == 1 and "septic_shock" in seen[0]
    assert not any(w in seen[0].lower() for w in ("hazard", "estimate", "event"))
    rec = [r for r in state["reviews"] if r["decision"].startswith("balance advice")]
    assert rec[0]["decision"] == "balance advice applied" and rec[0]["auto"]
    assert "septic_shock" not in state["analysis"]["confounders"]
    assert "Clinician balance advice" in state["report"]
    assert "Covariate balance (before any outcome analysis)" in state["report"]


def test_declined_advice_goes_on_to_the_analysis(synthetic, nct03872011_study):  # noqa: F811
    _, con = synthetic
    auto = _auto_with_shock()
    graph = build_replay(con, llm=_llm([]), context=CTX, balance_advice=True,
                         checkpointer=InMemorySaver(),
                         auto=auto.model_copy(update={"stages": {"concepts", "gate", "signoff"}})
                         )  # fmt: skip
    cfg = {"configurable": {"thread_id": "decline"}}
    graph.invoke({"study": nct03872011_study}, cfg)
    graph.invoke(Command(resume={"confounders": auto.design["confounders"],
                                 "reviewer": "Dr Test"}), cfg)  # fmt: skip
    ask = pending(graph, cfg)  # the audit fails first (A-f, no new-user exclusion)
    graph.invoke(Command(resume={"reviewer": "Dr Test", "override_audit": {
        "reason": "synthetic test data"}}), cfg)  # fmt: skip
    ask = pending(graph, cfg)
    assert ask["stage"] == "design" and "septic_shock" in ask["balance"]
    assert [c["name"] for c in ask["proposed_confounders"]] == [c["name"] for c in CONFOUNDERS]
    state = graph.invoke(Command(resume={"reviewer": "Dr Test"}), cfg)  # keep the set
    assert state["status"] == "auto_approved"
    assert "septic_shock" in state["analysis"]["confounders"]
    assert any(r["decision"] == "balance advice declined" and not r["auto"]
               for r in state["reviews"])
    routes = [(r["after"], r["to"]) for r in state["routes"]]
    assert ("design_review", "analyze") in routes


def test_loop_off_or_capped_reports_the_flags(synthetic, nct03872011_study):  # noqa: F811
    _, con = synthetic
    for kw in ({}, {"balance_advice": True, "budget": Budget(max_balance_rounds=0)}):
        seen: list = []
        state = build_replay(con, llm=_llm(seen), context=CTX, auto=_auto_with_shock(),
                             **kw).invoke({"study": nct03872011_study})  # fmt: skip
        assert not seen and state["status"] == "auto_approved"
        assert "septic_shock" in state["balance"]["flagged"]
        r = next(r for r in state["routes"] if r["after"] == "balance")
        assert r["to"] == "analyze" and "reported" in r["reason"]


def test_sheet_round_trips_as_json():
    sheet = BalanceSheet(model="m", items=[], notes=["x"])
    assert BalanceSheet.model_validate(json.loads(sheet.model_dump_json())) == sheet
