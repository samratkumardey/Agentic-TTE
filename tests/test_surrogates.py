"""Clinician surrogate answers (agentic_tte/agents/surrogates.py): which concepts are asked
about, the checks on the answer, concept IDs from the vocabulary only, and the pre-filled
concept-review decisions in the Replay graph, on synthetic OMOP."""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agentic_tte.agents.surrogates import SurrogateAnswer, Surrogates, check, unmapped
from agentic_tte.graph import AutoApprove, build_replay, pending
from agentic_tte.llm import ScriptedClient
from agentic_tte.omop import DataContext
from agentic_tte.omop.concept_sets import ConceptSet, ConceptSetBook
from agentic_tte.protocol import ConceptRef
from agentic_tte.synth import omop as so
from agentic_tte.synth.dgp import Scenario, generate
from tests.test_clinician import drafted  # noqa: F401
from tests.test_replay_graph import SPEC


@pytest.fixture(scope="module")
def synthetic():
    """Synthetic OMOP whose local concepts are standard, so the vocabulary search finds them."""
    cohort = generate(Scenario(n=1500, horizon_days=90.0), with_truth=False)
    return cohort, so.write_duckdb(so.to_omop(cohort, so.Profile(standard_local=True)))


def _book(p) -> ConceptSetBook:
    def cs(key, scope, name, domain, status="no_match"):
        return ConceptSet(key=key, scope=scope, criterion_text=name, concept_name=name,
                          domain=domain, status=status)  # fmt: skip

    return ConceptSetBook(protocol_id=p.protocol_id, protocol_version=p.version, sets=[
        cs("C:SAPS II", "confounder", "SAPS II", "Measurement"),
        cs("I9:plateau pressure", "eligibility", "plateau pressure", "Measurement"),
        cs("C:frailty", "confounder", "frailty", "Observation"),
        cs("C:time from randomization", "confounder", "time from randomization", "Observation"),
        cs("T:arm:hydrocortisone", "treatment", "hydrocortisone", "Drug"),
        cs("C:lactate", "confounder", "lactate", "Measurement", status="proposed")])


def _answer() -> SurrogateAnswer:
    return SurrogateAnswer.model_validate({"proposals": [
        {"key": "C:SAPS II", "substitute": "SOFA score", "domain": "Measurement",
         "rationale": "Both summarise organ dysfunction at admission.",
         "limitation": "SOFA weights organs differently from SAPS II."},
        {"key": "I9:plateau pressure", "substitute": "hydrocortisone", "domain": "Drug",
         "rationale": "planted: the treatment"},
        {"key": "C:frailty", "substitute": None, "rationale": "not recorded"},
        {"key": "C:time from randomization", "substitute": "lactate after randomization",
         "domain": "Measurement", "rationale": "planted: post-baseline"},
        {"key": "C:unknown", "substitute": "heart rate", "domain": "Measurement"},
    ]})


def test_only_unmapped_eligibility_and_confounders_are_asked(drafted):  # noqa: F811
    keys = [s.key for s in unmapped(_book(drafted))]
    assert keys == ["C:SAPS II", "I9:plateau pressure", "C:frailty",
                    "C:time from randomization"]  # treatment and mapped concepts are not


def test_checks_and_vocabulary(synthetic, drafted):  # noqa: F811
    _, con = synthetic
    p = drafted.model_copy(deep=True)
    book = _book(p)
    sheet = check(_answer(), p, unmapped(book), con, "scripted")
    by = {i.key: i for i in sheet.items}
    assert by["C:SAPS II"].status == "suggested"
    assert by["C:SAPS II"].candidates[0].concept_id == so.SOFA  # from the vocabulary
    assert by["I9:plateau pressure"].status == "dropped"
    assert by["I9:plateau pressure"].note == "the trial's treatment"
    assert by["C:frailty"].status == "none"
    assert by["C:time from randomization"].status == "dropped"
    assert any("unknown key" in n for n in sheet.notes)
    d = sheet.decisions()
    assert list(d) == ["C:SAPS II"] and d["C:SAPS II"]["action"] == "surrogate"
    assert d["C:SAPS II"]["concept_ids"] == [so.SOFA] and "Limitation" in d["C:SAPS II"]["note"]
    assert "SOFA score" in sheet.to_markdown()


def test_no_call_when_everything_maps(synthetic, drafted):  # noqa: F811
    _, con = synthetic
    llm = ScriptedClient([])
    sheet = Surrogates(llm).propose(drafted, ConceptSetBook(protocol_id="x", protocol_version=1),
                                    con)  # fmt: skip
    assert not llm.requests and not sheet.items and sheet.notes


ANSWER = json.dumps({"proposals": [
    {"key": "C:SAPS II", "substitute": "SOFA score", "domain": "Measurement",
     "rationale": "Organ dysfunction at admission.", "limitation": "Different weights."}]})


def _respond(messages, schema):
    assert messages[0].content.startswith("You are the Clinician agent")
    assert "key C:SAPS II" in messages[1].content
    return ANSWER


def _graph(con, auto=None):
    return build_replay(con, llm=ScriptedClient(_respond, max_retries=1), surrogates=True,
                        checkpointer=InMemorySaver(), auto=auto,
                        context=DataContext(data_classification="synthetic"))  # fmt: skip


def _protocol(drafted):  # noqa: F811
    p = drafted.model_copy(update={"assignment": drafted.assignment.model_copy(update={
        "confounders": [ConceptRef(name="age", domain="Demographic"),
                        ConceptRef(name="SAPS II", domain="Measurement")]})})
    return json.loads(p.model_dump_json())


def test_replay_prefills_the_surrogate_for_review(synthetic, drafted):  # noqa: F811
    _, con = synthetic
    graph = _graph(con)
    cfg = {"configurable": {"thread_id": "s1"}}
    graph.invoke({"protocol": _protocol(drafted)}, cfg)
    from langgraph.types import Command

    graph.invoke(Command(resume={"reviewer": "Dr Test"}), cfg)  # design approved
    ask = pending(graph, cfg)
    assert ask["stage"] == "concepts" and "SOFA score" in ask["surrogates"]
    d = ask["decisions_template"]["decisions"]["C:SAPS II"]
    assert d["action"] == "surrogate" and d["concept_ids"] == [so.SOFA]


def test_auto_approve_does_not_accept_surrogates_by_default(synthetic, drafted):  # noqa: F811
    _, con = synthetic
    for accept, status in ((False, "unmappable"), (True, "surrogate")):
        auto = AutoApprove(stages={"design", "concepts"}, spec=SPEC, accept_surrogates=accept)
        graph = _graph(con, auto)
        cfg = {"configurable": {"thread_id": f"auto-{accept}"}}
        graph.invoke({"protocol": _protocol(drafted)}, cfg)
        book = graph.get_state(cfg).values["book"]
        s = next(x for x in book["sets"] if x["key"] == "C:SAPS II")
        assert s["status"] == status and s["note"].startswith("auto: Clinician surrogate")
