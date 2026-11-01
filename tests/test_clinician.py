"""Clinician agent (agentic_tte/agents/clinician.py): result-blind literature, deterministic
retrieval, and the checks on its confounders (verbatim citations, no treatment, outcome or
post-baseline variable); then in the Replay graph on synthetic OMOP."""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agentic_tte.agents import Clinician, Trialist
from agentic_tte.agents.clinician import (
    ClinicianAnswer,
    Literature,
    build_literature,
    check,
    retrieve,
    verbatim,
)
from agentic_tte.graph import build_replay, pending
from agentic_tte.llm import ScriptedClient
from agentic_tte.omop import DataContext
from agentic_tte.protocol import ConceptRef, TargetTrialProtocol
from agentic_tte.sources.pubmed import Article
from agentic_tte.sources.pubmed.client import parse_articles
from tests.test_replay_graph import SPEC, _auto, synthetic  # noqa: F401
from tests.test_trialist import _respond

LACTATE = ("Higher serum lactate at admission was independently associated with hospital "
           "mortality in septic shock.")  # fmt: skip
SOFA = "The SOFA score predicted mortality and the use of corticosteroids by clinicians."


def _articles():
    return [
        Article(pmid="1", title="Lactate and outcome", abstract=f"Background: cohort. {LACTATE} "
                "Clinicians gave adjunctive therapy more often to sicker patients.", year=2010),
        Article(pmid="2", title="Severity scores", abstract=f"{SOFA} Age was also a predictor.",
                year=2012),
        Article(pmid="3", title="The trial result", abstract="Vitamin C did not reduce death.",
                year=2019),
        Article(pmid="4", title="Secondary analysis", abstract="A post hoc analysis of "
                "NCT03872011 found no benefit.", year=2016, nct_ids=["NCT03872011"]),
        Article(pmid="5", title="Meta-analysis", abstract="Including VICTAS, no effect.",
                year=2016),
        Article(pmid="6", title="Later cohort", abstract="Lactate predicted death.", year=2024),
    ]


class FakePubMed:
    def __init__(self):
        self.queries = []

    def search(self, term, *, retmax=40, maxdate=None):
        self.queries.append((term, maxdate))
        return ["1", "2", "4", "5", "6"]

    def fetch(self, pmids):
        by = {a.pmid: a for a in _articles()}
        return [by[x] for x in pmids if x in by]  # in the order asked, as the real client


@pytest.fixture
def drafted(nct03872011_study):
    return Trialist(ScriptedClient(_respond, max_retries=1)).draft_from_study(nct03872011_study)


def _study(study):
    s = json.loads(json.dumps(study))
    ps = s.setdefault("protocolSection", s)
    ps.setdefault("identificationModule", {})["acronym"] = "VICTAS"
    ps.setdefault("statusModule", {})["primaryCompletionDateStruct"] = {"date": "2019-12"}
    ps["referencesModule"] = {"references": [{"pmid": "3", "type": "RESULT"},
                                             {"pmid": "2", "type": "BACKGROUND"}]}
    return s


def test_literature_is_result_blind(nct03872011_study, drafted):
    client = FakePubMed()
    lit = build_literature(_study(nct03872011_study), drafted, client)
    assert [a.pmid for a in lit.articles] == ["2", "1"]  # background first, then searches
    why = {e.pmid: e.reason for e in lit.excluded}
    assert "trial" in why["3"] and "NCT ID" in why["4"] and "acronym" in why["5"]
    assert "after primary completion" in why["6"]
    assert lit.cutoff == "2019-12" and all(m == "2019/12/28" for _, m in client.queries)


def test_parse_pubmed_xml():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>9</PMID><Article>
      <Journal><JournalIssue><PubDate><Year>2015</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>A <i>cohort</i> study</ArticleTitle><Abstract>
      <AbstractText Label="RESULTS">Lactate predicted death.</AbstractText></Abstract>
      <DataBankList><DataBank><AccessionNumberList><AccessionNumber>NCT01234567
      </AccessionNumber></AccessionNumberList></DataBank></DataBankList>
      <PublicationTypeList><PublicationType>Journal Article</PublicationType>
      </PublicationTypeList></Article></MedlineCitation></PubmedArticle>
      <PubmedArticle><MedlineCitation><PMID>10</PMID><Article><ArticleTitle>No abstract
      </ArticleTitle></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    (a,) = parse_articles(xml)
    assert (a.pmid, a.title, a.year) == ("9", "A cohort study", 2015)
    assert a.abstract == "RESULTS: Lactate predicted death." and a.nct_ids == ["NCT01234567"]


def test_retrieval_is_deterministic(drafted):
    lit = Literature(articles=_articles()[:2])
    a, b = retrieve(drafted, lit), retrieve(drafted, lit)
    assert [x.id for x in a] == [x.id for x in b] and a
    assert {x.pmid for x in a} <= {"1", "2"}


def _answer(drafted):
    outcome = next(o for o in drafted.outcomes if o.primary).name
    return ClinicianAnswer.model_validate({"confounders": [
        {"name": "serum lactate", "domain": "Measurement", "rationale": "severity",
         "citations": [{"pmid": "1", "quote": LACTATE.lower()},
                       {"pmid": "1", "quote": "lactate was protective"}]},
        {"name": "Serum lactate", "domain": "Measurement", "rationale": "duplicate",
         "citations": [{"pmid": "2", "quote": "The SOFA score predicted mortality"}]},
        {"name": "SOFA score", "domain": "Measurement", "rationale": "severity",
         "citations": [{"pmid": "99", "quote": "a paper that is not in the literature"}]},
        {"name": "hydrocortisone", "domain": "Drug", "rationale": "co-treatment"},
        {"name": outcome, "domain": "Observation", "rationale": "the outcome"},
        {"name": "time from shock onset to study drug", "domain": "Observation",
         "rationale": "timing"},
        {"name": "Age", "domain": "Demographic", "rationale": "older patients die more"},
    ]})


def test_checks_on_the_answer(drafted):
    lit = Literature(articles=_articles()[:2])
    sheet = check(_answer(drafted), drafted, lit, "scripted", [])
    items = {i.name: i for i in sheet.items}
    assert list(items) == ["serum lactate", "SOFA score", "Age", "sex"]
    lac = items["serum lactate"]
    # the verbatim quote is kept (case-insensitive), the invented one dropped; the duplicate's
    # citation is merged in
    assert [c.pmid for c in lac.citations] == ["1", "2"] and lac.status == "cited"
    assert any("not found verbatim" in n for n in lac.notes)
    assert items["SOFA score"].status == "uncited"
    assert any("not in the literature" in n for n in items["SOFA score"].notes)
    assert items["sex"].status == "standard" and items["Age"].status == "uncited"
    dropped = {d.name: d.reason for d in sheet.dropped}
    assert "F-h" in dropped["hydrocortisone"]
    assert any(r == "the primary outcome" for r in dropped.values())
    assert "post-baseline" in dropped["time from shock onset to study drug"]
    md = sheet.to_markdown()
    assert "Dropped by the checks" in md and "not a decision" in md


def test_clinician_asks_once_with_the_passages(drafted):
    llm = ScriptedClient([json.dumps({"confounders": [
        {"name": "serum lactate", "domain": "Measurement", "rationale": "severity",
         "citations": [{"pmid": "1", "quote": LACTATE}]}]})])
    sheet = Clinician(llm).propose(drafted, Literature(articles=_articles()[:2]))
    (msgs, schema), = llm.requests
    assert "[PMID 1]" in msgs[1].content and "Time zero:" in msgs[1].content
    assert schema is not None and sheet.items[0].status == "cited"
    assert sheet.passages and sheet.model == "scripted"


CLINICIAN = json.dumps({"confounders": [
    {"name": "sofa", "domain": "Measurement", "rationale": "severity drives steroid use and "
     "death", "citations": [{"pmid": "2", "quote": SOFA}]},
    {"name": "lactate", "domain": "Measurement", "rationale": "severity",
     "citations": [{"pmid": "1", "quote": LACTATE}]},
    {"name": "hydrocortisone", "domain": "Drug", "rationale": "planted: the treatment"}]})


def _with_clinician(messages, schema):
    if messages[0].content.startswith("You are the Clinician agent"):
        return CLINICIAN
    return _respond(messages, schema)


def test_replay_with_the_clinician(synthetic, nct03872011_study):  # noqa: F811
    """The Clinician's confounders reach design review with their sheet; approving keeps them,
    and they are the ones the Statistician adjusts for."""
    _, con = synthetic
    lit = Literature(articles=_articles()[:2])
    graph = build_replay(con, llm=ScriptedClient(_with_clinician, max_retries=1),
                         context=DataContext(data_classification="synthetic"),
                         checkpointer=InMemorySaver(), literature=lit,
                         auto=_auto(stages={"concepts", "gate", "signoff"}))  # fmt: skip
    cfg = {"configurable": {"thread_id": "clin"}}
    graph.invoke({"study": nct03872011_study}, cfg)
    ask = pending(graph, cfg)
    assert ask["stage"] == "design" and "Confounders proposed by the Clinician" in ask["clinician"]
    p = TargetTrialProtocol.model_validate(ask["protocol"])
    assert [c.name for c in p.assignment.confounders] == ["sofa", "lactate", "age", "sex"]
    assert p.assignment.origin == "agent" and "Clinician (scripted)" in p.provenance.created_by
    graph.invoke(Command(resume={"reviewer": "Dr Test"}), cfg)  # approve as proposed
    ask = pending(graph, cfg)  # the Auditor's A-f on this draft, as without the Clinician
    assert ask["stage"] == "design" and ask.get("audit")
    state = graph.invoke(Command(resume={"reviewer": "Dr Test", "override_audit": {
        "reason": "synthetic test data: no one has the treatment before time zero"}}), cfg)
    assert state["analysis"]["confounders"] == ["sofa", "lactate", "age", "female"]
    assert "hydrocortisone: the treatment itself (F-h)" in state["report"]
    assert "confounders proposed by the Clinician (4, 2 cited, 1 dropped" in state["report"]


def test_reviewer_given_confounders_are_not_replaced(synthetic, drafted):  # noqa: F811
    _, con = synthetic
    p = drafted.model_copy(update={"assignment": drafted.assignment.model_copy(update={
        "confounders": [ConceptRef(name="age", domain="Demographic")]})})
    graph = build_replay(con, llm=ScriptedClient([]), checkpointer=InMemorySaver(),
                         literature=Literature(), context=DataContext(
                             data_classification="synthetic"))  # fmt: skip
    cfg = {"configurable": {"thread_id": "given"}}
    graph.invoke({"protocol": json.loads(p.model_dump_json())}, cfg)
    ask = pending(graph, cfg)
    assert ask["stage"] == "design" and "clinician" not in ask
    assert any("Clinician skipped" in m for m in graph.get_state(cfg).values["log"])


def test_queries_use_the_main_condition_and_filter_the_population(drafted):
    p = drafted.model_copy(update={"source": drafted.source.model_copy(update={
        "conditions": ["Hydrocortisone", "Septic Shock", "COVID-19"]})})
    q = __import__("agentic_tte.agents.clinician", fromlist=["x"]).literature_queries(p)
    assert q and all('"Septic Shock"' in x and "COVID" not in x for x in q)
    assert all("NOT (animals[mh] NOT humans[mh])" in x and "pediatric" in x for x in q)


def test_acronym_variants():
    from agentic_tte.agents.clinician import acronym_pattern

    assert acronym_pattern("(BICAR-ICU)").search("the BICAR-ICU trial")
    assert acronym_pattern("CAPE_COD").search("In CAPE COD, hydrocortisone")
    assert not acronym_pattern("CHEST").search("chest radiograph")
    assert acronym_pattern("EvK") is None


def test_score_clinician_against_the_references(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    ev = Path(__file__).parents[1] / "eval"
    sys.path.insert(0, str(ev))
    spec = importlib.util.spec_from_file_location("score_clinician", ev / "score_clinician.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    for ref in sorted((ev / "reference_protocols").glob("NCT*.adjudicated.json")):
        nct = ref.name.split(".")[0]
        names = [c.name for c in sc.load(ref).assignment.confounders]
        for rep, extra in (("rep1", []), ("rep2", ["something else"])):
            sheet = {"model": "m", "prompt_version": "clinician-v1", "dropped": [],
                     "items": [{"name": n, "status": "cited"} for n in names + extra]}
            d = tmp_path / "m" / rep
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{nct}.clinician.json").write_text(json.dumps(sheet), encoding="utf-8")
    rows = sc.score(tmp_path, "m")
    assert len(rows) == 12 and all(r["prf"].recall == 1 for r in rows)
    assert not any(r["rep_identical"] for r in rows)
    assert "Disclosure" in sc.render(rows, tmp_path, "m")


def test_answer_schema_is_bounded():
    # clinician-v1 had no bounds and ran past the request timeout on the core trials; the
    # bounds are in the JSON schema the decoder is constrained by.
    from agentic_tte.llm.base import inline_refs

    s = json.dumps(inline_refs(ClinicianAnswer.model_json_schema()))
    assert '"maxItems": 20' in s and '"maxItems": 2' in s and '"maxLength": 300' in s
    with pytest.raises(ValueError):
        ClinicianAnswer.model_validate({"confounders": [
            {"name": f"v{i}", "domain": "Measurement", "rationale": "r"} for i in range(21)]})


def test_quote_check_v3():
    text = ("Plasma citrulline ≤ 10 µmol/l at 24 h (OR 2.1) and the SOFA score were "
            "independent predictors of 28-day mortality. Exclusion criteria were COPD, "
            "liver cirrhosis and pregnancy.")
    # punctuation and symbols are ignored, words must be whole and in order
    assert verbatim("plasma citrulline 10 umol/l at 24 h", text) is False  # a different word
    assert verbatim("Plasma citrulline <= 10 µmol/l at 24 h", text)
    assert verbatim("SOFA score were independent predictors", text)
    assert not verbatim("ore were independent predictors", text)  # a part-word
    # "..." joins fragments that are each four or more words, in order
    assert verbatim("Plasma citrulline ≤ 10 µmol/l ... were independent predictors of", text)
    assert not verbatim("Plasma citrulline ≤ 10 ... were independent predictors of", text)
    assert not verbatim("were independent predictors of 28-day ... Plasma citrulline ≤ 10",
                        text)  # fmt: skip
    assert not verbatim("Exclusion criteria ... liver cirrhosis", text)  # short fragments
    assert not verbatim("SOFA predicted mortality", text)
    assert not verbatim("...", text)
