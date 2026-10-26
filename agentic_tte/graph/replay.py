"""Replay as a LangGraph workflow (PLAN.md §3, §13; Phase 2 exit criterion).

    draft -> design_review* -> propose_concepts -> concept_review* -> compile -> gate
          -> [red: stop] [amber: gate_review*] -> analyze -> report

``*`` marks a human review. Each one is a LangGraph ``interrupt()``: the run stops, its state is
saved by the checkpointer, and it resumes with the reviewer's answer
(``Command(resume=...)``):
- design_review: the drafted protocol. The answer can change protocol elements and name the
  confounders. Any change makes a new protocol version (created by the user).
- concept_review: the concept-set review sheet, a decisions file to fill in, and the compile
  spec (time zero and anchors as concept sets). Answer ``{"decisions": ..., "spec": ...}``.
- gate_review: only when the gate is amber. Answer ``{"proceed": bool, "reviewer": ...}``.

The LLM is used only by the Trialist, on registry text. The database connection is closed over
when the graph is built, never put in the state. The state holds the protocol, concept sets,
SQL and aggregates (suppressed counts, estimates): no patient rows. The supervisor is the
fixed routing below for now; an LLM supervisor can replace it later.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agentic_tte.omop.attrition import attrition
from agentic_tte.omop.compiler import CompiledCohort, CompileSpec, compile_protocol, spec_template
from agentic_tte.omop.concept_sets import (
    ConceptSetBook,
    apply_decisions,
    apply_to_protocol,
    decisions_template,
    propose,
    review_sheet,
)
from agentic_tte.omop.gate import DataContext, GateStatus, run_gate
from agentic_tte.protocol import ConceptRef, Origin, TargetTrialProtocol, new_version


class ReplayState(TypedDict, total=False):
    nct_id: str
    study: dict  # a registry record, instead of fetching nct_id
    protocol: dict  # current version (JSON)
    versions: Annotated[list[dict], operator.add]  # every version, in order
    book: dict
    spec: dict
    compiled: dict
    attrition: dict  # public (suppressed)
    attrition_md: str
    gate: dict
    gate_md: str
    analysis: dict
    analysis_md: str
    report: str
    status: str  # running, blocked, stopped, done
    log: Annotated[list[str], operator.add]


def _p(state: ReplayState) -> TargetTrialProtocol:
    return TargetTrialProtocol.model_validate(state["protocol"])


def _version(p: TargetTrialProtocol, why: str) -> dict:
    d = p.model_dump(mode="json")
    return {"protocol": d, "versions": [d], "log": [f"protocol v{p.version}: {why}"]}


def build_replay(con, *, llm=None, ctgov=None, context: DataContext | None = None,
                 design_hr: float | None = None, checkpointer=None):  # fmt: skip
    """Compile the Replay graph over an open OMOP connection. A checkpointer is needed for
    the human reviews (``InMemorySaver`` in tests, ``SqliteSaver`` for the CLI)."""
    context = context or DataContext()

    def draft(state: ReplayState) -> dict:
        if state.get("protocol"):
            p = _p(state)
            return {**_version(p, "given"), "status": "running"}
        from agentic_tte.agents import Trialist

        if llm is None:
            raise ValueError("drafting needs an LLM client (or pass a protocol)")
        t = Trialist(llm, ctgov)
        p = t.draft_from_study(state["study"]) if state.get("study") else t.draft(state["nct_id"])
        return {**_version(p, f"drafted by the Trialist ({llm.model})"), "status": "running"}

    def design_review(state: ReplayState) -> dict:
        p = _p(state)
        answer = interrupt({
            "stage": "design",
            "protocol": state["protocol"],
            "completeness": p.completeness(),
            "element_status": {k: str(v) for k, v in p.element_status().items()},
            "ask": "Approve with {\"reviewer\": \"...\"}, or also change elements: "
                   "{\"changes\": {field: value}, \"confounders\": [{\"name\", \"domain\"}]}",
        }) or {}  # fmt: skip
        changes = dict(answer.get("changes") or {})
        if answer.get("confounders") is not None:
            refs = [ConceptRef.model_validate(c).model_dump(mode="json")
                    for c in answer["confounders"]]
            changes["assignment"] = {**p.assignment.model_dump(mode="json"), "confounders": refs,
                                     "origin": Origin.USER.value}  # fmt: skip
        if not changes:
            return {"log": [f"design approved by {answer.get('reviewer') or 'reviewer'}"]}
        v = new_version(p, created_by=answer.get("reviewer") or "user",
                        notes="design review: " + ", ".join(sorted(changes)), **changes)
        return _version(v, f"design review changed {', '.join(sorted(changes))}")

    def propose_concepts(state: ReplayState) -> dict:
        book = propose(con, _p(state))
        return {"book": book.model_dump(mode="json"), "log": [f"concepts proposed: "
                                                              f"{book.summary()}"]}  # fmt: skip

    def concept_review(state: ReplayState) -> dict:
        book = ConceptSetBook.model_validate(state["book"])
        p = _p(state)
        answer = interrupt({
            "stage": "concepts",
            "review_sheet": review_sheet(book),
            "decisions_template": decisions_template(book),
            "spec_template": spec_template(p),
            "ask": "Answer {\"decisions\": <filled decisions file>, \"spec\": <compile spec>}",
        })  # fmt: skip
        reviewed = apply_decisions(book, answer["decisions"])
        spec = CompileSpec.model_validate({**answer["spec"], "concept_sets": {
            **reviewed.gate_sets(), **(answer["spec"].get("concept_sets") or {})}})
        v = apply_to_protocol(p, reviewed)
        return {**_version(v, f"concept sets reviewed: {reviewed.summary()}"),
                "book": reviewed.model_dump(mode="json"), "spec": spec.model_dump(mode="json")}

    def compile_node(state: ReplayState) -> dict:
        spec = CompileSpec.model_validate(state["spec"])
        tables = {r[0].lower() for r in con.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}  # fmt: skip
        spec = spec.model_copy(update={"descendants": "concept_ancestor" in tables})
        cc = compile_protocol(_p(state), spec)
        n = {s: sum(c.status == s for c in cc.criteria)
             for s in ("compiled", "not_compiled", "not_emulable")}  # fmt: skip
        return {"compiled": cc.model_dump(mode="json"), "log": [f"compiled: {n}"]}

    def gate(state: ReplayState) -> dict:
        cc = CompiledCohort.model_validate(state["compiled"])
        p = _p(state)
        rep = attrition(con, cc)
        report = run_gate(con, p, concept_sets=cc.spec.concept_sets, context=context,
                          attrition=rep, design_hr=design_hr)  # fmt: skip
        out = {"attrition": rep.public(), "attrition_md": rep.to_markdown(),
               "gate": report.model_dump(mode="json"), "gate_md": report.to_markdown(),
               "log": [f"gate: {report.overall}"]}  # fmt: skip
        if report.blocks_run:
            out["status"] = "blocked"
        return out

    def after_gate(state: ReplayState) -> str:
        overall = state["gate"]["checks"]
        statuses = {c["status"] for c in overall}
        if GateStatus.RED in statuses:
            return "report"
        return "gate_review" if GateStatus.AMBER in statuses else "analyze"

    def gate_review(state: ReplayState) -> dict:
        answer = interrupt({"stage": "gate", "gate": state["gate_md"],
                            "attrition": state["attrition_md"],
                            "ask": "Amber: proceed with these limitations? "
                                   "{\"proceed\": true|false, \"reviewer\": \"...\", "
                                   "\"note\": \"...\"}"}) or {}  # fmt: skip
        who = answer.get("reviewer") or "reviewer"
        if not answer.get("proceed"):
            return {"status": "stopped", "log": [f"stopped at the gate by {who}"]}
        return {"log": [f"gate limitations accepted by {who}: {answer.get('note') or ''}"]}

    def after_gate_review(state: ReplayState) -> str:
        return "report" if state.get("status") == "stopped" else "analyze"

    def analyze_node(state: ReplayState) -> dict:
        from agentic_tte.agents.statistician import analyze

        cc = CompiledCohort.model_validate(state["compiled"])
        if cc.frame_sql is None:
            return {"log": ["analysis skipped: no analysis frame (" + "; ".join(cc.frame_notes)
                            + ")"]}  # fmt: skip
        frame = con.execute(cc.frame_sql).df()  # stays here; only aggregates go to the state
        confounders = [c for c in cc.frame_columns[1:]
                       if c not in ("treated", "treat_start", "time", "event", "end_reason")]
        res = analyze(frame, _p(state), confounders=confounders, horizon_days=cc.horizon_days)
        p = res.primary
        msg = f"primary {res.primary_method}: HR {p.hr:.3f} ({p.lower:.3f}-{p.upper:.3f})" \
            if p else "no estimate"  # fmt: skip
        return {"analysis": res.model_dump(mode="json"), "analysis_md": res.to_markdown(),
                "log": [msg]}

    def report(state: ReplayState) -> dict:
        p = _p(state)
        status = state.get("status") or "running"
        if status == "running":
            status = "done"
        cc = CompiledCohort.model_validate(state["compiled"]) if state.get("compiled") else None
        parts = [f"# Replay: {p.source.nct_id or p.protocol_id}"
                 + (f" — {p.source.title}" if p.source.title else ""), "",
                 f"Status: **{status}**. Protocol v{p.version} (hash "
                 f"`{p.content_hash()[:12]}`); data classification "
                 f"{context.data_classification}.", "",
                 "## Run log", ""] + [f"- {m}" for m in state.get("log", [])]  # fmt: skip
        if state.get("gate_md"):
            parts += ["", state["gate_md"].replace("# Data Support Gate", "## Data Support Gate")]
        if state.get("attrition_md"):
            parts += ["", state["attrition_md"].replace("# Attrition", "## Attrition")]
        if cc is not None:
            parts += ["", cc.to_markdown().replace("# Compiled cohort", "## Compiled cohort")
                      .replace("\n## ", "\n### ")]  # fmt: skip
        if state.get("analysis_md"):
            parts += ["", state["analysis_md"]]
        parts += ["", "## Limitations", "",
                  "- Accuracy against the trial's published result is not scored here.",
                  "- Criteria that were not compiled or applied are listed above; the cohort "
                  "is wider than the protocol by them."]  # fmt: skip
        return {"report": "\n".join(parts) + "\n", "status": status}

    g = StateGraph(ReplayState)
    for name, fn in (("draft", draft), ("design_review", design_review),
                     ("propose_concepts", propose_concepts), ("concept_review", concept_review),
                     ("compile", compile_node), ("gate", gate), ("gate_review", gate_review),
                     ("analyze", analyze_node), ("report", report)):  # fmt: skip
        g.add_node(name, fn)
    g.add_edge(START, "draft")
    g.add_edge("draft", "design_review")
    g.add_edge("design_review", "propose_concepts")
    g.add_edge("propose_concepts", "concept_review")
    g.add_edge("concept_review", "compile")
    g.add_edge("compile", "gate")
    g.add_conditional_edges("gate", after_gate, ["report", "gate_review", "analyze"])
    g.add_conditional_edges("gate_review", after_gate_review, ["report", "analyze"])
    g.add_edge("analyze", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


def pending(graph, config: dict) -> Any | None:
    """The open review (the interrupt's payload), or None when the run is not waiting."""
    st = graph.get_state(config)
    return st.interrupts[0].value if st.interrupts else None
