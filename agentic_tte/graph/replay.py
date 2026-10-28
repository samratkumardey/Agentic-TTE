"""Replay as a LangGraph workflow (PLAN.md §3, §13; Phase 2 exit criterion).

    draft -> design_review* -> propose_concepts -> concept_review* -> compile -> audit
          -> [fail: back to design_review*] -> gate
          -> [red: report, stop] [amber: gate_review*] -> analyze -> report -> signoff*

``*`` marks a human review. Each one is a LangGraph ``interrupt()``: the run stops, its state is
saved by the checkpointer, and it resumes with the reviewer's answer
(``Command(resume=...)``):
- design_review: the drafted protocol. The answer can change protocol elements and name the
  confounders. Any change makes a new protocol version (created by the user). When the
  Protocol Auditor has failed the design, the review shows its findings; the answer either
  changes the design (it is compiled and audited again) or overrides the failed checks with
  ``{"override_audit": {"reason": "..."}}``, which is recorded and shown in the report. If the
  new design has no new concepts, it goes straight back to compile; otherwise its concepts
  are proposed and reviewed again.
- concept_review: the concept-set review sheet, a decisions file to fill in, and the compile
  spec (time zero and anchors as concept sets). Answer ``{"decisions": ..., "spec": ...}``.
- gate_review: only when the gate is amber. Answer ``{"proceed": bool, "reviewer": ...}``.
- signoff: the report of an analysed run. Answer ``{"approve": bool, "reviewer": ...,
  "note": ...}``: status ``signed_off``, or ``returned`` with the note.

Every review is recorded (``reviews``: stage, reviewer, automatic or not, time, decision) and
listed in the report. ``AutoApprove`` answers chosen reviews by a fixed policy, for batch
benchmarking: such a run is labelled as auto-approved at the top of its report, its sign-off
gives status ``auto_approved`` (never ``signed_off``), and it is not for clinical use.

The Protocol Auditor (``agentic_tte.agents.auditor``) runs after compile and before the gate:
the design is checked before any data-quality question. ``MAX_AUDIT_ROUNDS`` failed rounds
without an override stop the run. Under ``AutoApprove`` a failed audit is overridden
automatically (``audit="override"``, labelled in the report) or stops the run
(``audit="stop"``).

The LLM is used only by the Trialist, on registry text. The database connection is closed over
when the graph is built, never put in the state. The state holds the protocol, concept sets,
SQL and aggregates (suppressed counts, estimates): no patient rows. The supervisor is the
fixed routing below for now; an LLM supervisor can replace it later.
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agentic_tte.agents.auditor import audit, data_summary
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
    audit: dict  # the latest Protocol Auditor report
    audit_md: str
    audit_rounds: int  # failed audits so far
    audit_override: dict  # {"checks": [...], "reason": ..., "reviewer": ..., "auto": bool}
    report: str
    status: str  # running, blocked, stopped, done, signed_off, returned, auto_approved
    log: Annotated[list[str], operator.add]
    reviews: Annotated[list[dict], operator.add]  # audit trail of every review


Stage = Literal["design", "concepts", "gate", "signoff"]
AUTO_REVIEWER = "auto-approve (benchmark; no human review)"
MAX_AUDIT_ROUNDS = 3


class AutoApprove(BaseModel):
    """A fixed answer policy for batch benchmarking. It replaces human judgement, so every
    use is recorded and shown, and an auto-approved report is never "signed off".

    - design: approve, with ``design`` (e.g. confounders, changes) applied as given;
    - concepts: approve every vocabulary proposal, mark the rest unmappable, apply
      ``overrides`` (decisions by key); ``spec`` (time zero, anchors) must be given;
    - gate: proceed on amber if ``proceed_on_amber``;
    - signoff: approve if ``sign_off``;
    - a failed Protocol Auditor: ``audit="override"`` accepts the findings (labelled in the
      report), ``"stop"`` ends the run."""

    stages: set[Stage] = Field(default_factory=lambda: {"design", "concepts", "gate",
                                                        "signoff"})  # fmt: skip
    reviewer: str = AUTO_REVIEWER
    design: dict = Field(default_factory=dict)
    spec: dict | None = None
    overrides: dict[str, dict] = Field(default_factory=dict)
    proceed_on_amber: bool = True
    sign_off: bool = True
    audit: Literal["override", "stop"] = "override"

    def answer(self, stage: Stage, payload: dict) -> dict:
        if stage == "design":
            return {**self.design, "reviewer": self.reviewer}
        if stage == "concepts":
            if self.spec is None:
                raise ValueError("auto-approving concepts needs a compile spec (time zero)")
            decisions = {}
            for key, d in payload["decisions_template"]["decisions"].items():
                if key in self.overrides:
                    decisions[key] = self.overrides[key]
                elif d["action"] == "approve":
                    decisions[key] = {"action": "approve", "concept_ids": d["concept_ids"],
                                      "note": "auto: vocabulary proposal accepted"}
                else:
                    decisions[key] = {"action": "unmappable",
                                      "note": "auto: no vocabulary proposal"}
            return {"decisions": {"reviewer": self.reviewer, "decisions": decisions},
                    "spec": self.spec}
        if stage == "gate":
            return {"proceed": self.proceed_on_amber, "reviewer": self.reviewer,
                    "note": "auto: amber accepted" if self.proceed_on_amber else "auto: stop"}
        return {"approve": self.sign_off, "reviewer": self.reviewer,
                "note": "auto-approved for benchmarking; not reviewed by a person"}


def _p(state: ReplayState) -> TargetTrialProtocol:
    return TargetTrialProtocol.model_validate(state["protocol"])


def _version(p: TargetTrialProtocol, why: str) -> dict:
    d = p.model_dump(mode="json")
    return {"protocol": d, "versions": [d], "log": [f"protocol v{p.version}: {why}"]}


def _record(stage: str, answer: dict, auto: bool, decision: str) -> dict:
    who = answer.get("reviewer") or (answer.get("decisions") or {}).get("reviewer")
    return {"stage": stage, "reviewer": who or "reviewer", "auto": auto, "decision": decision,
            "note": answer.get("note") or "",
            "at": datetime.now(UTC).isoformat(timespec="seconds")}


def build_replay(con, *, llm=None, ctgov=None, context: DataContext | None = None,
                 design_hr: float | None = None, checkpointer=None,
                 auto: AutoApprove | None = None, abbreviations=None,
                 n_bootstrap: int = 0):  # fmt: skip
    """Compile the Replay graph over an open OMOP connection. A checkpointer is needed for
    the human reviews (``InMemorySaver`` in tests, ``SqliteSaver`` for the CLI); with
    ``auto`` covering every stage, the run needs none. ``abbreviations``: an
    ``AbbreviationList`` whose approved entries the concept search may expand.
    ``n_bootstrap``: resamples for the risk-measure CIs (eval/SCORING.md R4: 500 for scored
    runs; 0 skips them)."""
    context = context or DataContext()

    def review(stage: Stage, payload: dict) -> tuple[dict, bool]:
        if auto is not None and stage in auto.stages:
            return auto.answer(stage, payload), True
        return interrupt(payload) or {}, False

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
        failed = (state.get("audit") or {}).get("failed") or []
        payload = {
            "stage": "design",
            "protocol": state["protocol"],
            "completeness": p.completeness(),
            "element_status": {k: str(v) for k, v in p.element_status().items()},
            "ask": "Approve with {\"reviewer\": \"...\"}, or also change elements: "
                   "{\"changes\": {field: value}, \"confounders\": [{\"name\", \"domain\"}]}",
        }  # fmt: skip
        if failed:
            payload["audit"] = state["audit_md"]
            payload["ask"] = ("The Protocol Auditor failed checks " + ", ".join(failed) + ". "
                              "Change the design ({\"changes\": ...}), or override with "
                              "{\"override_audit\": {\"reason\": \"...\"}, \"reviewer\": \"...\"}")
        answer, is_auto = review("design", payload)
        if failed and answer.get("override_audit"):
            reason = (answer["override_audit"] or {}).get("reason") or ""
            if not reason.strip():
                raise ValueError("overriding the Protocol Auditor needs a reason")
            who = answer.get("reviewer") or "reviewer"
            return {"audit_override": {"checks": failed, "reason": reason, "reviewer": who,
                                       "auto": is_auto},
                    "reviews": [_record("design", {**answer, "note": reason}, is_auto,
                                        "audit overridden: " + ", ".join(failed))],
                    "log": [f"audit checks {', '.join(failed)} overridden by {who}"]}
        changes = dict(answer.get("changes") or {})
        if answer.get("confounders") is not None:
            refs = [ConceptRef.model_validate(c).model_dump(mode="json")
                    for c in answer["confounders"]]
            changes["assignment"] = {**p.assignment.model_dump(mode="json"), "confounders": refs,
                                     "origin": Origin.USER.value}  # fmt: skip
        what = ("changed " + ", ".join(sorted(changes))) if changes else "approved as drafted"
        rec = {"reviews": [_record("design", answer, is_auto, what)]}
        if not changes:
            return {**rec, "log": [f"design approved by {answer.get('reviewer') or 'reviewer'}"]}
        v = new_version(p, created_by=answer.get("reviewer") or "user",
                        notes="design review: " + ", ".join(sorted(changes)), **changes)
        return {**_version(v, f"design review {what}"), **rec}

    def after_design_review(state: ReplayState) -> str:
        """Back to compile when the reviewed concept sets still cover the design; otherwise
        its concepts are proposed and reviewed again."""
        if not (state.get("spec") and state.get("book")):
            return "propose_concepts"
        from agentic_tte.omop.concept_sets import _entries

        have = {s["key"] for s in state["book"]["sets"]}
        keys = {k for k, *_ in _entries(_p(state))}
        return "compile" if keys <= have else "propose_concepts"

    def propose_concepts(state: ReplayState) -> dict:
        book = propose(con, _p(state), abbreviations=abbreviations)
        return {"book": book.model_dump(mode="json"), "log": [f"concepts proposed: "
                                                              f"{book.summary()}"]}  # fmt: skip

    def concept_review(state: ReplayState) -> dict:
        book = ConceptSetBook.model_validate(state["book"])
        p = _p(state)
        answer, is_auto = review("concepts", {
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
                "book": reviewed.model_dump(mode="json"), "spec": spec.model_dump(mode="json"),
                "reviews": [_record("concepts", answer, is_auto, str(reviewed.summary()))]}

    def compile_node(state: ReplayState) -> dict:
        spec = CompileSpec.model_validate(state["spec"])
        tables = {r[0].lower() for r in con.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}  # fmt: skip
        spec = spec.model_copy(update={"descendants": "concept_ancestor" in tables})
        cc = compile_protocol(_p(state), spec)
        n = {s: sum(c.status == s for c in cc.criteria)
             for s in ("compiled", "not_compiled", "not_emulable")}  # fmt: skip
        return {"compiled": cc.model_dump(mode="json"), "log": [f"compiled: {n}"]}

    def audit_node(state: ReplayState) -> dict:
        cc = CompiledCohort.model_validate(state["compiled"])
        p = _p(state)
        summary = data_summary(con, p, cc.spec, cc)
        ids = sorted({i for k, v in cc.spec.concept_sets.items() if k.startswith("T:") for i in v})
        try:  # A-j: the treatment concepts' names, from the vocabulary
            names = dict(con.execute(f"SELECT concept_id, concept_name FROM concept WHERE "
                                     f"concept_id IN ({', '.join(map(str, ids))})").fetchall())
        except Exception:  # noqa: BLE001 - no vocabulary: A-j reports that it could not check
            names = None
        surrogates = {x["key"] for x in (state.get("book") or {}).get("sets", [])
                      if x.get("status") == "surrogate"}
        rep = audit(p, cc.spec, compiled=cc, summary=summary,
                    concept_names=names if ids and names else None, surrogate_keys=surrogates)
        out = {"audit": {**rep.model_dump(mode="json"), "failed": rep.failed},
               "audit_md": rep.to_markdown(), "log": [
                   "audit: passed" if rep.passed else f"audit: failed {', '.join(rep.failed)}"]}
        if rep.passed:
            return out
        ov = state.get("audit_override") or {}
        if set(rep.failed) <= set(ov.get("checks") or []):
            out["log"] = [f"audit: failed {', '.join(rep.failed)}, overridden by "
                          f"{ov.get('reviewer')}: {ov.get('reason')}"]
            return out
        if auto is not None and "design" in auto.stages:
            if auto.audit == "stop":
                return {**out, "status": "stopped"}
            reason = "auto: audit findings accepted for benchmarking; not reviewed by a person"
            rec = {"reviewer": auto.reviewer, "note": reason}
            return {**out, "audit_override": {"checks": rep.failed, "reason": reason,
                                              "reviewer": auto.reviewer, "auto": True},
                    "reviews": [_record("audit", rec, True,
                                        "audit overridden: " + ", ".join(rep.failed))]}
        rounds = (state.get("audit_rounds") or 0) + 1
        out["audit_rounds"] = rounds
        if rounds >= MAX_AUDIT_ROUNDS:
            out["status"] = "stopped"
            out["log"] = out["log"] + [f"stopped: the design failed the audit {rounds} times"]
        return out

    def after_audit(state: ReplayState) -> str:
        if state.get("status") == "stopped":
            return "report"
        failed = set((state.get("audit") or {}).get("failed") or [])
        overridden = set((state.get("audit_override") or {}).get("checks") or [])
        return "gate" if failed <= overridden else "design_review"

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
        answer, is_auto = review("gate", {"stage": "gate", "gate": state["gate_md"],
                            "attrition": state["attrition_md"],
                            "ask": "Amber: proceed with these limitations? "
                                   "{\"proceed\": true|false, \"reviewer\": \"...\", "
                                   "\"note\": \"...\"}"})  # fmt: skip
        who = answer.get("reviewer") or "reviewer"
        if not answer.get("proceed"):
            return {"status": "stopped", "log": [f"stopped at the gate by {who}"],
                    "reviews": [_record("gate", answer, is_auto, "stop")]}
        return {"log": [f"gate limitations accepted by {who}: {answer.get('note') or ''}"],
                "reviews": [_record("gate", answer, is_auto, "proceed")]}

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
        res = analyze(frame, _p(state), confounders=confounders, horizon_days=cc.horizon_days,
                      n_bootstrap=n_bootstrap)  # fmt: skip
        p = res.primary
        msg = f"primary {res.primary_method}: HR {p.hr:.3f} ({p.lower:.3f}-{p.upper:.3f})" \
            if p else "no estimate"  # fmt: skip
        return {"analysis": res.model_dump(mode="json"), "analysis_md": res.to_markdown(),
                "log": [msg]}

    def render(state: ReplayState, status: str, reviews: list[dict]) -> str:
        p = _p(state)
        cc = CompiledCohort.model_validate(state["compiled"]) if state.get("compiled") else None
        parts = [f"# Replay: {p.source.nct_id or p.protocol_id}"
                 + (f" — {p.source.title}" if p.source.title else ""), ""]
        autos = [r["stage"] for r in reviews if r["auto"]]
        if autos:
            parts += [f"> **AUTO-APPROVED BENCHMARK RUN.** No person reviewed: "
                      f"{', '.join(autos)}. Not for clinical use.", ""]
        parts += [f"Status: **{status}**. Protocol v{p.version} (hash "
                  f"`{p.content_hash()[:12]}`); data classification "
                  f"{context.data_classification}.", "",
                  "## Reviews", "", "| Stage | Reviewer | Automatic | Decision | Note | Time |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        parts += [f"| {r['stage']} | {r['reviewer']} | {'yes' if r['auto'] else 'no'} | "
                  f"{r['decision']} | {r['note'] or '–'} | {r['at']} |" for r in reviews]
        parts += ["", "## Run log", ""] + [f"- {m}" for m in state.get("log", [])]
        if state.get("audit_md"):
            parts += ["", state["audit_md"]]
            ov = state.get("audit_override")
            if ov:
                parts += [f"**Audit overridden** ({', '.join(ov['checks'])}) by {ov['reviewer']}"
                          + (" (automatic)" if ov.get("auto") else "") + f": {ov['reason']}"]
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
        return "\n".join(parts) + "\n"

    def report(state: ReplayState) -> dict:
        status = state.get("status") or "running"
        if status == "running":
            status = "done"
        return {"report": render(state, status, state.get("reviews", [])), "status": status}

    def after_report(state: ReplayState) -> str:
        return "signoff" if state.get("status") == "done" else END

    def signoff(state: ReplayState) -> dict:
        answer, is_auto = review("signoff", {
            "stage": "signoff", "report": state["report"],
            "ask": "Sign off the report? {\"approve\": true|false, \"reviewer\": \"...\", "
                   "\"note\": \"...\"}"})  # fmt: skip
        ok = bool(answer.get("approve"))
        status = ("auto_approved" if is_auto else "signed_off") if ok else "returned"
        rec = _record("signoff", answer, is_auto, "approve" if ok else "return")
        reviews = [*state.get("reviews", []), rec]
        return {"report": render(state, status, reviews), "status": status, "reviews": [rec],
                "log": [f"report {status} by {rec['reviewer']}"]}

    g = StateGraph(ReplayState)
    for name, fn in (("draft", draft), ("design_review", design_review),
                     ("propose_concepts", propose_concepts), ("concept_review", concept_review),
                     ("compile", compile_node), ("audit", audit_node), ("gate", gate),
                     ("gate_review", gate_review),
                     ("analyze", analyze_node), ("report", report),
                     ("signoff", signoff)):  # fmt: skip
        g.add_node(name, fn)
    g.add_edge(START, "draft")
    g.add_edge("draft", "design_review")
    g.add_conditional_edges("design_review", after_design_review, ["propose_concepts", "compile"])
    g.add_edge("propose_concepts", "concept_review")
    g.add_edge("concept_review", "compile")
    g.add_edge("compile", "audit")
    g.add_conditional_edges("audit", after_audit, ["gate", "design_review", "report"])
    g.add_conditional_edges("gate", after_gate, ["report", "gate_review", "analyze"])
    g.add_conditional_edges("gate_review", after_gate_review, ["report", "analyze"])
    g.add_edge("analyze", "report")
    g.add_conditional_edges("report", after_report, ["signoff", END])
    g.add_edge("signoff", END)
    return g.compile(checkpointer=checkpointer)


def pending(graph, config: dict) -> Any | None:
    """The open review (the interrupt's payload), or None when the run is not waiting."""
    st = graph.get_state(config)
    return st.interrupts[0].value if st.interrupts else None
