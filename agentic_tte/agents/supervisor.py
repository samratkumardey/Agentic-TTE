"""The Supervisor: routing, loop caps and budgets for the Replay graph (PLAN.md §3, §17.2).

Every agent and review node returns to the Supervisor, which picks the next node from the run
state and records why (``Route``). The routing is a fixed, deterministic policy, not an LLM:
the same state always gives the same route, so a run can be replayed and audited, and no model
can skip a review or the Protocol Auditor. An LLM supervisor would change only the choice
between allowed routes; the reviews, the audit and the gate are never optional.

The loops and their caps (``Budget``):
- Protocol Auditor -> design review: ``max_audit_rounds`` failed audits without an override
  stop the run;
- design review: ``max_design_reviews`` reviews in all (first review plus audit returns);
- concept proposals (re-proposed when a design change adds concepts): ``max_concept_rounds``;
- LLM requests made by the agents (cache hits are free): ``max_llm_calls``, checked before
  each step that may call the model; ``None`` means no cap;
- the Statistician <-> Clinician loop (balance flags -> Clinician advice -> design review):
  ``max_balance_rounds`` rounds of advice; after that the analysis runs with the imbalance
  reported (not a stop: residual imbalance is a caveat, not a failure);
- ``max_steps`` routing decisions in all, as a last guard against a loop.

A cap that is reached stops the run with the reason in the log and the report, which is
written as for any stopped run. The Statistician <-> Clinician loop is result-blind: it acts on
the balance check, which runs before the outcome analysis and reads no outcome. An effect
estimate never routes the run back to the design.

The Supervisor reads only routing fields of the state (the last node, statuses, the gate's
check statuses, audit findings, counts); it never sees patient rows.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Nodes that may call the LLM, given how the graph was built.
LLM_NODES = ("draft", "clinician", "propose_concepts", "balance_advice")


class Budget(BaseModel):
    max_audit_rounds: int = Field(default=3, ge=1)
    max_design_reviews: int = Field(default=8, ge=1)
    max_concept_rounds: int = Field(default=4, ge=1)
    max_balance_rounds: int = Field(default=2, ge=0)
    max_llm_calls: int | None = Field(default=None, ge=0)
    max_steps: int = Field(default=60, ge=1)


class Route(BaseModel):
    step: int
    after: str  # the node that just ran ("start" before the first)
    to: str  # the next node, or "end"
    reason: str


def _statuses(state: dict) -> set[str]:
    return {str(c["status"]) for c in (state.get("gate") or {}).get("checks", [])}


class Supervisor:
    """The routing policy. ``llm_nodes``: the nodes that call the model in this graph (the
    draft only when there is no protocol, the Clinician, concept search with surrogates)."""

    def __init__(self, budget: Budget | None = None, *, clinician: bool = False,
                 balance_advice: bool = False, llm_nodes: set[str] | None = None):  # fmt: skip
        self.budget = budget or Budget()
        self.clinician = clinician
        self.balance_advice = balance_advice
        self.llm_nodes = set(llm_nodes or ())

    # -- the fixed order of the Replay ------------------------------------------------------
    def _next(self, state: dict, last: str) -> tuple[str, str]:
        status = state.get("status")
        if last == "start":
            return "draft", "start: the protocol is drafted or taken as given"
        if last == "draft":
            if self.clinician:
                return "clinician", "the Clinician proposes confounders before design review"
            return "design_review", "the draft goes to design review"
        if last == "clinician":
            return "design_review", "the draft and the Clinician's confounders go to review"
        if last == "design_review":
            if state.get("balance_kept"):
                return "analyze", "the reviewer kept the confounders: imbalance reported"
            if not (state.get("spec") and state.get("book")):
                return "propose_concepts", "the design's concepts are searched in the vocabulary"
            from agentic_tte.omop.concept_sets import _entries
            from agentic_tte.protocol import TargetTrialProtocol

            have = {s["key"] for s in state["book"]["sets"]}
            keys = {k for k, *_ in _entries(TargetTrialProtocol.model_validate(state["protocol"]))}
            if keys <= have:
                return "compile", "the reviewed concept sets still cover the design"
            return "propose_concepts", f"the design has {len(keys - have)} new concept(s)"
        if last == "propose_concepts":
            return "concept_review", "the concept proposals go to review"
        if last == "concept_review":
            return "compile", "the reviewed concept sets are compiled"
        if last == "compile":
            return "audit", "the Protocol Auditor checks the design before the data"
        if last == "audit":
            if status == "stopped":
                return "report", "stopped at the audit"
            failed = set((state.get("audit") or {}).get("failed") or [])
            overridden = set((state.get("audit_override") or {}).get("checks") or [])
            if failed <= overridden:
                return "gate", ("the audit passed" if not failed
                                else "the audit's findings were overridden")  # fmt: skip
            return "design_review", "the audit failed " + ", ".join(sorted(failed))
        if last == "gate":
            s = _statuses(state)
            if "red" in s:
                if status == "excluded_s8":
                    return "report", "S8 not met at the first gate: the trial leaves the set"
                return "report", "the Data Support Gate is red"
            if "amber" in s:
                return "gate_review", "the Data Support Gate is amber"
            return "balance", "the Data Support Gate is green"
        if last == "gate_review":
            if status == "stopped":
                return "report", "stopped at the gate review"
            return "balance", "the gate's limitations were accepted"
        if last == "balance":
            flagged = list((state.get("balance") or {}).get("flagged") or {})
            if not flagged:
                return "analyze", "no confounder flagged by the balance check"
            if not self.balance_advice:
                return "analyze", f"flagged {', '.join(flagged)}: reported (Clinician loop off)"
            done = (state.get("visits") or {}).get("balance_advice", 0)
            if done >= self.budget.max_balance_rounds:
                return "analyze", (f"flagged {', '.join(flagged)} after {done} round(s) of "
                                   "advice: reported")  # fmt: skip
            return "balance_advice", f"flagged {', '.join(flagged)}: the Clinician advises"
        if last == "balance_advice":
            if state.get("balance_pending"):
                return "design_review", "the Clinician advises changes: design review decides"
            return "analyze", "the Clinician advises keeping the confounders"
        if last == "analyze":
            return "report", "the estimates go into the report"
        if last == "report":
            if status == "done":
                return "signoff", "the report waits for sign-off"
            return "end", f"the run ended {status}"
        if last == "signoff":
            return "end", f"the report is {status}"
        raise ValueError(f"the Supervisor has no route after {last!r}")

    # -- caps and budgets --------------------------------------------------------------------
    def _cap(self, state: dict, to: str, visits: dict[str, int]) -> str | None:
        b = self.budget
        if to in ("report", "signoff", "end"):
            return None  # a stopped run still gets its report
        if (state.get("audit_rounds") or 0) >= b.max_audit_rounds and to == "design_review":
            return f"the design failed the audit {state['audit_rounds']} times"
        if to == "design_review" and visits.get(to, 0) >= b.max_design_reviews:
            return f"{b.max_design_reviews} design reviews without a passing design"
        if to == "propose_concepts" and visits.get(to, 0) >= b.max_concept_rounds:
            return f"{b.max_concept_rounds} rounds of concept proposals"
        used = state.get("llm_calls") or 0
        calls_llm = to in self.llm_nodes and not (to == "draft" and state.get("protocol"))
        if b.max_llm_calls is not None and calls_llm and used >= b.max_llm_calls:
            return f"the LLM budget is spent ({used} of {b.max_llm_calls} requests)"
        if (state.get("steps") or 0) >= b.max_steps:
            return f"{b.max_steps} routing steps"
        return None

    def route(self, state: dict) -> tuple[Route, dict]:
        """The next node and the state update (route record, visit counts, a stop)."""
        last = state.get("last") or "start"
        step = (state.get("steps") or 0) + 1
        visits = dict(state.get("visits") or {})
        to, reason = self._next(state, last)
        update: dict = {}
        stop = self._cap(state, to, visits)
        if stop:  # the report needs a protocol; with none yet, the run just ends
            to = "report" if state.get("protocol") else "end"
            reason = f"stopped by the Supervisor: {stop}"
            update |= {"status": "stopped", "stop_reason": stop,
                       "log": [f"stopped by the Supervisor: {stop}"]}  # fmt: skip
        visits[to] = visits.get(to, 0) + 1
        if last == "design_review" and state.get("balance_kept"):
            update["balance_kept"] = False
        r = Route(step=step, after=last, to=to, reason=reason)
        update |= {"routes": [r.model_dump()], "steps": step, "visits": visits}
        return r, update

    def to_markdown(self, state: dict) -> str:
        b = self.budget
        llm = state.get("llm_calls") or 0
        lines = ["## Supervisor", "",
                 f"Fixed routing policy. Budget: audit rounds {b.max_audit_rounds}, design "
                 f"reviews {b.max_design_reviews}, concept rounds {b.max_concept_rounds}, "
                 f"balance rounds {b.max_balance_rounds}, LLM "
                 f"requests {b.max_llm_calls if b.max_llm_calls is not None else 'no cap'}, "
                 f"steps {b.max_steps}. Used: {state.get('steps') or 0} steps, {llm} LLM "
                 f"requests ({state.get('llm_tokens') or 0} tokens)."]  # fmt: skip
        if state.get("stop_reason"):
            lines += ["", f"**Stopped by the Supervisor:** {state['stop_reason']}."]
        routes = state.get("routes") or []
        if routes:
            lines += ["", "| Step | After | Next | Why |", "|---|---|---|---|"]
            lines += [f"| {r['step']} | {r['after']} | {r['to']} | {r['reason']} |"
                      for r in routes]
        return "\n".join(lines) + "\n"
