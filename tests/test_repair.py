"""The Repair agent (agentic_tte/agents/repair.py, SCORING.md Phase 3 item 33): fixed rules that
make a drafted design emulable, on hand-built protocols and on four of the core-trial drafts
(qwen3:32b, the Phase 0 core run; public registry text, no patient data)."""

import json
from pathlib import Path

from agentic_tte.agents.repair import RULES_VERSION, repair_design
from agentic_tte.agents.supervisor import Supervisor
from agentic_tte.omop.compiler import design_problem
from agentic_tte.protocol import (
    ConceptRef,
    Criterion,
    CriterionAtom,
    Duration,
    TargetTrialProtocol,
    TemporalConstraint,
    TreatmentComponent,
    TreatmentStrategy,
    ValueConstraint,
)

FIX = Path(__file__).parent / "fixtures" / "protocols"


def _draft(nct: str) -> TargetTrialProtocol:
    return TargetTrialProtocol.model_validate_json(
        (FIX / f"{nct}.core-H100.qwen3-32b.json").read_text(encoding="utf-8"))


DAY = Duration(amount=1, unit="day")


def arm(label, *names, comparator=False, arm_type=None, description=None, grace=None):
    return TreatmentStrategy(label=label, arm_type=arm_type, description=description,
                             is_comparator=comparator, grace_period=grace,
                             components=[TreatmentComponent(concept=ConceptRef(
                                 name=n, domain="Drug")) for n in names])  # fmt: skip


def proto(arms, criteria=()):
    return TargetTrialProtocol(protocol_id="T", treatment_strategies=arms,
                               eligibility=list(criteria))  # fmt: skip


def shape(changes, p):
    arms = changes.get("treatment_strategies", p.model_dump(mode="json")["treatment_strategies"])
    return [(a["label"], "C" if a["is_comparator"] else "T",
             [c["concept"]["name"] for c in a["components"]]) for a in arms]  # fmt: skip


def test_a_valid_placebo_design_needs_no_repair():
    p = proto([arm("Drug", "hydrocortisone", grace=DAY),
               arm("Placebo", "placebo", comparator=True)])  # fmt: skip
    changes, rep = repair_design(p, "hydrocortisone vs placebo")
    assert changes == {} and rep.repairs == [] and rep.flagged == []
    assert rep.rules_version == RULES_VERSION and "No repair needed" in rep.to_markdown()


def test_null_arm_vehicle_removed_and_the_other_arm_is_the_treatment():
    p = proto([arm("Vitamin C", "ascorbic acid", comparator=True),
               arm("Control", "dextrose 5% in water", arm_type="PLACEBO_COMPARATOR",
                   comparator=True)])  # fmt: skip
    changes, rep = repair_design(p, "vitamin C vs placebo")
    assert shape(changes, p) == [("Vitamin C", "T", ["ascorbic acid"]), ("Control", "C", [])]
    assert {r.rule for r in rep.repairs} == {"R-arm-null", "R-arm-one", "R-grace"}
    assert "registry marks it PLACEBO_COMPARATOR" in rep.repairs[0].what
    TargetTrialProtocol.model_validate({**p.model_dump(mode="json"), **changes})


def test_an_active_comparator_is_flagged_not_changed():
    p = proto([arm("Etomidate", "etomidate", grace=DAY),
               arm("Ketamine", "ketamine", comparator=True)])  # fmt: skip
    changes, rep = repair_design(p, "etomidate vs ketamine")
    assert changes == {}
    assert rep.flagged[0].startswith("the comparator 'Ketamine' is an active treatment: the "
                                     "active-comparator frame is used")  # fmt: skip


def test_repair_v3_amendments_subgroups_and_names():
    def c(cid, text, kind="exclusion", name="x"):
        return Criterion(id=cid, kind=kind, text=text, atoms=[CriterionAtom(concepts=[
            ConceptRef(name=name, domain="Condition")])])  # fmt: skip

    p = proto([arm("Accelerated", "Accelerated RRT initiation"),
               arm("Standard", "Standard RRT initiation", comparator=True)], [
        c("I1", "acute kidney injury", "inclusion"),
        c("I2", "On the sub-group of patients included with COVID19 :", "inclusion"),
        c("I3", "Diagnosis of COVID19 by PCR", "inclusion", "covid19"),
        c("I4", "study drug within 24 hours", "inclusion"),
        c("E1", "Amendments to eligibility criteria were:"),
        c("E2", "On 27/03/2008: Changes in following exclusion criteria :"),
        c("E3", '"chronic liver disease" was clarified as "chronic liver disease with Child C".'),
        c("E4", 'and a new exclusion criteria was added: "a previous episode of sepsis"'),
    ])  # fmt: skip
    p = p.model_copy(update={"source": p.source.model_copy(update={
        "title": "Standard Versus Accelerated Initiation of Renal Replacement Therapy"})})
    changes, rep = repair_design(p, "accelerated vs standard RRT initiation")
    assert [x["id"] for x in changes["eligibility"]] == ["I1", "I4", "E4"]
    rules = [(r.target, r.rule) for r in rep.repairs]
    assert ("I2", "R-subgroup") in rules and ("I3", "R-subgroup") in rules
    assert {("E1", "R-amend"), ("E2", "R-amend"), ("E3", "R-amend")} <= set(rules)
    assert any(f.startswith("E3 says a criterion was changed") for f in rep.flagged)
    names = shape(changes, p)
    assert names[0] == ("Accelerated", "T", ["renal replacement therapy"])
    assert names[1] == ("Standard", "C", [])
    assert rep.rules_version == "repair-v3"
    # Without the title, the abbreviation is kept.
    changes, _ = repair_design(p.model_copy(update={"source": p.source.model_copy(
        update={"title": None})}), "accelerated vs standard RRT initiation")  # fmt: skip
    assert shape(changes, p)[0][2] == ["RRT"]


def test_standard_fluid_and_timing_comparators_are_no_initiation():
    p = proto([arm("Colloids", "colloids"), arm("Crystalloids", "crystalloids", comparator=True)])
    changes, rep = repair_design(p, "colloids vs crystalloids")
    assert shape(changes, p) == [("Colloids", "T", ["colloids"]), ("Crystalloids", "C", [])]
    assert [r.rule for r in rep.repairs] == ["R-arm-standard", "R-grace"] and not rep.flagged
    t = changes["treatment_strategies"][0]
    assert t["grace_period"] == {"amount": 24, "unit": "hour"}
    assert t["grace_period_origin"] == "default"
    p = proto([arm("Accelerated", "Accelerated RRT initiation"),
               arm("Standard", "Standard RRT initiation", comparator=True)])  # fmt: skip
    changes, rep = repair_design(p, "accelerated vs standard RRT initiation")
    assert shape(changes, p)[1] == ("Standard", "C", [])
    assert rep.repairs[0].rule == "R-arm-timing" and not rep.flagged
    # Saline with a drug is not a standard-fluid comparator.
    p = proto([arm("A", "drug a"), arm("B", "drug b", "sodium chloride", comparator=True)])
    _, rep = repair_design(p, "drug a vs drug b")
    assert "R-arm-standard" not in {r.rule for r in rep.repairs}


def test_an_anchor_named_after_a_relation_is_time_zero():
    c = Criterion(id="E1", kind="exclusion", text="E1", atoms=[CriterionAtom(
        concepts=[ConceptRef(name="burn", domain="Condition")],
        temporal=TemporalConstraint(relation="x_at_time_of_y", anchor="x_at_time_of_y"))])
    p = proto([arm("Drug", "d", grace=DAY), arm("Placebo", "placebo", comparator=True)], [c])
    changes, rep = repair_design(p)
    assert changes["eligibility"][0]["atoms"][0]["temporal"]["anchor"] == "time zero"
    assert [r.rule for r in rep.repairs] == ["R-anchor"]


def test_without_a_contrast_ambiguous_arms_are_left_and_flagged():
    p = proto([arm("A", "drug a", comparator=True), arm("B", "drug b", comparator=True)])
    changes, rep = repair_design(p, None)
    assert changes == {} and "no contrast is given" in rep.flagged[0]
    changes, rep = repair_design(p, "drug a and drug b")
    assert changes == {} and "is not 'A vs B'" in rep.flagged[0]


def test_amendment_text_is_dropped():
    def c(cid, text, kind="exclusion"):
        return Criterion(id=cid, kind=kind, text=text, atoms=[CriterionAtom(concepts=[
            ConceptRef(name="x", domain="Condition")])])  # fmt: skip

    p = proto([arm("Drug", "d"), arm("Placebo", "placebo", comparator=True)], [
        c("E1", "pregnancy"),
        c("E2", "On 18/04/2012: the following exclusion criteria were removed:"),
        c("E3", "any surgery in the past 12 hours"),
        c("E4", "platelets < 30000/mm3"),
        c("E5", "On 01/02/2013: a new exclusion criterion was added"),
        c("E6", "an inclusion criterion was removed"),
        c("E7", "allergy to the study drug"),
    ])
    changes, rep = repair_design(p)
    assert [x["id"] for x in changes["eligibility"]] == ["E1", "E5", "E7"]
    assert [(r.target, r.rule) for r in rep.repairs if r.rule != "R-grace"] == [
        ("E2", "R-amend"), ("E3", "R-amend"), ("E4", "R-amend"), ("E6", "R-amend")]
    assert len(rep.flagged) == 1 and rep.flagged[0].startswith("E6 says a criterion was removed")


def test_inclusion_values_the_compiler_cannot_read_are_repaired():
    def inc(cid, concept, domain, value=None, temporal=None):
        return Criterion(id=cid, kind="inclusion", text=cid, atoms=[
            CriterionAtom(concepts=[ConceptRef(name=concept, domain=domain)], value=value,
                          temporal=temporal)])  # fmt: skip

    dur = ValueConstraint(comparator="<", value=24, unit="hours", text="for less than 24 hours")
    proven = ValueConstraint(comparator="present", text="proven")
    cumulative = TemporalConstraint(relation="cumulative_duration", anchor="time_zero",
                                    window=Duration(amount=24, unit="hour"))
    p = proto([arm("Drug", "d"), arm("Placebo", "placebo", comparator=True)], [
        inc("I1", "septic shock", "Condition", value=dur),
        inc("I2", "infection", "Condition", value=proven),
        inc("I3", "lactate", "Measurement", value=ValueConstraint(comparator=">", value=2)),
        inc("I4", "vasopressor", "Drug", temporal=cumulative),
        Criterion(id="E1", kind="exclusion", text="E1", atoms=[CriterionAtom(
            concepts=[ConceptRef(name="burn", domain="Condition")], value=dur)]),
    ])
    changes, rep = repair_design(p)
    by = {c["id"]: c for c in changes["eligibility"]}
    assert by["I1"]["atoms"][0]["value"] is None and by["I2"]["atoms"][0]["value"] is None
    assert by["I3"]["atoms"][0]["value"]["value"] == 2  # a measurement level compiles as it is
    assert by["I4"]["atoms"][0]["temporal"]["relation"] == "x_within_t_before_y"
    assert by["E1"]["atoms"][0]["value"] is not None  # exclusions are left as drafted
    assert [r.target for r in rep.repairs if r.rule == "R-value"] == ["I1", "I2", "I4"]
    repaired = TargetTrialProtocol.model_validate({**p.model_dump(mode="json"), **changes})
    for c in repaired.eligibility:
        assert c.kind == "exclusion" or design_problem(c, c.atoms[0]) is None, c.id


def test_aprocchss_factorial_draft():
    p = _draft("NCT00625209")
    changes, rep = repair_design(p, "hydrocortisone + fludrocortisone vs placebo")
    assert shape(changes, p) == [
        ("2", "T", ["hydrocortisone", "fludrocortisone"]),
        ("1", "C", ["placebo of hydrocortisone", "placebo of fludrocortisone",
                    "placebo of activated protein C"])]  # fmt: skip
    dropped = {r.target for r in rep.repairs if r.rule == "R-amend"}
    # repair-v3 also drops the amendment headings (E16, E17) and the changed or clarified
    # criteria (E18-E21); E23, a criterion an amendment added, is kept.
    assert dropped == {f"E{i}" for i in range(16, 31)} - {"E23"}
    assert {r.target for r in rep.repairs if r.rule == "R-value"} >= {"I2", "I3"}
    assert any(f.startswith("E22 says a criterion was removed") for f in rep.flagged)
    assert any(f.startswith("I4:") for f in rep.flagged)  # multi-domain value: for a person


def test_core_drafts_get_one_treatment_and_one_comparator():
    cases = {"NCT02643381": ("etomidate vs ketamine",
                             [("Etomidate", "T", ["Etomidate"]), ("Ketamine", "C", ["Ketamine"])]),
             "NCT03637205": ("ECLS vs usual care",
                             [("ECLS", "T", ["extracorporeal life support"]),
                              ("No ECLS", "C", [])]),
             "NCT03680274": ("vitamin C vs placebo",
                             [("Vitamin C", "T", ["Ascorbic acid"]), ("Control", "C", [])])}
    for nct, (contrast, want) in cases.items():
        p = _draft(nct)
        changes, rep = repair_design(p, contrast)
        assert shape(changes, p) == want, nct
        active = [f for f in rep.flagged if "active treatment" in f]
        assert bool(active) == (nct == "NCT02643381"), nct
    _, rep = repair_design(_draft("NCT03637205"), "ECLS vs usual care")
    bg = next(r for r in rep.repairs if r.target == "ECLS")
    assert bg.rule == "R-arm-shared" and "PCI" in bg.what and "No ECLS" in bg.what


def test_the_supervisor_routes_through_repair_before_review():
    sup = Supervisor(repair=True)
    assert sup.route({"last": "draft"})[0].to == "repair"
    assert sup.route({"last": "repair"})[0].to == "design_review"
    assert Supervisor(repair=True, clinician=True).route({"last": "repair"})[0].to == "clinician"
    assert Supervisor().route({"last": "draft"})[0].to == "design_review"


def test_the_replay_records_the_repair_as_a_version():
    from langgraph.checkpoint.memory import InMemorySaver

    from agentic_tte.graph import build_replay, pending
    from agentic_tte.omop import DataContext

    p = _draft("NCT03680274")
    graph = build_replay(None, context=DataContext(data_classification="synthetic"),
                         checkpointer=InMemorySaver(), repair=True,
                         contrast="vitamin C vs placebo")  # fmt: skip
    cfg = {"configurable": {"thread_id": "t"}}
    graph.invoke({"protocol": p.model_dump(mode="json")}, cfg)
    ask = pending(graph, cfg)  # waits at design review, with the repair report
    assert ask["stage"] == "design" and "## Repair agent" in ask["repair"]
    state = graph.get_state(cfg).values
    assert [r["to"] for r in state["routes"]] == ["draft", "repair", "design_review"]
    assert state["versions"][-1]["provenance"]["created_by"] == f"Repair agent ({RULES_VERSION})"
    assert state["repair"]["contrast"] == "vitamin C vs placebo"
    assert [a["label"] for a in ask["protocol"]["treatment_strategies"]] == ["Vitamin C",
                                                                            "Control"]
    json.dumps(state, default=str)
