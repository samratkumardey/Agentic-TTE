import importlib.util
import sys
from pathlib import Path

from agentic_tte.protocol import (
    ConceptRef,
    Criterion,
    CriterionAtom,
    Duration,
    TargetTrialProtocol,
    TemporalConstraint,
    TimeZero,
    ValueConstraint,
)

EVAL = Path(__file__).resolve().parents[1] / "eval"
spec = importlib.util.spec_from_file_location("phase0_scoring", EVAL / "phase0_scoring.py")
sc = importlib.util.module_from_spec(spec)
sys.modules["phase0_scoring"] = sc
spec.loader.exec_module(sc)


def proto(criteria, anchor="ICU admission", rationale="r1"):
    return TargetTrialProtocol(protocol_id="X", eligibility=criteria,
                               time_zero=TimeZero(anchor_event=anchor, rationale=rationale))


def crit(cid, atoms, logic="all", text="t"):
    return Criterion(id=cid, kind="exclusion", text=text, atoms=atoms, atom_logic=logic)


def atom(*names, value=None, temporal=None):
    return CriterionAtom(concepts=[ConceptRef(name=n, domain="Condition") for n in names],
                         value=value, temporal=temporal)  # fmt: skip


LACTATE = ValueConstraint(comparator=">", value=2.0, unit="mmol/L")
WITHIN = TemporalConstraint(relation="x_within_t_before_y", window=Duration(amount=3, unit="month"),
                            anchor="randomization")  # fmt: skip


def test_relaxed_and_strict_matching():
    gold = proto([crit("E1", [atom("Septic shock", value=LACTATE, temporal=WITHIN)])])
    same_but_spelled = proto([crit("E1", [atom("septic-shock", value=ValueConstraint(
        comparator=">", value=2.0, unit="mmol/l"), temporal=TemporalConstraint(
        relation="x_within_t_before_y", window=Duration(amount=3, unit="month"),
        anchor="time_zero"))])])  # fmt: skip
    assert sc.concept_prf(same_but_spelled, gold, strict=True).f1 == 1.0
    no_value = proto([crit("E1", [atom("septic shock", temporal=WITHIN)])])
    assert sc.concept_prf(no_value, gold, strict=False).f1 == 1.0
    assert sc.concept_prf(no_value, gold, strict=True).f1 == 0.0


def test_precision_recall_with_extra_and_missing_concepts():
    gold = proto([crit("E1", [atom("a", "b", "c")]), crit("E2", [atom("d")])])
    pred = proto([crit("E1", [atom("a", "b", "x")])])  # E2 missing entirely
    r = sc.concept_prf(pred, gold, strict=False)
    assert (r.tp, r.pred, r.ref) == (2, 3, 4)
    assert r.precision == 2 / 3 and r.recall == 0.5


def test_synonyms_apply():
    gold = proto([crit("E1", [atom("G6PD deficiency")])])
    pred = proto([crit("E1", [atom("glucose 6 phosphate dehydrogenase deficiency")])])
    syn = {sc.norm_text("glucose-6-phosphate dehydrogenase deficiency"):
           sc.norm_text("G6PD deficiency")}  # fmt: skip
    assert sc.concept_prf(pred, gold, strict=False, synonyms=syn).f1 == 1.0


def test_identity_ignores_free_text_but_not_logic():
    a = proto([crit("E1", [atom("a"), atom("b")], logic="any")], rationale="one wording")
    b = proto([crit("E1", [atom("a"), atom("b")], logic="any")], rationale="other wording")
    assert all(sc.element_identity(a, b).values())
    c = proto([crit("E1", [atom("a"), atom("b")], logic="all")])
    ident = sc.element_identity(a, c)
    assert not ident["eligibility"] and ident["time_zero"]
    same, n, flips = sc.criterion_identity(a, c)
    assert (same, n, flips) == (0, 1, [("E1", "any", "all")])


def test_logic_over_a_single_atom_is_not_a_difference():
    # Exit run, qwen3 NCT01223690 E1 "Presence of HIV infection": any vs all over one atom.
    a = proto([crit("E1", [atom("HIV infection")], logic="any")])
    b = proto([crit("E1", [atom("HIV infection")], logic="all")])
    assert sc.element_identity(a, b)["eligibility"]
    assert sc.criterion_identity(a, b) == (1, 1, [])
