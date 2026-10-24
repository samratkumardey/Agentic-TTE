"""Regression against upstream EmulatRx's published Trialist output for NCT03872011."""

from agentic_tte.protocol.emulatrx_import import load_emulatrx_cache
from tests.conftest import EMULATRX_NCT03872011


def test_import_nct03872011():
    p = load_emulatrx_cache(EMULATRX_NCT03872011)
    assert p.source.nct_id == "NCT03872011"
    assert len(p.inclusion) == 2 and len(p.exclusion) == 9

    age, onset = p.inclusion
    assert age.atoms[0].value.comparator == ">=" and age.atoms[0].value.value == 18
    assert age.concepts[0].omop_concept_id == 4265453  # upstream Athena lookup
    shock = onset.concepts[0]
    assert (shock.name, shock.omop_concept_id, shock.status) == ("septic shock", 196236, "mapped")
    assert onset.atoms[0].temporal.window.amount == 12

    steroids = p.exclusion[0]
    t = steroids.atoms[0].temporal
    assert t.relation == "x_within_t_before_y"
    assert t.window.amount == 3 and t.window.unit == "month"

    allergy = next(c for c in p.exclusion if "allergy" in c.text)
    assert len(allergy.concepts) == 3  # upstream's multi-concept prompt fix

    # query_ex_criteria.json is empty upstream: exclusion concepts stay unmapped, not invented.
    assert all(c.status == "unmapped" for crit in p.exclusion for c in crit.concepts)

    assert [o.name for o in p.outcomes if o.primary] == ["90-day mortality"]
    assert p.follow_up.horizon.amount == 90
