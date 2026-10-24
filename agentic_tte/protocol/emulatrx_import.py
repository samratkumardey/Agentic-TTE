"""Import an EmulatRx Trialist cache directory as a TargetTrialProtocol.

Reads the layout written by EmulatRx ``trialistUtils/main.py`` (MIT, TrialLab)::

    <dir>/api_info.json               raw CT.gov v2 record
    <dir>/info.json                   LLM-split criteria, treatments, outcomes
    <dir>/parsed_{in,ex}_criteria.json  tagged annotations per criterion
    <dir>/query_{in,ex}_criteria.json   concept -> OMOP concept_id lookups (may be empty)
    <dir>/parsed_pr_outcomes.json

This lets us (a) regression-test against upstream's published parses and (b) compare our
Trialist with theirs on the same trials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_tte.protocol.annotation import parse_annotation
from agentic_tte.protocol.models import (
    Criterion,
    FollowUp,
    MappingStatus,
    Outcome,
    Provenance,
    TargetTrialProtocol,
    TrialSource,
)
from agentic_tte.protocol.normalize import parse_duration, parse_temporal, parse_value
from agentic_tte.sources.ctgov.extract import extract_sections


def _load(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _criteria(parsed: list[dict] | None, queried: list[dict] | None, kind: str,
              prefix: str) -> list[Criterion]:  # fmt: skip
    lookups: dict[str, dict[str, dict]] = {}
    for row in queried or []:
        lookups[row["text"]] = {q["ori_name"].lower(): q for q in row.get("query_results", [])}
    out = []
    for i, row in enumerate(parsed or [], 1):
        ann = parse_annotation(row.get("parsed_result", ""))
        hits = lookups.get(row["text"], {})
        for c in ann.concepts:
            q = hits.get(c.name.lower())
            if q and q.get("concept_id"):
                c.omop_concept_id = int(q["concept_id"])
                c.standard_name = q.get("stand_name")
                c.status = MappingStatus.MAPPED
        out.append(
            Criterion(
                id=f"{prefix}{i}",
                kind=kind,
                text=row["text"],
                concepts=ann.concepts,
                concept_logic="any",  # upstream does not record it
                value=parse_value(ann.values[0]) if ann.values else None,
                temporal=parse_temporal(ann.temporals),
                negated=bool(ann.negations),
                source="registry_text",
                notes="imported from EmulatRx cache",
            )
        )
    return out


def load_emulatrx_cache(directory: Path | str) -> TargetTrialProtocol:
    d = Path(directory)
    api = _load(d / "api_info.json")
    source = extract_sections(api).source if api else TrialSource(nct_id=d.name)

    eligibility = _criteria(
        _load(d / "parsed_in_criteria.json"), _load(d / "query_in_criteria.json"), "inclusion", "I"
    ) + _criteria(
        _load(d / "parsed_ex_criteria.json"), _load(d / "query_ex_criteria.json"), "exclusion", "E"
    )

    info = _load(d / "info.json") or {}
    outcomes = []
    for key, primary in (("Primary Outcome Measures", True), ("Secondary Outcome Measures", False)):
        for o in info.get(key, []):
            outcomes.append(
                Outcome(
                    name=o.get("Outcome Measure", ""),
                    description=o.get("Measure Description"),
                    time_frame=o.get("Time Frame"),
                    primary=primary,
                    horizon=parse_duration(o.get("Time Frame")),
                )
            )
    primary = next((o for o in outcomes if o.primary), None)
    return TargetTrialProtocol(
        protocol_id=f"{source.nct_id or d.name}-emulatrx",
        source=source,
        eligibility=eligibility,
        outcomes=outcomes,
        follow_up=FollowUp(horizon=primary.horizon if primary else None),
        provenance=Provenance(created_by="emulatrx-import", llm_model="gpt-4o (upstream)"),
    )
