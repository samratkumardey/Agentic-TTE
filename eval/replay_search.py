"""Phase 3 Replay: the pre-registered ClinicalTrials.gov search (eval/SCORING.md, R3).

Runs the registry search once and saves a compact copy of every hit, so the screening in
eval/reference/replay_selection.json can be redone from a fixed record:

    uv run python eval/replay_search.py

Writes eval/reference/replay_search_<date>.json (public registry data only).
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

import httpx

TERMS = ["sepsis", "septic shock", "acute respiratory distress syndrome",
         "acute respiratory failure", "mechanical ventilation", "acute kidney injury",
         "delirium", "critical illness", "intensive care", "acute heart failure"]  # fmt: skip
_T = " OR ".join(f'"{t}"' for t in TERMS)
QUERY = (f"(AREA[Condition]({_T}) OR AREA[Keyword]({_T})) AND AREA[StudyType]INTERVENTIONAL "
         "AND AREA[DesignAllocation]RANDOMIZED AND AREA[OverallStatus]COMPLETED "
         "AND AREA[EnrollmentCount]RANGE[100, MAX] AND AREA[StdAge]ADULT")  # fmt: skip
API = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = ",".join([
    "protocolSection.identificationModule", "protocolSection.statusModule.completionDateStruct",
    "protocolSection.conditionsModule", "protocolSection.designModule",
    "protocolSection.armsInterventionsModule", "protocolSection.outcomesModule.primaryOutcomes",
    "protocolSection.eligibilityModule.minimumAge", "protocolSection.eligibilityModule.stdAges",
    "protocolSection.referencesModule", "hasResults",
    "resultsSection.outcomeMeasuresModule"])  # fmt: skip


def _years(age: str | None) -> float | None:
    m = re.match(r"(\d+(?:\.\d+)?)\s*(Year|Month|Week|Day)", age or "")
    if not m:
        return None
    per = {"Year": 1, "Month": 12, "Week": 52.18, "Day": 365.25}[m.group(2)]
    return float(m.group(1)) / per


def compact(study: dict) -> dict:
    """The fields the screening needs; primary-outcome result analyses only."""
    p = study["protocolSection"]
    design = p.get("designModule", {})
    info = design.get("designInfo", {})
    arms = p.get("armsInterventionsModule", {})
    primary = p.get("outcomesModule", {}).get("primaryOutcomes", [])
    analyses = []
    for om in study.get("resultsSection", {}).get("outcomeMeasuresModule", {}).get(
            "outcomeMeasures", []):  # fmt: skip
        if om.get("type") != "PRIMARY":
            continue
        for a in om.get("analyses", []):
            analyses.append({"outcome": om.get("title"), "time_frame": om.get("timeFrame"),
                             "param_type": a.get("paramType"), "value": a.get("paramValue"),
                             "ci_pct": a.get("ciPctValue"), "ci_lower": a.get("ciLowerLimit"),
                             "ci_upper": a.get("ciUpperLimit"), "p": a.get("pValue"),
                             "groups": a.get("groupIds"),
                             "groups_desc": a.get("groupDescription")})  # fmt: skip
    refs = p.get("referencesModule", {}).get("references", [])
    return {
        "nct_id": p["identificationModule"]["nctId"],
        "title": p["identificationModule"].get("briefTitle"),
        "acronym": p["identificationModule"].get("acronym"),
        "conditions": p.get("conditionsModule", {}).get("conditions", []),
        "phases": design.get("phases", []),
        "model": info.get("interventionModel"),
        "enrollment": design.get("enrollmentInfo", {}).get("count"),
        "min_age_years": _years(p.get("eligibilityModule", {}).get("minimumAge")),
        "std_ages": p.get("eligibilityModule", {}).get("stdAges", []),
        "completion": p.get("statusModule", {}).get("completionDateStruct", {}).get("date"),
        "interventions": [{"type": i.get("type"), "name": i.get("name")}
                          for i in arms.get("interventions", [])],
        "arms": [{"type": a.get("type"), "label": a.get("label")}
                 for a in arms.get("armGroups", [])],
        "primary_outcomes": [{"measure": o.get("measure"), "time_frame": o.get("timeFrame")}
                             for o in primary],
        "has_results": study.get("hasResults", False),
        "primary_analyses": analyses,
        "result_refs": [r.get("pmid") for r in refs if r.get("type") == "RESULT" and r.get("pmid")],
    }  # fmt: skip


def search() -> list[dict]:
    out, token = [], None
    with httpx.Client(timeout=120) as client:
        while True:
            params = {"query.term": QUERY, "fields": FIELDS, "pageSize": 200, "countTotal": "true"}
            if token:
                params["pageToken"] = token
            r = client.get(API, params=params)
            r.raise_for_status()
            page = r.json()
            out += [compact(s) for s in page.get("studies", [])]
            token = page.get("nextPageToken")
            if not token:
                return out


def main() -> None:
    hits = search()
    day = date.today().isoformat()
    path = Path(__file__).parent / "reference" / f"replay_search_{day}.json"
    path.write_text(json.dumps({"_about": "Pre-registered registry search, eval/SCORING.md R3",
                                "date": day, "api": API, "query": QUERY, "count": len(hits),
                                "studies": sorted(hits, key=lambda s: s["nct_id"])},
                               indent=1), encoding="utf-8")  # fmt: skip
    print(f"{len(hits)} studies -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
