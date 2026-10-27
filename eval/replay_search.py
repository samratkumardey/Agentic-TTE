"""Phase 3 Replay: the pre-registered ClinicalTrials.gov search (eval/SCORING.md, R3).

Runs the registry search once and saves a compact copy of every hit, so the screening in
eval/reference/replay_selection.json can be redone from a fixed record:

    uv run python eval/replay_search.py            # wave 1 (R3)
    uv run python eval/replay_search.py --wave 2   # wave 2 (Phase 3 changelog item 5)
    uv run python eval/replay_search.py --pubmed   # result links through PubMed (item 6)

Writes eval/reference/replay_search[_wave2]_<date>.json and replay_pubmed_links_<date>.json
(public registry and PubMed data only).
"""

import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import httpx

TERMS = ["sepsis", "septic shock", "acute respiratory distress syndrome",
         "acute respiratory failure", "mechanical ventilation", "acute kidney injury",
         "delirium", "critical illness", "intensive care", "acute heart failure"]  # fmt: skip
# Wave 2 (Phase 3 changelog, item 5): the wave-1 terms plus these.
TERMS_WAVE2 = TERMS + ["shock", "cardiogenic shock", "cardiac arrest", "critically ill",
                       "pneumonia", "gastrointestinal hemorrhage", "acute pancreatitis",
                       "cardiac surgery", "pulmonary embolism", "subarachnoid hemorrhage",
                       "intracerebral hemorrhage", "traumatic brain injury"]  # fmt: skip


def query(terms: list[str]) -> str:
    t = " OR ".join(f'"{x}"' for x in terms)
    return (f"(AREA[Condition]({t}) OR AREA[Keyword]({t})) AND AREA[StudyType]INTERVENTIONAL "
            "AND AREA[DesignAllocation]RANDOMIZED AND AREA[OverallStatus]COMPLETED "
            "AND AREA[EnrollmentCount]RANGE[100, MAX] AND AREA[StdAge]ADULT")  # fmt: skip


QUERY = query(TERMS)
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


def search(q: str = QUERY) -> list[dict]:
    out, token = [], None
    with httpx.Client(timeout=120) as client:
        while True:
            params = {"query.term": q, "fields": FIELDS, "pageSize": 200, "countTotal": "true"}
            if token:
                params["pageToken"] = token
            r = client.get(API, params=params)
            r.raise_for_status()
            page = r.json()
            out += [compact(s) for s in page.get("studies", [])]
            token = page.get("nextPageToken")
            if not token:
                return out


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def pubmed_rct(nct_id: str, client: httpx.Client) -> list[str]:
    """PMIDs of articles PubMed indexes with this NCT number and type RCT (item 6)."""
    term = f"{nct_id}[si] AND randomized controlled trial[pt]"
    for attempt in range(6):
        r = client.get(EUTILS, params={"db": "pubmed", "term": term, "retmode": "json"})
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()["esearchresult"]["idlist"]
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()
    return []


def link_pubmed(ref: Path, day: str) -> None:
    """PubMed links for every saved hit that passes the other stage-A screens."""
    sys.path.insert(0, str(Path(__file__).parent))
    from replay_screen import stage_a_without_s2

    studies = {}
    for f in sorted(ref.glob("replay_search*_*.json")):
        studies.update({s["nct_id"]: s for s in json.loads(f.read_text("utf-8"))["studies"]})
    todo = sorted(i for i, s in studies.items() if stage_a_without_s2(s) is None)
    links = {}
    with httpx.Client(timeout=60) as client:
        for n, i in enumerate(todo, 1):
            links[i] = pubmed_rct(i, client)
            time.sleep(0.4)  # NCBI allows 3 requests a second without a key
            if n % 100 == 0:
                print(f"{n}/{len(todo)}", file=sys.stderr)
    path = ref / f"replay_pubmed_links_{day}.json"
    path.write_text(json.dumps({"_about": "PubMed RCT articles per NCT number (SCORING.md "
                                "Phase 3 changelog item 6)", "date": day,
                                "term": "<NCT>[si] AND randomized controlled trial[pt]",
                                "links": links}, indent=1), encoding="utf-8")  # fmt: skip
    print(f"{sum(bool(v) for v in links.values())}/{len(links)} linked -> {path}", file=sys.stderr)


def main() -> None:
    day = date.today().isoformat()
    ref = Path(__file__).parent / "reference"
    if "--pubmed" in sys.argv:
        link_pubmed(ref, day)
        return
    wave2 = "--wave" in sys.argv and sys.argv[sys.argv.index("--wave") + 1] == "2"
    q = query(TERMS_WAVE2) if wave2 else QUERY
    hits = search(q)
    path = ref / f"replay_search{'_wave2' if wave2 else ''}_{day}.json"
    about = ("Wave-2 registry search, eval/SCORING.md Phase 3 changelog item 5" if wave2
             else "Pre-registered registry search, eval/SCORING.md R3")  # fmt: skip
    path.write_text(json.dumps({"_about": about, "date": day, "api": API, "query": q,
                                "count": len(hits),
                                "studies": sorted(hits, key=lambda s: s["nct_id"])},
                               indent=1), encoding="utf-8")  # fmt: skip
    print(f"{len(hits)} studies -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
