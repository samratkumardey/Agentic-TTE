"""Check a finished Phase 0 exit run before scoring it (eval/SCORING.md, Design).

    uv run python eval/check_exit_run.py runs/exit-H100 [--ctgov DIR]

Reads every runs/exit-<GPU>/<model>/rep<k>/ directory and reports, per run: model, replicate,
GPU, commit, trials present, LLM calls, cache hits, tokens and failures. Exits non-zero if the
run cannot be scored as designed: more than one GPU type, commit or Ollama version; a dirty
commit; missing trials; protocols that are not schema v2 / trialist-v2; or cache hits that the
run's own requests do not explain.

Cache hits: a fresh replicate can still hit its own cache when two trials share a criterion
word for word ("Age < 18 years", "Pregnant"). The checker rebuilds every first-attempt request
of the run, in trial order, from the CT.gov records (``--ctgov``, default
``$ATTE_PROJECT_DIR/cache/ctgov`` or ``.cache/agentic_tte/ctgov``) and allows exactly those
repeats. Anything beyond them means the cache was not fresh.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

LLM_LINE = re.compile(
    r"\[llm\] calls=(\d+) cache_hits=(\d+) prompt_tokens=(\d+) completion_tokens=(\d+)"
)
LLM_KEYS = ("calls", "cache_hits", "prompt_tokens", "completion_tokens")


def read_manifest(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    # The last line is "finished=<date> failed=<n>".
    m = re.search(r"failed=(\d+)", out.get("finished", ""))
    if m:
        out["finished"] = out["finished"].split()[0]
        out["failed"] = m.group(1)
    return out


def default_ctgov_dir() -> Path:
    if os.environ.get("ATTE_PROJECT_DIR"):
        return Path(os.environ["ATTE_PROJECT_DIR"]) / "cache" / "ctgov"
    return Path(".cache/agentic_tte/ctgov")


def expected_repeats(trials: list[str], ctgov: Path | None) -> dict[str, int] | None:
    """Per trial: how many of its first-attempt requests already occurred earlier in the run.
    None if the CT.gov records are not available."""
    if ctgov is None or not all((ctgov / f"{t}.json").exists() for t in trials):
        return None
    from agentic_tte.agents.trialist import first_requests
    from agentic_tte.sources.ctgov import extract_sections

    seen: set[str] = set()
    out: dict[str, int] = {}
    for t in trials:
        study = json.loads((ctgov / f"{t}.json").read_text(encoding="utf-8"))
        repeats = 0
        for req in first_requests(extract_sections(study)):
            key = json.dumps([m.model_dump() for m in req], sort_keys=True)
            repeats += key in seen
            seen.add(key)
        out[t] = repeats
    return out


def failure_reason(log_text: str) -> str:
    for name in ("ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError"):
        if name in log_text:
            return name
    if "Traceback" in log_text or "Error" in log_text:
        return "error"
    return "no output"


def check_run(run: Path, ctgov: Path | None = None) -> tuple[dict, list[str]]:
    m = read_manifest(run / "manifest.txt")
    trials = m.get("trials", "").split()
    row = {
        "run": str(run),
        "arm": run.parent.name,  # model + reasoning level, e.g. gpt-oss_20b-low
        "model": m.get("model"),
        "think": m.get("think"),
        "replicate": m.get("replicate"),
        "gpu": m.get("gpu"),
        "git_commit": m.get("git_commit"),
        "ollama_version": m.get("ollama_version"),
        "restart_count": int(m.get("restart_count") or 0),
        "finished": "finished" in m,
        "trials": len(trials),
        "protocols": 0,
        "calls": 0,
        "cache_hits": 0,
        "expected_repeats": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "parse_failed": 0,
        "dropped": 0,
        "missing": {},  # trial -> failure reason
    }
    problems: list[str] = []
    repeats = expected_repeats(trials, ctgov)
    if repeats is not None:
        row["expected_repeats"] = sum(repeats.values())
    for t in trials:
        proto, log = run / f"{t}.protocol.json", run / f"{t}.log"
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if not proto.exists():
            reason = failure_reason(text)
            row["missing"][t] = reason
            problems.append(f"{run}: missing {proto.name} ({reason})")
        else:
            row["protocols"] += 1
            p = json.loads(proto.read_text(encoding="utf-8"))
            if p.get("schema_version") != 2:
                problems.append(f"{proto}: schema_version {p.get('schema_version')}, expected 2")
            if p.get("provenance", {}).get("prompt_version") != "trialist-v2":
                problems.append(f"{proto}: prompt_version is not trialist-v2")
        hits = 0
        for hit in LLM_LINE.finditer(text):
            for key, value in zip(LLM_KEYS, hit.groups(), strict=True):
                row[key] += int(value)
            hits += int(hit.group(2))
        row["parse_failed"] += text.count("[trialist] parse_failed")
        row["dropped"] += text.count("[trialist] dropped")
        if hits and not row["restart_count"]:
            allowed = repeats.get(t) if repeats is not None else 0
            if hits > allowed:
                why = "cache not fresh" if repeats is not None else "no CT.gov records to check"
                problems.append(
                    f"{run}: {t} has {hits} cache hits, {allowed} explained by repeated "
                    f"requests within the run ({why})"
                )
    if not row["finished"]:
        problems.append(f"{run}: no 'finished' line in manifest (still running or killed)")
    if (row["git_commit"] or "").endswith("-dirty"):
        problems.append(f"{run}: commit {row['git_commit']} has uncommitted changes")
    return row, problems


def check(root: Path, ctgov: Path | None = None) -> tuple[list[dict], list[str]]:
    runs = sorted(p.parent for p in root.glob("*/rep*/manifest.txt"))
    if not runs:
        return [], [f"no runs found under {root} (expected <model>/rep<k>/manifest.txt)"]
    rows, problems = [], []
    for run in runs:
        row, probs = check_run(run, ctgov)
        rows.append(row)
        problems += probs
    for key in ("gpu", "git_commit", "ollama_version"):
        values = {r[key] for r in rows}
        if len(values) > 1:
            problems.append(f"runs differ in {key}: {sorted(map(str, values))}")
    return rows, problems


def main(argv: list[str]) -> int:
    args = list(argv)
    ctgov = default_ctgov_dir()
    if "--ctgov" in args:
        i = args.index("--ctgov")
        ctgov = Path(args[i + 1])
        del args[i : i + 2]
    if len(args) != 1:
        print(__doc__)
        return 2
    rows, problems = check(Path(args[0]), ctgov)
    cols = ["arm", "replicate", "protocols", "calls", "cache_hits", "expected_repeats",
            "prompt_tokens", "completion_tokens", "parse_failed", "dropped",
            "restart_count"]  # fmt: skip
    if rows:
        print("gpu:", rows[0]["gpu"], "| commit:", rows[0]["git_commit"],
              "| ollama:", rows[0]["ollama_version"])  # fmt: skip
        print("\t".join(cols))
        for r in rows:
            print("\t".join(str(r[c]) for c in cols))
    for p in problems:
        print("PROBLEM:", p)
    print("OK: ready to score" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
