"""Check a finished Phase 0 exit run before scoring it (eval/SCORING.md, Design).

    uv run python eval/check_exit_run.py runs/exit-A100

Reads every runs/exit-<GPU>/<model>/rep<k>/ directory and reports, per run: model, replicate,
GPU, commit, trials present, LLM calls, cache hits, tokens and failures. Exits non-zero if the
run cannot be scored as designed: more than one GPU type, commit or Ollama version; a dirty
commit; missing trials; protocols that are not schema v2 / trialist-v2; or cache hits in a run
that was never preempted.
"""

from __future__ import annotations

import json
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


def check_run(run: Path) -> tuple[dict, list[str]]:
    m = read_manifest(run / "manifest.txt")
    trials = m.get("trials", "").split()
    row = {
        "run": str(run),
        "model": m.get("model"),
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
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "parse_failed": 0,
        "dropped": 0,
    }
    problems: list[str] = []
    for t in trials:
        proto, log = run / f"{t}.protocol.json", run / f"{t}.log"
        if not proto.exists():
            problems.append(f"{run}: missing {proto.name}")
        else:
            row["protocols"] += 1
            p = json.loads(proto.read_text(encoding="utf-8"))
            if p.get("schema_version") != 2:
                problems.append(f"{proto}: schema_version {p.get('schema_version')}, expected 2")
            if p.get("provenance", {}).get("prompt_version") != "trialist-v2":
                problems.append(f"{proto}: prompt_version is not trialist-v2")
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            for hit in LLM_LINE.finditer(text):
                for key, value in zip(LLM_KEYS, hit.groups(), strict=True):
                    row[key] += int(value)
            row["parse_failed"] += text.count("[trialist] parse_failed")
            row["dropped"] += text.count("[trialist] dropped")
    if not row["finished"]:
        problems.append(f"{run}: no 'finished' line in manifest (still running or killed)")
    if row["cache_hits"] and not row["restart_count"]:
        problems.append(
            f"{run}: {row['cache_hits']} cache hits without a restart (cache not fresh)"
        )
    if (row["git_commit"] or "").endswith("-dirty"):
        problems.append(f"{run}: commit {row['git_commit']} has uncommitted changes")
    return row, problems


def check(root: Path) -> tuple[list[dict], list[str]]:
    runs = sorted(p.parent for p in root.glob("*/rep*/manifest.txt"))
    if not runs:
        return [], [f"no runs found under {root} (expected <model>/rep<k>/manifest.txt)"]
    rows, problems = [], []
    for run in runs:
        row, probs = check_run(run)
        rows.append(row)
        problems += probs
    for key in ("gpu", "git_commit", "ollama_version"):
        values = {r[key] for r in rows}
        if len(values) > 1:
            problems.append(f"runs differ in {key}: {sorted(map(str, values))}")
    return rows, problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    rows, problems = check(Path(argv[0]))
    cols = ["model", "replicate", "protocols", "calls", "cache_hits", "prompt_tokens",
            "completion_tokens", "parse_failed", "dropped", "restart_count"]  # fmt: skip
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
