"""Pre-commit guard: refuse to commit files that look like patient-level data.

The repository must never hold MIMIC-IV, INSIGHT or Oracle EHR rows (PhysioNet DUA, HIPAA).
Synthetic data is regenerated from code, so it does not need to be committed either.
"""

from __future__ import annotations

import sys
from pathlib import PurePosixPath

BLOCKED_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather", ".pkl", ".pickle", ".duckdb", ".sav"}
BLOCKED_DIRS = {"data", "trials", "runs"}
# Small hand-written fixtures are allowed here.
ALLOWED_PREFIXES = ("tests/fixtures/",)
MAX_BYTES = 5 * 1024 * 1024


def violations(paths: list[str], sizes: dict[str, int] | None = None) -> list[str]:
    problems = []
    for raw in paths:
        path = PurePosixPath(raw.replace("\\", "/"))
        posix = str(path)
        if posix.startswith(ALLOWED_PREFIXES):
            continue
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            problems.append(f"{posix}: data file type {path.suffix} is not allowed in git")
        elif BLOCKED_DIRS & set(path.parts[:-1]):
            problems.append(f"{posix}: files under data/, trials/ or runs/ are not allowed in git")
        elif sizes and sizes.get(raw, 0) > MAX_BYTES:
            problems.append(f"{posix}: larger than {MAX_BYTES // (1024 * 1024)} MB")
    return problems


def main(argv: list[str]) -> int:
    from os.path import getsize

    sizes = {p: getsize(p) for p in argv}
    problems = violations(argv, sizes)
    for problem in problems:
        print(f"[no-patient-data] {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
