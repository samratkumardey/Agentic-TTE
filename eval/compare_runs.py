"""Compare protocols trial by trial between two runs (e.g. np1 vs np4, forward vs reverse order).

    uv run python eval/compare_runs.py RUN_A RUN_B [--rep rep1]

RUN_A and RUN_B are run roots with <model arm>/rep<k>/ directories (runs/exit-<GPU>,
runs/repro-<GPU>/<config>). For each model arm present in both and each trial present in both:
  =  identical on every field (content_hash)
  ~  substantively identical (free text may differ; eval/phase0_scoring.py)
  x  the differing TTE elements
Exits non-zero if any trial differs substantively.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import phase0_scoring as sc  # noqa: E402


def compare(a_root: Path, b_root: Path, rep: str = "rep1") -> list[tuple[str, str, str, list]]:
    rows = []
    arms = sorted({p.name for p in a_root.iterdir() if (p / rep).is_dir()}
                  & {p.name for p in b_root.iterdir() if (p / rep).is_dir()})  # fmt: skip
    for arm in arms:
        a, b = sc.load_protocols(a_root / arm / rep), sc.load_protocols(b_root / arm / rep)
        for t in sorted(set(a) & set(b)):
            if a[t].content_hash() == b[t].content_hash():
                rows.append((arm, t, "=", []))
                continue
            diff = [e for e, same in sc.element_identity(a[t], b[t]).items() if not same]
            rows.append((arm, t, "x" if diff else "~", diff))
    return rows


def main(argv: list[str]) -> int:
    args = list(argv)
    rep = "rep1"
    if "--rep" in args:
        i = args.index("--rep")
        rep = args[i + 1]
        del args[i : i + 2]
    if len(args) != 2:
        print(__doc__)
        return 2
    rows = compare(Path(args[0]), Path(args[1]), rep)
    for arm, t, mark, diff in rows:
        print(f"{arm}\t{t}\t{mark}\t{','.join(diff)}")
    n_diff = sum(mark == "x" for _, _, mark, _ in rows)
    n_same = sum(mark == "=" for _, _, mark, _ in rows)
    print(f"{len(rows)} trial(s): {n_same} identical, {len(rows) - n_same - n_diff} substantively "
          f"identical, {n_diff} different")  # fmt: skip
    return 1 if n_diff else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
