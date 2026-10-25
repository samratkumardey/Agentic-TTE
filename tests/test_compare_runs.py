import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
spec = importlib.util.spec_from_file_location("compare_runs", ROOT / "eval" / "compare_runs.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

FIXTURE = ROOT / "tests" / "fixtures" / "protocols" / "NCT04134403.trialist-v1.qwen3-32b.json"


def _put(root: Path, arm: str, text: str) -> None:
    run = root / arm / "rep1"
    run.mkdir(parents=True)
    (run / "NCT04134403.protocol.json").write_text(text, encoding="utf-8")


def test_identical_free_text_and_substantive_differences(tmp_path):
    base = FIXTURE.read_text(encoding="utf-8")
    _put(tmp_path / "a", "qwen3_32b", base)
    _put(tmp_path / "b", "qwen3_32b", base)
    _put(tmp_path / "a", "medgemma", base)
    _put(tmp_path / "b", "medgemma", base.replace("Time zero is defined", "Time zero is set"))
    _put(tmp_path / "a", "gpt-oss_20b", base)
    _put(tmp_path / "b", "gpt-oss_20b", base.replace('"amount": 28.0', '"amount": 30.0', 1))
    rows = {arm: (mark, diff) for arm, _, mark, diff in cr.compare(tmp_path / "a", tmp_path / "b")}
    assert rows["qwen3_32b"] == ("=", [])
    assert rows["medgemma"] == ("~", [])  # only the time-zero rationale differs
    assert rows["gpt-oss_20b"][0] == "x"
    assert cr.main([str(tmp_path / "a"), str(tmp_path / "b")]) == 1
