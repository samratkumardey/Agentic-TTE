import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "guard", Path(__file__).resolve().parents[1] / "scripts" / "check_no_patient_data.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_blocks_data_files_and_dirs():
    bad = guard.violations(
        ["trials/NCT00475852/dataframes/big_df.csv", "data/omop/person.parquet",
         "runs/x/report.md", "notes\\cohort.pkl"]  # fmt: skip
    )
    assert len(bad) == 4


def test_allows_code_and_fixtures():
    assert guard.violations(
        ["agentic_tte/cli.py", "tests/fixtures/tiny.csv", "PLAN.md",
         "third_party/emulatrx/trialistUtils/results/trial_info/NCT03872011/info.json"]  # fmt: skip
    ) == []


def test_blocks_large_files():
    assert guard.violations(["big.json"], {"big.json": 6 * 1024 * 1024})
