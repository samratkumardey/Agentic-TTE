import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EMULATRX_NCT03872011 = (
    ROOT / "third_party" / "emulatrx" / "trialistUtils" / "results" / "trial_info" / "NCT03872011"
)


@pytest.fixture
def nct03872011_study() -> dict:
    """Real CT.gov v2 record cached by upstream EmulatRx (public registry data)."""
    return json.loads((EMULATRX_NCT03872011 / "api_info.json").read_text(encoding="utf-8"))
