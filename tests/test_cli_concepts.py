"""End-to-end: propose -> clinician decisions -> apply -> gate, through the CLI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from agentic_tte.cli import app
from agentic_tte.protocol import TargetTrialProtocol
from agentic_tte.synth.dgp import Scenario, generate
from agentic_tte.synth.omop import to_omop, write_duckdb

ROOT = Path(__file__).parent
VOCAB = ROOT / "fixtures" / "vocab"
PROTOCOL = ROOT / "fixtures" / "protocols" / "NCT04134403.trialist-v1.qwen3-32b.json"


def test_review_workflow(tmp_path):
    db = tmp_path / "synth.duckdb"
    write_duckdb(to_omop(generate(Scenario(n=150), with_truth=False)), db).close()
    protocol = tmp_path / "p.json"
    protocol.write_text(PROTOCOL.read_text(encoding="utf-8"), encoding="utf-8")
    run = CliRunner().invoke

    r = run(app, ["concepts", "propose", str(protocol), "--db", str(db), "--vocab", str(VOCAB),
                  "-o", str(tmp_path / "review")])  # fmt: skip
    assert r.exit_code == 0, r.output
    review = tmp_path / "review"
    assert "septic shock" in (review / "review.md").read_text(encoding="utf-8")

    decisions = json.loads((review / "decisions.json").read_text(encoding="utf-8"))
    decisions["reviewer"] = "Test Reviewer"
    (review / "decisions.json").write_text(json.dumps(decisions), encoding="utf-8")
    r = run(app, ["concepts", "apply", str(review / "book.json"), str(review / "decisions.json"),
                  "--protocol", str(protocol), "-o", str(review / "reviewed.json")])  # fmt: skip
    assert r.exit_code == 0, r.output
    v2 = TargetTrialProtocol.model_validate_json((tmp_path / "p.v2.json").read_text("utf-8"))
    assert v2.version == 2 and "Test Reviewer" in v2.provenance.notes

    r = run(app, ["gate", str(tmp_path / "p.v2.json"), "--db", str(db), "--vocab", str(VOCAB),
                  "--book", str(review / "reviewed.json"), "--classification", "synthetic"])
    assert "Data Support Gate" in r.output
    assert "Protocol concepts" in r.output
    # Undeclared classification blocks the run (exit 1).
    r = run(app, ["gate", str(protocol), "--db", str(db), "--classification", "unknown"])
    assert r.exit_code == 1 and "RED" in r.output
