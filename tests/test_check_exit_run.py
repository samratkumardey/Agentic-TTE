import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_exit_run", Path(__file__).resolve().parents[1] / "eval" / "check_exit_run.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

LOG = (
    "[trialist] criterion_parsed: {}\n[trialist] dropped: {\"element\": \"grace_period\"}\n"
    "[llm] calls=15 cache_hits={hits} prompt_tokens=100 completion_tokens=20\n"
)


def _run(root, model, rep, *, gpu="NVIDIA H100 80GB HBM3, 81559 MiB", commit="f9f7185",
         hits=0, restart=0, schema=2, trials=("NCT1", "NCT2"), finished=True):  # fmt: skip
    run = root / model / f"rep{rep}"
    run.mkdir(parents=True)
    lines = [f"model={model}", f"replicate={rep}", f"gpu={gpu}", f"git_commit={commit}",
             "ollama_version={\"version\":\"0.40.3\"}", f"restart_count={restart}",
             f"trials={' '.join(trials)}", "started=2026-10-24T10:00:00"]  # fmt: skip
    if finished:
        lines.append("finished=2026-10-24T11:00:00 failed=0")
    (run / "manifest.txt").write_text("\n".join(lines) + "\n")
    for t in trials:
        p = {"schema_version": schema, "provenance": {"prompt_version": "trialist-v2"}}
        (run / f"{t}.protocol.json").write_text(json.dumps(p))
        (run / f"{t}.log").write_text(LOG.replace("{hits}", str(hits)))


def test_clean_run_passes(tmp_path):
    for rep in (1, 2):
        _run(tmp_path, "qwen3_32b", rep)
        _run(tmp_path, "gpt-oss_20b", rep)
    rows, problems = mod.check(tmp_path)
    assert problems == []
    assert len(rows) == 4
    assert rows[0]["calls"] == 30 and rows[0]["dropped"] == 2 and rows[0]["protocols"] == 2


def test_detects_mixed_gpu_cache_hits_v1_and_dirty(tmp_path):
    _run(tmp_path, "qwen3_32b", 1)
    _run(tmp_path, "qwen3_32b", 2, gpu="NVIDIA A100-SXM4-80GB, 81920 MiB", hits=3)
    _run(tmp_path, "medgemma", 1, schema=1, commit="f9f7185-dirty", finished=False)
    _, problems = mod.check(tmp_path)
    text = "\n".join(problems)
    assert "runs differ in gpu" in text
    assert "cache hits without a restart" in text
    assert "schema_version 1" in text
    assert "uncommitted changes" in text
    assert "no 'finished' line" in text


def test_cache_hits_allowed_after_preemption(tmp_path):
    _run(tmp_path, "qwen3_32b", 1, hits=5, restart=1)
    _, problems = mod.check(tmp_path)
    assert problems == []
