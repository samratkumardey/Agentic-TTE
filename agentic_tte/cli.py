"""Command line entry point: ``agentic-tte --help``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from agentic_tte.config import get_settings

app = typer.Typer(no_args_is_help=True, help="Agentic Target Trial Emulation.")

# Registry text contains symbols such as "≥"; Windows consoles default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


def _ctgov():
    from agentic_tte.sources.ctgov import CTGovClient

    s = get_settings()
    return CTGovClient(s.ctgov_base_url, cache_dir=s.cache_dir, timeout_s=s.ctgov_timeout_s)


@app.command()
def fetch(nct_id: str, refresh: bool = typer.Option(False, help="Ignore the local cache.")):
    """Fetch a ClinicalTrials.gov record and print the extracted TTE sections."""
    from agentic_tte.sources.ctgov import extract_sections

    sections = extract_sections(_ctgov().get_study(nct_id, refresh=refresh))
    typer.echo(sections.model_dump_json(indent=2))


@app.command()
def draft(
    nct_id: str,
    out: Path = typer.Option(None, "--out", "-o", help="Write the protocol JSON here."),
    model: str = typer.Option(None, help="Override ATTE_MODEL."),
    no_cache: bool = typer.Option(False, help="Bypass the LLM response cache."),
):
    """Draft a target trial protocol from a registered trial (Replay mode)."""
    from agentic_tte.agents import Trialist
    from agentic_tte.llm import make_client

    s = get_settings()
    if model:
        s.model = model
    llm = make_client(s, use_cache=not no_cache)

    def log(kind: str, payload: dict) -> None:
        typer.echo(f"[trialist] {kind}: {json.dumps(payload, default=str)}", err=True)

    protocol = Trialist(llm, _ctgov(), on_event=log).draft(nct_id)
    text = protocol.model_dump_json(indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}", err=True)
    else:
        typer.echo(text)
    u = llm.usage
    typer.echo(
        f"[llm] calls={u.calls} cache_hits={u.cache_hits} prompt_tokens={u.prompt_tokens} "
        f"completion_tokens={u.completion_tokens} failures={u.failures}",
        err=True,
    )
    status = protocol.element_status()
    counts: dict[str, int] = {}
    for v in status.values():
        counts[v] = counts.get(v, 0) + 1
    summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
    typer.echo(f"[protocol] TTE elements by origin: {summary}", err=True)
    unspecified = [k for k, ok in protocol.completeness().items() if not ok]
    if unspecified:
        typer.echo(f"[protocol] default or missing: {', '.join(unspecified)}", err=True)


@app.command("llm-check")
def llm_check():
    """Check that the configured Ollama server is reachable and the model is pulled."""
    from agentic_tte.llm import OllamaClient

    s = get_settings()
    client = OllamaClient(s.model, host=s.ollama_host, timeout_s=10)
    version = client.server_version()
    if version is None:
        typer.echo(f"Ollama not reachable at {s.ollama_host}", err=True)
        raise typer.Exit(1)
    names = [m.get("name") for m in client.list_models()]
    typer.echo(f"Ollama {version} at {s.ollama_host}")
    typer.echo(f"model {s.model}: {'present' if client.model_digest() else 'NOT PULLED'}")
    typer.echo("available: " + ", ".join(n for n in names if n))
    if not client.model_digest():
        raise typer.Exit(1)


concepts_app = typer.Typer(no_args_is_help=True,
                           help="OMOP concept sets: propose from the vocabulary, then review.")
app.add_typer(concepts_app, name="concepts")


def _open_db(db: Path, vocab: Path | None):
    """Open an OMOP database read-only. With ``vocab``, the clinical tables are exposed as views
    and the vocabulary tables come from the Athena download instead of the database."""
    import duckdb

    from agentic_tte.omop.vocab import ATHENA_FILES, load_athena

    if vocab is None:
        return duckdb.connect(str(db), read_only=True)
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{db.as_posix()}' AS cdm (READ_ONLY)")
    vocab_tables = {f.lower() for f in ATHENA_FILES}
    for (t,) in con.execute("SELECT table_name FROM information_schema.tables "
                            "WHERE table_catalog = 'cdm'").fetchall():  # fmt: skip
        if t.lower() not in vocab_tables:
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM cdm.{t}")
    load_athena(con, vocab)
    return con


@concepts_app.command("propose")
def concepts_propose(
    protocol: Path,
    db: Path = typer.Option(..., help="OMOP CDM database (DuckDB file)."),
    out: Path = typer.Option(..., "--out", "-o", help="Directory for the review files."),
    vocab: Path = typer.Option(None, help="Athena vocabulary download to search instead."),
):
    """Propose concept sets for a protocol. Writes book.json, review.md (for the clinician)
    and decisions.json (to fill in)."""
    from agentic_tte.omop.concept_sets import decisions_template, propose, review_sheet
    from agentic_tte.protocol import TargetTrialProtocol

    p = TargetTrialProtocol.model_validate_json(protocol.read_text(encoding="utf-8"))
    book = propose(_open_db(db, vocab), p)
    out.mkdir(parents=True, exist_ok=True)
    (out / "book.json").write_text(book.model_dump_json(indent=2), encoding="utf-8")
    (out / "review.md").write_text(review_sheet(book), encoding="utf-8")
    (out / "decisions.json").write_text(json.dumps(decisions_template(book), indent=2),
                                        encoding="utf-8")  # fmt: skip
    typer.echo(f"{book.summary()} -> {out}/review.md; fill in {out}/decisions.json", err=True)


@concepts_app.command("apply")
def concepts_apply(
    book: Path,
    decisions: Path,
    protocol: Path = typer.Option(None, help="Also write a mapped protocol version."),
    out: Path = typer.Option(..., "--out", "-o", help="Reviewed book (JSON)."),
):
    """Apply a clinician's decisions to a concept-set book."""
    from agentic_tte.omop.concept_sets import ConceptSetBook, apply_decisions, apply_to_protocol
    from agentic_tte.protocol import TargetTrialProtocol

    b = ConceptSetBook.model_validate_json(book.read_text(encoding="utf-8"))
    reviewed = apply_decisions(b, json.loads(decisions.read_text(encoding="utf-8")))
    out.write_text(reviewed.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"{reviewed.summary()}; fully reviewed: {reviewed.fully_reviewed}", err=True)
    if protocol is not None:
        p = TargetTrialProtocol.model_validate_json(protocol.read_text(encoding="utf-8"))
        v = apply_to_protocol(p, reviewed)
        target = protocol.with_name(f"{protocol.stem}.v{v.version}.json")
        target.write_text(v.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"wrote {target}", err=True)


@app.command()
def gate(
    protocol: Path,
    db: Path = typer.Option(..., help="OMOP CDM database (DuckDB file)."),
    book: Path = typer.Option(None, help="Reviewed concept-set book (approved sets are used)."),
    classification: str = typer.Option(..., help="synthetic, deidentified, limited, identified"),
    approved: bool = typer.Option(False, help="Storage/compute approved for this classification."),
    vocab: Path = typer.Option(None, help="Athena vocabulary download."),
):
    """Run the Data Support Gate and print its report. Exits 1 if the run is blocked."""
    from agentic_tte.omop import DataContext, run_gate
    from agentic_tte.omop.concept_sets import ConceptSetBook
    from agentic_tte.protocol import TargetTrialProtocol

    p = TargetTrialProtocol.model_validate_json(protocol.read_text(encoding="utf-8"))
    sets = None
    if book is not None:
        sets = ConceptSetBook.model_validate_json(book.read_text(encoding="utf-8")).gate_sets()
    report = run_gate(_open_db(db, vocab), p, concept_sets=sets,
                      context=DataContext(data_classification=classification,
                                          approved_by_rcss=approved))  # fmt: skip
    typer.echo(report.to_markdown())
    if report.blocks_run:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
