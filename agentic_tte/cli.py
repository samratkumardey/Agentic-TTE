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
        f"completion_tokens={u.completion_tokens}",
        err=True,
    )
    missing = [k for k, ok in protocol.completeness().items() if not ok]
    if missing:
        typer.echo(f"[protocol] incomplete TTE elements: {', '.join(missing)}", err=True)


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


if __name__ == "__main__":
    app()
