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


def _abbreviations(path):
    from agentic_tte.omop.abbreviations import AbbreviationList

    if path is None:
        return None
    lst = AbbreviationList.load(path)
    n, k = len(lst.entries), len(lst.approved())
    typer.echo(f"abbreviations: {k} of {n} entries approved in {path}"
               + ("; unapproved entries are not used" if k < n else ""), err=True)
    return lst


etl_app = typer.Typer(no_args_is_help=True, help="Convert source data to OMOP CDM 5.4.")
app.add_typer(etl_app, name="etl")


@etl_app.command("mimic")
def etl_mimic(
    source: Path = typer.Option(..., help="MIMIC-IV 3.1 directory (hosp/, icu/)."),
    vocab: Path = typer.Option(..., help="Athena vocabulary download (CSV files)."),
    out: Path = typer.Option(..., help="DuckDB file to create (never overwritten)."),
    sample_percent: float = typer.Option(None, help="Pilot: keep this % of subjects."),
    diagnosis_time: str = typer.Option("admission", help="admission or discharge"),
    death_followup_days: int = typer.Option(365, help="Observation after the last discharge."),
    threads: int = typer.Option(None),
    memory_limit: str = typer.Option(None, help="DuckDB memory limit, e.g. 100GB."),
    temp_directory: str = typer.Option(None, help="Node-local scratch for spills."),
    concept_map: Path = typer.Option(None, help="mimic-code mimic-iv/concepts/concept_map: "
                                                "curated chartevents and procedureevents maps."),
    item_map: list[Path] = typer.Option(None, help="Reviewed item map (labevents or "
                                                   "inputevents); repeat for each."),
):
    """MIMIC-IV -> OMOP CDM 5.4. Writes the CDM, a manifest and a suppressed report next to
    it; only the report and manifest may leave the cluster."""
    import os
    import platform
    import subprocess

    from agentic_tte.etl.mimic_omop import EtlConfig, build, sha256sums_digest

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()  # fmt: skip
    except Exception:  # noqa: BLE001 - outside a checkout
        commit = None
    cfg = EtlConfig(source=source, vocab=vocab, out=out, sample_percent=sample_percent,
                    diagnosis_time=diagnosis_time, death_followup_days=death_followup_days,
                    threads=threads, memory_limit=memory_limit,
                    temp_directory=temp_directory, concept_map=concept_map,
                    item_maps=item_map or [])  # fmt: skip
    map_commit = None
    if concept_map is not None:
        try:
            map_commit = subprocess.run(["git", "-C", str(concept_map), "rev-parse", "HEAD"],
                                        capture_output=True, text=True,
                                        check=True).stdout.strip()  # fmt: skip
        except Exception:  # noqa: BLE001 - not a git checkout
            map_commit = "unknown"
    rep = build(cfg, git_commit=commit)
    stem = out.with_suffix("")
    Path(f"{stem}.report.md").write_text(rep.to_markdown(), encoding="utf-8")
    Path(f"{stem}.report.json").write_text(json.dumps(rep.public(), indent=2), "utf-8")
    manifest = {
        "job": os.environ.get("SLURM_JOB_ID", "interactive"), "node": platform.node(),
        "git_commit": commit, "source": str(source),
        "source_sha256sums_digest": sha256sums_digest(source),
        "vocabulary": str(vocab), "vocabulary_version": rep.vocabulary_version,
        "concept_map": str(concept_map) if concept_map else None,
        "concept_map_commit": map_commit,
        "item_maps": ",".join(str(x) for x in item_map or []) or None,
        "duckdb": rep.duckdb_version, "options": cfg.model_dump_json(),
        "started": rep.started_at.isoformat(), "finished": rep.finished_at.isoformat(),
    }  # fmt: skip
    Path(f"{stem}.manifest.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in manifest.items()) + "\n", encoding="utf-8")
    typer.echo(f"wrote {out}; report {stem}.report.md; manifest {stem}.manifest.txt", err=True)


items_app = typer.Typer(no_args_is_help=True,
                        help="Reviewed item maps for labevents and inputevents.")
etl_app.add_typer(items_app, name="items")


@items_app.command("propose")
def items_propose(
    source: Path = typer.Option(..., help="MIMIC-IV 3.1 directory (hosp/, icu/)."),
    vocab: Path = typer.Option(..., help="Athena vocabulary download."),
    kind: str = typer.Option(..., help="labevents or inputevents"),
    out: Path = typer.Option(..., "--out", "-o", help="Directory for the review files."),
    min_rows: int = typer.Option(11, help="Leave out items with fewer events."),
):
    """Propose a standard concept per item from the vocabulary search on its label. Writes
    map.json, review.md and decisions.json; keep them on the cluster."""
    import duckdb

    from agentic_tte.etl.item_maps import (
        decisions_template,
        item_counts,
        propose_items,
        review_sheet,
    )
    from agentic_tte.omop.vocab import load_athena

    if kind not in ("labevents", "inputevents"):
        typer.echo("kind must be labevents or inputevents", err=True)
        raise typer.Exit(2)
    module, dictionary = ("hosp", "d_labitems") if kind == "labevents" else ("icu", "d_items")
    for f in (source / module / f"{kind}.csv.gz", source / module / f"{dictionary}.csv.gz"):
        if not f.exists():  # check before the slow vocabulary load
            typer.echo(f"missing {f.resolve()}" + (" (is ATTE_MIMIC_SOURCE set? `source "
                       "hpc/env.sh`)" if str(source) == "." else ""), err=True)  # fmt: skip
            raise typer.Exit(2)
    con = duckdb.connect()
    try:
        load_athena(con, vocab)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e

    def csv(module_, table):
        path = (source / module_ / f"{table}.csv.gz").as_posix()
        return f"read_csv('{path}', header = true, all_varchar = true)"

    events = csv(module, kind)
    items, total = item_counts(con, kind, events, csv(module, dictionary), min_rows=min_rows)
    m = propose_items(con, kind, items, total_rows=total)
    out.mkdir(parents=True, exist_ok=True)
    (out / "map.json").write_text(m.model_dump_json(indent=2), encoding="utf-8")
    (out / "review.md").write_text(review_sheet(m), encoding="utf-8")
    (out / "decisions.json").write_text(json.dumps(decisions_template(m), indent=2), "utf-8")
    typer.echo(f"{len(m.entries)} items ({m.summary()}) -> {out}/review.md; fill in "
               f"{out}/decisions.json, then `agentic-tte etl items apply`", err=True)


@items_app.command("apply")
def items_apply(
    map_file: Path,
    decisions: Path,
    out: Path = typer.Option(..., "--out", "-o", help="Reviewed item map (JSON)."),
):
    """Apply a reviewer's decisions to an item map."""
    from agentic_tte.etl.item_maps import DecisionError, ItemMap, apply_decisions

    m = ItemMap.model_validate_json(map_file.read_text(encoding="utf-8"))
    try:
        reviewed = apply_decisions(m, json.loads(decisions.read_text(encoding="utf-8")))
    except DecisionError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e
    out.write_text(reviewed.model_dump_json(indent=2), encoding="utf-8")
    undecided = sum(e.status in ("proposed", "no_match") for e in reviewed.entries)
    typer.echo(f"{reviewed.summary()}; {undecided} items undecided (not converted) -> {out}",
               err=True)


abbr_app = typer.Typer(no_args_is_help=True,
                       help="Abbreviation list for concept search; entries need approval.")
app.add_typer(abbr_app, name="abbreviations")


@abbr_app.command("init")
def abbreviations_init(out: Path):
    """Copy the seed list (all entries unapproved) to a file you own and review."""
    from agentic_tte.omop.abbreviations import AbbreviationList

    if out.exists():
        typer.echo(f"{out} exists; not overwritten", err=True)
        raise typer.Exit(2)
    AbbreviationList.seed().save(out)
    typer.echo(f"wrote {out}: review each expansion, then `agentic-tte abbreviations approve "
               f"{out} --reviewer NAME`", err=True)  # fmt: skip


@abbr_app.command("show")
def abbreviations_show(path: Path = typer.Argument(None, help="List file; default: the seed.")):
    """Print the list with each entry's approval."""
    from agentic_tte.omop.abbreviations import AbbreviationList

    lst = AbbreviationList.load(path) if path else AbbreviationList.seed()
    typer.echo("| Abbreviation | Expansions | Domain | Approved by |")
    typer.echo("|---|---|---|---|")
    for e in lst.entries:
        typer.echo(f"| {e.abbr} | {' / '.join(e.expansions)} | {e.domain or 'any'} | "
                   f"{e.approved_by or 'NOT APPROVED'} |")


@abbr_app.command("approve")
def abbreviations_approve(
    path: Path,
    reviewer: str = typer.Option(..., help="Who reviewed the expansions."),
    only: str = typer.Option(None, help="Comma-separated abbreviations; default: all."),
):
    """Record a reviewer's approval in the list file."""
    from agentic_tte.omop.abbreviations import AbbreviationList

    lst = AbbreviationList.load(path)
    try:
        n = lst.approve(reviewer, {a.strip() for a in only.split(",")} if only else None)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e
    lst.save(path)
    typer.echo(f"{n} entries approved by {reviewer}; {len(lst.approved())} of "
               f"{len(lst.entries)} approved in {path}", err=True)


@concepts_app.command("propose")
def concepts_propose(
    protocol: Path,
    db: Path = typer.Option(..., help="OMOP CDM database (DuckDB file)."),
    out: Path = typer.Option(..., "--out", "-o", help="Directory for the review files."),
    vocab: Path = typer.Option(None, help="Athena vocabulary download to search instead."),
    abbreviations: Path = typer.Option(None, help="Abbreviation list; only its approved "
                                                  "entries are expanded."),
):
    """Propose concept sets for a protocol. Writes book.json, review.md (for the clinician)
    and decisions.json (to fill in)."""
    from agentic_tte.omop.concept_sets import decisions_template, propose, review_sheet
    from agentic_tte.protocol import TargetTrialProtocol

    p = TargetTrialProtocol.model_validate_json(protocol.read_text(encoding="utf-8"))
    book = propose(_open_db(db, vocab), p, abbreviations=_abbreviations(abbreviations))
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


@app.command("compile")
def compile_cmd(
    protocol: Path,
    out: Path = typer.Option(..., "--out", "-o", help="Directory for the SQL and the report."),
    spec: Path = typer.Option(None, help="Compile spec (time zero, anchors). Without one, a "
                                         "template is written to fill in."),
    book: Path = typer.Option(None, help="Reviewed concept-set book (approved sets are used)."),
    db: Path = typer.Option(None, help="OMOP database: check the SQL runs and count eligible."),
    vocab: Path = typer.Option(None, help="Athena vocabulary download."),
):
    """Compile a protocol's criteria and analysis frame to OMOP SQL (no LLM involved)."""
    from agentic_tte.omop.compiler import CompileSpec, compile_protocol, spec_template
    from agentic_tte.omop.concept_sets import ConceptSetBook
    from agentic_tte.omop.gate import suppress
    from agentic_tte.protocol import TargetTrialProtocol

    p = TargetTrialProtocol.model_validate_json(protocol.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    if spec is None:
        target = out / "spec.json"
        target.write_text(json.dumps(spec_template(p), indent=2), encoding="utf-8")
        typer.echo(f"no spec given: fill in {target} (time zero and anchors as concept sets), "
                   "then rerun with --spec", err=True)  # fmt: skip
        raise typer.Exit(1)
    s = CompileSpec.model_validate_json(spec.read_text(encoding="utf-8"))
    if book is not None:
        sets = ConceptSetBook.model_validate_json(book.read_text(encoding="utf-8")).gate_sets()
        s = s.model_copy(update={"concept_sets": {**sets, **s.concept_sets}})
    con = _open_db(db, vocab) if db is not None else None
    if con is not None:
        tables = {r[0].lower() for r in con.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}  # fmt: skip
        s = s.model_copy(update={"descendants": "concept_ancestor" in tables})
    cc = compile_protocol(p, s)
    (out / "cohort.sql").write_text(cc.cohort_sql + "\n", encoding="utf-8")
    if cc.frame_sql:
        (out / "frame.sql").write_text(cc.frame_sql + "\n", encoding="utf-8")
    (out / "compile.md").write_text(cc.to_markdown(), encoding="utf-8")
    (out / "compile.json").write_text(cc.model_dump_json(indent=2), encoding="utf-8")
    n = {st: sum(c.status == st for c in cc.criteria)
         for st in ("compiled", "not_compiled", "not_emulable")}  # fmt: skip
    typer.echo(f"{n} -> {out}/compile.md", err=True)
    if con is not None:
        from agentic_tte.omop.attrition import attrition

        rep = attrition(con, cc)
        (out / "attrition.md").write_text(rep.to_markdown(), encoding="utf-8")
        (out / "attrition.json").write_text(json.dumps(rep.public(), indent=2),
                                            encoding="utf-8")  # fmt: skip
        typer.echo(f"index persons {suppress(rep.steps[1].remaining)}, eligible "
                   f"{suppress(rep.eligible)} -> {out}/attrition.md", err=True)


replay_app = typer.Typer(no_args_is_help=True,
                         help="Replay a registered trial, with clinician reviews in between.")
app.add_typer(replay_app, name="replay")


def _replay_graph(run: dict, llm=None):
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    from agentic_tte.graph import AutoApprove, build_replay
    from agentic_tte.omop import DataContext

    out = Path(run["out"])
    saver = SqliteSaver(sqlite3.connect(str(out / "checkpoints.sqlite"), check_same_thread=False))
    con = _open_db(Path(run["db"]), Path(run["vocab"]) if run.get("vocab") else None)
    ctx = DataContext(data_classification=run["classification"], approved_by_rcss=run["approved"])
    auto = AutoApprove.model_validate(run["auto"]) if run.get("auto") else None
    graph = build_replay(con, llm=llm, ctgov=_ctgov() if llm else None, context=ctx,
                         design_hr=run.get("design_hr"), checkpointer=saver, auto=auto,
                         abbreviations=_abbreviations(run.get("abbreviations")))  # fmt: skip
    return graph, {"configurable": {"thread_id": "replay"}}


def _replay_pause(graph, cfg, out: Path) -> None:
    """Write the open review's files, or the report when the run has finished."""
    from agentic_tte.graph import pending

    ask = pending(graph, cfg)
    if ask is None:
        state = graph.get_state(cfg).values
        (out / "report.md").write_text(state.get("report", ""), encoding="utf-8")
        typer.echo(f"run {state.get('status')}: report in {out}/report.md", err=True)
        return
    stage = ask["stage"]
    if stage == "design":
        (out / "protocol.json").write_text(json.dumps(ask["protocol"], indent=2), "utf-8")
        tmpl = {"reviewer": "", "confounders": None, "changes": {}}
        files = ["protocol.json", "design.json"]
    elif stage == "concepts":
        (out / "review.md").write_text(ask["review_sheet"], encoding="utf-8")
        (out / "decisions.json").write_text(json.dumps(ask["decisions_template"], indent=2),
                                            encoding="utf-8")  # fmt: skip
        (out / "spec.json").write_text(json.dumps(ask["spec_template"], indent=2), "utf-8")
        tmpl, files = None, ["review.md", "decisions.json", "spec.json"]
    elif stage == "gate":
        (out / "gate.md").write_text(ask["gate"] + "\n" + ask["attrition"], encoding="utf-8")
        tmpl = {"proceed": False, "reviewer": "", "note": ""}
        files = ["gate.md", "gate.json"]
    else:
        (out / "report_draft.md").write_text(ask["report"], encoding="utf-8")
        tmpl = {"approve": False, "reviewer": "", "note": ""}
        files = ["report_draft.md", "signoff.json"]
    if tmpl is not None:
        (out / f"{stage}.json").write_text(json.dumps(tmpl, indent=2), encoding="utf-8")
    (out / "pending.txt").write_text(stage, encoding="utf-8")
    typer.echo(f"waiting for the {stage} review: {ask['ask']}\nfiles in {out}: "
               f"{', '.join(files)}\nthen run: agentic-tte replay resume --out {out}",
               err=True)  # fmt: skip


@replay_app.command("start")
def replay_start(
    out: Path = typer.Option(..., "--out", "-o", help="Run directory (checkpoints, reviews)."),
    db: Path = typer.Option(..., help="OMOP CDM database (DuckDB file)."),
    classification: str = typer.Option(..., help="synthetic, deidentified, limited, identified"),
    nct: str = typer.Option(None, help="Draft from ClinicalTrials.gov (needs the LLM)."),
    protocol: Path = typer.Option(None, help="Start from an existing protocol instead."),
    approved: bool = typer.Option(False, help="Storage/compute approved for this classification."),
    design_hr: float = typer.Option(None, help="Hazard ratio to power the feasibility check."),
    vocab: Path = typer.Option(None, help="Athena vocabulary download."),
    auto: Path = typer.Option(None, help="Auto-approve policy (JSON) for benchmarking: its "
                                         "stages are answered without a person, and the "
                                         "report says so."),
    abbreviations: Path = typer.Option(None, help="Abbreviation list; only its approved "
                                                  "entries are expanded in concept search."),
):
    """Start a Replay. It stops at each clinician review; continue with `replay resume`."""
    if (nct is None) == (protocol is None):
        typer.echo("give exactly one of --nct or --protocol", err=True)
        raise typer.Exit(2)
    policy = None
    if auto is not None:
        from agentic_tte.graph import AutoApprove

        policy = AutoApprove.model_validate_json(auto.read_text(encoding="utf-8"))
        typer.echo(f"AUTO-APPROVE for {', '.join(sorted(policy.stages))}: no human review of "
                   "those stages; the report will say so", err=True)  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoints.sqlite").exists():
        typer.echo(f"{out} already holds a run: use `replay resume` or a new directory", err=True)
        raise typer.Exit(2)
    run = {"out": str(out), "db": str(db), "vocab": str(vocab) if vocab else None,
           "classification": classification, "approved": approved, "design_hr": design_hr,
           "auto": policy.model_dump(mode="json") if policy else None,
           "abbreviations": str(abbreviations) if abbreviations else None}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    llm = None
    state: dict = {}
    if protocol is not None:
        state["protocol"] = json.loads(protocol.read_text(encoding="utf-8"))
    else:
        from agentic_tte.llm import make_client

        llm, state["nct_id"] = make_client(get_settings()), nct
    graph, cfg = _replay_graph(run, llm)
    graph.invoke(state, cfg)
    _replay_pause(graph, cfg, out)


@replay_app.command("resume")
def replay_resume(
    out: Path = typer.Option(..., "--out", "-o", help="Run directory from `replay start`."),
    answer: Path = typer.Option(None, help="Answer file; by default the review's own files."),
):
    """Continue a Replay with the reviewer's answer to the open review."""
    from langgraph.types import Command

    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    stage = (out / "pending.txt").read_text(encoding="utf-8").strip()
    if answer is not None:
        value = json.loads(answer.read_text(encoding="utf-8"))
    elif stage == "concepts":
        value = {"decisions": json.loads((out / "decisions.json").read_text(encoding="utf-8")),
                 "spec": json.loads((out / "spec.json").read_text(encoding="utf-8"))}
    else:
        value = json.loads((out / f"{stage}.json").read_text(encoding="utf-8"))
    who = value.get("reviewer") if stage != "concepts" else value["decisions"].get("reviewer")
    if not (who or "").strip():
        typer.echo(f"the {stage} answer needs a reviewer", err=True)
        raise typer.Exit(2)
    graph, cfg = _replay_graph(run)
    graph.invoke(Command(resume=value), cfg)
    (out / "pending.txt").unlink()
    _replay_pause(graph, cfg, out)


@app.command()
def gate(
    protocol: Path,
    db: Path = typer.Option(..., help="OMOP CDM database (DuckDB file)."),
    book: Path = typer.Option(None, help="Reviewed concept-set book (approved sets are used)."),
    classification: str = typer.Option(..., help="synthetic, deidentified, limited, identified"),
    approved: bool = typer.Option(False, help="Storage/compute approved for this classification."),
    vocab: Path = typer.Option(None, help="Athena vocabulary download."),
    compiled: Path = typer.Option(None, help="compile.json from `agentic-tte compile`: runs "
                                             "the feasibility check."),
    design_hr: float = typer.Option(None, help="Hazard ratio to power the feasibility check."),
):
    """Run the Data Support Gate and print its report. Exits 1 if the run is blocked."""
    from agentic_tte.omop import DataContext, run_gate
    from agentic_tte.omop.attrition import attrition
    from agentic_tte.omop.compiler import CompiledCohort
    from agentic_tte.omop.concept_sets import ConceptSetBook
    from agentic_tte.protocol import TargetTrialProtocol

    p = TargetTrialProtocol.model_validate_json(protocol.read_text(encoding="utf-8"))
    sets = None
    if book is not None:
        sets = ConceptSetBook.model_validate_json(book.read_text(encoding="utf-8")).gate_sets()
    con = _open_db(db, vocab)
    rep = None
    if compiled is not None:
        cc = CompiledCohort.model_validate_json(compiled.read_text(encoding="utf-8"))
        if cc.protocol_hash != p.content_hash():
            typer.echo("compile.json was compiled from a different protocol version", err=True)
            raise typer.Exit(2)
        rep = attrition(con, cc)
    report = run_gate(con, p, concept_sets=sets, attrition=rep, design_hr=design_hr,
                      context=DataContext(data_classification=classification,
                                          approved_by_rcss=approved))  # fmt: skip
    typer.echo(report.to_markdown())
    if report.blocks_run:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
