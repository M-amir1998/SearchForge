"""``searchforge`` command-line interface (Typer). CLI objects never leak into services."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any

import psutil
import typer
from rich.console import Console
from rich.table import Table

from searchforge import __version__
from searchforge.errors import QualityGateFailure, SearchForgeError

app = typer.Typer(
    name="searchforge",
    help="Retrieval engineering platform: index, search, evaluate.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
corpus_app = typer.Typer(help="Inspect and prepare corpora.", no_args_is_help=True)
index_app = typer.Typer(help="Build, update, inspect and validate index versions.", no_args_is_help=True)
storage_app = typer.Typer(help="Disk usage and safe cleanup.", no_args_is_help=True)
hn_app = typer.Typer(help="Hard-negative mining.", no_args_is_help=True)
incident_app = typer.Typer(help="Turn retrieval incidents into regression cases.", no_args_is_help=True)
review_app = typer.Typer(help="Blinded human relevance review.", no_args_is_help=True)
qrels_app = typer.Typer(help="Relevance judgment utilities.", no_args_is_help=True)
cache_app = typer.Typer(help="Embedding cache management.", no_args_is_help=True)
datasets_app = typer.Typer(help="Public evaluation datasets (explicit download only).", no_args_is_help=True)
demo_app = typer.Typer(help="CPU-only end-to-end demonstrations.", invoke_without_command=True)
for sub, name in (
    (corpus_app, "corpus"),
    (index_app, "index"),
    (storage_app, "storage"),
    (hn_app, "hard-negatives"),
    (incident_app, "incident"),
    (review_app, "review"),
    (qrels_app, "qrels"),
    (cache_app, "cache"),
    (datasets_app, "datasets"),
    (demo_app, "demo"),
):
    app.add_typer(sub, name=name)

console = Console()
ConfigArg = Annotated[Path, typer.Argument(help="SearchForge YAML config", exists=True, dir_okay=False)]
ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="SearchForge YAML config", exists=True, dir_okay=False)]


def _fail(exc: Exception, code: int = 2) -> None:
    console.print(f"[bold red]error:[/] {type(exc).__name__}: {exc}")
    raise typer.Exit(code)


def _load(path: Path) -> Any:
    from searchforge.config.loader import load_config

    return load_config(path)


def _service(path: Path, index_ref: str | None = None) -> tuple[Any, Any, Any]:
    from searchforge.indexes.handle import open_index
    from searchforge.service.search_service import SearchService

    loaded = _load(path)
    handle = open_index(loaded.workspace, index_ref or loaded.config.index.name, loaded.config)
    return loaded, handle, SearchService(handle, loaded.config)


def _mb(n: float) -> str:
    return f"{n / 1024 / 1024:,.2f} MB"


# ----------------------------------------------------------------------------- doctor
@app.command()
def doctor() -> None:
    """Report environment readiness (CPU-only is the normal, supported setup)."""
    import sqlite3

    from searchforge.indexes.sqlite_util import fts5_available

    def has(mod: str) -> str:
        import importlib.util

        return "installed" if importlib.util.find_spec(mod) else "not installed (optional)"

    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    rows = [
        ("SearchForge", __version__),
        ("Python", f"{platform.python_version()} ({sys.executable})"),
        ("Platform", platform.platform(terse=True)),
        ("CPU", f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count(logical=True)} threads"),
        ("RAM", f"{vm.total / 1024**3:.1f} GB total, {vm.available / 1024**3:.1f} GB available"),
        ("Disk (cwd)", f"{disk.free / 1024**3:.1f} GB free of {disk.total / 1024**3:.1f} GB"),
        ("SQLite", f"{sqlite3.sqlite_version}, FTS5 {'available' if fts5_available() else 'MISSING'}"),
        ("GPU", "not required - CPU-only operation is the default"),
        ("sentence-transformers", has("sentence_transformers")),
        ("qdrant-client", has("qdrant_client")),
        ("opensearch-py", has("opensearchpy")),
        ("opentelemetry", has("opentelemetry")),
        ("Docker", shutil.which("docker") or "not found (optional)"),
    ]
    table = Table(title="searchforge doctor", show_header=False)
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)
    if not fts5_available():
        raise typer.Exit(1)


# ----------------------------------------------------------------------------- corpus
@corpus_app.command("inspect")
def corpus_inspect(config: ConfigArg, sample: int = 3) -> None:
    """Stream the corpus and report counts, formats, attributes and warnings."""
    from collections import Counter

    from searchforge.ingestion.readers import iter_corpus
    from searchforge.ingestion.safety import IngestionReport

    loaded = _load(config)
    report = IngestionReport()
    formats: Counter[str] = Counter()
    attrs: Counter[str] = Counter()
    chars = 0
    examples: list[tuple[str, str | None, int]] = []
    for doc in iter_corpus(loaded, report):
        formats[doc.source.format.value] += 1
        attrs.update(doc.metadata.attributes.keys())
        chars += len(doc.text)
        if len(examples) < sample:
            examples.append((doc.document_id, doc.metadata.title, len(doc.text)))
    console.print(
        f"corpus [bold]{loaded.config.corpus.id}[/]: {report.documents_emitted} documents, "
        f"{chars:,} characters, {report.bytes_read:,} bytes read"
    )
    console.print(f"formats: {dict(formats)}  attributes: {dict(attrs.most_common(12))}")
    console.print(
        f"skipped files: {report.files_skipped}, skipped records: {report.records_skipped}, "
        f"duplicate ids: {report.duplicate_ids}"
    )
    for w in report.warnings[:10]:
        console.print(f"  [yellow]warning[/] {w}")
    for e in examples:
        console.print(f"  {e[0]}  {e[1]!r}  ({e[2]} chars)")


@corpus_app.command("prepare")
def corpus_prepare(config: ConfigArg) -> None:
    """Chunk the corpus and write a Parquet export to <workspace>/prepared/ (streaming)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from searchforge.chunking.chunkers import build_chunker
    from searchforge.ingestion.readers import iter_corpus

    loaded = _load(config)
    chunker = build_chunker(loaded.config.chunking)
    out = loaded.workspace / "prepared" / loaded.config.corpus.id / f"chunks-{chunker.fingerprint()}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("chunk_id", pa.string()),
            ("document_id", pa.string()),
            ("sequence", pa.int32()),
            ("start_char", pa.int64()),
            ("end_char", pa.int64()),
            ("section", pa.string()),
            ("text", pa.string()),
        ]
    )
    n = 0
    with pq.ParquetWriter(out, schema) as writer:
        batch: list[dict[str, Any]] = []
        for doc in iter_corpus(loaded):
            for c in chunker.chunk(doc):
                batch.append(
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "sequence": c.sequence,
                        "start_char": c.metadata.start_char,
                        "end_char": c.metadata.end_char,
                        "section": c.metadata.section,
                        "text": c.text,
                    }
                )
                if len(batch) >= 2048:
                    writer.write_table(pa.Table.from_pylist(batch, schema))
                    n += len(batch)
                    batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema))
            n += len(batch)
    console.print(f"wrote {n} chunks ({chunker.name}) -> {out}")


# ----------------------------------------------------------------------------- index
@index_app.command("build")
def index_build(config: ConfigArg, force: bool = typer.Option(False, help="rebuild even if identical")) -> None:
    """Full build into a new immutable version (reused when the build fingerprint is unchanged)."""
    from searchforge.indexes.builder import IndexBuilder

    try:
        res = IndexBuilder(_load(config)).build(force=force)
    except SearchForgeError as exc:
        _fail(exc)
        return
    m = res.manifest
    state = "reused existing" if res.reused else "built"
    console.print(
        f"{state} [bold]{m.index_id}[/]: {m.document_count} documents, {m.chunk_count} chunks, "
        f"{_mb(m.size_bytes())}, {m.build_seconds}s, peak RSS {m.peak_rss_mb} MB"
    )
    for w in res.report.warnings[:10]:
        console.print(f"  [yellow]warning[/] {w}")


@index_app.command("update")
def index_update(
    config: ConfigArg,
    keep_missing: bool = typer.Option(False, help="do not delete docs absent from the corpus"),
    delete: Annotated[list[str] | None, typer.Option(help="document id to delete")] = None,
) -> None:
    """Incremental update: only new/changed/deleted documents are processed (new version)."""
    from searchforge.indexes.builder import IndexBuilder

    try:
        res = IndexBuilder(_load(config)).update(delete_missing=not keep_missing, delete_ids=delete or [])
    except SearchForgeError as exc:
        _fail(exc)
        return
    if res.reused:
        console.print(f"no changes; CURRENT stays {res.manifest.index_id}")
    else:
        console.print(
            f"published {res.manifest.index_id} (parent {res.manifest.parent_version}): {res.manifest.update_summary}"
        )


def _workspace_opt(workspace: Path | None, config: Path | None) -> Path:
    if workspace is not None:
        return workspace.resolve()
    if config is not None:
        return Path(_load(config).workspace)
    return (Path.cwd() / ".searchforge").resolve()


@index_app.command("list")
def index_list(workspace: Path | None = None, config: Path | None = typer.Option(None, "--config", "-c")) -> None:
    """List index names and versions in a workspace."""
    from searchforge.indexes.lifecycle import IndexLayout

    layout = IndexLayout(_workspace_opt(workspace, config))
    table = Table("index", "version", "current", "docs", "chunks", "size", "created")
    for name in layout.names():
        cur = layout.current_version(name)
        for m in layout.versions(name):
            table.add_row(
                name,
                m.version,
                "*" if m.version == cur else "",
                str(m.document_count),
                str(m.chunk_count),
                _mb(m.size_bytes()),
                m.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
    console.print(table)


@index_app.command("inspect")
def index_inspect(
    index: str, workspace: Path | None = None, config: Path | None = typer.Option(None, "--config", "-c")
) -> None:
    """Print the manifest of INDEX (``name`` for CURRENT or ``name@version``)."""
    from searchforge.indexes.lifecycle import IndexLayout

    try:
        manifest, _ = IndexLayout(_workspace_opt(workspace, config)).resolve(index)
    except SearchForgeError as exc:
        _fail(exc)
        return
    data = manifest.model_dump(mode="json")
    data["size_by_component"] = manifest.size_by_component()
    data.pop("files", None)
    console.print_json(json.dumps(data, default=str))


@index_app.command("validate")
def index_validate(
    index: str, workspace: Path | None = None, config: Path | None = typer.Option(None, "--config", "-c")
) -> None:
    """Verify file checksums and counts against the manifest."""
    from searchforge.indexes.lifecycle import IndexLayout, validate_index_dir

    try:
        manifest, vdir = IndexLayout(_workspace_opt(workspace, config)).resolve(index)
        problems = validate_index_dir(vdir)
    except SearchForgeError as exc:
        _fail(exc)
        return
    if problems:
        for p in problems:
            console.print(f"[red]FAIL[/] {p}")
        raise typer.Exit(1)
    console.print(f"[green]OK[/] {manifest.index_id} is valid ({len(manifest.files)} files verified)")


# ----------------------------------------------------------------------------- search
def _parse_filters(filters: str | None, filter_kv: list[str] | None) -> dict[str, Any] | None:
    out: dict[str, Any] = json.loads(filters) if filters else {}
    for item in filter_kv or []:
        key, _, value = item.partition("=")
        out[key.strip()] = value.strip()
    return out or None


@app.command()
def search(
    query: str,
    config: ConfigOpt = Path("configs/evaluations/fixture.yaml"),
    profile: str | None = typer.Option(None, "--profile", "-p"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
    filters: str | None = typer.Option(None, help='JSON filters, e.g. \'{"year": {"gte": 2025}}\''),
    filter_kv: Annotated[list[str] | None, typer.Option("--filter", "-f", help="field=value equality")] = None,
    explain: bool = False,
    as_json: bool = typer.Option(False, "--json"),
    index: str | None = None,
) -> None:
    """Search an index and print ranked, explainable results."""
    try:
        _, handle, service = _service(config, index)
    except SearchForgeError as exc:
        _fail(exc)
        return
    try:
        resp = service.search(
            query, top_k=top_k, filters=_parse_filters(filters, filter_kv), profile=profile, explain=explain
        )
    except SearchForgeError as exc:
        _fail(exc)
        return
    finally:
        service.close()
        handle.close()
    if as_json:
        print(
            json.dumps(
                {"results": [r.as_dict() for r in resp.results], "trace": resp.trace.as_dict()}, default=str, indent=2
            )
        )
        return
    d = resp.trace.planner_decision
    console.print(
        f"[bold]{resp.index_id}[/] profile={resp.trace.profile} -> {resp.profile} "
        f"(class {d.get('query_class')}) {resp.latency_ms:.2f} ms"
        + (f" [yellow]DEGRADED {resp.degradation_reasons}[/]" if resp.degraded else "")
    )
    table = Table("#", "document", "score", "sources", "title / section", "excerpt")
    for r in resp.results:
        head = " > ".join(p for p in (r.metadata.get("title"), r.metadata.get("section")) if p)
        table.add_row(
            str(r.rank),
            r.document_id,
            f"{r.score:.4f}",
            "+".join(r.retrieval_sources),
            head[:50],
            (r.text or "").replace("\n", " ")[:80],
        )
    console.print(table)
    if explain:
        console.print_json(
            json.dumps(
                {"trace": resp.trace.as_dict(), "signals": [r.explanation for r in resp.results[:3]]}, default=str
            )
        )


# ----------------------------------------------------------------------------- evaluate / compare
def _parse_splits(split: list[str] | None) -> list[Any] | None:
    from searchforge.domain.judgment import QuerySplit

    if not split:
        return None
    try:
        return [QuerySplit(s) for s in split]
    except ValueError as exc:
        raise typer.BadParameter("--split must be one of: train, validation, test") from exc


def _eval_inputs(loaded: Any) -> tuple[Any, Any, Any]:
    from searchforge.evaluation.datasets import load_qrels, load_queries

    ev = loaded.config.evaluation
    if ev is None:
        raise typer.BadParameter("config has no evaluation section")
    return ev, load_queries(loaded.resolve(ev.queries)), load_qrels(loaded.resolve(ev.qrels))


@app.command()
def evaluate(
    config: ConfigArg,
    profile: Annotated[list[str] | None, typer.Option("--profile", "-p")] = None,
    split: Annotated[list[str] | None, typer.Option(help="train/validation/test (default: config)")] = None,
    out: Path = typer.Option(Path("results"), help="results root directory"),
    repeats: int | None = None,
) -> None:
    """Evaluate profiles against qrels and write results/<run-id>/ for each."""
    from searchforge.evaluation.failures import analyze_failures
    from searchforge.evaluation.runner import evaluate as run_eval
    from searchforge.evaluation.runner import select_queries
    from searchforge.evaluation.store import write_run

    split_arg = _parse_splits(split)
    try:
        loaded, handle, service = _service(config)
    except SearchForgeError as exc:
        _fail(exc)
        return
    try:
        ev, queries, qrels = _eval_inputs(loaded)
        splits = split_arg or ev.splits
        selected = select_queries(queries, splits)
        table = Table("profile", "queries", "nDCG@10", "Recall@20", "MRR", "MAP", "p50 ms", "p95 ms", "run dir")
        for prof in profile or ev.profiles:
            run = run_eval(
                service,
                selected,
                qrels,
                profile=prof,
                k_values=ev.k_values,
                repeats=repeats or ev.latency_repeats,
                threshold=ev.relevance_threshold,
            )
            failures = analyze_failures(
                run, qrels, handle.chunk_store, handle.filter_schema, threshold=ev.relevance_threshold
            )
            run_dir = write_run(
                run, out, config=loaded.raw_dict(), manifest=handle.manifest.model_dump(mode="json"), failures=failures
            )
            agg, lat = run.aggregate(), run.latency()
            table.add_row(
                prof,
                str(len(run.scored)),
                f"{agg.get('ndcg@10', 0):.3f}",
                f"{agg.get('recall@20', 0):.3f}",
                f"{agg['mrr']:.3f}",
                f"{agg['map']:.3f}",
                f"{lat['p50_ms']:.2f}",
                f"{lat['p95_ms']:.2f}",
                str(run_dir),
            )
        console.print(table)
    except SearchForgeError as exc:
        _fail(exc)
    finally:
        service.close()
        handle.close()


@app.command()
def compare(
    baseline: str,
    candidate: str,
    config: ConfigOpt = Path("configs/evaluations/fixture.yaml"),
    candidate_config: Path | None = typer.Option(None, help="evaluate the candidate profile on another config/index"),
    split: Annotated[list[str] | None, typer.Option()] = None,
    gates: bool = typer.Option(True, help="enforce evaluation.gates (exit 1 on failure)"),
    out: Path | None = typer.Option(None, help="write comparison JSON here"),
) -> None:
    """Per-query regression analysis BASELINE vs CANDIDATE profile, with quality gates."""
    from searchforge.evaluation.regression import compare_runs, evaluate_gates
    from searchforge.evaluation.runner import evaluate as run_eval
    from searchforge.evaluation.runner import select_queries

    split_arg = _parse_splits(split)
    try:
        loaded, handle, service = _service(config)
        _cand_loaded, cand_handle, cand_service = (
            _service(candidate_config) if candidate_config else (loaded, handle, service)
        )
    except SearchForgeError as exc:
        _fail(exc)
        return
    try:
        ev, queries, qrels = _eval_inputs(loaded)
        splits = split_arg or ev.splits
        selected = select_queries(queries, splits)
        base_run = run_eval(
            service,
            selected,
            qrels,
            profile=baseline,
            k_values=ev.k_values,
            repeats=ev.latency_repeats,
            threshold=ev.relevance_threshold,
        )
        cand_run = run_eval(
            cand_service,
            selected,
            qrels,
            profile=candidate,
            name=f"candidate:{candidate}",
            k_values=ev.k_values,
            repeats=ev.latency_repeats,
            threshold=ev.relevance_threshold,
        )
        relevant = {q.query_id: sorted(qrels.relevant(q.query_id, ev.relevance_threshold)) for q in selected}
        comp = compare_runs(base_run, cand_run, relevant_by_query=relevant, epsilon=ev.gates.regression_epsilon)
        sm = comp.summary
        console.print(
            f"[bold]{baseline}[/] vs [bold]{candidate}[/] on {sm['queries']} queries: "
            f"{sm['improved']} improved, {sm['regressed']} regressed, {sm['unchanged']} unchanged"
        )
        console.print(
            {k: round(v, 4) for k, v in sm["mean_deltas"].items() if k in ("ndcg@10", "recall@20", "mrr", "map")}
        )
        console.print(
            f"latency delta p50 {sm['latency']['p50_delta_ms']:+.2f} ms, p95 {sm['latency']['p95_delta_ms']:+.2f} ms"
        )
        for d in comp.regressions[:10]:
            console.print(
                f"  [red]v[/] {d.query_id} ({d.query_type}, {d.severity}) {d.baseline:.3f} -> "
                f"{d.candidate:.3f}; expected {d.relevant}; old top {d.baseline_top[:3]} new top "
                f"{d.candidate_top[:3]}"
            )
        report = evaluate_gates(comp, cand_run, ev.gates)
        for c in report.checks:
            mark = "[green]PASS[/]" if c.passed else "[red]FAIL[/]"
            console.print(f"  gate {c.name}: {mark} observed={c.observed} threshold={c.threshold}")
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps({"summary": sm, "gates": report.as_dict()}, indent=2, default=str), encoding="utf-8"
            )
        if gates and not report.passed:
            raise QualityGateFailure("quality gate failed")
    except QualityGateFailure as exc:
        _fail(exc, code=1)
    except SearchForgeError as exc:
        _fail(exc)
    finally:
        pairs = [(service, handle)] + ([(cand_service, cand_handle)] if cand_service is not service else [])
        for svc, hdl in pairs:
            svc.close()
            hdl.close()


@app.command()
def tune(
    config: ConfigArg, base_profile: str = "hybrid", strategy: str = "weighted_rrf", out: Path | None = None
) -> None:
    """Tune fusion weights on train, select on validation, report once on test."""
    from searchforge.config.models import FusionName
    from searchforge.evaluation.tuning import tune_fusion_weights

    loaded, handle, service = _service(config)
    try:
        ev, queries, qrels = _eval_inputs(loaded)
        res = tune_fusion_weights(
            service, queries, qrels, base_profile=base_profile, strategy=FusionName(strategy), k_values=ev.k_values
        )
    except SearchForgeError as exc:
        _fail(exc)
        return
    finally:
        service.close()
        handle.close()
    console.print(
        f"best weights {res.best_weights} (validation {res.metric}={res.validation_score:.4f}); "
        f"TEST {res.metric}: tuned {res.test_score:.4f} vs {base_profile} {res.baseline_test_score:.4f}"
    )
    for n in res.notes:
        console.print(f"  {n}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metric": res.metric,
            "strategy": res.strategy,
            "grid": res.grid,
            "best_weights": res.best_weights,
            "validation": res.validation_score,
            "test": res.test_score,
            "baseline_test": res.baseline_test_score,
            "notes": res.notes,
        }
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ----------------------------------------------------------------------------- hard negatives
@hn_app.command("build")
def hard_negatives_build(
    config: ConfigArg,
    profile: str = "hybrid",
    out: Path = Path("results/hard_negatives.jsonl"),
    max_rank: int = 10,
    per_query: int = 5,
) -> None:
    """Mine high-ranked non-relevant results into a hard-negative set (JSONL or Parquet)."""
    from searchforge.evaluation.runner import evaluate as run_eval
    from searchforge.hard_negatives.mining import mine_hard_negatives, write_hard_negatives

    loaded, handle, service = _service(config)
    try:
        ev, queries, qrels = _eval_inputs(loaded)
        run = run_eval(service, queries, qrels, profile=profile, k_values=ev.k_values, threshold=ev.relevance_threshold)
        rows = mine_hard_negatives(run, qrels, max_rank=max_rank, per_query=per_query, threshold=ev.relevance_threshold)
        write_hard_negatives(rows, out)
    finally:
        service.close()
        handle.close()
    judged = sum(r.judged for r in rows)
    console.print(f"{len(rows)} hard negatives ({judged} judged non-relevant, {len(rows) - judged} unjudged) -> {out}")


# ----------------------------------------------------------------------------- benchmark / report
@app.command()
def benchmark(
    spec: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: Path = typer.Option(Path("results/benchmark")),
    workspace: Path | None = None,
) -> None:
    """Run an experiment matrix (quality, latency, throughput, memory, index size) and write a report."""
    from searchforge.benchmarking.runner import run_benchmark
    from searchforge.reporting.report import write_benchmark_report

    try:
        result = run_benchmark(spec, workspace=workspace)
    except SearchForgeError as exc:
        _fail(exc)
        return
    path = write_benchmark_report(result, out)
    table = Table("system", "nDCG@10", "Recall@20", "MRR", "p95 ms", "QPS")
    for s in result.systems:
        r = s.row()
        table.add_row(
            s.spec.name,
            f"{r['ndcg@10']:.3f}",
            f"{r['recall@20']:.3f}",
            f"{r['mrr']:.3f}",
            f"{r['p95_ms']:.2f}",
            f"{r['qps']:.1f}" if r["qps"] else "-",
        )
    console.print(table)
    console.print(f"report -> {path}")


@app.command()
def report(run: Annotated[Path, typer.Argument(help="results/<run-id> or a benchmark output dir")]) -> None:
    """Show (or render) the report for an evaluation run or benchmark directory."""
    md = run / "report.md"
    metrics = run / "metrics.json"
    if not md.exists():
        if not metrics.exists():
            _fail(FileNotFoundError(f"no report.md or metrics.json in {run}"))
        data = json.loads(metrics.read_text(encoding="utf-8"))
        lines = [
            f"# Evaluation run {data['run_id']}",
            "",
            f"- profile: `{data['profile']}`",
            f"- index: `{data['index_id']}`",
            f"- queries: {data['queries']} (excluded {data['excluded']})",
            "",
            "| metric | value |",
            "|---|---|",
        ]
        lines += [f"| {k} | {v:.4f} |" for k, v in sorted(data["metrics"].items())]
        lines += ["", "| latency | ms |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in data["latency"].items() if k != "samples"]
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(md.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- serve
@app.command()
def serve(config: ConfigArg, host: str = "127.0.0.1", port: int = 8080, index: str | None = None) -> None:
    """Serve the search API (FastAPI + Uvicorn). Binds to localhost by default."""
    import uvicorn

    from searchforge.api.app import create_app

    _, _, service = _service(config, index)
    uvicorn.run(create_app(service), host=host, port=port, log_level="info", access_log=False)


@app.command("serve-fixture")
def serve_fixture(host: str = "0.0.0.0", port: int = 8080) -> None:  # noqa: S104 - container entrypoint
    """Build (or reuse) the bundled fixture index in ./.searchforge and serve it (Docker default)."""
    import uvicorn

    from searchforge.api.app import create_app
    from searchforge.config.loader import load_config
    from searchforge.datasets import fixture_config_path
    from searchforge.indexes.builder import IndexBuilder
    from searchforge.indexes.handle import open_index
    from searchforge.service.search_service import SearchService

    loaded = load_config(fixture_config_path(), overrides={"storage": {"reserve_free_gb": 0.5}})
    built = IndexBuilder(loaded).build()
    console.print(f"serving {built.manifest.index_id} on {host}:{port}")
    handle = open_index(loaded.workspace, loaded.config.index.name, loaded.config)
    uvicorn.run(create_app(SearchService(handle, loaded.config)), host=host, port=port, access_log=False)


# ----------------------------------------------------------------------------- storage
@storage_app.command("status")
def storage_status(
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = None,
    results: Path = Path("results"),
) -> None:
    """Disk usage of sources, indexes, embedding cache, results and free disk."""
    from searchforge.storage.manage import storage_status as status

    ws = _workspace_opt(workspace, config)
    sources: dict[str, Path] = {}
    if config is not None:
        loaded = _load(config)
        sources = {s.id: loaded.resolve(s.path) for s in loaded.config.corpus.sources}
    st = status(ws, sources, results.resolve())
    table = Table("component", "size")
    for k, v in st.sources.items():
        table.add_row(f"source data: {k}", _mb(v))
    for k, v in st.indexes.items():
        table.add_row(f"index: {k}", _mb(v))
    table.add_row("embedding cache", _mb(st.embedding_cache))
    table.add_row("prepared exports", _mb(st.prepared))
    table.add_row("partial builds", _mb(st.partial_builds))
    table.add_row("results", _mb(st.results))
    table.add_row("free disk", f"{st.free_disk / 1024**3:.1f} GB")
    console.print(f"workspace: {ws}")
    console.print(table)


@storage_app.command("clean")
def storage_clean(
    dry_run: bool = typer.Option(True, "--dry-run/--yes", help="default: only list what would be removed"),
    include_old_versions: bool = typer.Option(False, help="also remove non-CURRENT index versions"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = None,
) -> None:
    """Remove project-owned disposable artifacts in the workspace (never source data)."""
    from searchforge.storage.manage import execute_cleanup, plan_cleanup

    ws = _workspace_opt(workspace, config)
    plan = plan_cleanup(ws, include_old_versions=include_old_versions)
    if not plan.targets:
        console.print("nothing to clean")
        return
    for path, size, reason in plan.targets:
        console.print(f"  {'would remove' if dry_run else 'removing'} {path} ({_mb(size)}): {reason}")
    if dry_run:
        console.print(f"dry run: {_mb(plan.bytes)} reclaimable; re-run with --yes to delete")
        return
    console.print(f"freed {_mb(execute_cleanup(ws, plan))}")


# ----------------------------------------------------------------------------- cache
@cache_app.command("stats")
def cache_stats(config: Path | None = typer.Option(None, "--config", "-c"), workspace: Path | None = None) -> None:
    from searchforge.embeddings.cache import EmbeddingCache

    path = _workspace_opt(workspace, config) / "cache" / "embeddings"
    if not (path / EmbeddingCache.FILENAME).exists():
        console.print("no embedding cache")
        return
    cache = EmbeddingCache(path)
    st = cache.stats()
    cache.close()
    console.print(f"embedding cache {path}: {st.entries} vectors, {_mb(st.bytes)}")


@cache_app.command("clear")
def cache_clear(config: Path | None = typer.Option(None, "--config", "-c"), workspace: Path | None = None) -> None:
    from searchforge.embeddings.cache import EmbeddingCache

    path = _workspace_opt(workspace, config) / "cache" / "embeddings"
    if not (path / EmbeddingCache.FILENAME).exists():
        console.print("no embedding cache")
        return
    cache = EmbeddingCache(path)
    n = cache.clear()
    cache.close()
    console.print(f"removed {n} cached vectors")


# ----------------------------------------------------------------------------- incidents / review / qrels
@incident_app.command("add")
def incident_add(
    query: str,
    expected: Annotated[list[str], typer.Option("--expected", "-e", help="expected document id")],
    config: ConfigOpt = Path("configs/evaluations/fixture.yaml"),
    queries_out: Path = typer.Option(Path("evaluation/incident_queries.jsonl")),
    qrels_out: Path = typer.Option(Path("evaluation/incident_qrels.jsonl")),
    severity: str = "important",
    ref: str = "",
) -> None:
    """Sanitize a failed query and record it (with judgments) as a new regression case."""
    from searchforge.domain.judgment import QuerySeverity
    from searchforge.evaluation.incidents import IncidentCase, append_incident

    _loaded, handle, service = _service(config)
    try:
        known = set(handle.chunk_store.document_hashes())
        q = append_incident(
            IncidentCase(query, expected, QuerySeverity(severity), incident_ref=ref), queries_out, qrels_out, known
        )
    except SearchForgeError as exc:
        _fail(exc)
        return
    finally:
        service.close()
        handle.close()
    console.print(f"recorded {q.query_id}: {q.text!r} -> {expected} ({queries_out}, {qrels_out})")


@review_app.command("export")
def review_export(
    config: ConfigArg,
    out: Path = Path("evaluation/review.csv"),
    profile: Annotated[list[str] | None, typer.Option("--profile", "-p")] = None,
    depth: int = 10,
) -> None:
    """Export a blinded, shuffled review pool (system identity removed)."""
    from searchforge.evaluation.review import export_review_pool
    from searchforge.evaluation.runner import evaluate as run_eval

    loaded, handle, service = _service(config)
    try:
        _, queries, qrels = _eval_inputs(loaded)
        runs = [run_eval(service, queries, qrels, profile=p) for p in (profile or ["lexical", "dense", "hybrid"])]
        n = export_review_pool(runs, queries, handle.chunk_store, out, depth=depth, skip_judged=qrels)
    finally:
        service.close()
        handle.close()
    console.print(f"{n} unjudged (query, document) pairs -> {out}; grade 0-3 in the 'relevance' column")


@review_app.command("import")
def review_import(path: Path, out: Path = Path("evaluation/reviewed_qrels.jsonl")) -> None:
    from searchforge.evaluation.datasets import save_qrels
    from searchforge.evaluation.review import import_review

    try:
        qrels = import_review(path)
    except SearchForgeError as exc:
        _fail(exc)
        return
    save_qrels(qrels, out)
    console.print(f"{len(qrels)} judgments -> {out}")


@qrels_app.command("convert")
def qrels_convert(src: Path, dst: Path) -> None:
    """Convert qrels between .jsonl, .tsv (TREC) and .parquet."""
    from searchforge.evaluation.datasets import load_qrels, save_qrels

    qrels = load_qrels(src)
    save_qrels(qrels, dst)
    console.print(f"{len(qrels)} judgments for {len(qrels.query_ids())} queries -> {dst}")


# ----------------------------------------------------------------------------- datasets
@datasets_app.command("fetch-beir")
def datasets_fetch_beir(
    name: str = typer.Argument("scifact"), out: Path = Path("data/beir"), max_mb: int = 200
) -> None:
    """Download a BEIR dataset zip (explicit opt-in; never used by tests or CI)."""
    from searchforge.evaluation.beir import fetch_beir

    try:
        path = fetch_beir(name, out, max_bytes=max_mb * 1024 * 1024)
    except SearchForgeError as exc:
        _fail(exc)
        return
    console.print(f"{name} -> {path} (licence terms of the dataset apply; do not commit it)")


# ----------------------------------------------------------------------------- demo
@demo_app.callback()
def demo(
    ctx: typer.Context,
    out: Path = typer.Option(Path("results/local-demo")),
    workspace: Path = typer.Option(Path(".searchforge-demo")),
) -> None:
    """Full CPU E2E: fixture corpus -> indexes -> hybrid -> rerank -> evaluate -> compare -> report."""
    if ctx.invoked_subcommand is not None:
        ctx.obj = {"out": out, "workspace": workspace}
        return
    from searchforge.demo import run_demo

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    summary = run_demo(out.resolve(), workspace.resolve())
    table = Table("system", "nDCG@10", "Recall@20", "MRR", "p95 ms")
    for s in summary.benchmark.systems:
        if s.spec.group == "core":
            r = s.row()
            table.add_row(
                s.spec.name, f"{r['ndcg@10']:.3f}", f"{r['recall@20']:.3f}", f"{r['mrr']:.3f}", f"{r['p95_ms']:.2f}"
            )
    console.print("[bold]LOCAL DEMONSTRATION RESULTS[/] (synthetic fixture, CPU only - not a benchmark)")
    console.print(table)
    for e in summary.examples:
        console.print(f"  {e['query']!r} -> {e['planned_profile']} ({e['query_class']}): {e['top']}")
    status = "[green]PASSED[/]" if summary.gate.passed else "[red]FAILED[/]"
    console.print(f"quality gates for {summary.gate_candidate} vs B0-bm25: {status}")
    console.print(f"report -> {summary.report}")


@demo_app.command("adaptive")
def demo_adaptive(ctx: typer.Context) -> None:
    """Identifier -> lexical, semantic -> hybrid, complex -> hybrid+rerank; measured vs always hybrid+rerank."""
    from searchforge.demo import run_adaptive_demo

    opts = ctx.obj or {"out": Path("results/local-demo"), "workspace": Path(".searchforge-demo")}
    report_path, summary = run_adaptive_demo(Path(opts["out"]).resolve(), Path(opts["workspace"]).resolve())
    for cls, a in summary["by_class"]["adaptive"].items():
        f = summary["by_class"]["fixed"].get(cls, {})
        console.print(
            f"  {cls:<18} n={a['n']:<3} -> {','.join(a['profiles']):<14} nDCG@10 {f.get('ndcg@10', 0):.3f}"
            f" -> {a['ndcg@10']:.3f}; mean {f.get('mean_ms', 0):.2f} -> {a['mean_ms']:.2f} ms"
        )
    console.print(f"report -> {report_path}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
