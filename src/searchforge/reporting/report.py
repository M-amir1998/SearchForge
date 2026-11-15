"""Search-quality report generation (Markdown + SVG + JSON) from measured benchmark data only."""

from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import psutil

from searchforge import __version__
from searchforge.benchmarking.runner import BenchmarkResult, SystemResult
from searchforge.evaluation.store import write_run
from searchforge.hard_negatives.mining import mine_hard_negatives, write_hard_negatives
from searchforge.reporting.svg import Point, scatter_svg

QUERY_TYPES = ("keyword", "semantic", "identifier", "ambiguous", "filter")

DEFAULT_BANNER = """**LOCAL DEMONSTRATION RESULTS.** Measured on the small synthetic fixture collection shipped with
SearchForge on a CPU-only machine. These numbers demonstrate the methodology; they are **not**
industry benchmarks and do not predict behaviour on other corpora."""

DEFAULT_LIMITATIONS = (
    "The fixture collection is tiny (tens of documents, ~50 queries) and synthetic; differences of a few "
    "hundredths in nDCG are within noise. No significance testing is performed at this size.",
    "The dense baseline is TF-IDF + LSA fitted on the corpus itself — a classical, CPU-only baseline, not a "
    "neural embedding model. Neural embeddings/cross-encoders are optional extras and not measured here.",
    "Latency is measured in-process (no network) on a single shared developer machine after warm-up; "
    "sub-10 ms differences are within run-to-run noise.",
    "The rules-based planner and lexical-feature reranker were designed alongside this collection; validate "
    "them on an independent collection (e.g. a BEIR subset) before drawing conclusions.",
)


def _f(v: Any, digits: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _kb(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def environment() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "ram_gb": round(vm.total / 1024**3, 1),
        "gpu": "none (CPU-only run)",
        "searchforge": __version__,
    }


def _group(result: BenchmarkResult, group: str) -> list[SystemResult]:
    return [s for s in result.systems if s.spec.group == group]


def render_report(result: BenchmarkResult, plot_rel: str = "plots/quality_latency.svg") -> str:
    spec = result.spec
    env = environment()
    core = _group(result, "core") or result.systems
    first = result.systems[0]
    split = ",".join(s.value for s in spec.splits) if spec.splits else "all"
    lines: list[str] = [
        f"# SearchForge Search-Quality Report — `{spec.name}`",
        "",
        *[f"> {line}" for line in (spec.banner or DEFAULT_BANNER).strip().splitlines()],
        "",
        "## Setup",
        "",
        f"- Corpus: `{result.base_config.config.corpus.id}` — {first.document_count} documents",
        f"- Query split: `{split}` — {len(result.queries)} queries "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(Counter(q.query_type for q in result.queries).items()))})",
        f"- Latency: {spec.repeats} measured runs per query after warm-up, query cache bypassed; "
        f"throughput with {spec.throughput.threads} client threads",
        f"- Environment: Python {env['python']}, {env['platform']}, {env['logical_cpus']} logical CPUs, "
        f"{env['ram_gb']} GB RAM, GPU: {env['gpu']}",
        f"- Total benchmark wall time: {result.seconds:.1f} s",
        "",
        "## Main results",
        "",
        f"![quality vs latency]({plot_rel})",
        "",
        _table(
            ["System", "nDCG@10", "Recall@20", "MRR", "p95 Latency", "Index Size"],
            [
                [
                    s.spec.name,
                    _f(s.row()["ndcg@10"]),
                    _f(s.row()["recall@20"]),
                    _f(s.row()["mrr"]),
                    f"{s.row()['p95_ms']:.2f} ms",
                    _kb(s.index_size_bytes),
                ]
                for s in core
            ],
        ),
        "",
        "### Latency, throughput and memory",
        "",
        _table(
            ["System", "p50", "p95", "p99", "QPS", "Peak RSS (eval)"],
            [
                [
                    s.spec.name,
                    f"{s.row()['p50_ms']:.2f} ms",
                    f"{s.row()['p95_ms']:.2f} ms",
                    f"{s.row()['p99_ms']:.2f} ms",
                    _f(s.qps, 1),
                    f"{s.eval_peak_rss_mb:.0f} MB",
                ]
                for s in result.systems
            ],
        ),
        "",
        "Peak RSS is sampled for the whole process (approximate; includes the Python runtime and all open indexes).",
        "",
        "## Quality by query type (nDCG@10)",
        "",
    ]
    present = {q.query_type for q in result.queries}
    types = [t for t in QUERY_TYPES if t in present] + sorted(present - set(QUERY_TYPES))
    rows = []
    for s in core:
        bt = s.run.by_query_type()
        rows.append([s.spec.name, *[_f(bt.get(t, {}).get("ndcg@10")) for t in types]])
    lines += [_table(["Configuration", *[t.capitalize() for t in types]], rows), ""]

    for group, title in (
        ("chunking", "Chunking experiment"),
        ("depth", "Candidate-depth experiment"),
        ("fusion", "Fusion experiment"),
    ):
        members = _group(result, group)
        if not members:
            continue
        lines += [f"## {title}", ""]
        lines += [
            _table(
                ["System", "Description", "Chunks", "nDCG@10", "Recall@20", "MRR", "p95", "Index Size"],
                [
                    [
                        s.spec.name,
                        s.spec.description or "–",
                        str(s.chunk_count),
                        _f(s.row()["ndcg@10"]),
                        _f(s.row()["recall@20"]),
                        _f(s.row()["mrr"]),
                        f"{s.row()['p95_ms']:.2f} ms",
                        _kb(s.index_size_bytes),
                    ]
                    for s in members
                ],
            ),
            "",
        ]

    adaptive = [s for s in result.systems if s.run.summary()["planned_profiles"] and s.spec.profile == "adaptive"]
    for s in adaptive:
        lines += [
            f"## Adaptive routing (`{s.spec.name}`)",
            "",
            "Planned profile per query (rules-based planner):",
            "",
            _table(
                ["Planned profile", "Queries"], [[k, str(v)] for k, v in s.run.summary()["planned_profiles"].items()]
            ),
            "",
        ]

    lines += ["## Index size by component", ""]
    seen: set[str] = set()
    idx_rows = []
    for s in result.systems:
        if s.index_id in seen:
            continue
        seen.add(s.index_id)
        comp = s.index_components
        idx_rows.append(
            [
                s.index_id.split("@")[0],
                str(s.chunk_count),
                _kb(comp.get("chunks", 0)),
                _kb(comp.get("lexical", 0)),
                _kb(comp.get("dense", 0)),
                _kb(s.index_size_bytes),
                _f(s.build_seconds, 2) + " s",
                _f(s.build_peak_rss_mb, 0) + " MB",
                s.embedding or "–",
            ]
        )
    lines += [
        _table(
            [
                "Index",
                "Chunks",
                "Chunk store",
                "Lexical (FTS5)",
                "Dense (+embedder)",
                "Total",
                "Build time",
                "Build peak RSS",
                "Embedding",
            ],
            idx_rows,
        ),
        "",
    ]

    if result.cache:
        c = result.cache
        lines += [
            "## Query cache",
            "",
            f"Same {c['queries']} queries issued twice with the cache enabled (`{c['profile']}`): "
            f"cold mean {c['cold_mean_ms']:.2f} ms, warm mean {c['warm_mean_ms']:.3f} ms, "
            f"hits {c['hits']}, misses {c['misses']}, {c['entries']} entries (~{_kb(int(c['bytes']))}), "
            f"latency saved {c['latency_saved_ms']:.1f} ms.",
            "",
        ]

    if result.comparisons:
        lines += [f"## Regressions vs baseline `{spec.baseline or first.spec.name}` ({spec.primary_metric})", ""]
        rows = []
        for name, comparison in result.comparisons.items():
            sm = comparison.summary
            md = sm["mean_deltas"]
            rows.append(
                [
                    name,
                    str(sm["improved"]),
                    str(sm["regressed"]),
                    str(sm["unchanged"]),
                    _f(md.get("ndcg@10")),
                    _f(md.get("recall@20")),
                    _f(md.get("mrr")),
                    f"{sm['latency']['p95_delta_ms']:+.2f} ms",
                    ", ".join(sm["critical_regressions"]) or "none",
                ]
            )
        lines += [
            _table(
                [
                    "Candidate",
                    "Improved",
                    "Regressed",
                    "Unchanged",
                    "ΔnDCG@10",
                    "ΔRecall@20",
                    "ΔMRR",
                    "Δp95",
                    "Critical regressions",
                ],
                rows,
            ),
            "",
        ]

    lines += [
        "## Failure analysis",
        "",
        "Categories are rule-based *hypotheses* derived from observed evidence (retriever ranks, filters, "
        "index membership); see `failures.parquet` per run for the evidence columns.",
        "",
    ]
    rows = []
    for s in core:
        counts = s.failures.counts() if s.failures else {}
        rows.append([s.spec.name, str(sum(counts.values())), ", ".join(f"{k}: {v}" for k, v in counts.items()) or "–"])
    lines += [_table(["System", "Missed relevant docs (top-10)", "By category"], rows), ""]

    lines += ["## Limitations", "", *[f"- {item}" for item in (spec.limitations or DEFAULT_LIMITATIONS)], ""]
    return "\n".join(lines)


def write_benchmark_report(result: BenchmarkResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    core = _group(result, "core") or result.systems
    pts = [Point(s.spec.name, s.row()["p95_ms"], float(s.row()["ndcg@10"] or 0.0)) for s in core]
    (out_dir / "plots" / "quality_latency.svg").write_text(
        scatter_svg(
            pts, title="Quality vs latency (fixture collection, CPU)", x_label="p95 latency (ms)", y_label="nDCG@10"
        ),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    summary = {
        "benchmark": result.spec.name,
        "environment": environment(),
        "seconds": result.seconds,
        "systems": [s.row() for s in result.systems],
        "by_query_type": {s.spec.name: s.run.by_query_type() for s in result.systems},
        "comparisons": {k: v.summary for k, v in result.comparisons.items()},
        "failures": {s.spec.name: (s.failures.counts() if s.failures else {}) for s in result.systems},
        "cache": result.cache,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    runs_dir = out_dir / "runs"
    for s in result.systems:
        write_run(
            s.run,
            runs_dir,
            config={"system": s.spec.model_dump(), "index_id": s.index_id},
            manifest={"index_id": s.index_id, "chunks": s.chunk_count, "bytes": s.index_size_bytes},
            failures=s.failures,
            comparison=result.comparisons.get(s.spec.name),
            performance={"latency": s.run.latency(), "qps": s.qps, "eval_peak_rss_mb": s.eval_peak_rss_mb},
        )
    negatives = [hn for s in core for hn in mine_hard_negatives(s.run, result.qrels)]
    write_hard_negatives(negatives, out_dir / "hard_negatives.jsonl")
    return out_dir / "report.md"
