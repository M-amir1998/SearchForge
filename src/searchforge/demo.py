"""CPU-only end-to-end demonstrations on the bundled fixture collection.

``run_demo``: corpus → chunk → lexical + dense indexes → hybrid retrieval → rerank →
evaluate (B0–B5 + fusion/depth/chunking experiments) → compare → quality gates →
citation context → report. No internet, GPU or API key is needed.

``run_adaptive_demo``: fixed hybrid+rerank vs adaptive query planning, broken down
by query class (quality, latency, avoided dense/rerank work).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from searchforge.benchmarking.runner import BenchmarkResult, run_benchmark
from searchforge.config.loader import load_config
from searchforge.context.builder import CitationContextBuilder
from searchforge.datasets import fixture_dir
from searchforge.evaluation.metrics import percentile
from searchforge.evaluation.regression import GateReport, evaluate_gates
from searchforge.evaluation.runner import RunResult
from searchforge.indexes.handle import open_index
from searchforge.integrations.reliability_lab import build_export, citation_validity
from searchforge.reporting.report import write_benchmark_report
from searchforge.service.search_service import SearchService

EXAMPLE_QUERIES = (
    ("ERR-4012", None),
    ("how often do we have to change our credentials", None),
    ("rollback", {"published": {"gte": "2024-01-01"}}),
)


@dataclass(slots=True)
class DemoSummary:
    report: Path
    gate: GateReport
    gate_candidate: str
    examples: list[dict[str, Any]]
    context_preview: str
    benchmark: BenchmarkResult


def _examples(workspace: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    loaded = load_config(fixture_dir() / "fixture.yaml", overrides={"workspace": str(workspace)})
    handle = open_index(loaded.workspace, loaded.config.index.name, loaded.config)
    service = SearchService(handle, loaded.config)
    try:
        examples = []
        for text, filters in EXAMPLE_QUERIES:
            resp = service.search(text, filters=filters, profile="adaptive", top_k=3, explain=True)
            examples.append(
                {
                    "query": text,
                    "filters": filters,
                    "planned_profile": resp.profile,
                    "query_class": resp.trace.planner_decision.get("query_class"),
                    "top": [(r.document_id, round(r.score, 4)) for r in resp.results],
                    "latency_ms": round(resp.latency_ms, 2),
                }
            )
        resp = service.search("what happens when a client sends too many requests", profile="hybrid_rerank", top_k=8)
        package = CitationContextBuilder(max_chars=1500, max_per_document=1).build(resp.results)
        validity = citation_validity(package, handle.chunk_store)
        return examples, package.render(), validity
    finally:
        service.close()
        handle.close()


def run_demo(out_dir: Path, workspace: Path) -> DemoSummary:
    result = run_benchmark(fixture_dir() / "benchmark.yaml", workspace=workspace)
    report = write_benchmark_report(result, out_dir)
    candidate = "B4-hybrid-rerank"
    comp = result.comparisons[candidate]
    cand_run = next(s.run for s in result.systems if s.spec.name == candidate)
    gates_cfg = result.base_config.config.evaluation.gates  # type: ignore[union-attr]
    gate = evaluate_gates(comp, cand_run, gates_cfg)
    examples, context, validity = _examples(workspace)
    export = build_export(
        [s.run for s in result.systems if s.spec.group == "core"], list(result.comparisons.values()), gate, validity
    )
    (out_dir / "reliability_export.json").write_text(json.dumps(export, indent=2, default=str), encoding="utf-8")
    (out_dir / "gates.json").write_text(
        json.dumps({"candidate": candidate, **gate.as_dict()}, indent=2), encoding="utf-8"
    )
    (out_dir / "examples.json").write_text(
        json.dumps({"examples": examples, "context": context, "citation_validity": validity}, indent=2),
        encoding="utf-8",
    )
    return DemoSummary(
        report=report, gate=gate, gate_candidate=candidate, examples=examples, context_preview=context, benchmark=result
    )


def _by_class(run: RunResult) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for o in run.scored:
        groups[o.query_class].append(o)
    out = {}
    for cls, items in sorted(groups.items()):
        lat = [ms for o in items for ms in o.latencies_ms]
        out[cls] = {
            "n": len(items),
            "ndcg@10": sum(o.metrics["ndcg@10"] for o in items) / len(items),
            "mean_ms": sum(lat) / len(lat),
            "p95_ms": percentile(lat, 95),
            "profiles": sorted({o.planned_profile for o in items}),
        }
    return out


def run_adaptive_demo(out_dir: Path, workspace: Path) -> tuple[Path, dict[str, Any]]:
    result = run_benchmark(fixture_dir() / "adaptive.yaml", workspace=workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {s.spec.name: s for s in result.systems}
    fixed, adaptive = runs["always-hybrid-rerank"], runs["adaptive"]
    f_cls, a_cls = _by_class(fixed.run), _by_class(adaptive.run)
    lines = [
        "# Adaptive Query Planning — Local Demonstration",
        "",
        "> **LOCAL DEMONSTRATION RESULTS** on the synthetic fixture collection (all splits, CPU only).",
        "> Not a benchmark; the planner rules were written for this kind of collection.",
        "",
        "Question: can query-aware routing keep search quality comparable to *always* running",
        "hybrid retrieval + reranking while skipping dense retrieval / reranking where lexical search suffices?",
        "",
        "| System | nDCG@10 | Recall@20 | MRR | p50 | p95 | QPS |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in result.systems:
        r = s.row()
        lines.append(
            f"| {s.spec.name} | {r['ndcg@10']:.3f} | {r['recall@20']:.3f} | {r['mrr']:.3f} | "
            f"{r['p50_ms']:.2f} ms | {r['p95_ms']:.2f} ms | {r['qps']:.1f} |"
        )
    lines += [
        "",
        "## By query class (as classified by the planner)",
        "",
        "| Query class | n | Routed to | nDCG@10 fixed | nDCG@10 adaptive | mean ms fixed | mean ms adaptive |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls, a in a_cls.items():
        f = f_cls.get(cls, {})
        lines.append(
            f"| {cls} | {a['n']} | {', '.join(a['profiles'])} | {f.get('ndcg@10', 0):.3f} | "
            f"{a['ndcg@10']:.3f} | {f.get('mean_ms', 0):.2f} | {a['mean_ms']:.2f} |"
        )
    routed = adaptive.run.summary()["planned_profiles"]
    skipped_rerank = sum(v for k, v in routed.items() if k != "hybrid_rerank")
    skipped_dense = routed.get("lexical", 0)
    total = sum(routed.values())
    comp = result.comparisons["adaptive"].summary
    lines += [
        "",
        f"Adaptive routing skipped the reranker for {skipped_rerank}/{total} queries and dense retrieval "
        f"for {skipped_dense}/{total} queries. Versus always-hybrid-rerank: "
        f"ΔnDCG@10 = {comp['mean_deltas']['ndcg@10']:+.3f}, Δp95 = {comp['latency']['p95_delta_ms']:+.2f} ms, "
        f"{comp['improved']} improved / {comp['regressed']} regressed / {comp['unchanged']} unchanged queries.",
        "",
        "CPU cost proxy: the share of queries that avoided dense retrieval and/or reranking (above). "
        "Wall-clock differences on a ~50-document collection are small in absolute terms; the relative",
        "cost matters once a neural encoder or cross-encoder replaces the CPU baselines.",
        "",
    ]
    report = out_dir / "adaptive_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "systems": [s.row() for s in result.systems],
        "by_class": {"fixed": f_cls, "adaptive": a_cls},
        "routing": routed,
        "comparison": comp,
    }
    (out_dir / "adaptive_metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return report, summary
