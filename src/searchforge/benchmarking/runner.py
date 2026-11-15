"""Benchmark / experiment-matrix runner.

A benchmark spec lists *systems*. Each system = a search profile (+ optional
profile overrides) on an index built from the base config (+ optional config
overrides, e.g. a different chunking strategy → a separately named index). For
every system we measure, on the selected query split:

* retrieval quality (nDCG@k, Recall@k, MRR, MAP, ...), overall and per query type,
* latency p50/p95/p99 over ``repeats`` measured runs per query (cache bypassed),
* throughput (queries/second with N concurrent client threads),
* approximate peak RSS during the system's evaluation,
* index build time, index size by component, build peak RSS (from the manifest).

Systems are compared against a baseline system (per-query regression analysis).
All numbers are produced by this run — nothing is hard-coded.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from searchforge.benchmarking.memory import PeakRSSMonitor
from searchforge.config.loader import LoadedConfig, deep_merge, load_config, load_raw, validate_config
from searchforge.config.models import SearchProfile
from searchforge.domain.judgment import EvalQuery, Qrels, QuerySplit
from searchforge.errors import ConfigurationError
from searchforge.evaluation.datasets import load_qrels, load_queries
from searchforge.evaluation.failures import FailureReport, analyze_failures
from searchforge.evaluation.regression import ComparisonResult, compare_runs
from searchforge.evaluation.runner import RunResult, evaluate, select_queries
from searchforge.indexes.builder import IndexBuilder
from searchforge.indexes.handle import IndexHandle, open_index
from searchforge.service.search_service import SearchService


class SystemSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    profile: str
    description: str = ""
    group: str = "core"
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    profile_overrides: dict[str, Any] = Field(default_factory=dict)


class ThroughputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    threads: int = Field(default=4, ge=1, le=64)
    queries: int = Field(default=200, ge=1)


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_config: str
    systems: list[SystemSpec] = Field(min_length=1)
    baseline: str | None = None
    splits: list[QuerySplit] | None = Field(default_factory=lambda: [QuerySplit.TEST])
    repeats: int = Field(default=3, ge=1, le=50)
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    primary_metric: str = "ndcg@10"
    throughput: ThroughputSpec = ThroughputSpec()
    measure_cache: bool = True
    #: Report framing: a banner (Markdown) and limitation bullets; defaults describe the fixture collection.
    banner: str | None = None
    limitations: list[str] | None = None


@dataclass(slots=True)
class SystemResult:
    spec: SystemSpec
    run: RunResult
    index_id: str
    index_size_bytes: int
    index_components: dict[str, int]
    build_seconds: float | None
    build_peak_rss_mb: float | None
    eval_peak_rss_mb: float
    qps: float | None
    chunk_count: int
    document_count: int
    chunking: dict[str, Any]
    embedding: str | None
    failures: FailureReport | None = None

    def row(self) -> dict[str, Any]:
        agg = self.run.aggregate()
        lat = self.run.latency()
        return {
            "system": self.spec.name,
            "group": self.spec.group,
            "profile": self.spec.profile,
            "ndcg@10": agg.get("ndcg@10"),
            "recall@20": agg.get("recall@20"),
            "mrr": agg.get("mrr"),
            "map": agg.get("map"),
            "p50_ms": lat["p50_ms"],
            "p95_ms": lat["p95_ms"],
            "p99_ms": lat["p99_ms"],
            "qps": self.qps,
            "index_bytes": self.index_size_bytes,
            "chunks": self.chunk_count,
            "eval_peak_rss_mb": self.eval_peak_rss_mb,
            "build_seconds": self.build_seconds,
            "build_peak_rss_mb": self.build_peak_rss_mb,
        }


@dataclass(slots=True)
class BenchmarkResult:
    spec: BenchmarkSpec
    systems: list[SystemResult]
    comparisons: dict[str, ComparisonResult]
    queries: list[EvalQuery]
    qrels: Qrels
    base_config: LoadedConfig
    cache: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    seconds: float = 0.0


def load_spec(path: Path) -> tuple[BenchmarkSpec, Path]:
    raw = load_raw(path)
    return BenchmarkSpec.model_validate(raw), path.parent


def _effective_profile(service: SearchService, spec: SystemSpec) -> SearchProfile:
    if spec.profile not in service.profiles:
        raise ConfigurationError(f"system {spec.name}: unknown profile {spec.profile!r}")
    base = service.profiles[spec.profile]
    if not spec.profile_overrides:
        return base
    return SearchProfile.model_validate(deep_merge(base.model_dump(mode="json"), spec.profile_overrides))


def _throughput(service: SearchService, profile: str, queries: list[EvalQuery], spec: ThroughputSpec) -> float:
    work = [queries[i % len(queries)] for i in range(spec.queries)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=spec.threads) as pool:
        list(
            pool.map(
                lambda q: service.search(q.text, filters=q.filters or None, profile=profile, use_cache=False), work
            )
        )
    return len(work) / (time.perf_counter() - start)


def _cache_effect(service: SearchService, profile: str, queries: list[EvalQuery]) -> dict[str, Any]:
    service.cache.clear()
    before = (service.cache.stats.hits, service.cache.stats.misses)
    cold, warm = [], []
    for q in queries:
        st = time.perf_counter()
        service.search(q.text, filters=q.filters or None, profile=profile)
        cold.append((time.perf_counter() - st) * 1000)
    for q in queries:
        st = time.perf_counter()
        service.search(q.text, filters=q.filters or None, profile=profile)
        warm.append((time.perf_counter() - st) * 1000)
    stats = service.cache.stats
    return {
        "profile": profile,
        "queries": len(queries),
        "cold_mean_ms": sum(cold) / len(cold),
        "warm_mean_ms": sum(warm) / len(warm),
        "hits": stats.hits - before[0],
        "misses": stats.misses - before[1],
        "entries": stats.entries,
        "bytes": stats.bytes,
        "latency_saved_ms": round(stats.latency_saved_ms, 3),
    }


def run_benchmark(spec_path: Path, *, workspace: Path | None = None) -> BenchmarkResult:
    spec, spec_dir = load_spec(spec_path)
    base_path = (spec_dir / spec.base_config).resolve()
    base_raw = load_raw(base_path)
    if workspace is not None:
        base_raw["workspace"] = str(workspace)
    base_loaded = validate_config(base_raw, base_path.parent, base_path)
    ev = base_loaded.config.evaluation
    if ev is None:
        raise ConfigurationError("benchmark base_config needs an evaluation section")
    all_queries = load_queries(base_loaded.resolve(ev.queries))
    qrels = load_qrels(base_loaded.resolve(ev.qrels))
    queries = select_queries(all_queries, spec.splits)
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    handles: dict[str, tuple[LoadedConfig, IndexHandle, SearchService]] = {}
    results: list[SystemResult] = []
    try:
        for sys_spec in spec.systems:
            loaded = validate_config(deep_merge(base_raw, sys_spec.config_overrides), base_path.parent, base_path)
            key = loaded.config.index.name
            if key not in handles:
                IndexBuilder(loaded).build()
                handle = open_index(loaded.workspace, key, loaded.config)
                handles[key] = (loaded, handle, SearchService(handle, loaded.config))
            loaded, handle, service = handles[key]
            profile = _effective_profile(service, sys_spec)
            service.register_profile(sys_spec.name, profile)
            with PeakRSSMonitor() as mon:
                run = evaluate(
                    service,
                    queries,
                    qrels,
                    profile=sys_spec.name,
                    name=sys_spec.name,
                    k_values=spec.k_values,
                    repeats=spec.repeats,
                    threshold=ev.relevance_threshold,
                )
            qps = _throughput(service, sys_spec.name, queries, spec.throughput) if spec.throughput.enabled else None
            m = handle.manifest
            failures = analyze_failures(
                run, qrels, handle.chunk_store, handle.filter_schema, threshold=ev.relevance_threshold
            )
            results.append(
                SystemResult(
                    spec=sys_spec,
                    run=run,
                    index_id=m.index_id,
                    index_size_bytes=m.size_bytes(),
                    index_components=m.size_by_component(),
                    build_seconds=m.build_seconds,
                    build_peak_rss_mb=m.peak_rss_mb,
                    eval_peak_rss_mb=mon.peak_mb,
                    qps=qps,
                    chunk_count=m.chunk_count,
                    document_count=m.document_count,
                    chunking=m.chunking_config,
                    embedding=f"{m.embedding.provider}:{m.embedding.model}@{m.embedding.revision}/{m.embedding.dimension}"
                    if m.embedding
                    else None,
                    failures=failures,
                )
            )
        baseline_name = spec.baseline or spec.systems[0].name
        baseline = next(r for r in results if r.spec.name == baseline_name)
        relevant = {q.query_id: sorted(qrels.relevant(q.query_id, ev.relevance_threshold)) for q in queries}
        comparisons = {
            r.spec.name: compare_runs(
                baseline.run, r.run, primary_metric=spec.primary_metric, relevant_by_query=relevant
            )
            for r in results
            if r.spec.name != baseline_name
        }
        cache: dict[str, Any] = {}
        if spec.measure_cache:
            baseline_index = validate_config(
                deep_merge(base_raw, baseline.spec.config_overrides), base_path.parent, base_path
            ).config.index.name
            _, _, service = handles[baseline_index]
            cache = _cache_effect(service, baseline.spec.name, queries)
    finally:
        for _, handle, service in handles.values():
            service.close()
            handle.close()
    return BenchmarkResult(
        spec=spec,
        systems=results,
        comparisons=comparisons,
        queries=queries,
        qrels=qrels,
        base_config=base_loaded,
        cache=cache,
        started_at=started_at,
        seconds=round(time.perf_counter() - started, 2),
    )


def load_benchmark_base(spec_path: Path) -> LoadedConfig:
    spec, spec_dir = load_spec(spec_path)
    return load_config(spec_dir / spec.base_config)
