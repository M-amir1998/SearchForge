"""Offline evaluation: run a query set through a SearchService and score it against qrels.

Measurement rules:
* the query cache is bypassed (``use_cache=False``) so latency is real work;
* a short warm-up (first 5 judged queries) precedes measurement; each query then runs ``repeats`` times
  and latency percentiles are computed over all measured samples;
* rankings use ``ranking_depth`` (default 100 chunks) without altering the plan,
  then collapse to documents (see :mod:`searchforge.evaluation.metrics`);
* queries without any relevant judgment are excluded and reported.
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from searchforge.domain.ids import short_hash
from searchforge.domain.judgment import EvalQuery, Qrels, QuerySplit
from searchforge.errors import EvaluationFailure, SearchForgeError
from searchforge.evaluation.metrics import collapse_to_documents, evaluate_ranking, percentile
from searchforge.service.search_service import SearchService


@dataclass(slots=True)
class QueryOutcome:
    query: EvalQuery
    doc_ranking: list[str]
    chunk_ranking: list[tuple[str, str, float, tuple[str, ...]]]
    metrics: dict[str, float]
    latencies_ms: list[float]
    planned_profile: str
    query_class: str
    degraded: bool
    stage_docs: dict[str, list[str]] = field(default_factory=dict)
    reranked: bool = False
    rewrite_strategies: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        return percentile(self.latencies_ms, 50) if self.latencies_ms else 0.0


@dataclass(slots=True)
class RunResult:
    run_id: str
    name: str
    profile: str
    index_id: str
    k_values: list[int]
    threshold: float
    outcomes: list[QueryOutcome]
    excluded: list[str]
    created_at: str
    config: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def scored(self) -> list[QueryOutcome]:
        return [o for o in self.outcomes if o.error is None]

    def aggregate(self, outcomes: Sequence[QueryOutcome] | None = None) -> dict[str, float]:
        items = list(self.scored if outcomes is None else outcomes)
        if not items:
            return {}
        names = items[0].metrics.keys()
        return {n: sum(o.metrics[n] for o in items) / len(items) for n in names}

    def by_query_type(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[QueryOutcome]] = defaultdict(list)
        for o in self.scored:
            groups[o.query.query_type].append(o)
        return {t: {**self.aggregate(g), "n": float(len(g))} for t, g in sorted(groups.items())}

    def latency(self, outcomes: Sequence[QueryOutcome] | None = None) -> dict[str, float]:
        items = self.scored if outcomes is None else outcomes
        samples = [ms for o in items for ms in o.latencies_ms]
        if not samples:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "samples": 0.0}
        return {
            "p50_ms": percentile(samples, 50),
            "p95_ms": percentile(samples, 95),
            "p99_ms": percentile(samples, 99),
            "mean_ms": sum(samples) / len(samples),
            "samples": float(len(samples)),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "profile": self.profile,
            "index_id": self.index_id,
            "queries": len(self.scored),
            "excluded": len(self.excluded),
            "errors": sum(1 for o in self.outcomes if o.error),
            "metrics": self.aggregate(),
            "latency": self.latency(),
            "by_query_type": self.by_query_type(),
            "degraded_queries": sum(1 for o in self.scored if o.degraded),
            "planned_profiles": _count(o.planned_profile for o in self.scored),
            "created_at": self.created_at,
            **({"extra": self.extra} if self.extra else {}),
        }


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return dict(sorted(out.items()))


def select_queries(queries: Sequence[EvalQuery], splits: Sequence[QuerySplit] | None) -> list[EvalQuery]:
    if not splits:
        return list(queries)
    allowed = set(splits)
    return [q for q in queries if q.split in allowed]


def new_run_id(name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{name}-{short_hash(name + stamp + str(time.perf_counter_ns()), 6)}"


def evaluate(
    service: SearchService,
    queries: Sequence[EvalQuery],
    qrels: Qrels,
    *,
    profile: str,
    name: str | None = None,
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
    threshold: float = 1.0,
    repeats: int = 1,
    ranking_depth: int = 100,
    top_k: int = 10,
    warmup: bool = True,
) -> RunResult:
    if profile not in service.profiles:
        raise EvaluationFailure(f"unknown profile {profile!r}")
    judged = [q for q in queries if qrels.relevant(q.query_id, threshold)]
    excluded = [q.query_id for q in queries if not qrels.relevant(q.query_id, threshold)]
    if not judged:
        raise EvaluationFailure("no query in the selection has a relevant judgment")
    if warmup:
        for q in judged[: min(5, len(judged))]:
            with contextlib.suppress(SearchForgeError):
                service.search(
                    q.text,
                    filters=q.filters or None,
                    profile=profile,
                    use_cache=False,
                    top_k=top_k,
                    ranking_depth=ranking_depth,
                )
    outcomes: list[QueryOutcome] = []
    for q in judged:
        latencies: list[float] = []
        first = None
        error: str | None = None
        for _ in range(max(1, repeats)):
            st = time.perf_counter()
            try:
                resp = service.search(
                    q.text,
                    filters=q.filters or None,
                    profile=profile,
                    query_id=q.query_id,
                    use_cache=False,
                    top_k=top_k,
                    ranking_depth=ranking_depth,
                )
            except SearchForgeError as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
            latencies.append((time.perf_counter() - st) * 1000.0)
            if first is None:
                first = resp
        judgments = qrels.for_query(q.query_id)
        if first is None:
            outcomes.append(
                QueryOutcome(
                    q,
                    [],
                    [],
                    dict.fromkeys(evaluate_ranking([], judgments, k_values), 0.0),
                    [],
                    profile,
                    "error",
                    True,
                    error=error,
                )
            )
            continue
        docs = collapse_to_documents(r.document_id for r in first.results)
        stage_docs = {
            name_: collapse_to_documents(c.document_id for c in cands)
            for name_, cands in first.candidates_by_retriever.items()
        }
        stage_docs["fused"] = collapse_to_documents(c.document_id for c in first.fused_candidates)
        decision = first.trace.planner_decision
        outcomes.append(
            QueryOutcome(
                query=q,
                doc_ranking=docs,
                chunk_ranking=[(r.chunk_id, r.document_id, r.score, r.retrieval_sources) for r in first.results],
                metrics=evaluate_ranking(docs, judgments, k_values, threshold),
                latencies_ms=latencies,
                planned_profile=str(decision.get("profile", profile)),
                query_class=str(decision.get("query_class", "")),
                degraded=first.degraded,
                stage_docs=stage_docs,
                reranked="rerank" in first.trace.latency_breakdown(),
                rewrite_strategies=list((first.trace.rewrite or {}).get("strategies", [])),
            )
        )
    run_name = name or profile
    return RunResult(
        run_id=new_run_id(run_name),
        name=run_name,
        profile=profile,
        index_id=service.index_id,
        k_values=list(k_values),
        threshold=threshold,
        outcomes=outcomes,
        excluded=excluded,
        created_at=datetime.now(UTC).isoformat(),
    )
