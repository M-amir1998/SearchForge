"""Retrieval candidates, ranked results, explanations and traces."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from searchforge.domain.document import AttributeValue


@dataclass(frozen=True, slots=True)
class Candidate:
    """A scored hit from one retriever (or after fusion/reranking)."""

    chunk_id: str
    document_id: str
    score: float
    rank: int
    source: str


@dataclass(slots=True)
class RankingSignals:
    """Actual ranking signals behind a result — not model interpretability."""

    lexical_rank: int | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    per_retriever: dict[str, tuple[int, float]] = field(default_factory=dict)
    fusion_score: float | None = None
    fusion_rank: int | None = None
    reranker_score: float | None = None
    reranker_rank: int | None = None
    filter_reasons: list[str] = field(default_factory=list)
    diversity_note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lexical_rank": self.lexical_rank,
            "lexical_score": self.lexical_score,
            "dense_rank": self.dense_rank,
            "dense_score": self.dense_score,
            "per_retriever": {k: {"rank": r, "score": s} for k, (r, s) in self.per_retriever.items()},
            "fusion_score": self.fusion_score,
            "fusion_rank": self.fusion_rank,
            "reranker_score": self.reranker_score,
            "reranker_rank": self.reranker_rank,
            "filter_reasons": list(self.filter_reasons),
            "diversity_note": self.diversity_note,
        }


@dataclass(slots=True)
class ScoredChunk:
    """A candidate hydrated with chunk text/metadata, flowing through ranking stages."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    retrieval_sources: list[str]
    metadata: dict[str, Any]
    attributes: dict[str, AttributeValue]
    signals: RankingSignals = field(default_factory=RankingSignals)

    def with_score(self, score: float) -> ScoredChunk:
        return replace(self, score=score)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked result returned to callers."""

    chunk_id: str
    document_id: str
    score: float
    rank: int
    retrieval_sources: tuple[str, ...]
    metadata: dict[str, Any]
    text: str | None = None
    explanation: dict[str, Any] | None = None

    def as_dict(self, include_text: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "score": self.score,
            "rank": self.rank,
            "retrieval_sources": list(self.retrieval_sources),
            "metadata": self.metadata,
        }
        if include_text and self.text is not None:
            out["text"] = self.text
        if self.explanation is not None:
            out["explanation"] = self.explanation
        return out


@dataclass(slots=True)
class StageTiming:
    name: str
    ms: float


@dataclass(slots=True)
class SearchTrace:
    """Lightweight per-request trace. Holds a query *hash* by default, not text."""

    query_id: str
    query_capture: str | None
    capture_policy: str
    profile: str
    planner_decision: dict[str, Any] = field(default_factory=dict)
    retrievers: list[str] = field(default_factory=list)
    candidate_counts: dict[str, int] = field(default_factory=dict)
    fusion: str | None = None
    reranker: str | None = None
    timings: list[StageTiming] = field(default_factory=list)
    result_ids: list[str] = field(default_factory=list)
    degraded: bool = False
    degradation_reasons: list[str] = field(default_factory=list)
    cache_hit: bool = False
    rewrite: dict[str, Any] | None = None

    def add_timing(self, name: str, ms: float) -> None:
        self.timings.append(StageTiming(name=name, ms=ms))

    def latency_breakdown(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in self.timings:
            out[t.name] = round(out.get(t.name, 0.0) + t.ms, 3)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query_capture,
            "capture_policy": self.capture_policy,
            "profile": self.profile,
            "planner_decision": self.planner_decision,
            "retrievers": self.retrievers,
            "candidate_counts": self.candidate_counts,
            "fusion": self.fusion,
            "reranker": self.reranker,
            "latency_ms": self.latency_breakdown(),
            "result_ids": self.result_ids,
            "degraded": self.degraded,
            "degradation_reasons": self.degradation_reasons,
            "cache_hit": self.cache_hit,
            "rewrite": self.rewrite,
        }


@dataclass(slots=True)
class SearchResponse:
    """Complete answer to a search request."""

    query_id: str
    results: list[SearchResult]
    profile: str
    index_id: str
    latency_ms: float
    degraded: bool
    degradation_reasons: list[str]
    trace: SearchTrace
    candidates_by_retriever: dict[str, list[Candidate]] = field(default_factory=dict)
    fused_candidates: list[Candidate] = field(default_factory=list)
