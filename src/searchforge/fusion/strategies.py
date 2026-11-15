"""Rank fusion strategies.

* :class:`ReciprocalRankFusion` — ``score(d) = Σ_i w_i / (k + rank_i(d))`` (Cormack et al., 2009).
  Uses ranks only, so it needs no score calibration between BM25 and cosine.
* :class:`WeightedRRF` — RRF with explicit per-retriever weights (tune on train/validation only).
* :class:`NormalizedScoreFusion` — per-list min-max or z-score normalization, then a
  weighted sum. Raw BM25 (unbounded, query-dependent) and cosine ([-1, 1]) are on
  incomparable scales, so they are *never* summed without normalization.

Keys are chunk ids; ``Candidate.rank`` is 1-based. Ties are broken by best
contributing rank, then chunk id, so fusion is deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from searchforge.config.models import FusionConfig, FusionName
from searchforge.domain.result import Candidate


def _weight(weights: Mapping[str, float], list_name: str) -> float:
    """Weights are keyed by retriever; query-variant lists are named ``retriever#vN``."""
    if list_name in weights:
        return weights[list_name]
    base = list_name.split("#", 1)[0]
    return weights.get(base, 1.0)


def _finalize(scores: dict[str, float], best_rank: dict[str, int], docs: dict[str, str], limit: int) -> list[Candidate]:
    order = sorted(scores, key=lambda c: (-scores[c], best_rank[c], c))[:limit]
    return [
        Candidate(chunk_id=c, document_id=docs[c], score=scores[c], rank=i + 1, source="fusion")
        for i, c in enumerate(order)
    ]


@dataclass(slots=True)
class ReciprocalRankFusion:
    k: int = 60
    weights: dict[str, float] = field(default_factory=dict)
    name: str = "rrf"

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("RRF k must be >= 1")

    def fuse(self, ranked_lists: Mapping[str, Sequence[Candidate]], limit: int) -> list[Candidate]:
        scores: dict[str, float] = {}
        best: dict[str, int] = {}
        docs: dict[str, str] = {}
        for list_name, candidates in ranked_lists.items():
            w = _weight(self.weights, list_name)
            seen: set[str] = set()
            for c in candidates:
                if c.chunk_id in seen:
                    continue  # a list contributes at most once per item
                seen.add(c.chunk_id)
                scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + w / (self.k + c.rank)
                best[c.chunk_id] = min(best.get(c.chunk_id, c.rank), c.rank)
                docs[c.chunk_id] = c.document_id
        return _finalize(scores, best, docs, limit)


class WeightedRRF(ReciprocalRankFusion):
    """RRF where weights are mandatory (an explicit, tunable hypothesis)."""

    def __init__(self, k: int = 60, weights: dict[str, float] | None = None) -> None:
        if not weights:
            raise ValueError("WeightedRRF requires explicit retriever weights")
        super().__init__(k=k, weights=dict(weights), name="weighted_rrf")


def _normalize(values: list[float], method: str) -> tuple[list[float], float]:
    """Return normalized values and the value assigned to items absent from the list."""
    if not values:
        return [], 0.0
    if method == "zscore":
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        if std == 0:
            return [0.0 for _ in values], 0.0
        z = [(v - mean) / std for v in values]
        return z, min(z)
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values], 0.0
    return [(v - lo) / (hi - lo) for v in values], 0.0


@dataclass(slots=True)
class NormalizedScoreFusion:
    weights: dict[str, float] = field(default_factory=dict)
    method: str = "minmax"
    name: str = "normalized"

    def fuse(self, ranked_lists: Mapping[str, Sequence[Candidate]], limit: int) -> list[Candidate]:
        per_list: dict[str, tuple[dict[str, float], float]] = {}
        docs: dict[str, str] = {}
        best: dict[str, int] = {}
        for list_name, candidates in ranked_lists.items():
            uniq: dict[str, Candidate] = {}
            for c in candidates:
                uniq.setdefault(c.chunk_id, c)
            normed, missing = _normalize([c.score for c in uniq.values()], self.method)
            per_list[list_name] = (dict(zip(uniq.keys(), normed, strict=True)), missing)
            for c in uniq.values():
                docs[c.chunk_id] = c.document_id
                best[c.chunk_id] = min(best.get(c.chunk_id, c.rank), c.rank)
        scores: dict[str, float] = {}
        for chunk_id in docs:
            total = 0.0
            for list_name, (normed_map, missing) in per_list.items():
                total += _weight(self.weights, list_name) * normed_map.get(chunk_id, missing)
            scores[chunk_id] = total
        return _finalize(scores, best, docs, limit)


FusionImpl = ReciprocalRankFusion | NormalizedScoreFusion


def build_fusion(config: FusionConfig) -> FusionImpl:
    if config.strategy is FusionName.WEIGHTED_RRF:
        return WeightedRRF(k=config.k, weights=dict(config.weights))
    if config.strategy is FusionName.NORMALIZED:
        return NormalizedScoreFusion(weights=dict(config.weights), method=config.normalization)
    return ReciprocalRankFusion(k=config.k, weights=dict(config.weights))
