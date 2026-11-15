"""Result diversification and grouping.

*Relevance* is what the retrievers/rerankers estimate; *diversity* is a separate,
explicit trade-off applied afterwards:

* :func:`cap_per_document` — at most ``max_per_document`` chunks from one document
  in the head of the list (extras are demoted to the tail, not deleted).
* :func:`mmr` — Maximal Marginal Relevance (Carbonell & Goldstein, 1998):
  ``argmax λ·rel(d) − (1−λ)·max_{s∈S} sim(d, s)`` using dense vectors. λ=1 is pure
  relevance; lower λ trades relevance for novelty.
* :func:`group_results` — group by ``document_id`` or ``source`` for display.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from searchforge.domain.result import ScoredChunk, SearchResult


def cap_per_document(candidates: Sequence[ScoredChunk], max_per_document: int) -> list[ScoredChunk]:
    head: list[ScoredChunk] = []
    tail: list[ScoredChunk] = []
    counts: dict[str, int] = {}
    for c in candidates:
        n = counts.get(c.document_id, 0)
        if n < max_per_document:
            head.append(c)
        else:
            sig = replace(c.signals, diversity_note=f"demoted: >{max_per_document} chunks from document")
            tail.append(replace(c, signals=sig))
        counts[c.document_id] = n + 1
    return head + tail


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def mmr(
    candidates: Sequence[ScoredChunk],
    vectors: Mapping[str, NDArray[np.float32]],
    lambda_: float,
    k: int,
) -> list[ScoredChunk]:
    """Greedy MMR selection of ``k`` items; the remainder keeps its original order."""
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda must be within [0, 1]")
    rel = _minmax([c.score for c in candidates])
    remaining = list(range(len(candidates)))
    selected: list[int] = []
    while remaining and len(selected) < k:
        best_i, best_val = remaining[0], -np.inf
        for i in remaining:
            vi = vectors.get(candidates[i].chunk_id)
            redundancy = 0.0
            if vi is not None and selected:
                sims = [
                    float(vi @ vectors[candidates[s].chunk_id]) for s in selected if candidates[s].chunk_id in vectors
                ]
                redundancy = max(sims) if sims else 0.0
            val = lambda_ * rel[i] - (1.0 - lambda_) * redundancy
            if val > best_val:
                best_i, best_val = i, val
        selected.append(best_i)
        remaining.remove(best_i)
    out = [candidates[i] for i in selected] + [candidates[i] for i in remaining]
    return out


def group_results(results: Sequence[SearchResult], by: str = "document_id") -> dict[str, list[SearchResult]]:
    if by not in ("document_id", "source"):
        raise ValueError("group_by must be 'document_id' or 'source'")
    groups: dict[str, list[SearchResult]] = {}
    for r in results:
        key = r.document_id if by == "document_id" else str(r.metadata.get("source_id", "unknown"))
        groups.setdefault(key, []).append(r)
    return groups
