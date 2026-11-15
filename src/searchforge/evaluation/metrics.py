"""Classic IR metrics over *document-level* rankings.

Retrieval returns chunks, judgments are per document. Every ranked chunk list is
first collapsed to a document ranking by first occurrence
(:func:`collapse_to_documents`); a document's rank is the rank of its best chunk.
This keeps comparisons fair across chunking strategies (chunk ids differ, document
ids do not) and stops one document's many chunks from inflating P@k.

Definitions (see docs/methodology/retrieval-metrics.md) — for a ranking ``d_1..d_n``
and graded judgments ``rel(d)`` (unjudged = 0), with binary relevance
``rel(d) >= threshold``:

* P@k        = |relevant ∩ top_k| / k
* Recall@k   = |relevant ∩ top_k| / |relevant|
* HitRate@k  = 1 if any relevant document is in top_k else 0
* RR         = 1 / rank of the first relevant document (0 if none); MRR is its mean
* AP         = (1/|relevant|) Σ_{i: d_i relevant} P@i  (over the full ranking); MAP is its mean
* nDCG@k     = DCG@k / IDCG@k, DCG@k = Σ_{i≤k} gain(rel(d_i)) / log2(i + 1),
               gain = rel (linear, trec_eval-compatible) or 2^rel − 1 (exponential);
               IDCG uses all judged grades of the query sorted descending.

Queries without any relevant document are excluded from aggregates (reported).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def collapse_to_documents(document_ids: Iterable[str]) -> list[str]:
    """Deduplicate a ranked list of (chunk) document ids, keeping first occurrences."""
    return list(dict.fromkeys(document_ids))


def _relevant(judgments: Mapping[str, float], threshold: float) -> set[str]:
    return {d for d, r in judgments.items() if r >= threshold}


def precision_at_k(ranking: Sequence[str], judgments: Mapping[str, float], k: int, threshold: float = 1.0) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _relevant(judgments, threshold)
    return sum(1 for d in ranking[:k] if d in rel) / k


def recall_at_k(ranking: Sequence[str], judgments: Mapping[str, float], k: int, threshold: float = 1.0) -> float:
    rel = _relevant(judgments, threshold)
    if not rel:
        return 0.0
    return sum(1 for d in ranking[:k] if d in rel) / len(rel)


def hit_rate_at_k(ranking: Sequence[str], judgments: Mapping[str, float], k: int, threshold: float = 1.0) -> float:
    rel = _relevant(judgments, threshold)
    return 1.0 if any(d in rel for d in ranking[:k]) else 0.0


def reciprocal_rank(ranking: Sequence[str], judgments: Mapping[str, float], threshold: float = 1.0) -> float:
    rel = _relevant(judgments, threshold)
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranking: Sequence[str], judgments: Mapping[str, float], threshold: float = 1.0) -> float:
    rel = _relevant(judgments, threshold)
    if not rel:
        return 0.0
    hits = 0
    total = 0.0
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def _gain(grade: float, exponential: bool) -> float:
    return (2.0**grade - 1.0) if exponential else grade


def dcg_at_k(grades: Sequence[float], k: int, exponential: bool = False) -> float:
    return sum(_gain(g, exponential) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(ranking: Sequence[str], judgments: Mapping[str, float], k: int, *, exponential: bool = False) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    grades = [max(0.0, judgments.get(d, 0.0)) for d in ranking[:k]]
    ideal = sorted((max(0.0, g) for g in judgments.values()), reverse=True)
    idcg = dcg_at_k(ideal, k, exponential)
    if idcg == 0:
        return 0.0
    return dcg_at_k(grades, k, exponential) / idcg


def metric_names(k_values: Sequence[int]) -> list[str]:
    names: list[str] = []
    for k in k_values:
        names += [f"ndcg@{k}", f"recall@{k}", f"precision@{k}", f"hit@{k}"]
    return [*names, "mrr", "map"]


def evaluate_ranking(
    ranking: Sequence[str], judgments: Mapping[str, float], k_values: Sequence[int], threshold: float = 1.0
) -> dict[str, float]:
    """All metrics for one query. ``ranking`` must already be document-level."""
    out: dict[str, float] = {}
    for k in k_values:
        out[f"ndcg@{k}"] = ndcg_at_k(ranking, judgments, k)
        out[f"recall@{k}"] = recall_at_k(ranking, judgments, k, threshold)
        out[f"precision@{k}"] = precision_at_k(ranking, judgments, k, threshold)
        out[f"hit@{k}"] = hit_rate_at_k(ranking, judgments, k, threshold)
    out["mrr"] = reciprocal_rank(ranking, judgments, threshold)
    out["map"] = average_precision(ranking, judgments, threshold)
    return out


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy 'linear' method); q in [0, 100]."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
