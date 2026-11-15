"""IR metrics and rank fusion against hand-calculated fixtures."""

from __future__ import annotations

import math

import pytest

from searchforge.config.models import FusionConfig, FusionName
from searchforge.domain.result import Candidate
from searchforge.evaluation.metrics import (
    average_precision,
    collapse_to_documents,
    evaluate_ranking,
    hit_rate_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from searchforge.fusion.strategies import NormalizedScoreFusion, ReciprocalRankFusion, WeightedRRF, build_fusion

pytestmark = pytest.mark.unit

RANKING = ["A", "B", "C", "D", "E"]
JUDGMENTS = {"A": 3.0, "C": 2.0, "F": 1.0}


def test_precision_recall_hit() -> None:
    assert precision_at_k(RANKING, JUDGMENTS, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RANKING, JUDGMENTS, 5) == pytest.approx(2 / 5)
    assert recall_at_k(RANKING, JUDGMENTS, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKING, JUDGMENTS, 1) == pytest.approx(1 / 3)
    assert hit_rate_at_k(RANKING, JUDGMENTS, 1) == 1.0
    assert hit_rate_at_k(["B", "D"], JUDGMENTS, 2) == 0.0


def test_reciprocal_rank_and_average_precision() -> None:
    assert reciprocal_rank(RANKING, JUDGMENTS) == 1.0
    assert reciprocal_rank(["X", "A"], {"A": 1}) == 0.5
    assert reciprocal_rank(["X"], {"A": 1}) == 0.0
    # A@1 → 1/1, C@3 → 2/3, F never retrieved; divided by |relevant| = 3
    assert average_precision(RANKING, JUDGMENTS) == pytest.approx((1 + 2 / 3) / 3)
    assert average_precision(["X", "A"], {"A": 1}) == pytest.approx(0.5)


def test_ndcg_linear_gain_hand_computed() -> None:
    dcg = 3 / math.log2(2) + 0 + 2 / math.log2(4)
    idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKING, JUDGMENTS, 3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["X", "A"], {"A": 1}, 2) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["A", "C", "F"], JUDGMENTS, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["Z"], {}, 3) == 0.0


def test_ndcg_exponential_gain() -> None:
    got = ndcg_at_k(["C", "A"], {"A": 3, "C": 2}, 2, exponential=True)
    dcg = (2**2 - 1) / 1 + (2**3 - 1) / math.log2(3)
    idcg = (2**3 - 1) / 1 + (2**2 - 1) / math.log2(3)
    assert got == pytest.approx(dcg / idcg)


def test_threshold_and_bounds() -> None:
    assert recall_at_k(RANKING, JUDGMENTS, 5, threshold=2.0) == pytest.approx(1.0)
    m = evaluate_ranking(RANKING, JUDGMENTS, [1, 3, 10])
    assert all(0.0 <= v <= 1.0 for v in m.values())
    assert set(m) >= {"ndcg@10", "recall@3", "precision@1", "hit@3", "mrr", "map"}
    with pytest.raises(ValueError, match="positive"):
        precision_at_k(RANKING, JUDGMENTS, 0)


def test_collapse_to_documents_keeps_first_occurrence() -> None:
    assert collapse_to_documents(["d1", "d2", "d1", "d3", "d2"]) == ["d1", "d2", "d3"]


def test_percentile_linear() -> None:
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 95) == 5
    assert percentile([], 95) == 0.0
    assert percentile(list(range(101)), 95) == pytest.approx(95)


def _c(cid: str, rank: int, score: float = 0.0) -> Candidate:
    return Candidate(chunk_id=cid, document_id="doc-" + cid, score=score, rank=rank, source="x")


L1 = [_c("a", 1, 10.0), _c("b", 2, 5.0), _c("c", 3, 0.0)]
L2 = [_c("c", 1, 0.9), _c("a", 2, 0.5), _c("d", 3, 0.1)]


def test_rrf_known_values() -> None:
    fused = ReciprocalRankFusion(k=60).fuse({"lexical": L1, "dense": L2}, limit=10)
    scores = {c.chunk_id: c.score for c in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [c.chunk_id for c in fused] == ["a", "c", "b", "d"]
    assert [c.rank for c in fused] == [1, 2, 3, 4]
    assert fused[0].document_id == "doc-a"


def test_weighted_rrf_and_variant_weights() -> None:
    fused = WeightedRRF(k=60, weights={"lexical": 2.0, "dense": 1.0}).fuse({"lexical": L1, "dense": L2}, 10)
    scores = {c.chunk_id: c.score for c in fused}
    assert scores["a"] == pytest.approx(2 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(2 / 63 + 1 / 61)
    # multi-query variant lists inherit their retriever's weight
    v = WeightedRRF(k=60, weights={"lexical": 2.0}).fuse({"lexical#v1": [_c("z", 1)]}, 10)
    assert v[0].score == pytest.approx(2 / 61)
    with pytest.raises(ValueError, match="weights"):
        WeightedRRF(k=60, weights={})


def test_rrf_limit_and_duplicates_within_a_list() -> None:
    dup = [_c("a", 1), _c("a", 2), _c("b", 3)]
    fused = ReciprocalRankFusion(k=1).fuse({"x": dup}, limit=1)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 2)  # second "a" ignored


def test_normalized_minmax_fusion_known_values() -> None:
    fused = NormalizedScoreFusion(weights={"lexical": 0.5, "dense": 0.5}).fuse({"lexical": L1, "dense": L2}, 10)
    scores = {c.chunk_id: c.score for c in fused}
    assert scores == pytest.approx({"a": 0.75, "c": 0.5, "b": 0.25, "d": 0.0})
    assert [c.chunk_id for c in fused] == ["a", "c", "b", "d"]


def test_normalized_fusion_is_scale_invariant_unlike_raw_sums() -> None:
    """Multiplying BM25 scores by 1000 must not change the normalized fusion ranking."""
    scaled = [_c(c.chunk_id, c.rank, c.score * 1000) for c in L1]
    a = NormalizedScoreFusion().fuse({"lexical": L1, "dense": L2}, 10)
    b = NormalizedScoreFusion().fuse({"lexical": scaled, "dense": L2}, 10)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_zscore_and_build_fusion() -> None:
    fused = NormalizedScoreFusion(method="zscore").fuse({"lexical": L1, "dense": L2}, 10)
    assert fused[0].chunk_id == "a"
    assert build_fusion(FusionConfig(strategy=FusionName.RRF)).name == "rrf"
    assert build_fusion(FusionConfig(strategy=FusionName.WEIGHTED_RRF, weights={"lexical": 1.0})).name == "weighted_rrf"
    assert build_fusion(FusionConfig(strategy=FusionName.NORMALIZED)).name == "normalized"
