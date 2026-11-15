"""Built-in search profiles. User profiles with the same name override them."""

from __future__ import annotations

from searchforge.config.models import (
    DiversityConfig,
    FusionConfig,
    FusionName,
    RerankerConfig,
    RerankerName,
    RetrieverName,
    SearchConfig,
    SearchProfile,
)

LEX = RetrieverName.LEXICAL
DENSE = RetrieverName.DENSE


def builtin_profiles() -> dict[str, SearchProfile]:
    return {
        "lexical": SearchProfile(description="BM25 over SQLite FTS5 (B0)", retrievers=[LEX]),
        "dense": SearchProfile(description="Dense vector retrieval (B1)", retrievers=[DENSE]),
        "hybrid": SearchProfile(
            description="Lexical + dense, Reciprocal Rank Fusion k=60 (B2)",
            retrievers=[LEX, DENSE],
            fusion=FusionConfig(strategy=FusionName.RRF, k=60),
        ),
        "weighted_hybrid": SearchProfile(
            description="Lexical + dense, weighted RRF (B3); tune weights on train/validation only",
            retrievers=[LEX, DENSE],
            fusion=FusionConfig(strategy=FusionName.WEIGHTED_RRF, k=60, weights={"lexical": 1.0, "dense": 0.5}),
        ),
        "hybrid_normalized": SearchProfile(
            description="Lexical + dense, min-max normalized score fusion",
            retrievers=[LEX, DENSE],
            fusion=FusionConfig(strategy=FusionName.NORMALIZED, weights={"lexical": 0.5, "dense": 0.5}),
        ),
        "hybrid_rerank": SearchProfile(
            description="Hybrid RRF + CPU lexical-feature reranker over top-50 (B4)",
            retrievers=[LEX, DENSE],
            reranker=RerankerConfig(type=RerankerName.LEXICAL_FEATURES, top_n=50),
            diversity=DiversityConfig(max_per_document=3),
        ),
        "hybrid_cross_encoder": SearchProfile(
            description="Hybrid RRF + Sentence-Transformers cross-encoder (optional [neural])",
            retrievers=[LEX, DENSE],
            reranker=RerankerConfig(type=RerankerName.CROSS_ENCODER, top_n=50),
        ),
        "adaptive": SearchProfile(
            description="Query-planned adaptive retrieval (B5): rules route each query to a profile",
            retrievers=[LEX, DENSE],
            adaptive=True,
        ),
    }


def resolve_profiles(search: SearchConfig) -> dict[str, SearchProfile]:
    profiles = builtin_profiles()
    profiles.update(search.profiles)
    return profiles
