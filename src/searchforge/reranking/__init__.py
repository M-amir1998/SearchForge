"""Second-stage reranking."""

from __future__ import annotations

from searchforge.config.models import RerankerConfig, RerankerName
from searchforge.ports import Reranker
from searchforge.reranking.rerankers import CrossEncoderReranker, LexicalFeatureReranker, NoOpReranker


def build_reranker(config: RerankerConfig) -> Reranker:
    if config.type is RerankerName.LEXICAL_FEATURES:
        return LexicalFeatureReranker(config.feature_weights or None)
    if config.type is RerankerName.CROSS_ENCODER:
        return CrossEncoderReranker(
            config.model, config.revision, batch_size=config.batch_size, allow_download=config.allow_download
        )
    return NoOpReranker()


__all__ = ["CrossEncoderReranker", "LexicalFeatureReranker", "NoOpReranker", "build_reranker"]
