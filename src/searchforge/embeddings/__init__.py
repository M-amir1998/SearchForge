"""Embedding providers (separate from vector storage — ADR-003)."""

from __future__ import annotations

from searchforge.embeddings.cache import CachedEmbeddingProvider, EmbeddingCache, cache_key
from searchforge.embeddings.hashing import HashEmbeddingProvider
from searchforge.embeddings.tfidf import TfidfEmbeddingProvider

__all__ = [
    "CachedEmbeddingProvider",
    "EmbeddingCache",
    "HashEmbeddingProvider",
    "TfidfEmbeddingProvider",
    "cache_key",
]
