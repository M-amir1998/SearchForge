"""Caching: bounded in-memory query-result cache (embedding cache lives in embeddings/)."""

from __future__ import annotations

from searchforge.caching.lru import BoundedLRUCache, CacheStats
from searchforge.caching.query_cache import QueryResultCache, query_cache_key

__all__ = ["BoundedLRUCache", "CacheStats", "QueryResultCache", "query_cache_key"]
