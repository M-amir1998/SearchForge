"""Version-aware query-result cache.

The key includes everything that can change the answer: normalized query text,
canonical filters, *index id (name@version)*, planner version, profile, top-k and
the explain flag. Publishing a new index version therefore can never serve stale
results from the previous version — old entries simply stop being addressed and
age out by LRU/TTL.
"""

from __future__ import annotations

from typing import Any

from searchforge.caching.lru import BoundedLRUCache
from searchforge.config.models import CacheConfig
from searchforge.domain.filters import FilterExpression
from searchforge.domain.ids import canonical_json, sha256_hex
from searchforge.domain.result import SearchResponse

KEY_VERSION = "qc-v1"


def query_cache_key(
    *,
    text: str,
    filters: FilterExpression,
    index_id: str,
    planner_version: str,
    profile: str,
    top_k: int,
    explain: bool,
) -> str:
    payload: dict[str, Any] = {
        "v": KEY_VERSION,
        "q": " ".join(text.split()).lower(),
        "f": filters.canonical(),
        "i": index_id,
        "p": planner_version,
        "pr": profile,
        "k": top_k,
        "e": explain,
    }
    return sha256_hex(canonical_json(payload))


def response_size(resp: SearchResponse) -> int:
    """Approximate bytes: text + metadata JSON per result plus a fixed overhead."""
    total = 512
    for r in resp.results:
        total += 256 + len(r.text or "") * 2 + len(canonical_json(r.metadata))
        if r.explanation:
            total += len(canonical_json(r.explanation))
    return total


class QueryResultCache:
    def __init__(self, config: CacheConfig) -> None:
        self.enabled = config.enabled
        self._cache: BoundedLRUCache[SearchResponse] = BoundedLRUCache(
            max_entries=config.max_entries, max_bytes=config.max_bytes, ttl_s=config.ttl_s, sizeof=response_size
        )

    @property
    def stats(self) -> Any:
        return self._cache.stats

    def get(self, key: str) -> SearchResponse | None:
        return self._cache.get(key) if self.enabled else None

    def put(self, key: str, response: SearchResponse) -> None:
        if self.enabled:
            self._cache.put(key, response, cost_ms=response.latency_ms)

    def clear(self) -> None:
        self._cache.clear()
