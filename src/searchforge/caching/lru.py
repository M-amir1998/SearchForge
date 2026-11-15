"""Bounded, thread-safe LRU cache with TTL, entry and approximate-byte limits."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

V = TypeVar("V")


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    entries: int = 0
    bytes: int = 0
    latency_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def miss_rate(self) -> float:
        total = self.hits + self.misses
        return self.misses / total if total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
            "evictions": self.evictions,
            "expirations": self.expirations,
            "entries": self.entries,
            "bytes": self.bytes,
            "latency_saved_ms": round(self.latency_saved_ms, 3),
        }


@dataclass(slots=True)
class _Entry(Generic[V]):
    value: V
    size: int
    expires_at: float
    cost_ms: float


class BoundedLRUCache(Generic[V]):
    """Never grows beyond ``max_entries`` entries or ``max_bytes`` (approximate) bytes."""

    def __init__(
        self,
        max_entries: int,
        max_bytes: int,
        ttl_s: float,
        sizeof: Callable[[V], int],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1 or max_bytes < 1 or ttl_s <= 0:
            raise ValueError("cache limits must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl_s = ttl_s
        self._sizeof = sizeof
        self._clock = clock
        self._data: OrderedDict[str, _Entry[V]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, key: str) -> V | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            if entry.expires_at <= self._clock():
                self._remove(key)
                self.stats.expirations += 1
                self.stats.misses += 1
                return None
            self._data.move_to_end(key)
            self.stats.hits += 1
            self.stats.latency_saved_ms += entry.cost_ms
            return entry.value

    def put(self, key: str, value: V, cost_ms: float = 0.0) -> bool:
        """Insert; returns False if a single value exceeds ``max_bytes`` (not cached)."""
        size = max(1, int(self._sizeof(value)))
        if size > self.max_bytes:
            return False
        with self._lock:
            if key in self._data:
                self._remove(key)
            self._data[key] = _Entry(value, size, self._clock() + self.ttl_s, cost_ms)
            self._bytes += size
            while len(self._data) > self.max_entries or self._bytes > self.max_bytes:
                oldest = next(iter(self._data))
                self._remove(oldest)
                self.stats.evictions += 1
            self._sync()
        return True

    def _remove(self, key: str) -> None:
        entry = self._data.pop(key)
        self._bytes -= entry.size
        self._sync()

    def _sync(self) -> None:
        self.stats.entries = len(self._data)
        self.stats.bytes = self._bytes

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0
            self._sync()

    def __len__(self) -> int:
        return len(self._data)
