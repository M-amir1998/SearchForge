"""Persistent embedding cache (SQLite, float32 blobs).

Key = SHA-256 of (text hash, provider, model, model revision, preprocessing
version, dimension). A new model revision or preprocessing change therefore never
returns stale vectors. The cache lives in the workspace (``.searchforge/cache``),
is git-ignored, inspectable (:meth:`stats`) and removable (:meth:`clear`).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from searchforge.domain.ids import sha256_hex
from searchforge.domain.index import EmbeddingSpec
from searchforge.embeddings.base import FloatMatrix
from searchforge.ports import EmbeddingProvider


def cache_key(text: str, spec: EmbeddingSpec) -> str:
    return sha256_hex(
        "\x1f".join(
            [
                sha256_hex(text),
                spec.provider,
                spec.model,
                spec.revision,
                spec.preprocessing_version,
                str(spec.dimension),
            ]
        )
    )


@dataclass(slots=True)
class EmbeddingCacheStats:
    entries: int
    bytes: int
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class EmbeddingCache:
    FILENAME = "embeddings.sqlite"

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / self.FILENAME
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, "
            "vec BLOB NOT NULL, created REAL NOT NULL)"
        )
        self._conn.commit()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                batch = list(keys[start : start + 500])
                marks = ",".join("?" for _ in batch)
                for key, dim, blob in self._conn.execute(
                    f"SELECT key, dim, vec FROM emb WHERE key IN ({marks})", batch
                ):
                    vec = np.frombuffer(blob, dtype=np.float32)
                    if vec.shape[0] == dim:
                        out[str(key)] = vec
        return out

    def put_many(self, items: Sequence[tuple[str, np.ndarray]]) -> None:
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO emb (key, dim, vec, created) VALUES (?,?,?,?)",
                [(k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes(), now) for k, v in items],
            )
            self._conn.commit()

    def stats(self) -> EmbeddingCacheStats:
        with self._lock:
            entries, size = self._conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(vec)),0) FROM emb").fetchone()
        return EmbeddingCacheStats(entries=int(entries), bytes=int(size), hits=self.hits, misses=self.misses)

    def clear(self) -> int:
        with self._lock:
            n = int(self._conn.execute("SELECT COUNT(*) FROM emb").fetchone()[0])
            self._conn.execute("DELETE FROM emb")
            self._conn.commit()
            self._conn.execute("VACUUM")
        return n

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CachedEmbeddingProvider:
    """Wraps a provider; document embeddings are served from / written to the cache."""

    def __init__(self, inner: EmbeddingProvider, cache: EmbeddingCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def spec(self) -> EmbeddingSpec:
        return self.inner.spec

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        spec = self.inner.spec
        keys = [cache_key(t, spec) for t in texts]
        found = self.cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in found]
        self.cache.hits += len(texts) - len(missing)
        self.cache.misses += len(missing)
        if missing:
            fresh = self.inner.embed_documents([texts[i] for i in missing])
            self.cache.put_many([(keys[i], fresh[j]) for j, i in enumerate(missing)])
            for j, i in enumerate(missing):
                found[keys[i]] = fresh[j]
        if not texts:
            return np.zeros((0, spec.dimension), dtype=np.float32)
        return np.stack([found[k] for k in keys]).astype(np.float32)

    def embed_query(self, text: str) -> FloatMatrix:
        return self.inner.embed_query(text)
