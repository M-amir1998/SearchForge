"""Qdrant vector-index adapter (``pip install searchforge[qdrant]``).

Implements :class:`searchforge.ports.VectorIndex` using the Query API
(``query_points``; the legacy ``search`` family is deprecated since client 1.19).
Works against a server (``url``), an embedded on-disk store (``path``) or fully
in-memory (``location=":memory:"``) — the last one is what CI uses, so this
adapter is exercised without running any service.

Qdrant point ids must be unsigned ints or UUIDs, so chunk ids are mapped to
deterministic UUIDv5s; the original ids live in the payload. Fusion stays in
SearchForge (client-side), keeping RRF parameters identical across backends.
Each index version gets its own collection (immutability).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from searchforge.config.models import QdrantConfig
from searchforge.domain.document import AttributeValue
from searchforge.domain.filters import FilterExpression
from searchforge.domain.result import Candidate
from searchforge.errors import ConfigurationError, IndexVersionMismatch, RetrievalFailure
from searchforge.filtering.translate import to_qdrant_dict
from searchforge.ports import FloatMatrix, FloatVector

_NAMESPACE = uuid.UUID("8c1f6a3e-5d2b-4f7a-9a51-3c2b7d9e4f10")
VECTOR_NAME = "dense"
UPSERT_BATCH = 256


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


_IN_MEMORY_CLIENTS: dict[str, Any] = {}


def make_client(cfg: QdrantConfig) -> Any:
    """Create a client. ``:memory:`` clients are shared per process (one embedded store), so an
    index built in-memory can be searched by the same process; they never persist across runs."""
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ConfigurationError("qdrant-client is not installed; install searchforge[qdrant]") from exc
    api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
    if cfg.api_key_env and not api_key:
        raise ConfigurationError(f"environment variable {cfg.api_key_env} (Qdrant API key) is not set")
    if cfg.url:
        return QdrantClient(url=cfg.url, api_key=api_key, timeout=int(cfg.timeout_s))
    if cfg.path:
        return QdrantClient(path=cfg.path)
    location = cfg.location or ":memory:"
    if location == ":memory:":
        if location not in _IN_MEMORY_CLIENTS:
            _IN_MEMORY_CLIENTS[location] = QdrantClient(location=location)
        return _IN_MEMORY_CLIENTS[location]
    return QdrantClient(location=location)


def to_qdrant_filter(expr: FilterExpression) -> Any:
    """Convert the canonical filter dict into ``qdrant_client.models.Filter``."""
    if not expr:
        return None
    from qdrant_client import models

    raw = to_qdrant_dict(expr)

    def cond(item: dict[str, Any]) -> Any:
        key = item["key"]
        if "match" in item:
            m = item["match"]
            match = models.MatchAny(any=m["any"]) if "any" in m else models.MatchValue(value=m["value"])
            return models.FieldCondition(key=key, match=match)
        if "datetime_range" in item:
            return models.FieldCondition(key=key, range=models.DatetimeRange(**item["datetime_range"]))
        return models.FieldCondition(key=key, range=models.Range(**item["range"]))

    return models.Filter(
        must=[cond(i) for i in raw.get("must", [])] or None,
        must_not=[cond(i) for i in raw.get("must_not", [])] or None,
    )


class QdrantVectorIndex:
    """Implements :class:`searchforge.ports.VectorIndex` on Qdrant."""

    backend = "qdrant"

    def __init__(self, client: Any, collection: str, dimension: int, *, create: bool = False) -> None:
        from qdrant_client import models

        self._client = client
        self._models = models
        self.collection = collection
        self._dimension = dimension
        if create:
            if client.collection_exists(collection_name=collection):
                raise IndexVersionMismatch(f"Qdrant collection {collection!r} already exists")
            client.create_collection(
                collection_name=collection,
                vectors_config={VECTOR_NAME: models.VectorParams(size=dimension, distance=models.Distance.COSINE)},
            )
            client.create_payload_index(
                collection_name=collection, field_name="document_id", field_schema=models.PayloadSchemaType.KEYWORD
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(
        self,
        chunk_ids: Sequence[str],
        document_ids: Sequence[str],
        attributes: Sequence[Mapping[str, AttributeValue]],
        vectors: FloatMatrix,
    ) -> None:
        if vectors.shape[1] != self._dimension:
            raise IndexVersionMismatch("vector dimension mismatch")
        m = self._models
        for s in range(0, len(chunk_ids), UPSERT_BATCH):
            points = [
                m.PointStruct(
                    id=point_id(chunk_ids[i]),
                    vector={VECTOR_NAME: vectors[i].astype(float).tolist()},
                    payload={
                        "chunk_id": chunk_ids[i],
                        "document_id": document_ids[i],
                        "attributes": dict(attributes[i]),
                    },
                )
                for i in range(s, min(s + UPSERT_BATCH, len(chunk_ids)))
            ]
            self._client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        if not document_ids:
            return
        m = self._models
        selector = m.FilterSelector(
            filter=m.Filter(must=[m.FieldCondition(key="document_id", match=m.MatchAny(any=list(document_ids)))])
        )
        self._client.delete(collection_name=self.collection, points_selector=selector, wait=True)

    def search(self, vector: FloatVector, filters: FilterExpression, k: int) -> list[Candidate]:
        try:
            res = self._client.query_points(
                collection_name=self.collection,
                query=np.asarray(vector, dtype=np.float32).astype(float).tolist(),
                using=VECTOR_NAME,
                query_filter=to_qdrant_filter(filters),
                limit=k,
                with_payload=True,
            )
        except Exception as exc:
            raise RetrievalFailure(f"qdrant query failed: {type(exc).__name__}") from exc
        return [
            Candidate(
                chunk_id=str(p.payload["chunk_id"]),
                document_id=str(p.payload["document_id"]),
                score=float(p.score),
                rank=i + 1,
                source="dense",
            )
            for i, p in enumerate(res.points)
        ]

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, FloatVector]:
        if not chunk_ids:
            return {}
        pts = self._client.retrieve(
            collection_name=self.collection, ids=[point_id(c) for c in chunk_ids], with_vectors=True, with_payload=True
        )
        out: dict[str, FloatVector] = {}
        for p in pts:
            vec = p.vector[VECTOR_NAME] if isinstance(p.vector, dict) else p.vector
            out[str(p.payload["chunk_id"])] = np.asarray(vec, dtype=np.float32)
        return out

    def count(self) -> int:
        return int(self._client.count(collection_name=self.collection, exact=True).count)
