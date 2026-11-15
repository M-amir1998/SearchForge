"""OpenSearch adapters (``pip install searchforge[opensearch]``).

* :class:`OpenSearchLexicalIndex` — BM25 ``multi_match`` over ``body``/``title``
  with structured ``bool.filter`` clauses.
* :class:`OpenSearchVectorIndex` — ``knn_vector`` (HNSW, faiss engine, cosine)
  with efficient k-NN filtering.

All requests are structured Query DSL dicts; user text only ever appears as the
*value* of a ``query`` field, never inside a query-string syntax, so it cannot
inject operators. Fusion stays client-side (hybrid pipelines are optional).

OpenSearch is memory-hungry (JVM heap + k-NN native memory); it runs only behind
an explicit docker-compose profile with resource limits and never in standard CI.
Live tests run only when ``OPENSEARCH_URL`` is set.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from searchforge.config.models import OpenSearchConfig
from searchforge.domain.chunk import Chunk
from searchforge.domain.document import AttributeValue
from searchforge.domain.filters import FilterExpression
from searchforge.domain.result import Candidate
from searchforge.errors import ConfigurationError, RetrievalFailure
from searchforge.filtering.translate import to_opensearch
from searchforge.ports import FloatMatrix, FloatVector

BULK_BATCH = 500


def make_client(cfg: OpenSearchConfig) -> Any:
    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError("opensearch-py is not installed; install searchforge[opensearch]") from exc
    auth = None
    if cfg.username_env and cfg.password_env:
        user, pwd = os.environ.get(cfg.username_env), os.environ.get(cfg.password_env)
        if user and pwd:
            auth = (user, pwd)
    return OpenSearch(hosts=cfg.hosts, http_auth=auth, verify_certs=cfg.verify_certs, timeout=cfg.timeout_s)


def lexical_mapping() -> dict[str, Any]:
    return {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "dynamic_templates": [
                {
                    "attr_strings": {
                        "path_match": "attributes.*",
                        "match_mapping_type": "string",
                        "mapping": {"type": "keyword"},
                    }
                }
            ],
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "body": {"type": "text", "analyzer": "english"},
                "title": {"type": "text", "analyzer": "english"},
                "attributes": {"type": "object"},
            },
        },
    }


def knn_mapping(dimension: int) -> dict[str, Any]:
    return {
        "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "dynamic_templates": [
                {
                    "attr_strings": {
                        "path_match": "attributes.*",
                        "match_mapping_type": "string",
                        "mapping": {"type": "keyword"},
                    }
                }
            ],
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "attributes": {"type": "object"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "space_type": "cosinesimil",
                    "method": {"name": "hnsw", "engine": "faiss", "parameters": {"m": 16, "ef_construction": 128}},
                },
            },
        },
    }


def lexical_query(text: str, filters: FilterExpression, k: int) -> dict[str, Any]:
    return {
        "size": k,
        "_source": ["chunk_id", "document_id"],
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": text,
                            "fields": ["body", "title^2"],
                            "type": "best_fields",
                            "operator": "or",
                        }
                    }
                ],
                "filter": to_opensearch(filters),
            }
        },
    }


def knn_query(vector: Sequence[float], filters: FilterExpression, k: int) -> dict[str, Any]:
    knn: dict[str, Any] = {"vector": list(vector), "k": k}
    clauses = to_opensearch(filters)
    if clauses:
        knn["filter"] = {"bool": {"filter": clauses}}
    return {"size": k, "_source": ["chunk_id", "document_id"], "query": {"knn": {"embedding": knn}}}


def _hits_to_candidates(resp: Mapping[str, Any], source: str) -> list[Candidate]:
    hits = resp.get("hits", {}).get("hits", [])
    return [
        Candidate(
            chunk_id=str(h["_source"]["chunk_id"]),
            document_id=str(h["_source"]["document_id"]),
            score=float(h.get("_score") or 0.0),
            rank=i + 1,
            source=source,
        )
        for i, h in enumerate(hits)
    ]


class OpenSearchLexicalIndex:
    """Implements :class:`searchforge.ports.LexicalIndex` on OpenSearch BM25."""

    backend = "opensearch"

    def __init__(self, client: Any, index_name: str, *, create: bool = False) -> None:
        self._client = client
        self.index_name = index_name
        if create:
            client.indices.create(index=index_name, body=lexical_mapping())

    def add(self, chunks: Sequence[Chunk]) -> None:
        from opensearchpy import helpers

        actions = [
            {
                "_index": self.index_name,
                "_id": c.chunk_id,
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "body": c.text,
                "title": " | ".join(p for p in (c.metadata.title, c.metadata.section) if p),
                "attributes": dict(c.metadata.attributes),
            }
            for c in chunks
        ]
        for s in range(0, len(actions), BULK_BATCH):
            helpers.bulk(self._client, actions[s : s + BULK_BATCH])
        self._client.indices.refresh(index=self.index_name)

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        if document_ids:
            self._client.delete_by_query(
                index=self.index_name, body={"query": {"terms": {"document_id": list(document_ids)}}}, refresh=True
            )

    def search(self, text: str, filters: FilterExpression, k: int, phrases: Sequence[str] = ()) -> list[Candidate]:
        try:
            resp = self._client.search(index=self.index_name, body=lexical_query(text, filters, k))
        except Exception as exc:
            raise RetrievalFailure(f"opensearch lexical query failed: {type(exc).__name__}") from exc
        return _hits_to_candidates(resp, "lexical")

    def count(self) -> int:
        return int(self._client.count(index=self.index_name)["count"])


class OpenSearchVectorIndex:
    """Implements :class:`searchforge.ports.VectorIndex` on the OpenSearch k-NN plugin."""

    backend = "opensearch_knn"

    def __init__(self, client: Any, index_name: str, dimension: int, *, create: bool = False) -> None:
        self._client = client
        self.index_name = index_name
        self._dimension = dimension
        if create:
            client.indices.create(index=index_name, body=knn_mapping(dimension))

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
        from opensearchpy import helpers

        actions = [
            {
                "_index": self.index_name,
                "_id": chunk_ids[i],
                "chunk_id": chunk_ids[i],
                "document_id": document_ids[i],
                "attributes": dict(attributes[i]),
                "embedding": vectors[i].astype(float).tolist(),
            }
            for i in range(len(chunk_ids))
        ]
        for s in range(0, len(actions), BULK_BATCH):
            helpers.bulk(self._client, actions[s : s + BULK_BATCH])
        self._client.indices.refresh(index=self.index_name)

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        if document_ids:
            self._client.delete_by_query(
                index=self.index_name, body={"query": {"terms": {"document_id": list(document_ids)}}}, refresh=True
            )

    def search(self, vector: FloatVector, filters: FilterExpression, k: int) -> list[Candidate]:
        body = knn_query(np.asarray(vector, dtype=np.float32).astype(float).tolist(), filters, k)
        try:
            resp = self._client.search(index=self.index_name, body=body)
        except Exception as exc:
            raise RetrievalFailure(f"opensearch knn query failed: {type(exc).__name__}") from exc
        return _hits_to_candidates(resp, "dense")

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, FloatVector]:
        if not chunk_ids:
            return {}
        resp = self._client.mget(index=self.index_name, body={"ids": list(chunk_ids)})
        return {
            d["_id"]: np.asarray(d["_source"]["embedding"], dtype=np.float32)
            for d in resp.get("docs", [])
            if d.get("found")
        }

    def count(self) -> int:
        return int(self._client.count(index=self.index_name)["count"])
