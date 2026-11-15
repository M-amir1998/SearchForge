"""Remote embeddings over an OpenAI-compatible ``/embeddings`` HTTP endpoint.

Trust model (see SECURITY.md):

* The endpoint is an external service. Corpus text is transmitted **only** when
  ``allow_corpus_transmission: true`` is set explicitly; otherwise document
  embedding raises before any network I/O.
* API keys are read from an environment variable named in config — never stored in
  config files or index manifests.
* Calls are wrapped in a circuit breaker so a failing endpoint degrades search
  (lexical-only) instead of stalling every request.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np

from searchforge.config.models import RemoteEmbeddingConfig
from searchforge.domain.index import EmbeddingSpec
from searchforge.embeddings.base import EMBED_PREPROCESSING_VERSION, FloatMatrix, l2_normalize
from searchforge.errors import CircuitOpenError, EmbeddingFailure
from searchforge.service.circuit_breaker import CircuitBreaker


class RemoteEmbeddingProvider:
    """Implements :class:`searchforge.ports.EmbeddingProvider` over HTTP."""

    def __init__(
        self,
        config: RemoteEmbeddingConfig,
        *,
        client: httpx.Client | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.config = config
        self._client = client or httpx.Client(timeout=config.timeout_s)
        self._breaker = breaker or CircuitBreaker(name="remote-embeddings")
        self._dimension = config.dimension
        self._spec: EmbeddingSpec | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if not key:
                raise EmbeddingFailure(f"environment variable {self.config.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _post(self, texts: Sequence[str]) -> FloatMatrix:
        payload: dict[str, Any] = {"model": self.config.model, "input": list(texts)}
        try:
            with self._breaker.guard():
                resp = self._client.post(self.config.url, json=payload, headers=self._headers())
                resp.raise_for_status()
                body = resp.json()
        except CircuitOpenError as exc:
            raise EmbeddingFailure(str(exc)) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingFailure(f"remote embedding request failed: {type(exc).__name__}") from exc
        try:
            data = sorted(body["data"], key=lambda item: int(item.get("index", 0)))
            matrix = np.asarray([item["embedding"] for item in data], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingFailure("malformed embedding response") from exc
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise EmbeddingFailure("embedding response size mismatch")
        if self._dimension is None:
            self._dimension = int(matrix.shape[1])
        elif matrix.shape[1] != self._dimension:
            raise EmbeddingFailure(f"expected dimension {self._dimension}, got {matrix.shape[1]}")
        return l2_normalize(matrix)

    @property
    def spec(self) -> EmbeddingSpec:
        if self._spec is None:
            if self._dimension is None:
                self._post(["dimension probe"])
            assert self._dimension is not None
            self._spec = EmbeddingSpec(
                provider="remote",
                model=self.config.model,
                revision="remote-unversioned",
                dimension=self._dimension,
                preprocessing_version=EMBED_PREPROCESSING_VERSION,
            )
        return self._spec

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        if not self.config.allow_corpus_transmission:
            raise EmbeddingFailure(
                "refusing to send corpus text to a remote endpoint: set "
                "dense.remote.allow_corpus_transmission: true to opt in explicitly"
            )
        if not texts:
            return np.zeros((0, self.spec.dimension), dtype=np.float32)
        parts = [
            self._post(texts[i : i + self.config.batch_size]) for i in range(0, len(texts), self.config.batch_size)
        ]
        return np.vstack(parts).astype(np.float32)

    def embed_query(self, text: str) -> FloatMatrix:
        row: FloatMatrix = self._post([text])[0]
        return row
