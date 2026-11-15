"""Optional Sentence Transformers embeddings (``pip install searchforge[neural]``).

CPU-only by construction (``device="cpu"``). Models are never downloaded unless
``allow_download=True``; otherwise only locally cached checkpoints load
(``local_files_only``), so CI and offline runs fail fast instead of fetching.
Pin ``revision`` to a commit hash for reproducible indexes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from searchforge.domain.index import EmbeddingSpec
from searchforge.embeddings.base import EMBED_PREPROCESSING_VERSION, FloatMatrix, l2_normalize
from searchforge.errors import EmbeddingFailure


def _load_model(model: str, revision: str | None, allow_download: bool) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise EmbeddingFailure("sentence-transformers is not installed; install searchforge[neural]") from exc
    try:
        return SentenceTransformer(model, revision=revision, device="cpu", local_files_only=not allow_download)
    except Exception as exc:
        hint = "" if allow_download else " (model not cached locally and allow_download=false)"
        raise EmbeddingFailure(f"failed to load sentence-transformer {model!r}{hint}: {exc}") from exc


class SentenceTransformerEmbeddingProvider:
    """Implements :class:`searchforge.ports.EmbeddingProvider`."""

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        *,
        batch_size: int = 32,
        allow_download: bool = False,
        query_prefix: str = "",
        document_prefix: str = "",
        model_obj: Any | None = None,
    ) -> None:
        self.model_name = model
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model = model_obj if model_obj is not None else _load_model(model, revision, allow_download)
        get_dim = getattr(self._model, "get_sentence_embedding_dimension", None) or getattr(
            self._model, "get_embedding_dimension", None
        )
        dim = int(get_dim()) if get_dim is not None else 0
        if dim <= 0:
            dim = int(self._encode(["dimension probe"]).shape[1])
        self._spec = EmbeddingSpec(
            provider="sentence_transformer",
            model=model,
            revision=revision or "unpinned",
            dimension=dim,
            preprocessing_version=EMBED_PREPROCESSING_VERSION,
        )

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def _encode(self, texts: Sequence[str]) -> FloatMatrix:
        try:
            out = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingFailure(f"sentence-transformer encode failed: {exc}") from exc
        return l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            return np.zeros((0, self._spec.dimension), dtype=np.float32)
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> FloatMatrix:
        row: FloatMatrix = self._encode([self.query_prefix + text])[0]
        return row
