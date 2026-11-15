"""TF-IDF (+ optional LSA/TruncatedSVD) embeddings: the classical CPU dense baseline.

* Fitting uses scikit-learn (``TfidfVectorizer`` + ``TruncatedSVD``) on the indexed
  chunks; the fitted state is part of the index version.
* Persistence is pickle-free: vocabulary/params as JSON, ``idf`` and SVD
  components as ``.npy`` loaded with ``allow_pickle=False`` — a downloaded index
  cannot execute code.
* Transform uses scikit-learn's ``CountVectorizer`` with the frozen vocabulary for counting, then
  applies the TF-IDF math explicitly with SciPy sparse ops (identical to scikit-learn's
  ``TfidfTransformer``: sublinear ``1 + log(tf)``, idf weighting, L2), verified by a unit test.
* With SVD (LSA), vectors capture term co-occurrence ("semantic-ish"); without it,
  they are dense TF-IDF and essentially lexical. Either way this is a *baseline*;
  BM25 lexical retrieval is kept separate.

Incremental updates reuse the parent's fitted state (frozen vocabulary): new
terms are invisible to this provider until a full rebuild refits it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from searchforge.config.models import TfidfConfig
from searchforge.domain.ids import canonical_json, sha256_hex, short_hash
from searchforge.domain.index import EmbeddingSpec
from searchforge.embeddings.base import EMBED_PREPROCESSING_VERSION, FloatMatrix, l2_normalize
from searchforge.errors import EmbeddingFailure, IndexIntegrityError

TOKEN_PATTERN = r"(?u)\b\w+\b"


class TfidfEmbeddingProvider:
    """Implements :class:`searchforge.ports.EmbeddingProvider` after :meth:`fit`/:meth:`load`."""

    STATE_FILE = "tfidf.json"
    IDF_FILE = "idf.npy"
    SVD_FILE = "svd_components.npy"

    def __init__(self, config: TfidfConfig | None = None) -> None:
        self.config = config or TfidfConfig()
        self._vocabulary: dict[str, int] | None = None
        self._idf: np.ndarray | None = None
        self._components: np.ndarray | None = None
        self._spec: EmbeddingSpec | None = None
        self._vectorizer: Any = None

    # ------------------------------------------------------------------ fitting
    def fit(self, texts: Iterable[str]) -> TfidfEmbeddingProvider:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        cfg = self.config
        sample = []
        for i, t in enumerate(texts):
            if i >= cfg.fit_max_chunks:
                break
            sample.append(t)
        if not sample:
            raise EmbeddingFailure("cannot fit TF-IDF on an empty corpus")
        vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, cfg.ngram_max),
            sublinear_tf=cfg.sublinear_tf,
            min_df=cfg.min_df,
            max_features=cfg.max_features,
            stop_words="english",
            dtype=np.float32,
        )
        matrix = vec.fit_transform(sample)
        self._vocabulary = {str(k): int(v) for k, v in vec.vocabulary_.items()}
        self._idf = np.asarray(vec.idf_, dtype=np.float32)
        n_features = matrix.shape[1]
        if cfg.svd_components is not None:
            n_comp = min(cfg.svd_components, n_features - 1, matrix.shape[0] - 1)
            if n_comp >= 2:
                svd = TruncatedSVD(n_components=n_comp, random_state=cfg.random_state, algorithm="randomized", n_iter=7)
                svd.fit(matrix)
                self._components = np.asarray(svd.components_, dtype=np.float32)
        self._finalize()
        return self

    def _finalize(self) -> None:
        from sklearn.feature_extraction.text import CountVectorizer

        assert self._vocabulary is not None
        assert self._idf is not None
        self._vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, self.config.ngram_max),
            stop_words="english",
            vocabulary=self._vocabulary,
            dtype=np.float32,
        )
        dim = self._components.shape[0] if self._components is not None else len(self._vocabulary)
        state_hash = short_hash(
            canonical_json(sorted(self._vocabulary.items()))
            + sha256_hex(self._idf.tobytes())
            + (sha256_hex(self._components.tobytes()) if self._components is not None else ""),
            12,
        )
        self._spec = EmbeddingSpec(
            provider="tfidf",
            model="tfidf-lsa" if self._components is not None else "tfidf",
            revision=f"fit-{state_hash}",
            dimension=int(dim),
            preprocessing_version=EMBED_PREPROCESSING_VERSION,
        )

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        if self._vocabulary is None or self._idf is None:
            raise EmbeddingFailure("TF-IDF provider is not fitted")
        directory.mkdir(parents=True, exist_ok=True)
        state = {"config": self.config.model_dump(), "vocabulary": self._vocabulary}
        (directory / self.STATE_FILE).write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        np.save(directory / self.IDF_FILE, self._idf, allow_pickle=False)
        if self._components is not None:
            np.save(directory / self.SVD_FILE, self._components, allow_pickle=False)

    @classmethod
    def load(cls, directory: Path) -> TfidfEmbeddingProvider:
        state_path = directory / cls.STATE_FILE
        if not state_path.exists():
            raise IndexIntegrityError(f"missing TF-IDF state in {directory}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        provider = cls(TfidfConfig.model_validate(state["config"]))
        provider._vocabulary = {str(k): int(v) for k, v in state["vocabulary"].items()}
        provider._idf = np.load(directory / cls.IDF_FILE, allow_pickle=False).astype(np.float32)
        svd = directory / cls.SVD_FILE
        provider._components = np.load(svd, allow_pickle=False).astype(np.float32) if svd.exists() else None
        provider._finalize()
        return provider

    # ------------------------------------------------------------------ embedding
    @property
    def spec(self) -> EmbeddingSpec:
        if self._spec is None:
            raise EmbeddingFailure("TF-IDF provider is not fitted")
        return self._spec

    def _transform(self, texts: Sequence[str]) -> FloatMatrix:
        if self._vectorizer is None or self._idf is None:
            raise EmbeddingFailure("TF-IDF provider is not fitted")
        counts = sparse.csr_matrix(self._vectorizer.transform(list(texts)), dtype=np.float32)
        if self.config.sublinear_tf and counts.nnz:
            counts.data = np.log(counts.data) + 1.0
        weighted = counts @ sparse.diags(self._idf)
        weighted = sparse.csr_matrix(weighted, dtype=np.float32)
        norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        weighted = sparse.csr_matrix(sparse.diags(1.0 / norms) @ weighted, dtype=np.float32)
        if self._components is not None:
            dense = np.asarray(weighted @ self._components.T, dtype=np.float32)
        else:
            dense = np.asarray(weighted.toarray(), dtype=np.float32)
        return l2_normalize(dense)

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            return np.zeros((0, self.spec.dimension), dtype=np.float32)
        return self._transform(texts)

    def embed_query(self, text: str) -> FloatMatrix:
        row: FloatMatrix = self._transform([text])[0]
        return row
