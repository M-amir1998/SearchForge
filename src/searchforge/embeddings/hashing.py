"""Deterministic feature-hashing embeddings — FOR TESTS, FIXTURES AND CI ONLY.

These vectors are signed hashed bag-of-words (+ word bigrams). Their similarity is
purely lexical; they carry no semantic knowledge. They exist so dense code paths
(vector index, fusion, MMR, caching) can be exercised deterministically without a
model download. Never report them as semantic search quality.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from searchforge.domain.index import EmbeddingSpec
from searchforge.embeddings.base import EMBED_PREPROCESSING_VERSION, FloatMatrix, l2_normalize, tokenize


class HashEmbeddingProvider:
    """Implements :class:`searchforge.ports.EmbeddingProvider` (non-semantic)."""

    def __init__(self, dimension: int = 256, seed: int = 0, use_bigrams: bool = True) -> None:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        self.dimension = dimension
        self.seed = seed
        self.use_bigrams = use_bigrams
        self._spec = EmbeddingSpec(
            provider="hash",
            model="feature-hash",
            revision=f"v1-s{seed}-b{int(use_bigrams)}",
            dimension=dimension,
            preprocessing_version=EMBED_PREPROCESSING_VERSION,
        )

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(f"{self.seed}\x1f{feature}".encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dimension, (1.0 if (value >> 63) & 1 else -1.0)

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        toks = tokenize(text)
        feats: Counter[str] = Counter(toks)
        if self.use_bigrams:
            feats.update(f"{a} {b}" for a, b in itertools.pairwise(toks))
        for feat, tf in feats.items():
            idx, sign = self._bucket(feat)
            vec[idx] += sign * (1.0 + math.log(tf))
        return vec

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return l2_normalize(np.stack([self._embed(t) for t in texts]))

    def embed_query(self, text: str) -> FloatMatrix:
        return l2_normalize(self._embed(text))
