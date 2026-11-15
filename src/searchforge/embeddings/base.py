"""Shared embedding utilities."""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]

#: Bump when text preprocessing before embedding changes (participates in cache keys).
EMBED_PREPROCESSING_VERSION = "embed-pre-v1"

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def l2_normalize(matrix: NDArray[np.floating]) -> FloatMatrix:
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        norm = float(np.linalg.norm(m))
        return (m / norm).astype(np.float32) if norm > 0 else m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: FloatMatrix = (m / norms).astype(np.float32)
    return out
