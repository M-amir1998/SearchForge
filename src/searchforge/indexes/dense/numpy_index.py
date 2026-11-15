"""Exact (brute-force) inner-product vector index on NumPy.

Designed for the ``tiny``/``small``/``research`` data profiles on a CPU laptop:

* vectors are float32, L2-normalized → inner product == cosine similarity,
* persisted as ``vectors.npy`` (opened with ``mmap_mode="r"`` for search, so the OS
  pages vectors in on demand instead of loading them all into RAM),
* scoring is blockwise (``block_rows``) so peak temporary memory is bounded,
* filtering is *pre*-filtering via a boolean mask computed from canonical filters
  (masks are memoized per distinct filter),
* deletes are tombstones in memory; :meth:`save` compacts.

Exact search is the right baseline for evaluation (no ANN recall loss confounding
relevance metrics). For millions of vectors use an external ANN backend (Qdrant).
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from searchforge.domain.document import AttributeValue
from searchforge.domain.filters import FilterExpression
from searchforge.domain.ids import canonical_json
from searchforge.domain.result import Candidate
from searchforge.errors import IndexIntegrityError, IndexVersionMismatch

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]
BoolMask = NDArray[np.bool_]


class NumpyVectorIndex:
    """Implements :class:`searchforge.ports.VectorIndex`."""

    backend = "numpy"
    VECTORS = "vectors.npy"
    ROWS = "rows.jsonl"

    def __init__(self, dimension: int, block_rows: int = 65_536, mask_cache_size: int = 64) -> None:
        self._dimension = dimension
        self.block_rows = block_rows
        self._chunk_ids: list[str] = []
        self._document_ids: list[str] = []
        self._attributes: list[dict[str, AttributeValue]] = []
        self._base: FloatMatrix = np.zeros((0, dimension), dtype=np.float32)
        self._pending: list[FloatMatrix] = []
        self._alive: BoolMask = np.zeros(0, dtype=np.bool_)
        self._row_of: dict[str, int] = {}
        self._mask_cache: OrderedDict[str, BoolMask] = OrderedDict()
        self._mask_cache_size = mask_cache_size
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    # ------------------------------------------------------------------ build
    def add(
        self,
        chunk_ids: Sequence[str],
        document_ids: Sequence[str],
        attributes: Sequence[Mapping[str, AttributeValue]],
        vectors: FloatMatrix,
    ) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self._dimension:
            raise IndexVersionMismatch(
                f"vector dimension {vectors.shape} does not match index dimension {self._dimension}"
            )
        if not len(chunk_ids) == len(document_ids) == len(attributes) == vectors.shape[0]:
            raise ValueError("chunk_ids, document_ids, attributes and vectors must align")
        start = len(self._chunk_ids)
        for offset, cid in enumerate(chunk_ids):
            if cid in self._row_of and self._alive[self._row_of[cid]]:
                raise ValueError(f"duplicate chunk id {cid}")
            self._row_of[cid] = start + offset
        self._chunk_ids.extend(chunk_ids)
        self._document_ids.extend(document_ids)
        self._attributes.extend(dict(a) for a in attributes)
        self._pending.append(np.ascontiguousarray(vectors, dtype=np.float32))
        self._alive = np.concatenate([self._alive, np.ones(len(chunk_ids), dtype=np.bool_)])
        self._mask_cache.clear()

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        targets = set(document_ids)
        for i, did in enumerate(self._document_ids):
            if did in targets:
                self._alive[i] = False
        self._mask_cache.clear()

    def _matrix(self) -> FloatMatrix:
        if self._pending:
            with self._lock:
                if self._pending:
                    self._base = np.vstack([np.asarray(self._base), *self._pending]).astype(np.float32)
                    self._pending = []
        return self._base

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        """Write a compacted copy (tombstoned rows dropped)."""
        directory.mkdir(parents=True, exist_ok=True)
        matrix = self._matrix()
        keep = np.flatnonzero(self._alive)
        out = np.lib.format.open_memmap(
            directory / self.VECTORS, mode="w+", dtype=np.float32, shape=(len(keep), self._dimension)
        )
        for s in range(0, len(keep), self.block_rows):
            idx = keep[s : s + self.block_rows]
            out[s : s + len(idx)] = matrix[idx]
        out.flush()
        del out
        with open(directory / self.ROWS, "w", encoding="utf-8") as fh:
            for i in keep:
                fh.write(
                    canonical_json({"c": self._chunk_ids[i], "d": self._document_ids[i], "a": self._attributes[i]})
                    + "\n"
                )

    @classmethod
    def load(cls, directory: Path, *, mmap: bool = True, block_rows: int = 65_536) -> NumpyVectorIndex:
        vec_path = directory / cls.VECTORS
        rows_path = directory / cls.ROWS
        if not vec_path.exists() or not rows_path.exists():
            raise IndexIntegrityError(f"missing dense index files in {directory}")
        matrix = np.load(vec_path, mmap_mode="r" if mmap else None, allow_pickle=False)
        if matrix.dtype != np.float32 or matrix.ndim != 2:
            raise IndexIntegrityError("dense vectors must be a 2-D float32 array")
        index = cls(int(matrix.shape[1]), block_rows=block_rows)
        with open(rows_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                index._row_of[row["c"]] = len(index._chunk_ids)
                index._chunk_ids.append(row["c"])
                index._document_ids.append(row["d"])
                index._attributes.append(row["a"])
        if len(index._chunk_ids) != matrix.shape[0]:
            raise IndexIntegrityError("dense rows and vectors are misaligned")
        index._base = matrix if mmap else np.array(matrix, dtype=np.float32)
        index._alive = np.ones(matrix.shape[0], dtype=np.bool_)
        return index

    # ------------------------------------------------------------------ search
    def _filter_mask(self, filters: FilterExpression) -> BoolMask:
        if not filters:
            return self._alive
        key = canonical_json(filters.canonical())
        cached = self._mask_cache.get(key)
        if cached is not None:
            self._mask_cache.move_to_end(key)
            return cached
        mask = np.fromiter((filters.matches(a) for a in self._attributes), dtype=np.bool_, count=len(self._attributes))
        mask &= self._alive
        with self._lock:
            self._mask_cache[key] = mask
            while len(self._mask_cache) > self._mask_cache_size:
                self._mask_cache.popitem(last=False)
        return mask

    def search(self, vector: FloatVector, filters: FilterExpression, k: int) -> list[Candidate]:
        if k <= 0 or not self._chunk_ids:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dimension:
            raise IndexVersionMismatch(f"query vector dimension {q.shape[0]} != index dimension {self._dimension}")
        matrix = self._matrix()
        mask = self._filter_mask(filters)
        best_idx: list[NDArray[np.int64]] = []
        best_scores: list[FloatVector] = []
        n = matrix.shape[0]
        for s in range(0, n, self.block_rows):
            block_mask = mask[s : s + self.block_rows]
            if not block_mask.any():
                continue
            rows = np.flatnonzero(block_mask) + s
            scores = np.asarray(matrix[rows] @ q, dtype=np.float32)
            take = min(k, scores.shape[0])
            part = np.argpartition(-scores, take - 1)[:take]
            best_idx.append(rows[part])
            best_scores.append(scores[part])
        if not best_idx:
            return []
        idx = np.concatenate(best_idx)
        sc = np.concatenate(best_scores)
        # Deterministic order: score desc, then row asc for ties.
        order = np.lexsort((idx, -sc))[:k]
        return [
            Candidate(
                chunk_id=self._chunk_ids[int(idx[o])],
                document_id=self._document_ids[int(idx[o])],
                score=float(sc[o]),
                rank=r + 1,
                source="dense",
            )
            for r, o in enumerate(order)
        ]

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, FloatVector]:
        matrix = self._matrix()
        out: dict[str, FloatVector] = {}
        for cid in chunk_ids:
            row = self._row_of.get(cid)
            if row is not None and self._alive[row]:
                out[cid] = np.asarray(matrix[row], dtype=np.float32)
        return out

    def count(self) -> int:
        return int(self._alive.sum())

    def nbytes(self) -> int:
        return int(self.count() * self._dimension * 4)
