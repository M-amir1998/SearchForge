"""Citation-ready context packages for downstream generation (RAG is out of core scope).

Input: ranked results (with chunk text). Output: a :class:`ContextPackage` with
stable citation ids that map back to document, chunk and source metadata, built
within a character or estimated-token budget, with duplicate suppression:

* same-document overlap — chunks whose character ranges overlap an already
  selected chunk by more than ``overlap_threshold`` (overlapping windows),
* near-duplicate text — word-shingle Jaccard ≥ ``near_duplicate_threshold``
  (copies across documents),
* ``max_per_document`` — a single document cannot consume the whole budget.

Token counts are *estimates* (``ceil(chars / chars_per_token)``); a model-specific
tokenizer can be injected via ``token_counter``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from searchforge.domain.ids import make_citation_id
from searchforge.domain.result import SearchResult

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ContextItem:
    citation_id: str
    chunk_id: str
    document_id: str
    text: str
    rank: int
    title: str | None
    section: str | None
    source: str | None
    uri: str | None
    start_char: int | None
    end_char: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "rank": self.rank,
            "title": self.title,
            "section": self.section,
            "source": self.source,
            "uri": self.uri,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
        }


@dataclass(slots=True)
class ContextPackage:
    items: list[ContextItem]
    total_chars: int
    estimated_tokens: int
    budget_chars: int | None
    budget_tokens: int | None
    dropped: list[dict[str, str]] = field(default_factory=list)

    @property
    def citations(self) -> dict[str, dict[str, Any]]:
        return {i.citation_id: {k: v for k, v in i.as_dict().items() if k != "text"} for i in self.items}

    def render(self) -> str:
        blocks = []
        for item in self.items:
            head = " — ".join(p for p in (item.title, item.section) if p) or item.document_id
            blocks.append(f"[{item.citation_id}] {head}\n{item.text}")
        return "\n\n".join(blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [i.as_dict() for i in self.items],
            "total_chars": self.total_chars,
            "estimated_tokens": self.estimated_tokens,
            "budget_chars": self.budget_chars,
            "budget_tokens": self.budget_tokens,
            "dropped": self.dropped,
        }


@runtime_checkable
class ContextBuilder(Protocol):
    def build(self, results: Sequence[SearchResult]) -> ContextPackage: ...


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class CitationContextBuilder:
    def __init__(
        self,
        *,
        max_chars: int | None = 6000,
        max_tokens: int | None = None,
        chars_per_token: float = 4.0,
        max_per_document: int | None = 2,
        overlap_threshold: float = 0.5,
        near_duplicate_threshold: float = 0.8,
        separator_chars: int = 2,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if max_chars is None and max_tokens is None:
            raise ValueError("set max_chars and/or max_tokens")
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self.max_per_document = max_per_document
        self.overlap_threshold = overlap_threshold
        self.near_duplicate_threshold = near_duplicate_threshold
        self.separator_chars = separator_chars
        self._count = token_counter or (lambda t: math.ceil(len(t) / self.chars_per_token))

    def build(self, results: Sequence[SearchResult]) -> ContextPackage:
        items: list[ContextItem] = []
        dropped: list[dict[str, str]] = []
        per_doc: dict[str, int] = {}
        spans: dict[str, list[tuple[int, int]]] = {}
        shingles: list[set[tuple[str, ...]]] = []
        chars = 0
        tokens = 0
        for r in sorted(results, key=lambda x: x.rank):
            text = r.text or ""
            if not text:
                dropped.append({"chunk_id": r.chunk_id, "reason": "no_text"})
                continue
            if self.max_per_document is not None and per_doc.get(r.document_id, 0) >= self.max_per_document:
                dropped.append({"chunk_id": r.chunk_id, "reason": "max_per_document"})
                continue
            start = r.metadata.get("start_char")
            end = r.metadata.get("end_char")
            if isinstance(start, int) and isinstance(end, int) and end > start:
                overlap = max((min(end, e) - max(start, s) for s, e in spans.get(r.document_id, [])), default=0)
                if overlap / (end - start) > self.overlap_threshold:
                    dropped.append({"chunk_id": r.chunk_id, "reason": "overlapping_chunk"})
                    continue
            sh = _shingles(text)
            if any(_jaccard(sh, other) >= self.near_duplicate_threshold for other in shingles):
                dropped.append({"chunk_id": r.chunk_id, "reason": "near_duplicate"})
                continue
            add_chars = len(text) + (self.separator_chars if items else 0)
            add_tokens = self._count(text)
            if (self.max_chars is not None and chars + add_chars > self.max_chars) or (
                self.max_tokens is not None and tokens + add_tokens > self.max_tokens
            ):
                dropped.append({"chunk_id": r.chunk_id, "reason": "budget"})
                continue
            items.append(
                ContextItem(
                    citation_id=make_citation_id(r.chunk_id),
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    text=text,
                    rank=r.rank,
                    title=r.metadata.get("title"),
                    section=r.metadata.get("section"),
                    source=r.metadata.get("source_id"),
                    uri=r.metadata.get("uri"),
                    start_char=start if isinstance(start, int) else None,
                    end_char=end if isinstance(end, int) else None,
                )
            )
            chars += add_chars
            tokens += add_tokens
            per_doc[r.document_id] = per_doc.get(r.document_id, 0) + 1
            if isinstance(start, int) and isinstance(end, int):
                spans.setdefault(r.document_id, []).append((start, end))
            shingles.append(sh)
        return ContextPackage(
            items=items,
            total_chars=chars,
            estimated_tokens=tokens,
            budget_chars=self.max_chars,
            budget_tokens=self.max_tokens,
            dropped=dropped,
        )
