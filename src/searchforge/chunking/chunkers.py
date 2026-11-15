"""Chunkers: fixed approximate-token windows, sentence packing and section-aware.

All chunkers operate on *spans* over the normalized document text and therefore
guarantee ``document.text[start:end] == chunk.text`` (property-tested). Sizes are
configuration, never constants; see ``docs/methodology/chunking.md``.

``min_chars`` takes precedence over the maximum: a fragment shorter than
``min_chars`` is merged into its neighbour instead of being emitted (or dropped),
so no text is lost.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from searchforge.config.models import ChunkingConfig, ChunkStrategy
from searchforge.domain.chunk import Chunk, ChunkMetadata
from searchforge.domain.document import Document
from searchforge.domain.ids import canonical_json, make_chunk_id, short_hash
from searchforge.parsing.formats import markdown_headings

_WORD = re.compile(r"\S+")
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    section: str | None = None

    @property
    def length(self) -> int:
        return self.end - self.start


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def merge_short_spans(spans: Sequence[Span], min_chars: int) -> list[Span]:
    """Merge spans shorter than ``min_chars`` into a neighbour (no text is lost)."""
    out: list[Span] = []
    for span in spans:
        if out and span.length < min_chars:
            prev = out.pop()
            out.append(Span(prev.start, max(prev.end, span.end), prev.section))
        else:
            out.append(span)
    if len(out) >= 2 and out[0].length < min_chars:
        first, second = out[0], out[1]
        out[0:2] = [Span(first.start, second.end, second.section or first.section)]
    return out


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Sentence/line spans inside ``text[start:end]`` with whitespace trimmed."""
    stop = len(text) if end is None else end
    spans: list[Span] = []
    pos = start
    for m in _SENT_END.finditer(text, start, stop):
        s, e = _trim(text, pos, m.start())
        if e > s:
            spans.append(Span(s, e))
        pos = m.end()
    s, e = _trim(text, pos, stop)
    if e > s:
        spans.append(Span(s, e))
    return spans


def _hard_split(text: str, span: Span, max_chars: int) -> list[Span]:
    """Split an over-long span at whitespace near ``max_chars`` (or hard if none)."""
    out: list[Span] = []
    s = span.start
    while span.end - s > max_chars:
        cut = text.rfind(" ", s + max_chars // 2, s + max_chars)
        cut = cut if cut > s else s + max_chars
        a, b = _trim(text, s, cut)
        if b > a:
            out.append(Span(a, b, span.section))
        s = cut
    a, b = _trim(text, s, span.end)
    if b > a:
        out.append(Span(a, b, span.section))
    return out


def pack_sentences(
    text: str, units: Sequence[Span], max_chars: int, overlap_chars: int, section: str | None
) -> list[Span]:
    """Greedily pack sentence units into windows of at most ``max_chars``.

    Consecutive windows share trailing whole sentences totalling at most
    ``overlap_chars``; the next window always advances by at least one unit.
    """
    pieces: list[Span] = []
    for unit in units:
        pieces.extend(_hard_split(text, unit, max_chars) if unit.length > max_chars else [unit])
    out: list[Span] = []
    i = 0
    n = len(pieces)
    while i < n:
        j = i
        while j + 1 < n and pieces[j + 1].end - pieces[i].start <= max_chars:
            j += 1
        out.append(Span(pieces[i].start, pieces[j].end, section))
        if j + 1 >= n:
            break
        nxt = j + 1
        if overlap_chars > 0:
            k = j
            while k > i and pieces[j].end - pieces[k].start <= overlap_chars:
                k -= 1
            candidate = k + 1
            if candidate <= j and pieces[j].end - pieces[candidate].start <= overlap_chars:
                nxt = candidate
        i = max(nxt, i + 1)
    return out


class _BaseChunker:
    name: str = "base"

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def _params(self) -> dict[str, object]:
        raise NotImplementedError

    def fingerprint(self) -> str:
        return short_hash(canonical_json({"chunker": self.name, **self._params()}), 12)

    def _spans(self, document: Document) -> list[Span]:
        raise NotImplementedError

    def chunk(self, document: Document) -> Iterator[Chunk]:
        spans = merge_short_spans(self._spans(document), self.config.min_chars)
        fp = self.fingerprint()
        meta = document.metadata
        for seq, span in enumerate(spans):
            text = document.text[span.start : span.end]
            yield Chunk(
                chunk_id=make_chunk_id(document.document_id, fp, span.start, span.end, text),
                document_id=document.document_id,
                text=text,
                sequence=seq,
                metadata=ChunkMetadata(
                    source_id=document.source.source_id,
                    source_key=document.source.source_key,
                    start_char=span.start,
                    end_char=span.end,
                    title=meta.title,
                    section=span.section,
                    uri=document.source.uri,
                    created_at=meta.created_at,
                    updated_at=meta.updated_at,
                    attributes=dict(meta.attributes),
                ),
            )


class FixedTokenApproxChunker(_BaseChunker):
    """Windows of ``max_tokens`` whitespace tokens with ``overlap_tokens`` overlap.

    Whitespace tokens approximate model tokens without requiring a tokenizer; real
    subword counts are typically 1.2–1.5x higher for English prose.
    """

    name = "fixed_token"

    def _params(self) -> dict[str, object]:
        c = self.config
        return {"max_tokens": c.max_tokens, "overlap_tokens": c.overlap_tokens, "min_chars": c.min_chars}

    def _spans(self, document: Document) -> list[Span]:
        words = [(m.start(), m.end()) for m in _WORD.finditer(document.text)]
        if not words:
            return []
        size = self.config.max_tokens
        step = size - self.config.overlap_tokens
        out: list[Span] = []
        i = 0
        while True:
            j = min(i + size, len(words))
            out.append(Span(words[i][0], words[j - 1][1]))
            if j >= len(words):
                break
            i += step
        return out


class SentenceChunker(_BaseChunker):
    """Packs whole sentences up to ``max_chars`` with sentence-aligned overlap."""

    name = "sentence"

    def _params(self) -> dict[str, object]:
        c = self.config
        return {"max_chars": c.max_chars, "overlap_chars": c.overlap_chars, "min_chars": c.min_chars}

    def _spans(self, document: Document) -> list[Span]:
        units = sentence_spans(document.text)
        return pack_sentences(document.text, units, self.config.max_chars, self.config.overlap_chars, None)


class SectionChunker(_BaseChunker):
    """Markdown-section-aware chunking.

    Sections (ATX headings outside code fences) are never merged across a heading
    boundary unless one is shorter than ``min_chars``. Long sections are packed by
    sentences with overlap. Each chunk records its heading path, e.g.
    ``"Deployment > Rollback"``.
    """

    name = "section"

    def _params(self) -> dict[str, object]:
        c = self.config
        return {"max_chars": c.max_chars, "overlap_chars": c.overlap_chars, "min_chars": c.min_chars}

    def _sections(self, text: str) -> list[Span]:
        headings = markdown_headings(text)
        bounds: list[tuple[int, str | None]] = []
        if not headings or headings[0][0] > 0:
            bounds.append((0, None))
        stack: list[tuple[int, str]] = []
        for offset, level, title in headings:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            bounds.append((offset, " > ".join(t for _, t in stack)))
        out: list[Span] = []
        for idx, (start, label) in enumerate(bounds):
            end = bounds[idx + 1][0] if idx + 1 < len(bounds) else len(text)
            s, e = _trim(text, start, end)
            if e > s:
                out.append(Span(s, e, label))
        return out

    def _spans(self, document: Document) -> list[Span]:
        text = document.text
        out: list[Span] = []
        for section in self._sections(text):
            if section.length <= self.config.max_chars:
                out.append(section)
            else:
                units = sentence_spans(text, section.start, section.end)
                out.extend(
                    pack_sentences(text, units, self.config.max_chars, self.config.overlap_chars, section.section)
                )
        return out


def build_chunker(config: ChunkingConfig) -> _BaseChunker:
    if config.strategy is ChunkStrategy.FIXED_TOKEN:
        return FixedTokenApproxChunker(config)
    if config.strategy is ChunkStrategy.SENTENCE:
        return SentenceChunker(config)
    return SectionChunker(config)
