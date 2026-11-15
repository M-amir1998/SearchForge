"""Chunk model: the unit of indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from searchforge.domain.document import AttributeValue


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Everything needed to reconstruct a citation for a chunk.

    ``start_char``/``end_char`` index into the *normalized* document text, so
    ``document.text[start_char:end_char] == chunk.text`` always holds.
    """

    source_id: str
    source_key: str
    start_char: int
    end_char: int
    title: str | None = None
    section: str | None = None
    uri: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    attributes: dict[str, AttributeValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous span of a document."""

    chunk_id: str
    document_id: str
    text: str
    sequence: int
    metadata: ChunkMetadata

    @property
    def char_length(self) -> int:
        return len(self.text)
