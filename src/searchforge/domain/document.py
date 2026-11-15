"""Document model: the unit of ingestion, identity and relevance judgment."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

#: Scalar attribute values that may be stored, filtered and faceted on.
AttributeValue = str | int | float | bool


class SourceFormat(StrEnum):
    """Supported ingestion formats."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HTML = "html"


@dataclass(frozen=True, slots=True)
class DocumentSource:
    """Where a document came from.

    ``source_key`` is root-relative (never an absolute local path) so ids and
    citations are portable across machines.
    """

    source_id: str
    source_key: str
    format: SourceFormat
    uri: str | None = None
    record_key: str = ""


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Descriptive and filterable metadata attached to a document."""

    title: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    language: str | None = None
    attributes: dict[str, AttributeValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Document:
    """A normalized source document."""

    document_id: str
    text: str
    source: DocumentSource
    metadata: DocumentMetadata
    content_hash: str

    @property
    def title(self) -> str | None:
        return self.metadata.title
