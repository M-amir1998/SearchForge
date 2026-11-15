"""Streaming, bounded-memory ingestion."""

from __future__ import annotations

from searchforge.ingestion.readers import RecordMapper, detect_format, iter_corpus
from searchforge.ingestion.safety import IngestionReport, ensure_within, iter_source_files

__all__ = [
    "IngestionReport",
    "RecordMapper",
    "detect_format",
    "ensure_within",
    "iter_corpus",
    "iter_source_files",
]
