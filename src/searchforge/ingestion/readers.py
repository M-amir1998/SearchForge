"""Streaming document readers for text, Markdown, JSON, JSONL, CSV, Parquet and HTML.

Every reader is a generator: corpora are processed one record (or one Parquet
record batch) at a time so memory stays bounded regardless of corpus size.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from searchforge.config.loader import LoadedConfig
from searchforge.config.models import CorpusConfig, NormalizationConfig, SourceConfig
from searchforge.domain.document import (
    AttributeValue,
    Document,
    DocumentMetadata,
    DocumentSource,
    SourceFormat,
)
from searchforge.domain.ids import make_content_hash, make_document_id
from searchforge.errors import IngestionError
from searchforge.ingestion.safety import (
    IngestionReport,
    decode_bytes,
    iter_source_files,
    read_text_file,
    relative_key,
)
from searchforge.parsing.formats import html_to_text, parse_markdown
from searchforge.parsing.normalize import normalize_text

_SUFFIX_FORMATS: dict[str, SourceFormat] = {
    ".txt": SourceFormat.TEXT,
    ".text": SourceFormat.TEXT,
    ".md": SourceFormat.MARKDOWN,
    ".markdown": SourceFormat.MARKDOWN,
    ".json": SourceFormat.JSON,
    ".jsonl": SourceFormat.JSONL,
    ".ndjson": SourceFormat.JSONL,
    ".csv": SourceFormat.CSV,
    ".parquet": SourceFormat.PARQUET,
    ".html": SourceFormat.HTML,
    ".htm": SourceFormat.HTML,
}

PARQUET_BATCH_ROWS = 1024


def detect_format(path: Path, configured: SourceFormat | None) -> SourceFormat | None:
    if configured is not None:
        return configured
    return _SUFFIX_FORMATS.get(path.suffix.lower())


def parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _scalar(value: Any) -> AttributeValue | None:
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    return None


class RecordMapper:
    """Maps a raw record (dict) to a normalized :class:`Document`."""

    def __init__(self, source: SourceConfig, normalization: NormalizationConfig, max_chars: int) -> None:
        self.source = source
        self.normalization = normalization
        self.max_chars = max_chars
        self._reserved = {
            f
            for f in (
                source.id_field,
                source.text_field,
                source.title_field,
                source.uri_field,
                source.created_field,
                source.updated_field,
                source.language_field,
            )
            if f
        }

    def _attributes(self, record: Mapping[str, Any], language: str | None) -> dict[str, AttributeValue]:
        attrs: dict[str, AttributeValue] = {}
        keys = self.source.metadata_fields
        if keys is None:
            keys = [k for k in record if k not in self._reserved]
        for key in keys:
            if key in record:
                scalar = _scalar(record[key])
                if scalar is not None and isinstance(key, str) and key.isidentifier():
                    attrs[key.lower()] = scalar
        attrs["source"] = self.source.id
        if language:
            attrs["language"] = language
        return attrs

    def to_document(
        self,
        record: Mapping[str, Any],
        *,
        source_key: str,
        record_key: str,
        fmt: SourceFormat,
        report: IngestionReport,
        default_title: str | None = None,
        default_uri: str | None = None,
    ) -> Document | None:
        report.records_seen += 1
        raw_text = record.get(self.source.text_field)
        if not isinstance(raw_text, str) or not raw_text.strip():
            report.records_skipped += 1
            report.warn(f"{source_key}{record_key}: missing/empty text field")
            return None
        if len(raw_text) > self.max_chars:
            report.records_skipped += 1
            report.warn(f"{source_key}{record_key}: record exceeds max_record_chars")
            return None
        text = normalize_text(raw_text, self.normalization)
        if not text:
            report.records_skipped += 1
            return None
        explicit = record.get(self.source.id_field) if self.source.id_field else None
        explicit_id = None if explicit is None or explicit == "" else str(explicit)
        title_raw = record.get(self.source.title_field) if self.source.title_field else None
        title = normalize_text(str(title_raw), self.normalization) if title_raw else default_title
        uri_raw = record.get(self.source.uri_field) if self.source.uri_field else None
        uri = str(uri_raw) if uri_raw else default_uri
        lang_raw = record.get(self.source.language_field) if self.source.language_field else None
        language = str(lang_raw) if lang_raw else None
        created = parse_datetime(record.get(self.source.created_field)) if self.source.created_field else None
        updated = parse_datetime(record.get(self.source.updated_field)) if self.source.updated_field else None
        attributes = self._attributes(record, language)
        document_id = make_document_id(explicit_id=explicit_id, source_key=source_key, record_key=record_key)
        attributes["document_id"] = document_id
        hashed_meta = {
            "title": title,
            "uri": uri,
            "attributes": attributes,
            "created": created.isoformat() if created else None,
            "updated": updated.isoformat() if updated else None,
        }
        return Document(
            document_id=document_id,
            text=text,
            source=DocumentSource(
                source_id=self.source.id, source_key=source_key, format=fmt, uri=uri, record_key=record_key
            ),
            metadata=DocumentMetadata(
                title=title, created_at=created, updated_at=updated, language=language, attributes=attributes
            ),
            content_hash=make_content_hash(text, hashed_meta),
        )


def _uri_for(source: SourceConfig, key: str) -> str | None:
    return f"{source.uri_prefix.rstrip('/')}/{key}" if source.uri_prefix else None


def _iter_file_records(
    path: Path, fmt: SourceFormat, cfg: CorpusConfig, report: IngestionReport
) -> Iterator[tuple[str, dict[str, Any], str | None]]:
    """Yield ``(record_key, record, default_title)`` for one file."""
    limits = cfg.limits
    if fmt in (SourceFormat.TEXT, SourceFormat.MARKDOWN, SourceFormat.HTML):
        content = read_text_file(path, limits, report)
        if content is None:
            return
        if fmt is SourceFormat.MARKDOWN:
            md = parse_markdown(content)
            record: dict[str, Any] = {k: v for k, v in md.front_matter.items() if k != "text"}
            record["text"] = md.body
            yield "", record, md.title or path.stem
        elif fmt is SourceFormat.HTML:
            text, title = html_to_text(content)
            yield "", {"text": text}, title or path.stem
        else:
            first = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
            title = first if 0 < len(first) <= 120 else path.stem
            yield "", {"text": content}, title
        return
    if fmt is SourceFormat.JSON:
        content = read_text_file(path, limits, report)
        if content is None:
            return
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            report.files_skipped += 1
            report.warn(f"{path.name}: invalid JSON ({exc.msg})")
            return
        rows = data if isinstance(data, list) else [data]
        for i, row in enumerate(rows):
            if isinstance(row, dict):
                yield f"#{i}", row, None
            else:
                report.records_skipped += 1
        return
    if fmt is SourceFormat.JSONL:
        with open(path, "rb") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                report.bytes_read += len(raw_line)
                if not raw_line.strip():
                    continue
                line = decode_bytes(raw_line, limits, report, f"{path.name}:{line_no}")
                if line is None:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    report.records_skipped += 1
                    report.warn(f"{path.name}:{line_no}: invalid JSON line")
                    continue
                if isinstance(row, dict):
                    yield f"#L{line_no}", row, None
                else:
                    report.records_skipped += 1
        return
    if fmt is SourceFormat.CSV:
        content = read_text_file(path, limits, report)
        if content is None:
            return
        csv.field_size_limit(min(limits.max_record_chars, 2**31 - 1))
        reader = csv.DictReader(io.StringIO(content, newline=""))
        for i, row in enumerate(reader):
            yield f"#R{i}", {k: v for k, v in row.items() if k is not None}, None
        return
    if fmt is SourceFormat.PARQUET:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        offset = 0
        for batch in pf.iter_batches(batch_size=PARQUET_BATCH_ROWS):
            for i, row in enumerate(batch.to_pylist()):
                yield f"#R{offset + i}", row, None
            offset += batch.num_rows
        return
    raise IngestionError(f"unsupported format {fmt}")  # pragma: no cover


def iter_source_documents(loaded: LoadedConfig, source: SourceConfig, report: IngestionReport) -> Iterator[Document]:
    cfg = loaded.config.corpus
    root = loaded.resolve(source.path)
    mapper = RecordMapper(source, cfg.normalization, cfg.limits.max_record_chars)
    for path in iter_source_files(root, source.glob, cfg.limits, report):
        fmt = detect_format(path, source.format)
        if fmt is None:
            report.files_skipped += 1
            continue
        key = relative_key(root, path)
        for record_key, record, default_title in _iter_file_records(path, fmt, cfg, report):
            doc = mapper.to_document(
                record,
                source_key=key,
                record_key=record_key,
                fmt=fmt,
                report=report,
                default_title=default_title,
                default_uri=_uri_for(source, key),
            )
            if doc is not None:
                yield doc


def iter_corpus(loaded: LoadedConfig, report: IngestionReport | None = None) -> Iterator[Document]:
    """Stream all documents of a corpus. Duplicate ids: first occurrence wins (warned)."""
    rep = report if report is not None else IngestionReport()
    seen: set[str] = set()
    limit = loaded.config.corpus.limits.max_documents
    for source in loaded.config.corpus.sources:
        for doc in iter_source_documents(loaded, source, rep):
            if doc.document_id in seen:
                rep.duplicate_ids += 1
                rep.warn(f"duplicate document id {doc.document_id}; keeping first occurrence")
                continue
            seen.add(doc.document_id)
            rep.documents_emitted += 1
            yield doc
            if limit is not None and rep.documents_emitted >= limit:
                return
