"""SQLite chunk store: the source of truth for chunk text, offsets and citation metadata."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from searchforge.domain.chunk import Chunk, ChunkMetadata
from searchforge.domain.document import Document
from searchforge.indexes.sqlite_util import ThreadLocalConnections, connect

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id  TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    source_key   TEXT NOT NULL,
    title        TEXT,
    uri          TEXT,
    chunk_count  INTEGER NOT NULL,
    char_count   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    sequence    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    title       TEXT,
    section     TEXT,
    uri         TEXT,
    source_id   TEXT NOT NULL,
    source_key  TEXT NOT NULL,
    start_char  INTEGER NOT NULL,
    end_char    INTEGER NOT NULL,
    created_at  TEXT,
    updated_at  TEXT,
    attributes  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_document ON chunks(document_id, sequence);
"""

_COLUMNS = (
    "chunk_id, document_id, sequence, text, title, section, uri, source_id, source_key, "
    "start_char, end_char, created_at, updated_at, attributes"
)


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _row_to_chunk(row: sqlite3.Row | tuple[Any, ...]) -> Chunk:
    (
        chunk_id,
        document_id,
        sequence,
        text,
        title,
        section,
        uri,
        source_id,
        source_key,
        start,
        end,
        created,
        updated,
        attributes,
    ) = row
    return Chunk(
        chunk_id=str(chunk_id),
        document_id=str(document_id),
        text=str(text),
        sequence=int(sequence),
        metadata=ChunkMetadata(
            source_id=str(source_id),
            source_key=str(source_key),
            start_char=int(start),
            end_char=int(end),
            title=title if isinstance(title, str) else None,
            section=section if isinstance(section, str) else None,
            uri=uri if isinstance(uri, str) else None,
            created_at=_dt(created if isinstance(created, str) else None),
            updated_at=_dt(updated if isinstance(updated, str) else None),
            attributes=json.loads(str(attributes)),
        ),
    )


class SqliteChunkStore:
    """Implements :class:`searchforge.ports.ChunkStore`."""

    FILENAME = "chunks.sqlite"

    def __init__(self, path: Path, *, read_only: bool) -> None:
        self.path = path
        self.read_only = read_only
        if read_only:
            self._conns: ThreadLocalConnections | None = ThreadLocalConnections(path, read_only=True)
            self._writer: sqlite3.Connection | None = None
        else:
            self._conns = None
            self._writer = connect(path, read_only=False)
            self._writer.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        if self._writer is not None:
            return self._writer
        assert self._conns is not None
        return self._conns.get()

    # ------------------------------------------------------------------ writes
    def add_document(self, document: Document, chunks: Sequence[Chunk]) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)",
            (
                document.document_id,
                document.content_hash,
                document.source.source_id,
                document.source.source_key,
                document.metadata.title,
                document.source.uri,
                len(chunks),
                len(document.text),
            ),
        )
        conn.executemany(
            f"INSERT INTO chunks ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    c.chunk_id,
                    c.document_id,
                    c.sequence,
                    c.text,
                    c.metadata.title,
                    c.metadata.section,
                    c.metadata.uri,
                    c.metadata.source_id,
                    c.metadata.source_key,
                    c.metadata.start_char,
                    c.metadata.end_char,
                    c.metadata.created_at.isoformat() if c.metadata.created_at else None,
                    c.metadata.updated_at.isoformat() if c.metadata.updated_at else None,
                    json.dumps(c.metadata.attributes, sort_keys=True),
                )
                for c in chunks
            ],
        )

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        conn = self._conn()
        for start in range(0, len(document_ids), 500):
            batch = list(document_ids[start : start + 500])
            marks = ",".join("?" for _ in batch)
            conn.execute(f"DELETE FROM chunks WHERE document_id IN ({marks})", batch)
            conn.execute(f"DELETE FROM documents WHERE document_id IN ({marks})", batch)

    def commit(self) -> None:
        if self._writer is not None:
            self._writer.commit()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.commit()
            self._writer.close()
            self._writer = None
        if self._conns is not None:
            self._conns.close()

    # ------------------------------------------------------------------ reads
    def get_many(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        out: dict[str, Chunk] = {}
        conn = self._conn()
        ids = list(dict.fromkeys(chunk_ids))
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            marks = ",".join("?" for _ in batch)
            rows = conn.execute(f"SELECT {_COLUMNS} FROM chunks WHERE chunk_id IN ({marks})", batch)
            for row in rows:
                chunk = _row_to_chunk(row)
                out[chunk.chunk_id] = chunk
        return out

    def chunks_for_document(self, document_id: str) -> list[Chunk]:
        rows = self._conn().execute(
            f"SELECT {_COLUMNS} FROM chunks WHERE document_id = ? ORDER BY sequence",
            (document_id,),
        )
        return [_row_to_chunk(r) for r in rows]

    def iter_chunks(self, batch_size: int = 512) -> Iterator[list[Chunk]]:
        """Stream all chunks in insertion order, ``batch_size`` at a time (keyset paging)."""
        conn = self._conn()
        last = 0
        while True:
            rows = conn.execute(
                f"SELECT rowid, {_COLUMNS} FROM chunks WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (last, batch_size),
            ).fetchall()
            if not rows:
                return
            last = int(rows[-1][0])
            yield [_row_to_chunk(r[1:]) for r in rows]

    def iter_chunks_for_documents(self, document_ids: Sequence[str], batch_size: int = 512) -> Iterator[list[Chunk]]:
        ids = list(document_ids)
        conn = self._conn()
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            marks = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM chunks WHERE document_id IN ({marks}) ORDER BY document_id, sequence",
                batch,
            )
            chunks = [_row_to_chunk(r) for r in rows]
            if chunks:
                yield chunks

    def document_hashes(self) -> dict[str, str]:
        return {str(d): str(h) for d, h in self._conn().execute("SELECT document_id, content_hash FROM documents")}

    def document_info(self, document_id: str) -> dict[str, object] | None:
        row = (
            self._conn()
            .execute(
                "SELECT document_id, content_hash, source_id, source_key, title, uri, chunk_count, "
                "char_count FROM documents WHERE document_id = ?",
                (document_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        keys = ("document_id", "content_hash", "source_id", "source_key", "title", "uri", "chunk_count", "char_count")
        return dict(zip(keys, row, strict=True))

    def count_chunks(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def count_documents(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def total_chunk_chars(self) -> int:
        return int(self._conn().execute("SELECT COALESCE(SUM(LENGTH(text)),0) FROM chunks").fetchone()[0])
