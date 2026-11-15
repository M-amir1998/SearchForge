"""SQLite FTS5 lexical index with BM25 ranking and typed metadata filtering.

Safety: user text is *never* interpolated into SQL. The FTS5 ``MATCH`` expression
is generated from extracted word tokens, each wrapped as an FTS5 string literal
(``"token"`` with embedded quotes doubled), so FTS5 operators (``AND``, ``NEAR``,
``*``, column filters, ``^``) in user input are inert. The expression and all
filter values are bound parameters.

Scoring: FTS5 ``bm25()`` returns *lower-is-better* values (negative); we negate
it so larger is better, consistently with every other SearchForge retriever.
k1=1.2 and b=0.75 are fixed by SQLite; per-column weights are configurable.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from searchforge.domain.chunk import Chunk
from searchforge.domain.filters import FilterExpression
from searchforge.domain.result import Candidate
from searchforge.filtering.translate import to_sqlite
from searchforge.indexes.sqlite_util import ThreadLocalConnections, connect
from searchforge.parsing.stopwords import STOPWORDS

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MAX_MATCH_TERMS = 64

#: Small English stopword list. Removing these from the OR-query reduces candidate
#: noise; BM25's IDF would down-weight them anyway. Never applied inside phrases.

_TOKENIZERS = {
    "porter unicode61": "porter unicode61",
    "unicode61": "unicode61",
    "porter unicode61 remove_diacritics 2": "porter unicode61 remove_diacritics 2",
}


def _literal(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def build_match_expression(text: str, phrases: Sequence[str] = ()) -> str | None:
    """Turn free text into a safe FTS5 MATCH expression (OR of quoted terms/phrases)."""
    parts: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        toks = TOKEN_RE.findall(phrase.lower())
        if len(toks) >= 2:
            lit = _literal(" ".join(toks))
            if lit not in seen:
                seen.add(lit)
                parts.append(lit)
    tokens = list(TOKEN_RE.findall(text.lower()))
    content = [t for t in tokens if t not in STOPWORDS] or tokens
    for tok in content:
        lit = _literal(tok)
        if lit not in seen:
            seen.add(lit)
            parts.append(lit)
        if len(parts) >= MAX_MATCH_TERMS:
            break
    if not parts:
        return None
    return " OR ".join(parts)


class SqliteFtsLexicalIndex:
    """Implements :class:`searchforge.ports.LexicalIndex` on SQLite FTS5."""

    backend = "sqlite_fts"
    FILENAME = "lexical.sqlite"

    def __init__(
        self,
        path: Path,
        *,
        read_only: bool,
        tokenizer: str = "porter unicode61 remove_diacritics 2",
        text_weight: float = 1.0,
        title_weight: float = 2.0,
    ) -> None:
        if tokenizer not in _TOKENIZERS:
            raise ValueError(f"unsupported tokenizer {tokenizer!r}")
        self.path = path
        self.read_only = read_only
        self.text_weight = float(text_weight)
        self.title_weight = float(title_weight)
        self._writer: sqlite3.Connection | None = None
        self._conns: ThreadLocalConnections | None = None
        if read_only:
            self._conns = ThreadLocalConnections(path, read_only=True)
        else:
            self._writer = connect(path, read_only=False)
            tok = _TOKENIZERS[tokenizer]
            self._writer.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS lex_chunks (
                    rowid INTEGER PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    document_id TEXT NOT NULL,
                    attributes TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS lex_chunks_doc ON lex_chunks(document_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS lex_fts USING fts5(
                    body, title, tokenize = '{tok}'
                );
                """
            )

    def _conn(self) -> sqlite3.Connection:
        if self._writer is not None:
            return self._writer
        assert self._conns is not None
        return self._conns.get()

    def add(self, chunks: Sequence[Chunk]) -> None:
        conn = self._conn()
        for chunk in chunks:
            title_parts = [p for p in (chunk.metadata.title, chunk.metadata.section) if p]
            cur = conn.execute(
                "INSERT INTO lex_chunks (chunk_id, document_id, attributes) VALUES (?,?,?)",
                (chunk.chunk_id, chunk.document_id, json.dumps(chunk.metadata.attributes, sort_keys=True)),
            )
            conn.execute(
                "INSERT INTO lex_fts (rowid, body, title) VALUES (?,?,?)",
                (cur.lastrowid, chunk.text, " | ".join(title_parts)),
            )

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        conn = self._conn()
        ids = list(document_ids)
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            marks = ",".join("?" for _ in batch)
            rowids = [r[0] for r in conn.execute(f"SELECT rowid FROM lex_chunks WHERE document_id IN ({marks})", batch)]
            for rs in range(0, len(rowids), 500):
                rb = rowids[rs : rs + 500]
                rmarks = ",".join("?" for _ in rb)
                conn.execute(f"DELETE FROM lex_fts WHERE rowid IN ({rmarks})", rb)
                conn.execute(f"DELETE FROM lex_chunks WHERE rowid IN ({rmarks})", rb)

    def optimize(self) -> None:
        conn = self._conn()
        conn.execute("INSERT INTO lex_fts(lex_fts) VALUES ('optimize')")
        conn.commit()

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

    def search(self, text: str, filters: FilterExpression, k: int, phrases: Sequence[str] = ()) -> list[Candidate]:
        match = build_match_expression(text, phrases)
        if match is None or k <= 0:
            return []
        where, params = to_sqlite(filters, "c.attributes")
        sql = (
            "SELECT c.chunk_id, c.document_id, bm25(lex_fts, ?, ?) AS s "
            "FROM lex_fts JOIN lex_chunks c ON c.rowid = lex_fts.rowid "
            f"WHERE lex_fts MATCH ? AND {where} "
            "ORDER BY s ASC, c.rowid ASC LIMIT ?"
        )
        rows = self._conn().execute(sql, [self.text_weight, self.title_weight, match, *params, int(k)]).fetchall()
        return [
            Candidate(chunk_id=str(cid), document_id=str(did), score=-float(score), rank=i + 1, source="lexical")
            for i, (cid, did, score) in enumerate(rows)
        ]

    def count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM lex_chunks").fetchone()[0])

    def document_frequency(self, term: str) -> int:
        """Number of chunks containing ``term`` (diagnostics; term is bound, not interpolated)."""
        match = build_match_expression(term)
        if match is None:
            return 0
        return int(self._conn().execute("SELECT COUNT(*) FROM lex_fts WHERE lex_fts MATCH ?", (match,)).fetchone()[0])
