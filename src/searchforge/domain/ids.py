"""Deterministic identifiers and content hashes.

Rules (see ``docs/architecture/indexing.md``):

* ``document_id`` depends only on the *source identity* (explicit record id, or the
  source-relative path plus record key) — never on chunking/indexing configuration,
  and never on content (so an edited document keeps its id and is detected as an
  *update* through its content hash).
* ``content_hash`` depends only on normalized text plus indexed metadata.
* ``chunk_id`` depends on the document id, the chunker fingerprint and the chunk's
  character span/text, so a different chunking strategy yields different chunk ids
  while the document ids stay stable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-/]{0,199}$")


def sha256_hex(data: str | bytes) -> str:
    """Return the hex SHA-256 digest of ``data`` (strings are UTF-8 encoded)."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def short_hash(data: str | bytes, length: int = 16) -> str:
    """A truncated SHA-256 hex digest, used for compact deterministic ids."""
    return sha256_hex(data)[:length]


def canonical_json(value: Any) -> str:
    """Stable JSON serialization (sorted keys, no whitespace) used for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def is_safe_identifier(value: str) -> bool:
    """True when ``value`` can be used verbatim as an external id."""
    return bool(_SAFE_ID.match(value))


def make_document_id(*, explicit_id: str | None, source_key: str, record_key: str = "") -> str:
    """Build a stable document id.

    An explicit, safe source id is used verbatim so ids remain human-meaningful
    (e.g. ``KB-0042``). Otherwise the id is derived from the source-relative key,
    which never includes absolute filesystem paths.
    """
    if explicit_id is not None:
        candidate = explicit_id.strip()
        if candidate and is_safe_identifier(candidate):
            return candidate
        return "doc-" + short_hash(f"explicit\x1f{candidate}")
    return "doc-" + short_hash(f"{source_key}\x1f{record_key}")


def make_content_hash(text: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Hash of normalized text and indexed metadata; detects changed documents."""
    payload = text + "\x1e" + canonical_json(dict(metadata or {}))
    return sha256_hex(payload)


def make_chunk_id(document_id: str, chunker_fingerprint: str, start: int, end: int, text: str) -> str:
    """Deterministic chunk id (document + chunker + span + text)."""
    digest = short_hash(f"{document_id}\x1f{chunker_fingerprint}\x1f{start}:{end}\x1f{text}", 12)
    return f"{document_id}#{digest}"


def make_citation_id(chunk_id: str) -> str:
    """Stable, compact citation handle that maps back to exactly one chunk."""
    return "S" + short_hash(f"citation\x1f{chunk_id}", 10).upper()


def query_hash(text: str) -> str:
    """Hash used to correlate queries in telemetry without storing raw text."""
    return short_hash("query\x1f" + " ".join(text.split()).lower(), 16)
