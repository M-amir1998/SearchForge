"""Query-capture policy for traces/logs (ADR-010: raw queries are not persisted by default).

``NONE``      nothing about the query text is recorded
``HASH_ONLY`` a salted-free stable hash for correlation (default)
``REDACTED``  text with emails, URLs, IPs, long numbers, UUIDs and key-like tokens masked
``FULL``      verbatim text (opt-in, e.g. for local debugging on synthetic data)
"""

from __future__ import annotations

import re

from searchforge.config.models import CapturePolicy
from searchforge.domain.ids import query_hash

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<uuid>"),
    (re.compile(r"\b(?:sk|pk|ghp|gho|xox[bp]|AKIA)[A-Za-z0-9_\-]{8,}\b"), "<secret>"),
    (re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"), "<token>"),
    (re.compile(r"\+?\d[\d\s\-()]{7,}\d"), "<number>"),
    (re.compile(r"\b\d{5,}\b"), "<number>"),
)


def redact(text: str) -> str:
    out = text
    for pattern, placeholder in _REDACTIONS:
        out = pattern.sub(placeholder, out)
    return out


def capture_query(text: str, policy: CapturePolicy) -> str | None:
    if policy is CapturePolicy.NONE:
        return None
    if policy is CapturePolicy.HASH_ONLY:
        return "sha256:" + query_hash(text)
    if policy is CapturePolicy.REDACTED:
        return redact(text)
    return text
