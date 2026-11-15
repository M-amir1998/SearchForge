"""Ingestion safety: path containment, size limits and decoding policy."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from searchforge.config.models import IngestionLimits
from searchforge.errors import CorpusNotFound, IngestionError, UnsafePathError


@dataclass(slots=True)
class IngestionReport:
    """Counts and warnings collected while streaming a corpus (no document text)."""

    files_seen: int = 0
    files_skipped: int = 0
    records_seen: int = 0
    documents_emitted: int = 0
    records_skipped: int = 0
    duplicate_ids: int = 0
    decode_replacements: int = 0
    bytes_read: int = 0
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str, limit: int = 200) -> None:
        if len(self.warnings) < limit:
            self.warnings.append(message)


def ensure_within(root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and require it to stay under ``root`` (no traversal)."""
    root_r = root.resolve()
    cand_r = candidate.resolve()
    if cand_r != root_r and root_r not in cand_r.parents:
        raise UnsafePathError(f"path escapes source root: {candidate}")
    return cand_r


def relative_key(root: Path, path: Path) -> str:
    """POSIX-style root-relative key (portable; never an absolute path)."""
    base = root if root.is_dir() else root.parent
    return path.resolve().relative_to(base.resolve()).as_posix()


def iter_source_files(root: Path, pattern: str, limits: IngestionLimits, report: IngestionReport) -> Iterator[Path]:
    """Yield regular files under ``root`` matching ``pattern`` in sorted order."""
    if not root.exists():
        raise CorpusNotFound(f"corpus source not found: {root}")
    if ".." in Path(pattern).parts:
        raise UnsafePathError(f"glob pattern may not contain '..': {pattern!r}")
    if root.is_file():
        candidates: list[Path] = [root]
        base = root.parent
    else:
        candidates = sorted(root.glob(pattern))
        base = root
    for path in candidates:
        if path.is_dir():
            continue
        report.files_seen += 1
        if path.is_symlink() and not limits.follow_symlinks:
            report.files_skipped += 1
            report.warn(f"skipped symlink {path.name}")
            continue
        try:
            ensure_within(base, path)
        except UnsafePathError:
            report.files_skipped += 1
            report.warn(f"skipped path outside root: {path.name}")
            continue
        size = path.stat().st_size
        if size > limits.max_file_bytes:
            report.files_skipped += 1
            report.warn(f"skipped oversized file {path.name} ({size} bytes)")
            continue
        yield path


def decode_bytes(raw: bytes, limits: IngestionLimits, report: IngestionReport, name: str) -> str | None:
    """Decode UTF-8 (BOM tolerated) following the configured malformed-encoding policy."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        if limits.on_decode_error == "fail":
            raise IngestionError(f"{name}: invalid UTF-8 ({exc.reason})") from exc
        if limits.on_decode_error == "skip":
            report.files_skipped += 1
            report.warn(f"skipped undecodable file {name}")
            return None
        report.decode_replacements += 1
        report.warn(f"replaced invalid UTF-8 bytes in {name}")
        return raw.decode("utf-8-sig", errors="replace")


def read_text_file(path: Path, limits: IngestionLimits, report: IngestionReport) -> str | None:
    with open(path, "rb") as fh:
        raw = fh.read(limits.max_file_bytes + 1)
    if len(raw) > limits.max_file_bytes:
        report.files_skipped += 1
        report.warn(f"skipped file that grew beyond limit: {path.name}")
        return None
    report.bytes_read += len(raw)
    text = decode_bytes(raw, limits, report, path.name)
    # Normalize line endings before any format parsing (front matter, headings, CSV).
    return None if text is None else text.replace("\r\n", "\n").replace("\r", "\n")


def directory_size(path: Path) -> int:
    """Total size of regular files below ``path`` (0 if missing). Symlinks are not followed."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for fname in filenames:
            fp = Path(dirpath) / fname
            if not fp.is_symlink():
                try:
                    total += fp.stat().st_size
                except OSError:
                    continue
    return total
