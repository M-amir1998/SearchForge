"""Index versions on disk: layout, manifests, immutability, validation.

Layout::

    <workspace>/indexes/<name>/
        CURRENT                       # pointer to the serving version (atomic replace)
        <version>/                    # immutable once manifest.json exists
            manifest.json
            chunks.sqlite             # chunk store (text, offsets, citation metadata)
            lexical.sqlite            # FTS5 index (+ filter attributes)
            dense/vectors.npy         # float32, L2-normalized
            dense/rows.jsonl
            dense/embedder/...        # fitted provider state (TF-IDF), pickle-free
        <version>.partial/            # in-progress build; never served

A version directory is published by an atomic rename from ``.partial`` and its
files are then made read-only. Search opens SQLite with ``mode=ro``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from searchforge.domain.index import FileChecksum, IndexManifest
from searchforge.errors import IndexImmutableError, IndexIntegrityError, IndexNotFound

MANIFEST = "manifest.json"
CURRENT = "CURRENT"
PARTIAL_SUFFIX = ".partial"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checksum_files(directory: Path) -> list[FileChecksum]:
    out: list[FileChecksum] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        rel = path.relative_to(directory).as_posix()
        if rel == MANIFEST:
            continue
        out.append(FileChecksum(path=rel, sha256=file_sha256(path), bytes=path.stat().st_size))
    return out


def git_sha(cwd: Path | None = None) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        res = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [git, "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = res.stdout.strip()
    return sha if res.returncode == 0 and len(sha) == 40 else None


def new_version_id(fingerprint: str) -> str:
    return datetime.now(UTC).strftime("v%Y%m%dT%H%M%S%f") + "-" + fingerprint[:8]


def _make_writable(func: Callable[..., Any], path: str, _exc: Any) -> None:
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    func(path)


def remove_tree(path: Path) -> None:
    """Remove a directory tree, including files made read-only by publication."""
    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_make_writable)
    else:
        shutil.rmtree(path, onerror=_make_writable)


def make_read_only(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def make_writable(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD | stat.S_IWRITE)


class IndexLayout:
    """Filesystem layout of all index versions in a workspace."""

    def __init__(self, workspace: Path) -> None:
        self.root = workspace / "indexes"

    def name_dir(self, name: str) -> Path:
        return self.root / name

    def version_dir(self, name: str, version: str) -> Path:
        return self.root / name / version

    def partial_dir(self, name: str, version: str) -> Path:
        return self.root / name / (version + PARTIAL_SUFFIX)

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def versions(self, name: str) -> list[IndexManifest]:
        ndir = self.name_dir(name)
        if not ndir.exists():
            return []
        manifests = []
        for p in ndir.iterdir():
            if p.is_dir() and not p.name.endswith(PARTIAL_SUFFIX) and (p / MANIFEST).exists():
                manifests.append(read_manifest(p))
        return sorted(manifests, key=lambda m: m.created_at)

    def partials(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(p for p in self.root.glob(f"*/*{PARTIAL_SUFFIX}") if p.is_dir())

    def current_version(self, name: str) -> str | None:
        ptr = self.name_dir(name) / CURRENT
        if not ptr.exists():
            return None
        value = ptr.read_text(encoding="utf-8").strip()
        return value or None

    def set_current(self, name: str, version: str) -> None:
        if not (self.version_dir(name, version) / MANIFEST).exists():
            raise IndexNotFound(f"cannot point CURRENT at incomplete version {name}@{version}")
        ptr = self.name_dir(name) / CURRENT
        tmp = ptr.with_suffix(".tmp")
        tmp.write_text(version + "\n", encoding="utf-8")
        os.replace(tmp, ptr)

    def resolve(self, ref: str) -> tuple[IndexManifest, Path]:
        """Resolve ``name`` (CURRENT) or ``name@version`` to a manifest and directory."""
        name, _, version = ref.partition("@")
        if not version:
            current = self.current_version(name)
            if current is None:
                raise IndexNotFound(f"index {name!r} has no CURRENT version (build it first)")
            version = current
        vdir = self.version_dir(name, version)
        if not (vdir / MANIFEST).exists():
            raise IndexNotFound(f"index version {name}@{version} not found")
        return read_manifest(vdir), vdir

    def find_by_fingerprint(self, name: str, fingerprint: str) -> IndexManifest | None:
        for manifest in reversed(self.versions(name)):
            if manifest.build_fingerprint == fingerprint:
                return manifest
        return None

    def ensure_new_version_dir(self, name: str, version: str) -> Path:
        final = self.version_dir(name, version)
        if final.exists():
            raise IndexImmutableError(f"index version {name}@{version} already exists and is immutable")
        partial = self.partial_dir(name, version)
        if partial.exists():
            remove_tree(partial)
        partial.mkdir(parents=True)
        return partial

    def publish(self, name: str, version: str, manifest: IndexManifest, set_current: bool = True) -> Path:
        partial = self.partial_dir(name, version)
        write_manifest(partial, manifest)
        final = self.version_dir(name, version)
        os.replace(partial, final)
        make_read_only(final)
        if set_current:
            self.set_current(name, version)
        return final


def write_manifest(directory: Path, manifest: IndexManifest) -> None:
    path = directory / MANIFEST
    if path.exists():
        raise IndexImmutableError(f"manifest already exists in {directory}")
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def read_manifest(directory: Path) -> IndexManifest:
    path = directory / MANIFEST
    if not path.exists():
        raise IndexNotFound(f"no manifest in {directory}")
    try:
        return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise IndexIntegrityError(f"corrupt manifest in {directory}: {exc}") from exc


def validate_index_dir(directory: Path) -> list[str]:
    """Return a list of integrity problems (empty list == valid)."""
    problems: list[str] = []
    manifest = read_manifest(directory)
    recorded = {f.path: f for f in manifest.files}
    actual = {f.path: f for f in checksum_files(directory)}
    for rel, rec in recorded.items():
        got = actual.get(rel)
        if got is None:
            problems.append(f"missing file: {rel}")
        elif got.sha256 != rec.sha256:
            problems.append(f"checksum mismatch: {rel}")
    for rel in actual.keys() - recorded.keys():
        problems.append(f"unexpected file: {rel}")
    if not problems:
        from searchforge.indexes.chunk_store import SqliteChunkStore

        store = SqliteChunkStore(directory / SqliteChunkStore.FILENAME, read_only=True)
        try:
            if store.count_chunks() != manifest.chunk_count:
                problems.append("chunk count differs from manifest")
            if store.count_documents() != manifest.document_count:
                problems.append("document count differs from manifest")
        finally:
            store.close()
    return problems
