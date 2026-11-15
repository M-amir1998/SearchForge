"""Explicit, opt-in BEIR dataset download with archive-safety checks.

Never called by tests or CI. The archive is streamed with a hard size cap, and
extraction rejects absolute paths, ``..`` traversal, symlinks, too many members and
excessive uncompressed size (zip-bomb guard). Dataset licences are those of the
upstream publishers; the data is written under ``data/`` (git-ignored).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path, PurePosixPath

import httpx

from searchforge.errors import EvaluationFailure, UnsafePathError
from searchforge.evaluation.datasets import load_beir, save_queries

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
ALLOWED = frozenset({"scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid"})
MAX_MEMBERS = 10_000
MAX_UNCOMPRESSED = 2 * 1024**3


def safe_extract(
    archive: Path, dest: Path, *, max_members: int = MAX_MEMBERS, max_uncompressed: int = MAX_UNCOMPRESSED
) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > max_members:
            raise UnsafePathError(f"archive has too many members ({len(infos)})")
        if sum(i.file_size for i in infos) > max_uncompressed:
            raise UnsafePathError("archive uncompressed size exceeds the limit")
        for info in infos:
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or info.filename.startswith(("/", "\\")):
                raise UnsafePathError(f"unsafe archive member: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise UnsafePathError(f"symlink in archive: {info.filename}")
            target = (dest / name).resolve()
            if dest not in target.parents and target != dest:
                raise UnsafePathError(f"archive member escapes destination: {info.filename}")
        zf.extractall(dest)


def fetch_beir(name: str, out: Path, *, max_bytes: int = 200 * 1024 * 1024, split: str = "test") -> Path:
    if name not in ALLOWED:
        raise EvaluationFailure(f"unsupported BEIR dataset {name!r}; choose from {sorted(ALLOWED)}")
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"{name}.zip"
    received = 0
    with httpx.stream("GET", BEIR_URL.format(name=name), timeout=60.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(archive, "wb") as fh:
            for block in resp.iter_bytes(1 << 16):
                received += len(block)
                if received > max_bytes:
                    fh.close()
                    archive.unlink(missing_ok=True)
                    raise EvaluationFailure(f"download exceeded {max_bytes} bytes; aborting")
                fh.write(block)
    safe_extract(archive, out)
    archive.unlink()
    ds = out / name
    queries, _ = load_beir(ds, split)
    save_queries(queries, ds / "queries.searchforge.jsonl")
    (ds / "SEARCHFORGE_SOURCE.json").write_text(
        json.dumps({"url": BEIR_URL.format(name=name), "split": split, "bytes": received}, indent=2), encoding="utf-8"
    )
    return ds
