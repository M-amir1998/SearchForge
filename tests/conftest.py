"""Shared fixtures. Everything runs offline on CPU; no model downloads, no servers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from searchforge.config.loader import LoadedConfig, load_config, validate_config
from searchforge.datasets import fixture_config_path
from searchforge.indexes.builder import IndexBuilder
from searchforge.indexes.handle import IndexHandle, open_index
from searchforge.service.search_service import SearchService

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def write_md(path: Path, doc_id: str, body: str, **meta: Any) -> Path:
    lines = ["---", f"id: {doc_id}", *[f"{k}: {v}" for k, v in meta.items()], "---", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + body, encoding="utf-8")
    return path


def small_config(tmp: Path, corpus: Path, **overrides: Any) -> LoadedConfig:
    data: dict[str, Any] = {
        "workspace": str(tmp / "ws"),
        "corpus": {
            "id": "test",
            "sources": [{"id": "kb", "path": str(corpus), "glob": "*.md"}],
            "filterable_fields": {"category": "string", "published": "date", "year": "integer"},
        },
        "chunking": {"strategy": "section", "max_chars": 400, "overlap_chars": 50, "min_chars": 20},
        "index": {"name": "test", "dense": {"provider": "hash", "dimension": 128}},
        "search": {"cache": {"enabled": True}},
        "storage": {"reserve_free_gb": 0.0},
    }
    for key, value in overrides.items():
        data[key] = {**data.get(key, {}), **value} if isinstance(value, dict) else value
    return validate_config(data, tmp)


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    write_md(
        corpus / "rollback.md",
        "DOC-ROLLBACK",
        "# Rollback\n\nUse `lumen deploy rollback` to undo a bad release. Rollback restores the previous "
        "version within two minutes.\n\n## Canary\n\nCanary deployments shift traffic gradually.\n",
        category="operations",
        published="2025-01-15",
        year=2025,
    )
    write_md(
        corpus / "keys.md",
        "DOC-KEYS",
        "# API key rotation\n\nRotate API keys every 90 days. ERR-4012 means the rate limit was exceeded.\n",
        category="security",
        published="2025-02-10",
        year=2025,
    )
    write_md(
        corpus / "tls.md",
        "DOC-TLS",
        "# TLS certificates\n\nCertificates renew automatically thirty days before expiry.\n",
        category="security",
        published="2023-03-01",
        year=2023,
    )
    write_md(
        corpus / "legacy.md",
        "DOC-LEGACY",
        "# Legacy rollback\n\nThe legacy rollback re-runs an old Jenkins job. Deprecated.\n",
        category="operations",
        published="2022-06-01",
        year=2022,
    )
    return corpus


@pytest.fixture
def mini_index(tmp_path: Path, mini_corpus: Path) -> Iterator[tuple[LoadedConfig, IndexHandle, SearchService]]:
    loaded = small_config(tmp_path, mini_corpus)
    IndexBuilder(loaded).build()
    handle = open_index(loaded.workspace, "test", loaded.config)
    service = SearchService(handle, loaded.config)
    yield loaded, handle, service
    service.close()
    handle.close()


@pytest.fixture(scope="session")
def fixture_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("fixture-ws")


@pytest.fixture(scope="session")
def fixture_loaded(fixture_workspace: Path) -> LoadedConfig:
    return load_config(fixture_config_path(), overrides={"workspace": str(fixture_workspace)})


@pytest.fixture(scope="session")
def fixture_service(fixture_loaded: LoadedConfig) -> Iterator[tuple[IndexHandle, SearchService]]:
    IndexBuilder(fixture_loaded).build()
    handle = open_index(fixture_loaded.workspace, fixture_loaded.config.index.name, fixture_loaded.config)
    service = SearchService(handle, fixture_loaded.config)
    yield handle, service
    service.close()
    handle.close()
