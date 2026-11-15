"""Multi-component flows on local backends (SQLite FTS5 + NumPy, hash embeddings)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from searchforge.config.loader import LoadedConfig
from searchforge.config.models import SearchProfile
from searchforge.domain.judgment import EvalQuery, Qrels, RelevanceJudgment
from searchforge.domain.result import Candidate, ScoredChunk
from searchforge.errors import (
    IndexImmutableError,
    IndexVersionMismatch,
    InvalidFilter,
    InvalidPlan,
    RetrievalFailure,
    StorageBudgetExceeded,
)
from searchforge.evaluation.runner import evaluate
from searchforge.indexes.builder import IndexBuilder
from searchforge.indexes.handle import IndexHandle, open_index
from searchforge.indexes.lifecycle import IndexLayout, validate_index_dir
from searchforge.ports import RetrievalRequest
from searchforge.service.search_service import SearchService
from searchforge.storage.manage import execute_cleanup, plan_cleanup, storage_status
from tests.conftest import small_config, write_md

pytestmark = pytest.mark.integration


def test_documents_to_lexical_index_to_search_to_evaluation(tmp_path: Path, mini_corpus: Path) -> None:
    loaded = small_config(tmp_path, mini_corpus, index={"name": "lex", "dense": {"enabled": False}})
    res = IndexBuilder(loaded).build()
    m = res.manifest
    assert (m.document_count, m.index_types, m.embedding) == (4, ["lexical"], None)
    assert m.corpus_version and m.chunker_fingerprint and m.searchforge_version
    assert validate_index_dir(res.path) == []
    handle = open_index(loaded.workspace, "lex", loaded.config)
    service = SearchService(handle, loaded.config)
    try:
        resp = service.search("rollback a release", profile="lexical")
        assert resp.results[0].document_id == "DOC-ROLLBACK" and not resp.degraded
        queries = [EvalQuery("q1", "rollback a release"), EvalQuery("q2", "ERR-4012")]
        qrels = Qrels([RelevanceJudgment("q1", "DOC-ROLLBACK", 3), RelevanceJudgment("q2", "DOC-KEYS", 3)])
        run = evaluate(service, queries, qrels, profile="lexical", k_values=[1, 10])
        assert run.aggregate()["mrr"] == pytest.approx(1.0) and run.latency()["samples"] == 2
        with pytest.raises(InvalidPlan):
            service.search("x", profile="dense")  # index has no dense component
    finally:
        service.close()
        handle.close()


def test_hybrid_fusion_rerank_explain(mini_index: tuple[LoadedConfig, IndexHandle, SearchService]) -> None:
    _, handle, service = mini_index
    resp = service.search("undo release rollback", profile="hybrid_rerank", explain=True, top_k=3)
    assert resp.results[0].document_id == "DOC-ROLLBACK"
    trace = resp.trace.as_dict()
    assert trace["retrievers"] == ["lexical", "dense"] and trace["fusion"] == "rrf"
    assert trace["reranker"] == "lexical_features" and "rerank" in trace["latency_ms"]
    assert trace["query"].startswith("sha256:")  # HASH_ONLY default: no raw text in traces
    sig = resp.results[0].explanation["signals"]  # type: ignore[index]
    assert sig["fusion_score"] is not None and sig["reranker_score"] is not None
    assert set(resp.candidates_by_retriever) == {"lexical", "dense"}
    filtered = service.search("rollback", filters={"published": {"gte": "2024-01-01"}}, profile="hybrid")
    docs = {r.document_id for r in filtered.results}
    assert "DOC-ROLLBACK" in docs and not docs & {"DOC-LEGACY", "DOC-TLS"}  # pre-filtered, not post-hoc
    assert "DOC-LEGACY" in {r.document_id for r in service.search("rollback", profile="hybrid").results}
    inline = service.search("category:security certificates", profile="adaptive")
    assert inline.results and all(r.document_id in ("DOC-TLS", "DOC-KEYS") for r in inline.results)
    with pytest.raises(InvalidFilter):
        service.search("x", filters={"nope": 1})
    assert service.search("ERR-4012", profile="adaptive").profile == "lexical"
    vectors = handle.vectors
    assert vectors is not None and vectors.count() == handle.chunk_store.count_chunks()


def test_query_cache_hits_and_index_version_invalidation(tmp_path: Path, mini_corpus: Path) -> None:
    loaded = small_config(tmp_path, mini_corpus)
    IndexBuilder(loaded).build()
    handle = open_index(loaded.workspace, "test", loaded.config)
    service = SearchService(handle, loaded.config)
    first = service.search("rollback")
    again = service.search("rollback")
    assert not first.trace.cache_hit and again.trace.cache_hit
    assert service.cache.stats.hits == 1 and service.search("rollback", top_k=3).trace.cache_hit is False
    service.close()
    handle.close()
    write_md(
        mini_corpus / "new.md",
        "DOC-NEW",
        "# Rollback checklist\n\nA brand new rollback checklist.\n",
        category="operations",
        published="2025-06-01",
        year=2025,
    )
    upd = IndexBuilder(loaded).update()
    handle2 = open_index(loaded.workspace, "test", loaded.config)
    service2 = SearchService(handle2, loaded.config)
    service2.cache = service.cache  # shared cache across versions must not serve stale entries
    fresh = service2.search("rollback")
    assert not fresh.trace.cache_hit and fresh.index_id == upd.manifest.index_id
    assert "DOC-NEW" in {r.document_id for r in fresh.results}
    service2.close()
    handle2.close()


def test_incremental_add_update_delete_and_stale_documents_disappear(tmp_path: Path, mini_corpus: Path) -> None:
    loaded = small_config(tmp_path, mini_corpus)
    builder = IndexBuilder(loaded)
    v1 = builder.build()
    assert builder.build().reused  # identical fingerprint → reuse
    write_md(
        mini_corpus / "keys.md",
        "DOC-KEYS",
        "# API key rotation\n\nRotate API keys every 45 days now.\n",
        category="security",
        published="2025-07-01",
        year=2025,
    )
    (mini_corpus / "tls.md").unlink()
    write_md(
        mini_corpus / "backup.md",
        "DOC-BACKUP",
        "# Backups\n\nNightly snapshots with point in time recovery.\n",
        category="operations",
        published="2025-03-01",
        year=2025,
    )
    v2 = builder.update()
    assert v2.manifest.update_summary == {"added": 1, "updated": 1, "deleted": 1, "unchanged": 2}
    assert v2.manifest.parent_version == v1.manifest.version and validate_index_dir(v2.path) == []
    assert builder.update().reused  # no changes → no new version
    handle = open_index(loaded.workspace, "test", loaded.config)
    service = SearchService(handle, loaded.config)
    try:
        for profile in ("lexical", "dense", "hybrid"):
            docs = {r.document_id for r in service.search("certificates expiry renew", profile=profile).results}
            assert "DOC-TLS" not in docs
            texts = " ".join(r.text or "" for r in service.search("rotate api keys", profile=profile).results)
            assert "90 days" not in texts
        assert service.search("snapshots recovery", profile="lexical").results[0].document_id == "DOC-BACKUP"
        assert handle.chunk_store.count_documents() == 4
    finally:
        service.close()
        handle.close()
    # the parent version is untouched (immutability) and still opens
    old = open_index(loaded.workspace, f"test@{v1.manifest.version}", loaded.config)
    assert old.chunk_store.document_info("DOC-TLS") is not None
    old.close()
    with pytest.raises(IndexImmutableError):
        IndexLayout(loaded.workspace).ensure_new_version_dir("test", v1.manifest.version)
    changed = small_config(
        tmp_path, mini_corpus, chunking={"strategy": "sentence", "max_chars": 300, "overlap_chars": 0, "min_chars": 10}
    )
    with pytest.raises(IndexVersionMismatch, match="chunking"):
        IndexBuilder(changed).update()


class _Broken:
    name = "dense"

    def retrieve(self, request: RetrievalRequest) -> list[Candidate]:
        raise RetrievalFailure("vector backend down")


class _Slow:
    name = "slow"

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        time.sleep(0.5)
        return list(candidates)


def test_partial_failure_is_degraded_never_hidden(mini_index: tuple[LoadedConfig, IndexHandle, SearchService]) -> None:
    loaded, handle, _service = mini_index
    from searchforge.retrieval.retrievers import LexicalRetriever

    assert handle.lexical is not None
    degraded = SearchService(
        handle, loaded.config, retrievers={"lexical": LexicalRetriever(handle.lexical), "dense": _Broken()}
    )
    try:
        resp = degraded.search("rollback", profile="hybrid")
        assert resp.degraded and resp.results and any("dense" in r for r in resp.degradation_reasons)
        assert not degraded.search("rollback", profile="hybrid").trace.cache_hit  # degraded never cached
    finally:
        degraded.close()
    strict = loaded.config.model_copy(
        update={
            "search": loaded.config.search.model_copy(
                update={"timeouts": loaded.config.search.timeouts.model_copy(update={"allow_degraded": False})}
            )
        }
    )
    svc = SearchService(handle, strict, retrievers={"lexical": LexicalRetriever(handle.lexical), "dense": _Broken()})
    with pytest.raises(RetrievalFailure):
        svc.search("rollback", profile="hybrid")
    svc.close()


def test_reranker_timeout_falls_back_to_fused_order(
    mini_index: tuple[LoadedConfig, IndexHandle, SearchService],
) -> None:
    loaded, handle, _ = mini_index
    cfg = loaded.config.model_copy(
        update={
            "search": loaded.config.search.model_copy(
                update={"timeouts": loaded.config.search.timeouts.model_copy(update={"reranker_ms": 50.0})}
            )
        }
    )
    svc = SearchService(handle, cfg)
    try:
        prof = svc.profiles["hybrid_rerank"]
        svc.register_profile("slow_rerank", prof)
        key = (
            f"{prof.reranker.type.value}:{prof.reranker.model}:{prof.reranker.revision}:"
            f"{sorted(prof.reranker.feature_weights.items())}"
        )
        svc._rerankers[key] = _Slow()
        resp = svc.search("rollback", profile="slow_rerank")
        assert resp.degraded and "reranker_timeout" in resp.degradation_reasons and resp.results
    finally:
        svc.close()


def test_storage_budget_status_and_safe_cleanup(tmp_path: Path, mini_corpus: Path) -> None:
    loaded = small_config(tmp_path, mini_corpus, storage={"reserve_free_gb": 10_000_000.0})
    with pytest.raises(StorageBudgetExceeded):
        IndexBuilder(loaded).build()
    ok = small_config(tmp_path, mini_corpus)
    builder = IndexBuilder(ok)
    builder.build()
    write_md(mini_corpus / "x.md", "DOC-X", "# X\n\nanother document body text\n", category="ops")
    builder.update()
    ws = ok.workspace
    status = storage_status(ws, {"kb": mini_corpus})
    assert status.indexes["test"] > 0 and status.sources["kb"] > 0 and status.embedding_cache > 0
    plan = plan_cleanup(ws, include_old_versions=True)
    reasons = [r for _, _, r in plan.targets]
    assert any("embedding cache" in r for r in reasons) and any("non-current" in r for r in reasons)
    execute_cleanup(ws, plan)
    assert IndexLayout(ws).current_version("test") is not None and len(IndexLayout(ws).versions("test")) == 1
    assert mini_corpus.exists() and any(mini_corpus.iterdir())  # source data untouched
    large = small_config(tmp_path, mini_corpus, index={"name": "big", "data_profile": "large"})
    with pytest.raises(StorageBudgetExceeded, match="large"):
        IndexBuilder(large).build()


def test_profile_registration_and_candidate_override(
    mini_index: tuple[LoadedConfig, IndexHandle, SearchService],
) -> None:
    _, _, service = mini_index
    service.register_profile("tiny_depth", SearchProfile(retrievers=["lexical"], candidate_k=2))  # type: ignore[list-item]
    resp = service.search("rollback", profile="tiny_depth", top_k=5)
    assert resp.trace.planner_decision["candidate_k"] == 5  # raised to top_k, recorded in notes
    assert any("candidate_k raised" in n for n in resp.trace.planner_decision["notes"])


def _unused(*_: Any) -> None:  # pragma: no cover - keeps helper imports explicit for readers
    return None
