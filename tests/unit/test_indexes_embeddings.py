"""Lexical index, vector index, embedding providers and the embedding cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from searchforge.config.models import RemoteEmbeddingConfig, TfidfConfig
from searchforge.domain.chunk import Chunk, ChunkMetadata
from searchforge.domain.filters import FilterSchema, parse_filters
from searchforge.embeddings.cache import CachedEmbeddingProvider, EmbeddingCache, cache_key
from searchforge.embeddings.hashing import HashEmbeddingProvider
from searchforge.embeddings.remote import RemoteEmbeddingProvider
from searchforge.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from searchforge.embeddings.tfidf import TfidfEmbeddingProvider
from searchforge.errors import EmbeddingFailure, IndexIntegrityError, IndexVersionMismatch
from searchforge.indexes.dense.numpy_index import NumpyVectorIndex
from searchforge.indexes.lexical.sqlite_fts import SqliteFtsLexicalIndex, build_match_expression

pytestmark = pytest.mark.unit
SCHEMA = FilterSchema.from_mapping({"category": "string", "year": "integer"})


def _chunk(cid: str, doc: str, text: str, title: str = "", **attrs: Any) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id=doc,
        text=text,
        sequence=0,
        metadata=ChunkMetadata(
            source_id="s", source_key="k", start_char=0, end_char=len(text), title=title or None, attributes=attrs
        ),
    )


CHUNKS = [
    _chunk("c1", "d1", "rotate api keys every ninety days", "Key rotation", category="security", year=2025),
    _chunk("c2", "d2", "rollback a bad release with the deploy command", "Rollback", category="ops", year=2024),
    _chunk("c3", "d3", "certificates renew automatically before expiry", "TLS", category="security", year=2023),
    _chunk("c4", "d4", "rollback rollback rollback legacy jenkins job", "Legacy", category="ops", year=2022),
]


def test_match_expression_is_safe() -> None:
    assert (
        build_match_expression('ERR-4012 "quoted" AND NEAR(x) col:val *')
        == '"err" OR "4012" OR "quoted" OR "near" OR "x" OR "col" OR "val"'
    )
    assert build_match_expression("the and of") == '"the" OR "and" OR "of"'  # all stopwords → keep
    assert build_match_expression("!!!") is None
    assert build_match_expression("rotate keys", phrases=["api keys"]).startswith('"api keys"')


def test_fts_ranking_filters_and_delete(tmp_path: Path) -> None:
    idx = SqliteFtsLexicalIndex(tmp_path / "lex.sqlite", read_only=False)
    idx.add(CHUNKS)
    idx.commit()
    hits = idx.search("rollback release", parse_filters(None, SCHEMA), 10)
    assert [h.chunk_id for h in hits][:2] == ["c2", "c4"]  # both terms beat repeated single term
    assert all(h.score > 0 for h in hits) and [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert [h.chunk_id for h in idx.search("rollback", parse_filters({"year": {"gte": 2024}}, SCHEMA), 10)] == ["c2"]
    assert idx.search("key rotation", parse_filters(None, SCHEMA), 10)[0].chunk_id == "c1"  # title field + porter
    assert idx.search('") OR 1=1 --', parse_filters(None, SCHEMA), 10) == []
    idx.delete_documents(["d2"])
    assert "c2" not in {h.chunk_id for h in idx.search("rollback", parse_filters(None, SCHEMA), 10)}
    assert idx.count() == 3
    idx.close()
    ro = SqliteFtsLexicalIndex(tmp_path / "lex.sqlite", read_only=True)
    assert ro.count() == 3
    ro.close()


def test_numpy_index_rank_filter_persist(tmp_path: Path) -> None:
    vecs = np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    idx = NumpyVectorIndex(3, block_rows=2)
    idx.add(
        [c.chunk_id for c in CHUNKS], [c.document_id for c in CHUNKS], [c.metadata.attributes for c in CHUNKS], vecs
    )
    q = np.array([1, 0, 0], dtype=np.float32)
    assert [h.chunk_id for h in idx.search(q, parse_filters(None, SCHEMA), 3)] == ["c1", "c2", "c3"] or [
        h.chunk_id for h in idx.search(q, parse_filters(None, SCHEMA), 2)
    ] == ["c1", "c2"]
    top2 = idx.search(q, parse_filters(None, SCHEMA), 2)
    assert [h.chunk_id for h in top2] == ["c1", "c2"] and top2[0].score == pytest.approx(1.0)
    assert [h.chunk_id for h in idx.search(q, parse_filters({"category": "ops"}, SCHEMA), 5)] == ["c2", "c4"]
    idx.delete_documents(["d1"])
    assert idx.search(q, parse_filters(None, SCHEMA), 1)[0].chunk_id == "c2"
    idx.save(tmp_path / "dense")
    loaded = NumpyVectorIndex.load(tmp_path / "dense", mmap=True)
    assert loaded.count() == 3 and set(loaded.get_vectors(["c1", "c2"])) == {"c2"}
    with pytest.raises(IndexVersionMismatch):
        loaded.search(np.zeros(5, dtype=np.float32), parse_filters(None, SCHEMA), 1)
    with pytest.raises(IndexVersionMismatch):
        idx.add(["x"], ["y"], [{}], np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(IndexIntegrityError):
        NumpyVectorIndex.load(tmp_path / "missing")


def test_hash_embeddings_deterministic_and_normalized() -> None:
    p = HashEmbeddingProvider(dimension=64)
    a = p.embed_documents(["hello world", "hello world", ""])
    assert np.allclose(a[0], a[1]) and np.linalg.norm(a[0]) == pytest.approx(1.0)
    assert np.allclose(a[2], 0)
    assert np.allclose(p.embed_query("hello world"), a[0])
    assert p.spec.provider == "hash" and p.spec.dimension == 64


def test_tfidf_matches_sklearn_and_roundtrips(tmp_path: Path) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = ["rotate the api keys", "rollback the release now", "api release notes", "keys keys keys"]
    p = TfidfEmbeddingProvider(TfidfConfig(svd_components=None, ngram_max=2)).fit(texts)
    ref = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
    ).fit(texts)
    expected = ref.transform(["api keys rotate unknownword"]).toarray()[0]
    order = [ref.vocabulary_[t] for t in sorted(ref.vocabulary_, key=lambda t: p._vocabulary[t])]  # type: ignore[index]
    assert np.allclose(p.embed_query("api keys rotate unknownword"), expected[order], atol=1e-6)
    lsa = TfidfEmbeddingProvider(TfidfConfig(svd_components=2)).fit(texts)
    lsa.save(tmp_path / "emb")
    again = TfidfEmbeddingProvider.load(tmp_path / "emb")
    assert again.spec == lsa.spec and again.spec.dimension == 2
    assert np.allclose(again.embed_documents(texts), lsa.embed_documents(texts), atol=1e-6)
    with pytest.raises(EmbeddingFailure):
        TfidfEmbeddingProvider().fit([])
    with pytest.raises(EmbeddingFailure):
        _ = TfidfEmbeddingProvider().spec


def test_embedding_cache_hits_and_version_aware_keys(tmp_path: Path) -> None:
    inner = HashEmbeddingProvider(dimension=16)
    cache = EmbeddingCache(tmp_path / "c")
    cached = CachedEmbeddingProvider(inner, cache)
    first = cached.embed_documents(["a b", "c d"])
    second = cached.embed_documents(["a b", "c d", "e f"])
    assert np.allclose(first, second[:2])
    assert (cache.hits, cache.misses) == (2, 3)
    assert cache.stats().entries == 3 and cache.stats().bytes == 3 * 16 * 4
    other = HashEmbeddingProvider(dimension=16, seed=1)
    assert cache_key("a b", inner.spec) != cache_key("a b", other.spec)
    assert cache.clear() == 3 and cache.stats().entries == 0
    cache.close()


def _remote(handler: Any, allow: bool = True, key_env: str | None = None) -> RemoteEmbeddingProvider:
    cfg = RemoteEmbeddingConfig(
        url="https://embeddings.invalid/v1/embeddings",
        model="m",
        allow_corpus_transmission=allow,
        api_key_env=key_env,
        batch_size=2,
    )
    return RemoteEmbeddingProvider(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_remote_embeddings_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"auth": request.headers.get("authorization"), "n": len(body["input"])})
        data = [{"index": i, "embedding": [float(i + 1), 0.0, 1.0]} for i in range(len(body["input"]))][::-1]
        return httpx.Response(200, json={"data": data})

    monkeypatch.setenv("SF_TEST_KEY", "k-123")
    p = _remote(handler, key_env="SF_TEST_KEY")
    out = p.embed_documents(["a", "b", "c"])
    assert out.shape == (3, 3) and np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert out[0][0] < out[1][0]  # response re-ordered by index
    assert all(s["auth"] == "Bearer k-123" for s in seen) and [s["n"] for s in seen[-2:]] == [2, 1]
    with pytest.raises(EmbeddingFailure, match="allow_corpus_transmission"):
        _remote(handler, allow=False).embed_documents(["secret corpus text"])
    failing = _remote(lambda r: httpx.Response(500))
    with pytest.raises(EmbeddingFailure):
        failing.embed_query("x")
    monkeypatch.delenv("SF_TEST_KEY")
    with pytest.raises(EmbeddingFailure, match="not set"):
        _remote(handler, key_env="SF_TEST_KEY").embed_query("x")


def test_sentence_transformer_adapter_with_fake_model() -> None:
    class Fake:
        def get_sentence_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            assert kwargs["normalize_embeddings"] is True
            return np.array([[len(t), 1, 0, 0] for t in texts], dtype=np.float32)

    p = SentenceTransformerEmbeddingProvider("fake/model", "abc123", model_obj=Fake(), query_prefix="query: ")
    assert p.spec.dimension == 4 and p.spec.revision == "abc123"
    assert np.linalg.norm(p.embed_query("x")) == pytest.approx(1.0)
    assert p.embed_documents(["a", "bb"]).shape == (2, 4)
