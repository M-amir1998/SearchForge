"""FastAPI service, Qdrant (in-memory) and OpenSearch adapters, integrations, eval tooling."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from searchforge.api.app import create_app
from searchforge.config.loader import LoadedConfig
from searchforge.config.models import QdrantConfig
from searchforge.domain.filters import FilterSchema, parse_filters
from searchforge.domain.judgment import EvalQuery, Qrels, QuerySeverity, RelevanceJudgment
from searchforge.errors import EvaluationFailure, UnsafePathError
from searchforge.evaluation.beir import safe_extract
from searchforge.evaluation.clicks import ClickEvent, load_click_log, propensities, weak_labels
from searchforge.evaluation.datasets import load_qrels, save_qrels
from searchforge.evaluation.incidents import IncidentCase, append_incident, incident_to_eval
from searchforge.evaluation.review import export_review_pool, import_review
from searchforge.evaluation.runner import evaluate
from searchforge.indexes.builder import IndexBuilder
from searchforge.indexes.external import opensearch as os_adapter
from searchforge.indexes.handle import IndexHandle, open_index
from searchforge.integrations.agent_tool import TOOL_SCHEMA, make_search_tool
from searchforge.integrations.rag import SearchForgeRetriever
from searchforge.integrations.reliability_lab import build_export, citation_validity
from searchforge.service.search_service import SearchService
from tests.conftest import small_config

pytestmark = pytest.mark.integration
Mini = tuple[LoadedConfig, IndexHandle, SearchService]


def test_api_endpoints(mini_index: Mini) -> None:
    _, _, service = mini_index
    client = TestClient(create_app(service))
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready").json()
    assert ready["ready"] and ready["retrievers"] == ["dense", "lexical"]
    body = client.post("/search", json={"query": "rollback release", "top_k": 2, "profile": "hybrid"}).json()
    assert body["results"][0]["document_id"] == "DOC-ROLLBACK" and len(body["results"]) == 2
    assert body["index_version"].startswith("test@") and "trace" not in body and body["degraded"] is False
    explained = client.post("/search/explain", json={"query": "ERR-4012", "profile": "adaptive"}).json()
    assert explained["planned_profile"] == "lexical" and explained["trace"]["query"].startswith("sha256:")
    assert "explanation" in explained["results"][0]
    assert client.post("/search", json={"query": "x", "filters": {"nope": 1}}).status_code == 400
    assert client.post("/search", json={"query": "x", "profile": "nope"}).status_code == 400
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "unexpected": 1}).status_code == 422
    indexes = client.get("/indexes").json()
    assert indexes["indexes"][0]["current"] and "files" not in indexes["indexes"][0]
    version = indexes["indexes"][0]["version"]
    assert client.get(f"/indexes/{version}").json()["chunks"] > 0
    assert client.get("/indexes/v0-missing").status_code == 404
    metrics = client.get("/metrics").text
    assert "searchforge_search_requests_total" in metrics and "searchforge_stage_latency_seconds" in metrics
    assert "rollback" not in metrics  # no query text in labels


def test_rag_retriever_agent_tool_and_export(mini_index: Mini) -> None:
    _, handle, service = mini_index
    rag = SearchForgeRetriever(service, profile="hybrid", top_k=3)
    ev = rag.retrieve("rollback")
    assert ev and ev[0].citation_id.startswith("S") and ev[0].text
    pkg = rag.context("rollback")
    assert citation_validity(pkg, handle.chunk_store)["valid_rate"] == 1.0
    tool = make_search_tool(service)
    out = tool("ERR-4012", top_k=50)
    assert out["ok"] and len(out["results"]) <= 20 and TOOL_SCHEMA["input_schema"]["required"] == ["query"]
    assert tool("x", filters={"bad": 1})["ok"] is False
    run = evaluate(
        service, [EvalQuery("q", "rollback")], Qrels([RelevanceJudgment("q", "DOC-ROLLBACK", 3)]), profile="lexical"
    )
    export = build_export([run], citations=citation_validity(pkg, handle.chunk_store))
    assert export["schema_version"] == "searchforge.eval-export/v1" and export["runs"][0]["queries"] == 1


@pytest.mark.qdrant
def test_qdrant_adapter_in_memory(tmp_path: Path, mini_corpus: Path) -> None:
    pytest.importorskip("qdrant_client")
    from searchforge.indexes.external.qdrant import QdrantVectorIndex, make_client

    loaded = small_config(
        tmp_path,
        mini_corpus,
        index={
            "name": "qd",
            "dense": {"provider": "hash", "dimension": 64, "backend": "qdrant", "qdrant": {"location": ":memory:"}},
        },
    )
    client = make_client(QdrantConfig(location=":memory:"))
    idx = QdrantVectorIndex(client, "t1", 3, create=True)
    idx.add(
        ["a", "b", "c"],
        ["d1", "d2", "d3"],
        [{"category": "x"}, {"category": "y"}, {"category": "x"}],
        np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype=np.float32),
    )
    schema = FilterSchema.from_mapping({"category": "string"})
    hits = idx.search(np.array([1, 0, 0], dtype=np.float32), parse_filters(None, schema), 2)
    assert [h.chunk_id for h in hits] == ["a", "b"] and idx.count() == 3
    assert [
        h.chunk_id for h in idx.search(np.array([1, 0, 0], np.float32), parse_filters({"category": "x"}, schema), 5)
    ] == ["a", "c"]
    assert [
        h.chunk_id
        for h in idx.search(np.array([1, 0, 0], np.float32), parse_filters({"category": {"ne": "x"}}, schema), 5)
    ] == ["b"]
    idx.delete_documents(["d1"])
    assert idx.count() == 2 and set(idx.get_vectors(["a", "b"])) == {"b"}
    # end-to-end build through the builder (embedded in-memory Qdrant per process)
    res = IndexBuilder(loaded).build()
    assert res.manifest.vector_backend == "qdrant"


def test_opensearch_query_dsl_and_fake_client() -> None:
    schema = FilterSchema.from_mapping({"category": "string", "year": "integer"})
    expr = parse_filters({"category": "security", "year": {"gte": 2024}}, schema)
    q = os_adapter.lexical_query('ERR-4012 OR "x" AND *', expr, 5)
    assert q["query"]["bool"]["must"][0]["multi_match"]["query"] == 'ERR-4012 OR "x" AND *'  # value, not syntax
    assert {"range": {"attributes.year": {"gte": 2024}}} in q["query"]["bool"]["filter"]
    k = os_adapter.knn_query([0.1, 0.2], expr, 3)
    assert k["query"]["knn"]["embedding"]["k"] == 3 and "filter" in k["query"]["knn"]["embedding"]
    assert os_adapter.knn_mapping(8)["mappings"]["properties"]["embedding"]["dimension"] == 8

    class FakeClient:
        def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
            return {"hits": {"hits": [{"_score": 2.0, "_source": {"chunk_id": "c1", "document_id": "d1"}}]}}

    hits = os_adapter.OpenSearchLexicalIndex(FakeClient(), "idx").search("q", expr, 5)
    assert hits[0].chunk_id == "c1" and hits[0].source == "lexical"


@pytest.mark.opensearch
@pytest.mark.skipif(not os.environ.get("OPENSEARCH_URL"), reason="OPENSEARCH_URL not set (live test)")
def test_opensearch_live() -> None:  # pragma: no cover - requires the opensearch compose profile
    from searchforge.config.models import OpenSearchConfig

    client = os_adapter.make_client(OpenSearchConfig(hosts=[os.environ["OPENSEARCH_URL"]], verify_certs=False))
    assert client.info()["version"]["number"]


def test_incident_click_review_qrels_and_archive_safety(tmp_path: Path, mini_index: Mini) -> None:
    _, handle, service = mini_index
    q, judgments = incident_to_eval(
        IncidentCase(
            "why does jane@corp.example get ERR-4012 at 10.0.0.1",
            ["DOC-KEYS"],
            QuerySeverity.CRITICAL,
            incident_ref="INC-7",
        )
    )
    assert "jane" not in q.text and "<email>" in q.text and q.provenance.value == "failure_derived"
    assert judgments[0].relevance == 3 and q.notes == "incident:INC-7"
    qp, rp = tmp_path / "q.jsonl", tmp_path / "r.jsonl"
    rec = append_incident(IncidentCase("rollback broke", ["DOC-ROLLBACK"]), qp, rp, {"DOC-ROLLBACK"})
    assert load_qrels(rp).relevant(rec.query_id) == {"DOC-ROLLBACK"}
    with pytest.raises(EvaluationFailure):
        append_incident(IncidentCase("rollback broke", ["DOC-ROLLBACK"]), qp, rp)
    with pytest.raises(EvaluationFailure, match="not in the index"):
        incident_to_eval(IncidentCase("x", ["NOPE"]), {"DOC-ROLLBACK"})

    log = tmp_path / "clicks.jsonl"
    rows = [{"query_key": "h1", "document_id": "A", "position": 1, "clicked": i % 2 == 0} for i in range(10)]
    rows += [{"query_key": "h1", "document_id": "B", "position": 3, "clicked": i % 5 == 0} for i in range(10)]
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    events = load_click_log(log)
    assert propensities(events)[1] == 1.0 and propensities(events)[3] < 1.0
    labels = {w.document_id: w for w in weak_labels(events)}
    assert labels["B"].ips_clicks > labels["B"].clicks  # debiased upward for the low position
    assert all(w.label_source == "click_weak" for w in labels.values())
    (tmp_path / "raw.jsonl").write_text(
        json.dumps({"query": "raw text", "document_id": "A", "position": 1, "clicked": True}), encoding="utf-8"
    )
    with pytest.raises(EvaluationFailure, match="raw query"):
        load_click_log(tmp_path / "raw.jsonl")
    _ = ClickEvent

    run = evaluate(
        service, [EvalQuery("q1", "rollback")], Qrels([RelevanceJudgment("q1", "DOC-ROLLBACK", 3)]), profile="hybrid"
    )
    pool = tmp_path / "review.csv"
    n = export_review_pool([run], [EvalQuery("q1", "rollback")], handle.chunk_store, pool)
    content = pool.read_text(encoding="utf-8")
    assert n >= 1 and "hybrid" not in content and "score" not in content.splitlines()[0]
    pool.write_text(content.replace(",\n", ",2\n"), encoding="utf-8")
    assert len(import_review(pool)) == n
    pool.write_text(content.replace(",\n", ",7\n"), encoding="utf-8")
    with pytest.raises(EvaluationFailure):
        import_review(pool)

    qrels = Qrels([RelevanceJudgment("q", "d", 2)])
    for suffix in (".tsv", ".parquet", ".jsonl"):
        save_qrels(qrels, tmp_path / f"x{suffix}")
        assert load_qrels(tmp_path / f"x{suffix}").for_query("q") == {"d": 2.0}

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(UnsafePathError):
        safe_extract(evil, tmp_path / "out")
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("ds/file.txt", "ok")
    safe_extract(good, tmp_path / "out")
    assert (tmp_path / "out" / "ds" / "file.txt").read_text() == "ok"
    with pytest.raises(UnsafePathError):
        safe_extract(good, tmp_path / "out2", max_uncompressed=1)


def test_open_index_errors(tmp_path: Path, mini_corpus: Path) -> None:
    from searchforge.errors import IndexNotFound

    loaded = small_config(tmp_path, mini_corpus)
    with pytest.raises(IndexNotFound):
        open_index(loaded.workspace, "test", loaded.config)
