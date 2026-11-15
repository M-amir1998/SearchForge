"""FastAPI search service. FastAPI objects stay in this module (never in domain/services).

Endpoints: ``GET /health``, ``GET /ready``, ``POST /search``, ``POST /search/explain``,
``GET /indexes``, ``GET /indexes/{index_id}``, ``GET /metrics`` (Prometheus).

Responses never include configuration, filesystem paths or secrets; raw query text
is not logged (traces carry the policy-controlled capture, hash-only by default).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from searchforge import __version__
from searchforge.domain.query import MAX_QUERY_CHARS
from searchforge.domain.result import SearchResponse
from searchforge.errors import (
    IndexNotFound,
    InvalidFilter,
    InvalidPlan,
    InvalidQuery,
    RerankerFailure,
    RetrievalFailure,
    SearchForgeError,
    SearchTimeout,
)
from searchforge.indexes.lifecycle import IndexLayout
from searchforge.service.search_service import SearchService


class SearchRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] | None = None
    profile: str | None = Field(default=None, max_length=64)


def _status_for(exc: SearchForgeError) -> int:
    if isinstance(exc, InvalidFilter | InvalidQuery | InvalidPlan):
        return 400
    if isinstance(exc, IndexNotFound):
        return 404
    if isinstance(exc, SearchTimeout):
        return 504
    if isinstance(exc, RetrievalFailure | RerankerFailure):
        return 503
    return 500


def _manifest_view(m: Any, current: str | None) -> dict[str, Any]:
    return {
        "index_id": m.index_id,
        "name": m.index_name,
        "version": m.version,
        "current": m.version == current,
        "parent_version": m.parent_version,
        "corpus_id": m.corpus_id,
        "corpus_version": m.corpus_version,
        "documents": m.document_count,
        "chunks": m.chunk_count,
        "index_types": m.index_types,
        "embedding": m.embedding.model_dump() if m.embedding else None,
        "chunking": m.chunking_config,
        "created_at": m.created_at.isoformat(),
        "size_bytes": m.size_bytes(),
        "git_sha": m.git_sha,
    }


def _response(resp: SearchResponse, explain: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "query_id": resp.query_id,
        "results": [r.as_dict(include_text=True) for r in resp.results],
        "latency_ms": round(resp.latency_ms, 3),
        "profile": resp.trace.profile,
        "planned_profile": resp.profile,
        "index_version": resp.index_id,
        "degraded": resp.degraded,
        "degradation_reasons": resp.degradation_reasons,
    }
    if explain:
        out["trace"] = resp.trace.as_dict()
    return out


def create_app(service: SearchService) -> FastAPI:
    app = FastAPI(
        title="SearchForge",
        version=__version__,
        description="Hybrid retrieval, reranking and search-quality infrastructure.",
    )
    layout = IndexLayout(service.handle.path.parent.parent.parent)
    index_name = service.handle.manifest.index_name

    def run(body: SearchRequestBody, explain: bool) -> JSONResponse:
        try:
            resp = service.search(
                body.query, top_k=body.top_k, filters=body.filters, profile=body.profile, explain=explain
            )
        except SearchForgeError as exc:
            raise HTTPException(
                status_code=_status_for(exc), detail={"error": type(exc).__name__, "message": str(exc)[:500]}
            ) from exc
        return JSONResponse(_response(resp, explain))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready() -> JSONResponse:
        try:
            chunks = service.handle.chunk_store.count_chunks()
        except Exception:  # pragma: no cover - defensive
            chunks = 0
        ok = chunks > 0 and bool(service.available_retrievers)
        return JSONResponse(
            status_code=200 if ok else 503,
            content={
                "ready": ok,
                "index_version": service.index_id,
                "retrievers": sorted(service.available_retrievers),
                "profiles": sorted(p for p in service.profiles if not p.startswith("_")),
            },
        )

    @app.post("/search")
    def search(body: SearchRequestBody) -> JSONResponse:
        return run(body, explain=False)

    @app.post("/search/explain")
    def search_explain(body: SearchRequestBody) -> JSONResponse:
        return run(body, explain=True)

    @app.get("/indexes")
    def indexes() -> dict[str, Any]:
        current = layout.current_version(index_name)
        return {
            "indexes": [_manifest_view(m, current) for m in layout.versions(index_name)],
            "serving": service.index_id,
        }

    @app.get("/indexes/{index_id}")
    def index_detail(index_id: str) -> dict[str, Any]:
        ref = index_id if "@" in index_id else f"{index_name}@{index_id}"
        if not ref.startswith(index_name + "@"):
            raise HTTPException(status_code=404, detail={"error": "IndexNotFound"})
        try:
            manifest, _ = layout.resolve(ref)
        except IndexNotFound as exc:
            raise HTTPException(status_code=404, detail={"error": "IndexNotFound"}) from exc
        return _manifest_view(manifest, layout.current_version(index_name))

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=service.metrics.render(), media_type="text/plain; version=0.0.4")

    return app
