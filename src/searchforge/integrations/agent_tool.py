"""Tool-style API for agent runtimes: ``search(query, filters, top_k)``.

Returns JSON-serializable dicts and exposes a JSON-Schema tool definition. Inputs
are validated by the service (filters are typed; unknown fields are rejected), so an
agent cannot inject backend query syntax. No agent framework is imported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from searchforge.domain.ids import make_citation_id
from searchforge.errors import SearchForgeError
from searchforge.service.search_service import SearchService

TOOL_SCHEMA: dict[str, Any] = {
    "name": "search",
    "description": "Search the knowledge base. Returns ranked passages with stable citation ids.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 2000},
            "filters": {"type": "object", "description": 'Typed metadata filters, e.g. {"category": "security"}'},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def make_search_tool(
    service: SearchService, profile: str = "adaptive", max_chars: int = 1200
) -> Callable[..., dict[str, Any]]:
    def search(query: str, filters: Mapping[str, Any] | None = None, top_k: int = 5) -> dict[str, Any]:
        try:
            resp = service.search(query, filters=filters, top_k=max(1, min(int(top_k), 20)), profile=profile)
        except SearchForgeError as exc:
            return {"ok": False, "error": type(exc).__name__, "message": str(exc)[:300]}
        return {
            "ok": True,
            "degraded": resp.degraded,
            "results": [
                {
                    "citation_id": make_citation_id(r.chunk_id),
                    "document_id": r.document_id,
                    "rank": r.rank,
                    "title": r.metadata.get("title"),
                    "section": r.metadata.get("section"),
                    "uri": r.metadata.get("uri"),
                    "text": (r.text or "")[:max_chars],
                }
                for r in resp.results
            ],
        }

    return search
