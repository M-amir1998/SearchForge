"""Evidence for RAG and a tool for agents — SearchForge returns citations, never generations.

Run:  uv run python examples/rag_and_agent_tool.py
"""

from __future__ import annotations

import json

from searchforge.config.loader import load_config
from searchforge.context.builder import CitationContextBuilder
from searchforge.datasets import fixture_config_path
from searchforge.indexes.builder import IndexBuilder
from searchforge.indexes.handle import open_index
from searchforge.integrations.agent_tool import TOOL_SCHEMA, make_search_tool
from searchforge.integrations.rag import SearchForgeRetriever
from searchforge.integrations.reliability_lab import citation_validity
from searchforge.service.search_service import SearchService


def main() -> None:
    loaded = load_config(fixture_config_path(), overrides={"workspace": ".searchforge-examples"})
    IndexBuilder(loaded).build()
    handle = open_index(loaded.workspace, loaded.config.index.name, loaded.config)
    service = SearchService(handle, loaded.config)
    try:
        retriever = SearchForgeRetriever(
            service,
            profile="hybrid_rerank",
            top_k=8,
            context_builder=CitationContextBuilder(max_chars=1200, max_per_document=1),
        )
        package = retriever.context("how do I verify webhook signatures?")
        print(package.render())
        print("\ncitations valid:", citation_validity(package, handle.chunk_store)["valid_rate"])

        tool = make_search_tool(service, profile="adaptive")
        print("\ntool schema:", json.dumps(TOOL_SCHEMA["input_schema"], indent=2))
        print(json.dumps(tool("ERR-4031", top_k=2), indent=2)[:800])
    finally:
        service.close()
        handle.close()


if __name__ == "__main__":
    main()
