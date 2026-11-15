"""Optional, dependency-free integration surfaces (RAG retriever, agent tool, eval export).

SearchForge never serves LLMs; generation and model serving live in other systems.
"""

from __future__ import annotations

from searchforge.integrations.agent_tool import TOOL_SCHEMA, make_search_tool
from searchforge.integrations.rag import Evidence, EvidenceRetriever, SearchForgeRetriever
from searchforge.integrations.reliability_lab import EXPORT_SCHEMA, build_export, citation_validity

__all__ = [
    "EXPORT_SCHEMA",
    "TOOL_SCHEMA",
    "Evidence",
    "EvidenceRetriever",
    "SearchForgeRetriever",
    "build_export",
    "citation_validity",
    "make_search_tool",
]
