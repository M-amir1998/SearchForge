"""Post-retrieval ranking controls: diversity, MMR, grouping."""

from __future__ import annotations

from searchforge.ranking.diversity import cap_per_document, group_results, mmr

__all__ = ["cap_per_document", "group_results", "mmr"]
