"""Machine-readable evaluation export for downstream reliability tooling (e.g. AI Reliability Lab).

No dependency on any consumer: the export is plain JSON validated against
:data:`EXPORT_SCHEMA` (JSON Schema draft 2020-12) so consumers can rely on its shape.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import jsonschema

from searchforge import __version__
from searchforge.context.builder import ContextPackage
from searchforge.evaluation.regression import ComparisonResult, GateReport
from searchforge.evaluation.runner import RunResult
from searchforge.indexes.chunk_store import SqliteChunkStore

SCHEMA_VERSION = "searchforge.eval-export/v1"

EXPORT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema_version", "producer", "created_at", "runs"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "producer": {"type": "object", "required": ["name", "version"]},
        "created_at": {"type": "string"},
        "runs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "index_id", "queries", "retrieval_quality", "latency_ms"],
                "properties": {
                    "retrieval_quality": {"type": "object", "additionalProperties": {"type": "number"}},
                    "latency_ms": {"type": "object", "additionalProperties": {"type": "number"}},
                    "queries": {"type": "integer", "minimum": 0},
                },
            },
        },
        "regressions": {"type": "array"},
        "gates": {"type": ["object", "null"]},
        "citation_validity": {"type": ["object", "null"]},
    },
}


def citation_validity(package: ContextPackage, store: SqliteChunkStore) -> dict[str, Any]:
    """Every citation must resolve to an existing chunk whose text equals the cited text."""
    chunks = store.get_many([i.chunk_id for i in package.items])
    invalid = [
        i.citation_id
        for i in package.items
        if i.chunk_id not in chunks
        or chunks[i.chunk_id].text != i.text
        or chunks[i.chunk_id].document_id != i.document_id
    ]
    total = len(package.items)
    return {"citations": total, "invalid": invalid, "valid_rate": (total - len(invalid)) / total if total else 1.0}


def build_export(
    runs: list[RunResult],
    comparisons: list[ComparisonResult] | None = None,
    gates: GateReport | None = None,
    citations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "producer": {"name": "searchforge", "version": __version__},
        "created_at": datetime.now(UTC).isoformat(),
        "runs": [
            {
                "name": r.name,
                "index_id": r.index_id,
                "queries": len(r.scored),
                "retrieval_quality": r.aggregate(),
                "latency_ms": {k: v for k, v in r.latency().items() if k != "samples"},
            }
            for r in runs
        ],
        "regressions": [
            {
                "baseline": c.baseline,
                "candidate": c.candidate,
                "primary_metric": c.primary_metric,
                "improved": c.summary["improved"],
                "regressed": c.summary["regressed"],
                "unchanged": c.summary["unchanged"],
                "regressed_queries": [d.query_id for d in c.regressions],
                "critical_regressions": c.summary["critical_regressions"],
            }
            for c in (comparisons or [])
        ],
        "gates": gates.as_dict() if gates else None,
        "citation_validity": citations,
    }
    jsonschema.validate(doc, EXPORT_SCHEMA)
    return doc
