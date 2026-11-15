"""Persist evaluation runs: ``results/<run-id>/`` with config, manifest, metrics and Parquet tables.

Query text in persisted tables follows a text policy: synthetic and public-benchmark
queries are stored verbatim; anything production-derived is redacted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from searchforge.domain.judgment import QueryProvenance
from searchforge.domain.result import SearchResult  # noqa: F401  (re-exported type for callers)
from searchforge.evaluation.failures import FailureReport
from searchforge.evaluation.regression import ComparisonResult
from searchforge.evaluation.runner import QueryOutcome, RunResult
from searchforge.observability.privacy import redact

_VERBATIM = {QueryProvenance.SYNTHETIC, QueryProvenance.PUBLIC_BENCHMARK, QueryProvenance.MANUAL}


def report_text(outcome: QueryOutcome) -> str:
    q = outcome.query
    return q.text if q.provenance in _VERBATIM else redact(q.text)


def per_query_rows(run: RunResult) -> list[dict[str, Any]]:
    rows = []
    for o in run.outcomes:
        rows.append(
            {
                "run": run.name,
                "query_id": o.query.query_id,
                "query_text": report_text(o),
                "query_type": o.query.query_type,
                "split": o.query.split.value,
                "severity": o.query.severity.value,
                "planned_profile": o.planned_profile,
                "query_class": o.query_class,
                "degraded": o.degraded,
                "latency_p50_ms": o.latency_ms,
                "top10": ",".join(o.doc_ranking[:10]),
                "error": o.error or "",
                **{k.replace("@", "_at_"): v for k, v in o.metrics.items()},
            }
        )
    return rows


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        pq.write_table(pa.table({"empty": pa.array([], type=pa.int8())}), path)


def write_run(
    run: RunResult,
    root: Path,
    *,
    config: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    failures: FailureReport | None = None,
    comparison: ComparisonResult | None = None,
    performance: dict[str, Any] | None = None,
) -> Path:
    out = root / run.run_id
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config or run.config, sort_keys=True), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(manifest or {"index_id": run.index_id}, indent=2, default=str), encoding="utf-8"
    )
    (out / "metrics.json").write_text(json.dumps(run.summary(), indent=2, default=str), encoding="utf-8")
    _write_parquet(per_query_rows(run), out / "per_query.parquet")
    _write_parquet([vars_delta(d) for d in comparison.deltas] if comparison else [], out / "regressions.parquet")
    _write_parquet([r.as_row() for r in failures.records] if failures else [], out / "failures.parquet")
    (out / "performance.json").write_text(
        json.dumps(performance or {"latency": run.latency()}, indent=2, default=str), encoding="utf-8"
    )
    return out


def vars_delta(d: Any) -> dict[str, Any]:
    return {
        "query_id": d.query_id,
        "query_type": d.query_type,
        "severity": d.severity,
        "baseline": d.baseline,
        "candidate": d.candidate,
        "delta": d.delta,
        "status": d.status,
        "baseline_top": ",".join(d.baseline_top),
        "candidate_top": ",".join(d.candidate_top),
        "relevant": ",".join(d.relevant),
    }
