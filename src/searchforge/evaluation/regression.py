"""Baseline-vs-candidate retrieval regression analysis and release quality gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from searchforge.config.models import GateConfig
from searchforge.domain.judgment import QuerySeverity
from searchforge.evaluation.runner import RunResult


@dataclass(slots=True)
class QueryDelta:
    query_id: str
    query_type: str
    severity: str
    baseline: float
    candidate: float
    delta: float
    status: str
    deltas: dict[str, float]
    baseline_top: list[str]
    candidate_top: list[str]
    relevant: list[str]


@dataclass(slots=True)
class ComparisonResult:
    baseline: str
    candidate: str
    primary_metric: str
    deltas: list[QueryDelta]
    summary: dict[str, Any]
    missing_queries: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[QueryDelta]:
        return [d for d in self.deltas if d.status == "regressed"]

    @property
    def critical_regressions(self) -> list[QueryDelta]:
        return [d for d in self.regressions if d.severity == QuerySeverity.CRITICAL.value]


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    *,
    primary_metric: str = "ndcg@10",
    epsilon: float = 1e-9,
    relevant_by_query: dict[str, list[str]] | None = None,
) -> ComparisonResult:
    base = {o.query.query_id: o for o in baseline.scored}
    cand = {o.query.query_id: o for o in candidate.scored}
    common = sorted(base.keys() & cand.keys())
    missing = sorted(base.keys() ^ cand.keys())
    deltas: list[QueryDelta] = []
    for qid in common:
        b, c = base[qid], cand[qid]
        if primary_metric not in b.metrics:
            raise KeyError(f"metric {primary_metric} not computed in baseline run")
        d = c.metrics[primary_metric] - b.metrics[primary_metric]
        status = "improved" if d > epsilon else ("regressed" if d < -epsilon else "unchanged")
        deltas.append(
            QueryDelta(
                query_id=qid,
                query_type=b.query.query_type,
                severity=b.query.severity.value,
                baseline=b.metrics[primary_metric],
                candidate=c.metrics[primary_metric],
                delta=d,
                status=status,
                deltas={m: c.metrics[m] - b.metrics[m] for m in b.metrics if m in c.metrics},
                baseline_top=b.doc_ranking[:10],
                candidate_top=c.doc_ranking[:10],
                relevant=(relevant_by_query or {}).get(qid, []),
            )
        )
    n = len(deltas) or 1
    b_agg, c_agg = baseline.aggregate([base[q] for q in common]), candidate.aggregate([cand[q] for q in common])
    b_lat, c_lat = baseline.latency(), candidate.latency()
    mean_deltas = {m: c_agg[m] - b_agg[m] for m in b_agg if m in c_agg}
    summary = {
        "queries": len(deltas),
        "improved": sum(d.status == "improved" for d in deltas),
        "regressed": sum(d.status == "regressed" for d in deltas),
        "unchanged": sum(d.status == "unchanged" for d in deltas),
        "regression_rate": sum(d.status == "regressed" for d in deltas) / n,
        "baseline_metrics": b_agg,
        "candidate_metrics": c_agg,
        "mean_deltas": mean_deltas,
        "latency": {
            "baseline": b_lat,
            "candidate": c_lat,
            "p50_delta_ms": c_lat["p50_ms"] - b_lat["p50_ms"],
            "p95_delta_ms": c_lat["p95_ms"] - b_lat["p95_ms"],
        },
        "critical_regressions": [
            d.query_id for d in deltas if d.status == "regressed" and d.severity == QuerySeverity.CRITICAL.value
        ],
    }
    return ComparisonResult(
        baseline=baseline.name,
        candidate=candidate.name,
        primary_metric=primary_metric,
        deltas=deltas,
        summary=summary,
        missing_queries=missing,
    )


@dataclass(slots=True)
class GateCheck:
    name: str
    passed: bool
    observed: float | int | None
    threshold: float | int | None
    detail: str = ""


@dataclass(slots=True)
class GateReport:
    passed: bool
    checks: list[GateCheck]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "observed": c.observed,
                    "threshold": c.threshold,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


def evaluate_gates(comparison: ComparisonResult, candidate: RunResult, gates: GateConfig) -> GateReport:
    checks: list[GateCheck] = []
    md = comparison.summary["mean_deltas"]
    if gates.ndcg_at_10_min_delta is not None:
        obs = md.get("ndcg@10")
        checks.append(
            GateCheck(
                "ndcg_at_10_min_delta",
                obs is not None and obs >= gates.ndcg_at_10_min_delta,
                obs,
                gates.ndcg_at_10_min_delta,
                "mean ΔnDCG@10 (candidate − baseline)",
            )
        )
    if gates.recall_at_20_min is not None:
        obs = candidate.aggregate().get("recall@20")
        checks.append(
            GateCheck(
                "recall_at_20_min",
                obs is not None and obs >= gates.recall_at_20_min,
                obs,
                gates.recall_at_20_min,
                "candidate Recall@20",
            )
        )
    if gates.p95_ms_max is not None:
        obs = candidate.latency()["p95_ms"]
        checks.append(GateCheck("p95_ms_max", obs <= gates.p95_ms_max, obs, gates.p95_ms_max, "candidate p95 latency"))
    if gates.max_query_regression_rate is not None:
        obs = comparison.summary["regression_rate"]
        checks.append(
            GateCheck(
                "max_query_regression_rate",
                obs <= gates.max_query_regression_rate,
                obs,
                gates.max_query_regression_rate,
                "share of queries whose nDCG@10 dropped",
            )
        )
    if gates.block_on_critical_regression:
        crit = comparison.summary["critical_regressions"]
        checks.append(GateCheck("no_critical_regressions", not crit, len(crit), 0, ", ".join(crit) if crit else "none"))
    return GateReport(passed=all(c.passed for c in checks), checks=checks)
