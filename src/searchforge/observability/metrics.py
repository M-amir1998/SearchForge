"""Prometheus metrics for the search service.

Label policy (enforced here, documented in SECURITY.md):
* labels are drawn from small, closed sets — profile names from config, stage
  names from code, status/reason enums;
* raw query text, document ids, chunk ids or user ids are never labels.

QPS = ``rate(searchforge_search_requests_total[1m])``; p50/p95/p99 via
``histogram_quantile`` over ``searchforge_search_latency_seconds_bucket``.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
STAGES = frozenset(
    {"analysis", "rewrite", "lexical", "dense", "fusion", "hydrate", "rerank", "diversity", "cache_lookup", "total"}
)
STATUSES = frozenset({"ok", "degraded", "error", "cache_hit"})


class SearchMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "searchforge_search_requests", "Search requests", ["profile", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "searchforge_search_latency_seconds",
            "End-to-end search latency",
            ["profile"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.stage_latency = Histogram(
            "searchforge_stage_latency_seconds",
            "Per-stage latency",
            ["stage"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.cache_events = Counter(
            "searchforge_cache_events", "Cache hits/misses", ["cache", "event"], registry=self.registry
        )
        self.errors = Counter(
            "searchforge_errors", "Errors by stage and kind", ["stage", "kind"], registry=self.registry
        )
        self.degraded = Counter(
            "searchforge_degraded", "Degraded responses by reason", ["reason"], registry=self.registry
        )
        self.result_count = Histogram(
            "searchforge_result_count",
            "Results returned",
            buckets=(0, 1, 3, 5, 10, 20, 50, 100),
            registry=self.registry,
        )
        self.cache_entries = Gauge("searchforge_query_cache_entries", "Query cache entries", registry=self.registry)
        self.cache_bytes = Gauge(
            "searchforge_query_cache_bytes", "Query cache approximate bytes", registry=self.registry
        )

    def observe_stage(self, stage: str, seconds: float) -> None:
        self.stage_latency.labels(stage=stage if stage in STAGES else "other").observe(seconds)

    def observe_request(self, profile: str, status: str, seconds: float, n_results: int) -> None:
        status = status if status in STATUSES else "error"
        self.requests.labels(profile=profile, status=status).inc()
        self.latency.labels(profile=profile).observe(seconds)
        self.result_count.observe(n_results)

    def error(self, stage: str, kind: str) -> None:
        self.errors.labels(stage=stage if stage in STAGES else "other", kind=kind[:40]).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)
