"""SearchService: the application service that executes a query plan.

Pipeline::

    normalize → parse/validate filters → analyze (+inline filters) → plan → validate plan
      → cache lookup (version-aware key)
      → rewrite (recorded, optional multi-query variants)
      → retrievers in parallel (per-retriever timeout; failures → degraded, never hidden)
      → fusion (if >1 list) → hydrate from chunk store
      → rerank top-N (timeout → fused order fallback, flagged degraded)
      → diversity (per-document cap / MMR) → top-k results (+explanations)
      → trace, metrics, cache store

Timed-out work keeps running in its worker thread (Python threads cannot be
killed); its result is discarded. The executor is bounded (``max_workers``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import replace
from typing import Any

from searchforge.caching.query_cache import QueryResultCache, query_cache_key
from searchforge.config.models import (
    RerankerConfig,
    RerankerName,
    SearchForgeConfig,
    SearchProfile,
)
from searchforge.config.profiles import resolve_profiles
from searchforge.domain.filters import FilterExpression, merge_filters, parse_filters
from searchforge.domain.ids import query_hash
from searchforge.domain.query import QueryAnalysis, SearchQuery
from searchforge.domain.result import (
    Candidate,
    RankingSignals,
    ScoredChunk,
    SearchResponse,
    SearchResult,
    SearchTrace,
)
from searchforge.errors import (
    EmbeddingFailure,
    InvalidFilter,
    InvalidPlan,
    RerankerFailure,
    RetrievalFailure,
    SearchForgeError,
    SearchTimeout,
)
from searchforge.fusion.strategies import build_fusion
from searchforge.indexes.handle import IndexHandle
from searchforge.observability.metrics import SearchMetrics
from searchforge.observability.privacy import capture_query
from searchforge.observability.tracing import span
from searchforge.parsing.normalize import normalize_query_text
from searchforge.ports import Reranker, RetrievalRequest, Retriever
from searchforge.query.analysis import QueryAnalyzer
from searchforge.query.planning import QueryPlan, QueryPlanner, StaticQueryPlanner, validate_plan
from searchforge.query.rewriting import QueryRewriter
from searchforge.ranking.diversity import cap_per_document, mmr
from searchforge.reranking import build_reranker
from searchforge.retrieval.retrievers import DenseRetriever, LexicalRetriever


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


class SearchService:
    def __init__(
        self,
        handle: IndexHandle,
        config: SearchForgeConfig,
        *,
        metrics: SearchMetrics | None = None,
        planner: QueryPlanner | None = None,
        rerankers: Mapping[str, Reranker] | None = None,
        retrievers: Mapping[str, Retriever] | None = None,
    ) -> None:
        self.handle = handle
        self.config = config
        self.search_cfg = config.search
        self.profiles = resolve_profiles(config.search)
        self.static_planner = StaticQueryPlanner(self.profiles, config.search.planner)
        self.planner: QueryPlanner = planner or self.static_planner
        filterable = [f for f in handle.filter_schema.fields if f not in ("document_id",)]
        self.analyzer = QueryAnalyzer(filterable, extract_inline_filters=config.search.planner.extract_inline_filters)
        self.metrics = metrics or SearchMetrics()
        self.cache = QueryResultCache(config.search.cache)
        self._rerankers: dict[str, Reranker] = dict(rerankers or {})
        self._retrievers: dict[str, Retriever] = dict(retrievers or {})
        if not self._retrievers:
            if handle.lexical is not None:
                self._retrievers["lexical"] = LexicalRetriever(handle.lexical)
            if handle.vectors is not None and handle.embedder is not None:
                self._retrievers["dense"] = DenseRetriever(handle.embedder, handle.vectors)
        self.executor = ThreadPoolExecutor(max_workers=config.search.max_workers, thread_name_prefix="searchforge")

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    @property
    def available_retrievers(self) -> set[str]:
        return set(self._retrievers)

    @property
    def index_id(self) -> str:
        return self.handle.index_id

    def register_profile(self, name: str, profile: SearchProfile) -> None:
        """Add/replace a profile at runtime (experiments, tuning). Planner rules are unchanged."""
        self.profiles[name] = profile
        self.static_planner.profiles[name] = profile
        self.cache.clear()  # cached responses for a replaced profile must not be served

    def reranker_for(self, cfg: RerankerConfig) -> Reranker | None:
        if cfg.type is RerankerName.NONE:
            return None
        key = f"{cfg.type.value}:{cfg.model}:{cfg.revision}:{sorted(cfg.feature_weights.items())}"
        if key not in self._rerankers:
            self._rerankers[key] = build_reranker(cfg)
        return self._rerankers[key]

    # ------------------------------------------------------------------ planning
    def _plan(self, analysis: QueryAnalysis, profile: str, top_k: int) -> QueryPlan:
        plan = self.planner.plan(analysis, profile)
        adaptive = self.static_planner.profile(profile).adaptive
        missing = set(plan.retrievers) - self.available_retrievers
        if missing and adaptive:
            kept = tuple(r for r in plan.retrievers if r in self.available_retrievers)
            plan = replace(
                plan, retrievers=kept, notes=(*plan.notes, f"index lacks {sorted(missing)}; dropped from plan")
            )
        if plan.candidate_k < top_k:
            plan = replace(plan, candidate_k=top_k, notes=(*plan.notes, "candidate_k raised to top_k"))
        if plan.reranker.type is not RerankerName.NONE and plan.reranker.top_n < top_k:
            plan = replace(plan, reranker=plan.reranker.model_copy(update={"top_n": top_k}))
        validate_plan(plan, available_retrievers=self.available_retrievers, top_k=top_k)
        return plan

    # ------------------------------------------------------------------ search
    def search(
        self,
        text: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
        profile: str | None = None,
        query_id: str | None = None,
        explain: bool = False,
        use_cache: bool = True,
        ranking_depth: int | None = None,
    ) -> SearchResponse:
        """Execute a search.

        ``ranking_depth`` (evaluation use) returns up to that many ranked chunks
        *without changing the plan* (candidate depth and rerank depth are those of
        the profile), so metrics such as Recall@20 can be computed after collapsing
        chunks to documents. Responses with a ranking depth are never cached.
        """
        t0 = time.perf_counter()
        profile_name = profile or self.search_cfg.default_profile
        status = "error"
        n_results = 0
        try:
            resp = self._search(
                text,
                top_k or self.search_cfg.top_k,
                filters,
                profile_name,
                query_id,
                explain,
                use_cache and ranking_depth is None,
                t0,
                ranking_depth,
            )
            status = "cache_hit" if resp.trace.cache_hit else ("degraded" if resp.degraded else "ok")
            n_results = len(resp.results)
            return resp
        except SearchForgeError as exc:
            self.metrics.error("total", type(exc).__name__)
            raise
        finally:
            label = profile_name if profile_name in self.profiles else "invalid"
            self.metrics.observe_request(label, status, time.perf_counter() - t0, n_results)

    def _search(
        self,
        text: str,
        top_k: int,
        raw_filters: Mapping[str, Any] | None,
        profile_name: str,
        query_id: str | None,
        explain: bool,
        use_cache: bool,
        t0: float,
        ranking_depth: int | None = None,
    ) -> SearchResponse:
        deadline = t0 + self.search_cfg.timeouts.overall_ms / 1000.0
        qtext = normalize_query_text(text)
        qid = query_id or "q-" + query_hash(qtext)[:12]
        schema = self.handle.filter_schema
        explicit = parse_filters(raw_filters, schema)

        st = time.perf_counter()
        with span("query_analysis", query_hash=query_hash(qtext)):
            analysis = self.analyzer.analyze(qtext, explicit_filter_count=len(explicit.conditions))
        inline = FilterExpression()
        search_text = qtext
        if analysis.extracted_filters:
            try:
                inline = parse_filters(analysis.extracted_filters, schema)
                search_text = analysis.residual_text or qtext
            except InvalidFilter:
                inline = FilterExpression()
        filters = merge_filters(explicit, inline)
        query = SearchQuery(query_id=qid, text=search_text, filters=filters, requested_k=top_k)
        plan = self._plan(analysis, profile_name, top_k)
        analysis_ms = _ms(st)

        trace = SearchTrace(
            query_id=qid,
            query_capture=capture_query(qtext, self.search_cfg.capture_policy),
            capture_policy=self.search_cfg.capture_policy.value,
            profile=profile_name,
            planner_decision={
                **plan.as_dict(),
                "analysis": analysis.as_dict(),
                "planner_version": self.planner.version,
            },
        )
        trace.add_timing("analysis", analysis_ms)
        self.metrics.observe_stage("analysis", analysis_ms / 1000)

        cache_key = query_cache_key(
            text=query.text,
            filters=filters,
            index_id=self.index_id,
            planner_version=self.planner.version,
            profile=profile_name,
            top_k=top_k,
            explain=explain,
        )
        if use_cache and self.cache.enabled:
            cached = self.cache.get(cache_key)
            self.metrics.cache_events.labels(cache="query", event="hit" if cached else "miss").inc()
            if cached is not None:
                hit_trace = replace(cached.trace, cache_hit=True, query_id=qid, timings=[])
                hit_trace.add_timing("cache_lookup", _ms(t0))
                return replace(cached, query_id=qid, trace=hit_trace, latency_ms=_ms(t0))

        # ---------------------------------------------------------------- rewrite
        rewriter = QueryRewriter(plan.rewrite)
        record = rewriter.rewrite(query.text)
        trace.rewrite = {
            "original_hash": query_hash(record.original),
            "strategies": list(record.strategies),
            "variants": len(record.variants),
            "changed": record.changed,
        }
        texts = [record.rewritten] + (list(record.variants) if plan.rewrite.multi_query else [])

        # ---------------------------------------------------------------- retrieve
        degraded: list[str] = []
        lists = self._retrieve(plan, texts, query, analysis, deadline, trace, degraded)
        trace.retrievers = list(plan.retrievers)
        trace.candidate_counts = {name: len(c) for name, c in lists.items()}

        # ---------------------------------------------------------------- fuse
        st = time.perf_counter()
        fusion = build_fusion(plan.fusion)
        if len(lists) == 1:
            only = next(iter(lists.values()))
            fused = [replace(c, rank=i + 1) for i, c in enumerate(only[: plan.candidate_k])]
            trace.fusion = None
        else:
            with span("fusion", strategy=fusion.name):
                fused = fusion.fuse(lists, limit=plan.candidate_k)
            trace.fusion = fusion.name
        trace.candidate_counts["fused"] = len(fused)
        fusion_ms = _ms(st)
        trace.add_timing("fusion", fusion_ms)
        self.metrics.observe_stage("fusion", fusion_ms / 1000)

        # ---------------------------------------------------------------- hydrate
        st = time.perf_counter()
        scored = self._hydrate(fused, lists, filters, degraded)
        hydrate_ms = _ms(st)
        trace.add_timing("hydrate", hydrate_ms)
        self.metrics.observe_stage("hydrate", hydrate_ms / 1000)

        # ---------------------------------------------------------------- rerank
        try:
            reranker = self.reranker_for(plan.reranker)
        except RerankerFailure as exc:
            if not self.search_cfg.timeouts.reranker_fallback:
                raise
            self.metrics.error("rerank", type(exc).__name__)
            degraded.append("reranker_unavailable")
            reranker = None
        trace.reranker = reranker.name if reranker else None
        if reranker is not None and scored:
            scored = self._rerank(reranker, plan, query.text, scored, deadline, trace, degraded)

        # ---------------------------------------------------------------- diversity
        st = time.perf_counter()
        if plan.diversity.mmr_lambda is not None and self.handle.vectors is not None:
            pool = scored[: max(top_k * 3, top_k)]
            vecs = self.handle.vectors.get_vectors([c.chunk_id for c in pool])
            scored = mmr(pool, vecs, plan.diversity.mmr_lambda, top_k) + scored[len(pool) :]
        if plan.diversity.max_per_document is not None:
            scored = cap_per_document(scored, plan.diversity.max_per_document)
        div_ms = _ms(st)
        trace.add_timing("diversity", div_ms)
        self.metrics.observe_stage("diversity", div_ms / 1000)

        results = [
            self._result(c, rank, explain, plan)
            for rank, c in enumerate(scored[: max(top_k, ranking_depth or 0)], start=1)
        ]
        trace.result_ids = [r.chunk_id for r in results]
        trace.degraded = bool(degraded)
        trace.degradation_reasons = degraded
        total_ms = _ms(t0)
        trace.add_timing("total", total_ms)
        for reason in degraded:
            self.metrics.degraded.labels(reason=reason.split(":", 1)[0].split("#", 1)[0]).inc()
        response = SearchResponse(
            query_id=qid,
            results=results,
            profile=plan.profile,
            index_id=self.index_id,
            latency_ms=total_ms,
            degraded=bool(degraded),
            degradation_reasons=degraded,
            trace=trace,
            candidates_by_retriever=lists,
            fused_candidates=fused,
        )
        if use_cache and not degraded:
            self.cache.put(cache_key, response)
            stats = self.cache.stats
            self.metrics.cache_entries.set(stats.entries)
            self.metrics.cache_bytes.set(stats.bytes)
        return response

    # ------------------------------------------------------------------ stages
    def _retrieve(
        self,
        plan: QueryPlan,
        texts: list[str],
        query: SearchQuery,
        analysis: QueryAnalysis,
        deadline: float,
        trace: SearchTrace,
        degraded: list[str],
    ) -> dict[str, list[Candidate]]:
        futures: dict[str, tuple[str, Future[tuple[list[Candidate], float]]]] = {}
        for vi, variant in enumerate(texts):
            req = RetrievalRequest(
                text=variant, k=plan.candidate_k, filters=query.filters, phrases=analysis.quoted_phrases
            )
            for name in plan.retrievers:
                key = name if vi == 0 else f"{name}#v{vi}"
                futures[key] = (name, self.executor.submit(self._timed_retrieve, name, req))
        lists: dict[str, list[Candidate]] = {}
        errors: list[str] = []
        for key, (name, fut) in futures.items():
            remaining = deadline - time.perf_counter()
            timeout = max(0.0, min(self.search_cfg.timeouts.retriever_ms / 1000.0, remaining))
            try:
                cands, ms = fut.result(timeout=timeout)
                lists[key] = cands
                trace.add_timing(name, ms)
                self.metrics.observe_stage(name, ms / 1000)
            except FutureTimeout:
                fut.cancel()
                errors.append(f"{key}_timeout")
                self.metrics.error(name, "timeout")
            except (RetrievalFailure, EmbeddingFailure) as exc:
                errors.append(f"{key}_error:{type(exc).__name__}")
                self.metrics.error(name, type(exc).__name__)
        if not lists:
            if any("timeout" in e for e in errors):
                raise SearchTimeout("all retrievers timed out: " + ", ".join(errors))
            raise RetrievalFailure("all retrievers failed: " + ", ".join(errors))
        if errors:
            if not self.search_cfg.timeouts.allow_degraded:
                raise RetrievalFailure("retriever failure with allow_degraded=false: " + ", ".join(errors))
            degraded.extend(errors)
        return lists

    def _timed_retrieve(self, name: str, request: RetrievalRequest) -> tuple[list[Candidate], float]:
        st = time.perf_counter()
        with span(f"{name}_retrieval", k=request.k):
            out = self._retrievers[name].retrieve(request)
        return out, _ms(st)

    def _hydrate(
        self,
        fused: list[Candidate],
        lists: Mapping[str, list[Candidate]],
        filters: FilterExpression,
        degraded: list[str],
    ) -> list[ScoredChunk]:
        chunks = self.handle.chunk_store.get_many([c.chunk_id for c in fused])
        positions: dict[str, dict[str, tuple[int, float]]] = {}
        for list_name, cands in lists.items():
            for c in cands:
                positions.setdefault(c.chunk_id, {}).setdefault(list_name, (c.rank, c.score))
        out: list[ScoredChunk] = []
        missing = 0
        for c in fused:
            chunk = chunks.get(c.chunk_id)
            if chunk is None:
                missing += 1
                continue
            per = positions.get(c.chunk_id, {})
            sig = RankingSignals(per_retriever=dict(per), fusion_score=c.score, fusion_rank=c.rank)
            if "lexical" in per:
                sig.lexical_rank, sig.lexical_score = per["lexical"]
            if "dense" in per:
                sig.dense_rank, sig.dense_score = per["dense"]
            if filters:
                sig.filter_reasons = [f"passed: {cond.describe()}" for cond in filters.conditions]
            md = chunk.metadata
            out.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=c.score,
                    retrieval_sources=sorted({k.split("#", 1)[0] for k in per}),
                    metadata={
                        "title": md.title,
                        "section": md.section,
                        "uri": md.uri,
                        "source_id": md.source_id,
                        "source_key": md.source_key,
                        "start_char": md.start_char,
                        "end_char": md.end_char,
                        "sequence": chunk.sequence,
                    },
                    attributes=dict(md.attributes),
                    signals=sig,
                )
            )
        if missing:
            degraded.append(f"index_inconsistency:{missing}_chunks_missing")
        return out

    def _rerank(
        self,
        reranker: Reranker,
        plan: QueryPlan,
        text: str,
        scored: list[ScoredChunk],
        deadline: float,
        trace: SearchTrace,
        degraded: list[str],
    ) -> list[ScoredChunk]:
        top_n = min(plan.reranker.top_n, len(scored))
        head, tail = scored[:top_n], scored[top_n:]
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            degraded.append("deadline_before_rerank")
            return scored
        timeout = min(self.search_cfg.timeouts.reranker_ms / 1000.0, remaining)
        st = time.perf_counter()
        fut = self.executor.submit(reranker.rerank, text, head)
        try:
            with span("reranking", reranker=reranker.name, candidates=top_n):
                reranked = fut.result(timeout=timeout)
        except FutureTimeout:
            fut.cancel()
            self.metrics.error("rerank", "timeout")
            if not self.search_cfg.timeouts.reranker_fallback:
                raise RerankerFailure("reranker timed out") from None
            degraded.append("reranker_timeout")
            return scored
        except RerankerFailure as exc:
            self.metrics.error("rerank", type(exc).__name__)
            if not self.search_cfg.timeouts.reranker_fallback:
                raise
            degraded.append("reranker_error")
            return scored
        ms = _ms(st)
        trace.add_timing("rerank", ms)
        trace.candidate_counts["reranked"] = top_n
        self.metrics.observe_stage("rerank", ms / 1000)
        if {c.chunk_id for c in reranked} != {c.chunk_id for c in head}:
            raise RerankerFailure("reranker violated its contract (candidate set changed)")
        return reranked + tail

    def _result(self, c: ScoredChunk, rank: int, explain: bool, plan: QueryPlan) -> SearchResult:
        explanation = None
        if explain:
            explanation = {
                "signals": c.signals.as_dict(),
                "plan_profile": plan.profile,
                "query_class": plan.query_class,
            }
        return SearchResult(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            score=round(float(c.score), 6),
            rank=rank,
            retrieval_sources=tuple(c.retrieval_sources),
            metadata=c.metadata,
            text=c.text,
            explanation=explanation,
        )


def validate_profile_exists(service: SearchService, profile: str) -> None:
    if profile not in service.profiles:
        raise InvalidPlan(f"unknown search profile {profile!r}; available: {sorted(service.profiles)}")
