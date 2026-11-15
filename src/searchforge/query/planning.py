"""Query planning: decide retrievers, candidate depth, fusion and reranking per query.

* :class:`StaticQueryPlanner` — config-driven rules (query class → profile). Default.
* :class:`LLMQueryPlanner`    — optional; asks an injected ``complete(prompt) -> str``
  callable for a profile name. Its output is *untrusted*: it must name an existing,
  non-adaptive profile that passes validation, otherwise the static plan is used and
  the fallback is recorded. No LLM is required anywhere in SearchForge.

Every plan is validated (:func:`validate_plan`) before execution.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from searchforge.config.models import (
    DiversityConfig,
    FusionConfig,
    PlannerConfig,
    RerankerConfig,
    RerankerName,
    RetrieverName,
    RewriteConfig,
    SearchProfile,
)
from searchforge.domain.query import QueryAnalysis
from searchforge.errors import InvalidPlan

MAX_CANDIDATES = 1000


@dataclass(frozen=True, slots=True)
class QueryPlan:
    profile: str
    retrievers: tuple[str, ...]
    candidate_k: int
    fusion: FusionConfig
    reranker: RerankerConfig
    diversity: DiversityConfig
    rewrite: RewriteConfig
    query_class: str
    planner: str
    reason: str
    fallback: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "retrievers": list(self.retrievers),
            "candidate_k": self.candidate_k,
            "fusion": self.fusion.strategy.value if len(self.retrievers) > 1 else None,
            "reranker": self.reranker.type.value,
            "rerank_top_n": self.reranker.top_n if self.reranker.type is not RerankerName.NONE else 0,
            "query_class": self.query_class,
            "planner": self.planner,
            "reason": self.reason,
            "fallback": self.fallback,
            "notes": list(self.notes),
        }


def plan_from_profile(
    name: str, profile: SearchProfile, *, query_class: str, planner: str, reason: str, fallback: str | None = None
) -> QueryPlan:
    return QueryPlan(
        profile=name,
        retrievers=tuple(r.value for r in profile.retrievers),
        candidate_k=profile.candidate_k,
        fusion=profile.fusion,
        reranker=profile.reranker,
        diversity=profile.diversity,
        rewrite=profile.rewrite,
        query_class=query_class,
        planner=planner,
        reason=reason,
        fallback=fallback,
    )


def validate_plan(plan: QueryPlan, *, available_retrievers: set[str], top_k: int) -> None:
    if not plan.retrievers:
        raise InvalidPlan("plan has no retrievers")
    unknown = set(plan.retrievers) - {r.value for r in RetrieverName}
    if unknown:
        raise InvalidPlan(f"unknown retrievers: {sorted(unknown)}")
    missing = set(plan.retrievers) - available_retrievers
    if missing:
        raise InvalidPlan(f"index does not provide retrievers: {sorted(missing)}")
    if not 1 <= plan.candidate_k <= MAX_CANDIDATES:
        raise InvalidPlan("candidate_k out of range")
    if plan.candidate_k < top_k:
        raise InvalidPlan(f"candidate_k ({plan.candidate_k}) must be >= top_k ({top_k})")
    if plan.reranker.type is not RerankerName.NONE and plan.reranker.top_n < top_k:
        raise InvalidPlan("reranker.top_n must be >= top_k")


@runtime_checkable
class QueryPlanner(Protocol):
    name: str
    version: str

    def plan(self, analysis: QueryAnalysis, requested_profile: str) -> QueryPlan: ...


class StaticQueryPlanner:
    """Rules-based routing for ``adaptive`` profiles; pass-through for fixed profiles."""

    name = "static"

    def __init__(self, profiles: Mapping[str, SearchProfile], config: PlannerConfig) -> None:
        self.profiles = dict(profiles)
        self.config = config
        self.version = config.version
        for qclass, target in config.rules.items():
            if target not in self.profiles:
                raise InvalidPlan(f"planner rule {qclass.value} → unknown profile {target!r}")
            if self.profiles[target].adaptive:
                raise InvalidPlan(f"planner rule {qclass.value} → adaptive profile {target!r} (cycle)")

    def profile(self, name: str) -> SearchProfile:
        if name not in self.profiles:
            raise InvalidPlan(f"unknown search profile {name!r}; available: {sorted(self.profiles)}")
        return self.profiles[name]

    def plan(self, analysis: QueryAnalysis, requested_profile: str) -> QueryPlan:
        requested = self.profile(requested_profile)
        qclass = analysis.query_class.value
        if not requested.adaptive:
            return plan_from_profile(
                requested_profile, requested, query_class=qclass, planner=self.name, reason="fixed profile"
            )
        target = self.config.rules.get(analysis.query_class, self.config.default_profile)
        return plan_from_profile(
            target, self.profile(target), query_class=qclass, planner=self.name, reason=f"rule {qclass} → {target}"
        )


class LLMQueryPlanner:
    """Optional planner backed by an injected text-completion callable."""

    name = "llm"

    def __init__(self, complete: Callable[[str], str], fallback: StaticQueryPlanner, version: str = "llm-v1") -> None:
        self.complete = complete
        self.fallback = fallback
        self.version = version

    def _prompt(self, analysis: QueryAnalysis) -> str:
        choices = {n: p.description for n, p in self.fallback.profiles.items() if not p.adaptive}
        return (
            "Choose the retrieval profile for a search query. Respond with JSON "
            '{"profile": <name>, "reason": <short>}.\n'
            f"Profiles: {json.dumps(choices, sort_keys=True)}\n"
            f"Query features: {json.dumps(analysis.as_dict(), sort_keys=True)}\n"
            f"Query: {analysis.residual_text}"
        )

    def plan(self, analysis: QueryAnalysis, requested_profile: str) -> QueryPlan:
        base = self.fallback.plan(analysis, requested_profile)
        if not self.fallback.profile(requested_profile).adaptive:
            return base
        try:
            raw = json.loads(self.complete(self._prompt(analysis)))
            name = str(raw["profile"])
            reason = str(raw.get("reason", ""))[:200]
            profile = self.fallback.profile(name)
            if profile.adaptive:
                raise InvalidPlan("LLM chose an adaptive profile")
        except Exception as exc:  # untrusted output: any failure → static plan
            return QueryPlan(**{**_plan_kwargs(base), "fallback": f"llm_invalid:{type(exc).__name__}"})
        return plan_from_profile(
            name, profile, query_class=analysis.query_class.value, planner=self.name, reason=f"llm: {reason}"
        )


def _plan_kwargs(plan: QueryPlan) -> dict[str, Any]:
    return {f: getattr(plan, f) for f in QueryPlan.__dataclass_fields__}
