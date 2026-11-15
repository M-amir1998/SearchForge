"""Fusion-weight tuning with strict train / validation / test separation.

* the grid is scored on the **train** split,
* the best ``top_n`` settings are re-scored on **validation** and the winner selected there,
* the chosen setting is reported **once** on **test** — never tuned on it.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from searchforge.config.models import FusionConfig, FusionName, SearchProfile
from searchforge.domain.judgment import EvalQuery, Qrels, QuerySplit
from searchforge.errors import EvaluationFailure
from searchforge.evaluation.runner import RunResult, evaluate, select_queries
from searchforge.service.search_service import SearchService


@dataclass(slots=True)
class TuningResult:
    metric: str
    strategy: str
    grid: list[dict[str, Any]]
    best_weights: dict[str, float]
    validation_score: float
    test_score: float
    baseline_test_score: float
    test_run: RunResult | None = None
    notes: list[str] = field(default_factory=list)


def tune_fusion_weights(
    service: SearchService,
    queries: Sequence[EvalQuery],
    qrels: Qrels,
    *,
    base_profile: str = "hybrid",
    strategy: FusionName = FusionName.WEIGHTED_RRF,
    lexical_weights: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0),
    dense_weights: Sequence[float] = (0.25, 0.5, 1.0),
    metric: str = "ndcg@10",
    top_n: int = 3,
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
) -> TuningResult:
    train = select_queries(queries, [QuerySplit.TRAIN])
    val = select_queries(queries, [QuerySplit.VALIDATION])
    test = select_queries(queries, [QuerySplit.TEST])
    if not train or not val or not test:
        raise EvaluationFailure("tuning requires non-empty train, validation and test splits")
    base = service.profiles[base_profile]
    grid: list[dict[str, Any]] = []
    for lw, dw in itertools.product(lexical_weights, dense_weights):
        name = f"_tune_{strategy.value}_l{lw}_d{dw}"
        profile = base.model_copy(
            update={"fusion": FusionConfig(strategy=strategy, k=base.fusion.k, weights={"lexical": lw, "dense": dw})}
        )
        service.register_profile(name, profile)
        run = evaluate(service, train, qrels, profile=name, k_values=k_values, warmup=False)
        grid.append({"profile": name, "lexical": lw, "dense": dw, "train": run.aggregate()[metric]})
    grid.sort(key=lambda g: -g["train"])
    for g in grid[:top_n]:
        g["validation"] = evaluate(
            service, val, qrels, profile=g["profile"], k_values=k_values, warmup=False
        ).aggregate()[metric]
    best = max(grid[:top_n], key=lambda g: g["validation"])
    test_run = evaluate(
        service, test, qrels, profile=best["profile"], name=f"tuned-{strategy.value}", k_values=k_values, warmup=False
    )
    baseline = evaluate(service, test, qrels, profile=base_profile, k_values=k_values, warmup=False)
    return TuningResult(
        metric=metric,
        strategy=strategy.value,
        grid=grid,
        best_weights={"lexical": best["lexical"], "dense": best["dense"]},
        validation_score=best["validation"],
        test_score=test_run.aggregate()[metric],
        baseline_test_score=baseline.aggregate()[metric],
        test_run=test_run,
        notes=[
            f"train={len(train)} validation={len(val)} test={len(test)} queries",
            "weights chosen on validation; test reported once",
        ],
    )


def make_profile_variant(base: SearchProfile, **updates: Any) -> SearchProfile:
    return base.model_copy(update=updates)
