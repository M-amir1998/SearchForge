"""Offline search-quality evaluation: metrics, runs, regressions, gates, failures."""

from __future__ import annotations

from searchforge.evaluation.datasets import load_qrels, load_queries, save_qrels, save_queries
from searchforge.evaluation.metrics import collapse_to_documents, evaluate_ranking, ndcg_at_k
from searchforge.evaluation.regression import compare_runs, evaluate_gates
from searchforge.evaluation.runner import RunResult, evaluate, select_queries

__all__ = [
    "RunResult",
    "collapse_to_documents",
    "compare_runs",
    "evaluate",
    "evaluate_gates",
    "evaluate_ranking",
    "load_qrels",
    "load_queries",
    "ndcg_at_k",
    "save_qrels",
    "save_queries",
    "select_queries",
]
