"""Blinded human relevance review.

``export_review_pool`` pools the top-k documents of several runs per query,
de-duplicates them, shuffles deterministically and strips every trace of which
system retrieved what (no run names, ranks or scores) so reviewers cannot be
biased by the method. Reviewers assign 0/1/2/3:

    0 not relevant · 1 partially relevant · 2 relevant · 3 perfect answer

``import_review`` validates grades and returns qrels.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Sequence
from pathlib import Path

from searchforge.domain.ids import short_hash
from searchforge.domain.judgment import EvalQuery, Qrels, RelevanceJudgment
from searchforge.errors import EvaluationFailure
from searchforge.evaluation.runner import RunResult
from searchforge.indexes.chunk_store import SqliteChunkStore

FIELDS = ["review_id", "query_id", "query_text", "document_id", "title", "excerpt", "relevance"]
VALID_GRADES = {0, 1, 2, 3}


def export_review_pool(
    runs: Sequence[RunResult],
    queries: Sequence[EvalQuery],
    store: SqliteChunkStore,
    path: Path,
    *,
    depth: int = 10,
    seed: int = 7,
    excerpt_chars: int = 600,
    skip_judged: Qrels | None = None,
) -> int:
    texts = {q.query_id: q.text for q in queries}
    pool: dict[tuple[str, str], None] = {}
    for run in runs:
        for o in run.scored:
            for doc in o.doc_ranking[:depth]:
                if skip_judged is not None and doc in skip_judged.for_query(o.query.query_id):
                    continue
                pool[(o.query.query_id, doc)] = None
    items = sorted(pool)
    random.Random(seed).shuffle(items)  # noqa: S311 - deterministic blinding order, not security
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for qid, doc in items:
            chunks = store.chunks_for_document(doc)
            excerpt = " ".join(c.text for c in chunks)[:excerpt_chars]
            title = chunks[0].metadata.title if chunks else ""
            writer.writerow(
                {
                    "review_id": "r-" + short_hash(f"{qid}|{doc}", 10),
                    "query_id": qid,
                    "query_text": texts.get(qid, ""),
                    "document_id": doc,
                    "title": title or "",
                    "excerpt": excerpt,
                    "relevance": "",
                }
            )
    return len(items)


def import_review(path: Path, source: str = "human_review") -> Qrels:
    qrels = Qrels()
    with open(path, encoding="utf-8", newline="") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            raw = (row.get("relevance") or "").strip()
            if raw == "":
                continue
            try:
                grade = int(raw)
            except ValueError as exc:
                raise EvaluationFailure(f"line {n}: relevance must be an integer 0-3") from exc
            if grade not in VALID_GRADES:
                raise EvaluationFailure(f"line {n}: relevance must be one of 0, 1, 2, 3")
            qrels.add(RelevanceJudgment(row["query_id"], row["document_id"], grade, source=source))
    return qrels
