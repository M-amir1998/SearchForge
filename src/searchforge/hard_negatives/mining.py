"""Hard-negative mining: high-ranked results that are not relevant.

For each query, documents ranked within ``max_rank`` that are not judged relevant
become hard-negative candidates for future retriever/reranker training. Each row
records which retrievers surfaced the chunk, so negatives can be targeted (e.g.
"dense-only hard negatives"). ``judged`` distinguishes an explicit 0 judgment
(a confirmed negative) from an unjudged document (a *probable* negative that may
actually be relevant — review before training).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from searchforge.domain.judgment import Qrels
from searchforge.evaluation.runner import RunResult


@dataclass(frozen=True, slots=True)
class HardNegative:
    query_id: str
    document_id: str
    chunk_id: str
    rank: int
    score: float
    source_retrievers: str
    judged: bool
    run_name: str


def mine_hard_negatives(
    run: RunResult, qrels: Qrels, *, max_rank: int = 10, threshold: float = 1.0, per_query: int = 5
) -> list[HardNegative]:
    out: list[HardNegative] = []
    for o in run.scored:
        judgments = qrels.for_query(o.query.query_id)
        seen: set[str] = set()
        taken = 0
        for chunk_id, doc_id, score, sources in o.chunk_ranking:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank = len(seen)  # document-level rank (chunks collapsed, as in evaluation)
            if rank > max_rank:
                break
            if judgments.get(doc_id, 0.0) >= threshold:
                continue
            out.append(
                HardNegative(
                    query_id=o.query.query_id,
                    document_id=doc_id,
                    chunk_id=chunk_id,
                    rank=rank,
                    score=float(score),
                    source_retrievers="+".join(sources),
                    judged=doc_id in judgments,  # judged below the relevance threshold
                    run_name=run.name,
                )
            )
            taken += 1
            if taken >= per_query:
                break
    return out


def write_hard_negatives(rows: list[HardNegative], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]), path)
        return
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
