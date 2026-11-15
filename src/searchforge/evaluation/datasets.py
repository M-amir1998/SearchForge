"""Query sets and qrels: import/export in JSONL, TREC TSV and Parquet; BEIR loading."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from searchforge.domain.judgment import (
    EvalQuery,
    Qrels,
    QueryProvenance,
    QuerySeverity,
    QuerySplit,
    RelevanceJudgment,
)
from searchforge.errors import EvaluationFailure

# ----------------------------------------------------------------------------- queries


def query_from_dict(row: dict[str, Any]) -> EvalQuery:
    try:
        return EvalQuery(
            query_id=str(row["query_id"]),
            text=str(row["text"]),
            query_type=str(row.get("query_type", "unspecified")),
            split=QuerySplit(row.get("split", "test")),
            severity=QuerySeverity(row.get("severity", "normal")),
            filters=dict(row.get("filters") or {}),
            provenance=QueryProvenance(row.get("provenance", "manual")),
            notes=str(row.get("notes", "")),
        )
    except (KeyError, ValueError) as exc:
        raise EvaluationFailure(f"invalid query row {row!r}: {exc}") from exc


def query_to_dict(q: EvalQuery) -> dict[str, Any]:
    return {
        "query_id": q.query_id,
        "text": q.text,
        "query_type": q.query_type,
        "split": q.split.value,
        "severity": q.severity.value,
        "filters": q.filters,
        "provenance": q.provenance.value,
        "notes": q.notes,
    }


def load_queries(path: Path) -> list[EvalQuery]:
    if not path.exists():
        raise EvaluationFailure(f"query set not found: {path}")
    out: list[EvalQuery] = []
    seen: set[str] = set()
    for row in _iter_jsonl(path):
        q = query_from_dict(row)
        if q.query_id in seen:
            raise EvaluationFailure(f"duplicate query id {q.query_id}")
        seen.add(q.query_id)
        out.append(q)
    return out


def save_queries(queries: Iterable[EvalQuery], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for q in queries:
            fh.write(json.dumps(query_to_dict(q), ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- qrels


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationFailure(f"{path.name}:{n}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise EvaluationFailure(f"{path.name}:{n}: expected an object")
                yield row


def load_qrels(path: Path) -> Qrels:
    """Load qrels from ``.jsonl``, ``.parquet`` or TREC-style ``.tsv``/``.txt``.

    TSV accepts both TREC (``qid 0 docid rel``) and BEIR (``query-id corpus-id score`` with header).
    """
    if not path.exists():
        raise EvaluationFailure(f"qrels not found: {path}")
    suffix = path.suffix.lower()
    judgments: list[RelevanceJudgment] = []
    if suffix == ".jsonl":
        for row in _iter_jsonl(path):
            judgments.append(
                RelevanceJudgment(
                    str(row["query_id"]),
                    str(row["document_id"]),
                    float(row["relevance"]),
                    str(row.get("source", "manual")),
                )
            )
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        for row in pq.read_table(path).to_pylist():
            judgments.append(
                RelevanceJudgment(
                    str(row["query_id"]),
                    str(row["document_id"]),
                    float(row["relevance"]),
                    str(row.get("source") or "manual"),
                )
            )
    elif suffix in (".tsv", ".txt"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts or parts[0] in ("query-id", "qid"):
                    continue
                if len(parts) == 4:
                    qid, _, did, rel = parts
                elif len(parts) == 3:
                    qid, did, rel = parts
                else:
                    raise EvaluationFailure(f"unrecognized qrels line: {line!r}")
                judgments.append(RelevanceJudgment(qid, did, float(rel)))
    else:
        raise EvaluationFailure(f"unsupported qrels format: {suffix}")
    return Qrels(judgments)


def save_qrels(qrels: Qrels, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"query_id": j.query_id, "document_id": j.document_id, "relevance": j.relevance, "source": j.source}
        for j in qrels.judgments()
    ]
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    elif suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), path)
    elif suffix in (".tsv", ".txt"):
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
            for r in rows:
                rel = r["relevance"]
                writer.writerow([r["query_id"], 0, r["document_id"], int(rel) if float(rel).is_integer() else rel])
    else:
        raise EvaluationFailure(f"unsupported qrels format: {suffix}")


def load_beir(directory: Path, split: str = "test") -> tuple[list[EvalQuery], Qrels]:
    """Load BEIR ``queries.jsonl`` + ``qrels/<split>.tsv`` (corpus is ingested via a JSONL source).

    Only queries that have judgments in the split are returned.
    """
    qrels = load_qrels(directory / "qrels" / f"{split}.tsv")
    judged = set(qrels.query_ids())
    queries = [
        EvalQuery(
            query_id=str(row["_id"]),
            text=str(row["text"]),
            query_type="beir",
            split=QuerySplit.TEST,
            provenance=QueryProvenance.PUBLIC_BENCHMARK,
        )
        for row in _iter_jsonl(directory / "queries.jsonl")
        if str(row["_id"]) in judged
    ]
    return queries, qrels
