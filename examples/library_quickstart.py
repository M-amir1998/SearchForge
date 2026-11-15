"""Use SearchForge as a library: build the fixture index, search with explanations, evaluate.

Run:  uv run python examples/library_quickstart.py
CPU only, offline, no API keys. Writes indexes to ./.searchforge-examples.
"""

from __future__ import annotations

from searchforge.config.loader import load_config
from searchforge.datasets import fixture_config_path
from searchforge.domain.judgment import QuerySplit
from searchforge.evaluation.datasets import load_qrels, load_queries
from searchforge.evaluation.runner import evaluate, select_queries
from searchforge.indexes.builder import IndexBuilder
from searchforge.indexes.handle import open_index
from searchforge.service.search_service import SearchService


def main() -> None:
    loaded = load_config(fixture_config_path(), overrides={"workspace": ".searchforge-examples"})
    build = IndexBuilder(loaded).build()  # reused automatically when nothing changed
    print(f"index {build.manifest.index_id}: {build.manifest.chunk_count} chunks (reused={build.reused})")

    handle = open_index(loaded.workspace, loaded.config.index.name, loaded.config)
    service = SearchService(handle, loaded.config)
    try:
        for query in ("ERR-4012", "what happens when a client sends too many requests"):
            resp = service.search(query, profile="adaptive", top_k=3, explain=True)
            print(f"\n{query!r} -> planned profile {resp.profile} ({resp.latency_ms:.1f} ms)")
            for r in resp.results:
                sig = (r.explanation or {}).get("signals", {})
                print(
                    f"  {r.rank}. {r.document_id:<16} score={r.score:.4f} lexical_rank={sig.get('lexical_rank')} "
                    f"dense_rank={sig.get('dense_rank')}"
                )

        ev = loaded.config.evaluation
        assert ev is not None
        queries = select_queries(load_queries(loaded.resolve(ev.queries)), [QuerySplit.TEST])
        qrels = load_qrels(loaded.resolve(ev.qrels))
        for profile in ("lexical", "hybrid"):
            run = evaluate(service, queries, qrels, profile=profile)
            agg = run.aggregate()
            print(
                f"\n{profile:<8} nDCG@10={agg['ndcg@10']:.3f} Recall@20={agg['recall@20']:.3f} "
                f"MRR={agg['mrr']:.3f} p95={run.latency()['p95_ms']:.2f} ms  (fixture test split, local demo)"
            )
    finally:
        service.close()
        handle.close()


if __name__ == "__main__":
    main()
