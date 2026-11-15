# Current search/retrieval stack: reference notes (verified November 2026)

These notes were compiled on 2026-11-15 from official docs, PyPI JSON, GitHub READMEs/releases and
library source. Every claim has a source URL next to it. Anything we could not confirm from a fetched
source is marked **unverified**. Where a GitHub release page showed only a day and month, the year is
inferred from the neighbouring releases and marked "(year inferred)".

| Component | Latest version seen | License | Source |
|---|---|---|---|
| `qdrant-client` | 1.20.0 (22 Oct, year inferred 2026), Python >=3.10 | Apache-2.0 | https://pypi.org/pypi/qdrant-client/json, https://github.com/qdrant/qdrant-client/releases |
| Qdrant server | version not captured (**unverified**) | Apache-2.0 | https://github.com/qdrant/qdrant |
| `opensearch-py` | 3.2.0 (2026-04-27), Python >=3.10,<4 | Apache-2.0 | https://pypi.org/pypi/opensearch-py/json |
| OpenSearch server | 3.9.0 (29 Sep, year inferred 2026) | Apache-2.0 | https://github.com/opensearch-project/OpenSearch/releases |
| Elasticsearch | 9.5.6 marked "Latest" (2 Nov, year inferred 2026) | AGPLv3 / SSPL / ELv2 (source); binaries ELv2 | https://github.com/elastic/elasticsearch/releases |
| pgvector (ext) | 0.8.7 | PostgreSQL License | https://github.com/pgvector/pgvector |
| `pgvector` (Python) | 0.5.1, Python >=3.10 | MIT | https://pypi.org/pypi/pgvector/json |
| `sentence-transformers` | 6.3.0 (11 Nov, year inferred 2026), Python >=3.10 | Apache-2.0 | https://pypi.org/pypi/sentence-transformers/json, https://github.com/UKPLab/sentence-transformers/releases |
| `bm25s` | 0.3.16, Python >=3.8 | MIT | https://pypi.org/pypi/bm25s/json |
| `rank-bm25` | 0.2.2 (2022-02-16) | Apache-2.0 | https://pypi.org/pypi/rank-bm25/json |

---

## 1. Qdrant + `qdrant-client`

**Hybrid/fusion.** The Query API (`query_points`) takes nested `Prefetch` sub-requests and fuses them
server-side (https://qdrant.tech/documentation/concepts/hybrid-queries/):
- RRF: score = Σ 1/(k + rank), zero-based ranks. Use `models.RrfQuery(rrf=models.Rrf(k=..., weights=[...]))`
  (k since server v1.16, weights since v1.17). If `k` is not set the docs say it defaults to **2**, not 60,
  so set `k=60` explicitly to match the literature. `models.FusionQuery(fusion=models.Fusion.RRF)` still exists in
  the client models (`Fusion` enum = `rrf`, `dbsf`), but the docs now show `RrfQuery`
  (source: `qdrant_client/http/models/models.py` on master).
- DBSF (server v1.11+): normalises each prefetch with mean ± 3σ, then sums. Use `models.FusionQuery(fusion=models.Fusion.DBSF)`.
- Query API with prefetch/fusion has been available since server v1.10.
- Client 1.20.0 adds server-side BM25 via `TextQuery` on text-indexed payload fields. Local mode supports
  `TextQuery` only with default tokenization (release notes).
- Client 1.19.0 deprecated `search`, `search_batch`, `search_groups` and `recommend*` in favour of
  `query_points` / `query_batch_points` / `query_points_groups`. It removed the FastEmbed `add`/`query` helpers.

**Filtering.** `Filter(must=[...], should=[...], must_not=[...])` with `FieldCondition(key=..., match=MatchValue(value=...))`,
`MatchAny(any=[...])`, `Range(gte=..., lte=...)` and `DatetimeRange(gt=..., lte=...)` (RFC 3339 strings or `datetime`).
`Prefetch` has its own `filter=` field. The docs recommend creating payload indexes on filtered fields,
ideally before ingest (https://qdrant.tech/documentation/concepts/filtering/).

**Point IDs.** IDs must be "64-bit unsigned integers" or UUIDs (https://qdrant.tech/documentation/manage-data/points/).
Arbitrary string doc IDs (for example BEIR `_id` values) need a deterministic mapping such as `uuid.uuid5(NAMESPACE, doc_id)`,
with the original ID stored in the payload.

**Local deployment.** `QdrantClient(":memory:")` or `QdrantClient(path="path/to/db")` runs without a server
(https://github.com/qdrant/qdrant-client). The local implementation warns that it "is not recommended for collections
with more than 20,000 points" (`LARGE_DATA_THRESHOLD = 20_000` in `qdrant_client/local/local_collection.py`). Only one
client can open a `path=` folder at a time, enforced by a portalocker lock. In local mode `search_params` is ignored
with a warning (1.19.0 release notes). To run the server: `docker run -p 6333:6333 qdrant/qdrant`, which has no auth by default.

**CPU feasibility (16 GB, no GPU).** Good. Local mode needs no extra services for small corpora such as SciFact (5K docs)
or NFCorpus (3.6K docs). For FiQA (57K docs), use the Docker server. Resource figures for the server are **unverified**.

**API shapes (client 1.20.0; signatures checked against `qdrant_client/qdrant_client.py`):**
```python
from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")            # or path="./qdrant_db", or url="http://localhost:6333"

client.create_collection(
    collection_name="docs",
    vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
    # sparse_vectors_config={"sparse": models.SparseVectorParams()},  # optional
)

client.upsert(
    collection_name="docs",
    points=[
        models.PointStruct(
            id="5c56c793-69f3-4fbf-87e6-c4bf54c28c26",   # uint or UUID only
            vector={"dense": [0.1] * 384},
            payload={"doc_id": "d1", "lang": "en", "year": 2024},
        )
    ],
)

res = client.query_points(
    collection_name="docs",
    query=[0.1] * 384,
    using="dense",
    query_filter=models.Filter(
        must=[models.FieldCondition(key="lang", match=models.MatchValue(value="en"))]
    ),
    limit=10,
    with_payload=True,
)
for p in res.points:            # QueryResponse.points -> ScoredPoint(id, score, payload, ...)
    print(p.id, p.score, p.payload)

client.delete(
    collection_name="docs",
    points_selector=models.PointIdsList(points=["5c56c793-69f3-4fbf-87e6-c4bf54c28c26"]),
)

hybrid = client.query_points(
    collection_name="docs",
    prefetch=[
        models.Prefetch(query=models.SparseVector(indices=[1, 42], values=[0.22, 0.8]), using="sparse", limit=20),
        models.Prefetch(query=[0.1] * 384, using="dense", limit=20,
                        filter=models.Filter(must=[models.FieldCondition(
                            key="year", range=models.Range(gte=2020))])),
    ],
    query=models.RrfQuery(rrf=models.Rrf(k=60)),          # or models.FusionQuery(fusion=models.Fusion.RRF)
    limit=10,
)
```
Docs examples cover named-vector upsert, `PointIdsList`, `query_points` with `query_filter`, `search_params`
and `with_payload`, and prefetch + `RrfQuery`. The `filter=` field inside `Prefetch` comes from the model source.
The page examples did not show it.

---

## 2. OpenSearch + `opensearch-py`

**Lexical.** BM25 is the default similarity. We did not fetch a page stating this for 3.x, so it is **unverified** here,
although it is long-standing Lucene behaviour.

**k-NN plugin** (https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/):
- Engines: **Faiss (default)**, Lucene, and NMSLIB (deprecated legacy). JVector is available as a separate plugin.
- HNSW defaults: `m=16` and `ef_construction=100`. Faiss `ef_search` defaults to 100. Lucene uses k as its ef_search.
- Space types: `l2` (default), `cosinesimil` (Faiss since 2.19), `innerproduct` and `hamming`.
- Faiss also supports IVF (needs training). Lucene supports HNSW and, since 3.6, flat.
- A `knn_vector` field takes top-level `mode` (`in_memory` | `on_disk`) and `compression_level` (`1x`..`32x`)
  (https://docs.opensearch.org/latest/mappings/supported-field-types/knn-vector/).
- k-NN query: exactly one of `k`, `max_distance` or `min_score`. Optional `filter` and `method_parameters` (`ef_search`).
  A filter "can only be used with the `faiss` or `lucene` engines" (https://docs.opensearch.org/latest/query-dsl/specialized/k-nn/index/).

**Hybrid + fusion.**
- The `hybrid` compound query (https://docs.opensearch.org/latest/query-dsl/compound/hybrid/) takes at most 5 sub-queries,
  a top-level `filter` applied to all sub-queries, and `pagination_depth` (required when `from > 0`). It must run with a search pipeline.
- `normalization-processor` (since 2.10, https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/):
  - Normalization: `min_max` (default), `l2` or `z_score` (`z_score` works only with `arithmetic_mean`).
  - Combination: `arithmetic_mean` (default), `geometric_mean` or `harmonic_mean`.
  - Optional `parameters.weights`: one weight per query, each in [0, 1], summing to 1.0.
- `score-ranker-processor` (since 2.19, https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/):
  `combination.technique: "rrf"` with `rank_constant` and optional `weights`, which must also sum to 1.0.

```json
PUT /_search/pipeline/rrf-pipeline
{"phase_results_processors": [{"score-ranker-processor": {
  "combination": {"technique": "rrf", "rank_constant": 60}}}]}

PUT /_search/pipeline/minmax-pipeline
{"phase_results_processors": [{"normalization-processor": {
  "normalization": {"technique": "min_max"},
  "combination": {"technique": "arithmetic_mean", "parameters": {"weights": [0.3, 0.7]}}}}]}

GET /docs/_search?search_pipeline=rrf-pipeline
{"query": {"hybrid": {
  "queries": [{"match": {"text": "query terms"}},
              {"knn": {"emb": {"vector": [0.1, 0.2], "k": 50}}}],
  "filter": {"term": {"lang": "en"}}}}}
```

**Local deployment and memory.**
- Docker single node: `docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=<pw>" opensearchproject/opensearch:latest`.
  A strong admin password is mandatory since 2.12.
- For local development, `DISABLE_SECURITY_PLUGIN=true` and `DISABLE_INSTALL_DEMO_CONFIG=true` turn off security.
- Linux hosts need `vm.max_map_count=262144`. Docker Desktop should have at least 4 GB of memory
  (https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/).
- The default heap is `-Xms1g -Xmx1g`, and the docs recommend half of system RAM. The compose sample uses
  `OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m` (https://docs.opensearch.org/latest/install-and-configure/install-opensearch/index/).
- Faiss/NMSLIB graphs live in native (off-heap) memory, so heap is not the whole budget. The circuit-breaker setting name and
  default were not captured (**unverified**).
- On a 16 GB laptop, a 1–2 GB heap plus native k-NN memory fits comfortably for BEIR-small corpora. Still, each run costs a
  JVM, Docker, and a few GB of RAM.

**`opensearch-py` 3.2.0 shapes** (https://github.com/opensearch-project/opensearch-py/blob/main/USER_GUIDE.md,
`guides/bulk.md`, `samples/knn/knn_basics.py`). 3.1.0 raised the minimum Python to 3.10.
```python
from opensearchpy import OpenSearch, helpers

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}],
                    http_auth=("admin", "<pw>"), use_ssl=True, verify_certs=False)

client.indices.create(index="docs", body={
    "settings": {"index.knn": True},
    "mappings": {"properties": {
        "text": {"type": "text"},
        "lang": {"type": "keyword"},
        "emb": {"type": "knn_vector", "dimension": 384, "space_type": "cosinesimil",
                "method": {"name": "hnsw", "engine": "faiss",
                           "parameters": {"m": 16, "ef_construction": 100}}}}}})

helpers.bulk(client, [{"_index": "docs", "_id": "d1", "text": "...", "lang": "en", "emb": [0.1] * 384}])
client.indices.refresh(index="docs")

bm25 = client.search(index="docs", body={"size": 10, "query": {"bool": {
    "must": [{"match": {"text": "query terms"}}],
    "filter": [{"term": {"lang": "en"}}]}}})

knn = client.search(index="docs", body={"size": 10, "query": {"knn": {"emb": {
    "vector": [0.1] * 384, "k": 10, "filter": {"term": {"lang": "en"}}}}}})
hits = knn["hits"]["hits"]          # each hit: _id, _score, _source
```
`client.bulk(body=[{"index": {...}}, {...}, ...])` (an action/document list or NDJSON string) also works. The `bool` + `filter`
query shape is standard Query DSL and was not re-fetched from a 3.x page (**unverified** for 3.x specifically).

---

## 3. Elasticsearch

- **Retrievers / RRF** (https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever):
  `retriever.rrf` takes `retrievers: [{standard: {query}}, {knn: {field, query_vector, k, num_candidates}}]`,
  `rank_constant` (default 60, at least 1), `rank_window_size` (default 10), a `filter` applied to all children,
  per-retriever `weight` (GA 9.2) and a `query`/`fields` shorthand (GA 9.1).
- **dense_vector** (https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector):
  - `dims` up to 4096. `similarity` defaults to `cosine`.
  - `index_options.type` can be `hnsw`, `int8_hnsw`, `bbq_hnsw` and others, with `m=16` and `ef_construction=100` defaults.
  - The default quantization varies by version: `int8_hnsw` (9.0); `bbq_hnsw` for 384 or more dims (9.1+); `bbq_disk` in 9.4+ "when your license allows it".
- **Licensing caveats.**
  - Source is available under AGPLv3, SSPL 1.0 or ELv2 (AGPL option added Aug/Sep 2024). Releases ship under the Elastic License
    (https://www.elastic.co/pricing/faq/licensing).
  - An Elastic staff member said on the forum that the "RRF and linear retrievers are under Enterprise license". Basic clusters
    return a 403 "current license is non-compliant for [linear retriever]"
    (https://discuss.elastic.co/t/use-of-retrievers-in-the-search-api/379145).
  - The official subscriptions matrix could not be read in tier-resolved form, so the RRF tier is confirmed only by that forum
    reply (**unverified** against the matrix).
  - That dense_vector/kNN works on the free Basic tier is **unverified**: only third-party sources say so.
- **Implication.** Client-side RRF over separate `match` and `knn` requests avoids the Enterprise gate on a Basic cluster.

---

## 4. pgvector (PostgreSQL)

Source: https://github.com/pgvector/pgvector (v0.8.7, PostgreSQL License).

- **Distance operators:** `<->` L2, `<#>` negative inner product, `<=>` cosine distance, `<+>` L1, `<~>` Hamming, `<%>` Jaccard.
- **Indexes:**
  - `CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);` with `SET hnsw.ef_search = 100;` (default 40).
  - `... USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);`
  - Indexable dimensions: `vector` 2,000, `halfvec` 4,000, `bit` 64,000.
- **Filtering:** approximate indexes apply the filter after the scan, which can return fewer rows than requested. Since 0.8.0,
  `SET hnsw.iterative_scan = strict_order | relaxed_order;` keeps scanning, bounded by `hnsw.max_scan_tuples` (default 20,000).
  The README suggests B-tree indexes on filter columns, partial indexes, or partitioning.
- **Hybrid:** the README recommends combining with Postgres full-text search and merging via RRF or a cross-encoder. The official
  example (https://github.com/pgvector/pgvector-python/blob/master/examples/hybrid_search/rrf.py) builds a GIN full-text index,
  ranks with `ts_rank_cd` and fuses in SQL with k = 60.
- **Python** (https://github.com/pgvector/pgvector-python, `pgvector` 0.5.1, MIT):
```python
import numpy as np, psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect(dbname="sf", autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)
conn.execute("CREATE TABLE items (id text PRIMARY KEY, body text, emb vector(384), "
             "tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED)")
conn.execute("INSERT INTO items (id, body, emb) VALUES (%s, %s, %s)", ("d1", "text", np.random.rand(384)))
dense = conn.execute("SELECT id FROM items ORDER BY emb <=> %s LIMIT 10", (qvec,)).fetchall()
lex = conn.execute("SELECT id, ts_rank_cd(tsv, q) AS r FROM items, plainto_tsquery('english', %s) q "
                   "WHERE tsv @@ q ORDER BY r DESC LIMIT 10", (qtext,)).fetchall()
```
  The README's psycopg example uses `<->` and passes `Vector(np.array(...))` or a numpy array after `register_vector`. The `<=>` and
  `tsvector` lines above combine documented pieces. `ts_rank`/`ts_rank_cd` are not true BM25 (they ignore IDF), which matters when
  comparing results across adapters (**unverified**: this comes from PostgreSQL docs knowledge and was not fetched here).
- **CPU feasibility:** good. Postgres runs fine on a laptop, but it is an extra service to install and run.

---

## 5. Sentence Transformers

Sources: https://pypi.org/pypi/sentence-transformers/json and https://github.com/UKPLab/sentence-transformers/releases.

- **Version:** current major is **v6** (6.0.0 in Aug, 6.3.0 on 11 Nov, years inferred 2026).
- **v6 changes:** adds `MultiVectorEncoder` (ColBERT-style), moves to transformers v5 and PyTorch 2.2+, and requires
  `trust_remote_code=True` for local custom code.
- **Backends:** `torch`, `onnx` and `openvino`.
- **API reference:** https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html and
  https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu", revision="<commit-sha>",
                        local_files_only=False, backend="torch")      # cache_folder= / SENTENCE_TRANSFORMERS_HOME
E = m.encode(docs, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
q = m.encode_query(["query"], normalize_embeddings=True)   # uses the model's "query" prompt if defined
dim = m.get_embedding_dimension()                          # name documented in v6 reference

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu", revision="<sha>", max_length=512)
scores = ce.predict([(query, d) for d in docs], batch_size=32)    # np.ndarray of logits
ranked = ce.rank(query, docs, top_k=10, return_documents=False)   # [{'corpus_id': int, 'score': float}, ...] sorted
```
- `encode()` defaults: `batch_size=32`, `normalize_embeddings=False`, `convert_to_numpy=True` and `precision="float32"`.
  It also takes `truncate_dim`.
- `encode_query()` and `encode_document()` add the model's `query` or `document`/`passage`/`corpus` prompt when no prompt is given.
- The v6 reference documents only `get_embedding_dimension()`. Whether `get_sentence_embedding_dimension()` still exists as an alias
  is **unverified**.

**Small CPU-friendly models**

| Model | Dim | Params | Max len | License | Notes |
|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 22.7M | 256 | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| BAAI/bge-small-en-v1.5 | 384 | 33.4M | 512 | MIT | Optional query prefix "Represent this sentence for searching relevant passages: " https://huggingface.co/BAAI/bge-small-en-v1.5 |
| intfloat/e5-small-v2 | 384 | 33.4M (12 layers) | 512 | MIT | **Requires** "query: " / "passage: " prefixes https://huggingface.co/intfloat/e5-small-v2 |
| cross-encoder/ms-marco-MiniLM-L6-v2 | n/a | 22.7M | n/a | Apache-2.0 | 74.30 NDCG@10 (TREC DL19), 1,800 docs/s on a V100 GPU https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 |

CPU feasibility: all four are 22–34M-parameter BERT-small models. Encoding a SciFact/NFCorpus-sized corpus (about 5K docs) on a
laptop CPU is practical, and FiQA (57K) is practical but slower. Reranking top-100 per query with MiniLM-L6 is feasible for a few
hundred queries. We have no fetched CPU throughput figures, so treat these estimates as **unverified** and benchmark locally.

**Offline mode** (https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables):
- `HF_HUB_OFFLINE=1` stops all HTTP calls to the Hub. Only cached files are used, and an error is raised if a file is not cached.
  It also skips the "is there a newer version" check, so loading is faster.
- The variable is read at import time of `huggingface_hub`, so set it before importing.
- Cache locations are controlled by `HF_HOME` (default `~/.cache/huggingface`) and `HF_HUB_CACHE` (default `$HF_HOME/hub`).
- On Windows without developer mode, symlinks are not created and files get duplicated (`HF_HUB_DISABLE_SYMLINKS_WARNING`).
- Pinning `revision=` to a commit SHA plus `HF_HUB_OFFLINE=1` gives reproducible runs.

---

## 6. BM25 implementations

**SQLite FTS5** (https://www.sqlite.org/fts5.html):
- `bm25()`: "The better the match, the numerically smaller the value returned". FTS5 multiplies BM25 by −1, so sort ascending
  (`ORDER BY rank` or `ORDER BY bm25(ft)`) and negate to get a higher-is-better score.
- `k1 = 1.2` and `b = 0.75` are hard-coded.
- Column weights are passed positionally: `bm25(email, 10.0, 5.0)`. Missing weights default to 1.0. Weights can be set per query
  with `rank MATCH 'bm25(10.0, 5.0)'` or persistently with `INSERT INTO ft(ft, rank) VALUES('rank', 'bm25(10.0, 5.0)')`.
  Ordering by the hidden `rank` column is faster than calling `bm25()`.
- Tokenizers:
  - `unicode61` is the default; `remove_diacritics` defaults to 1, and 2 is more correct.
  - `porter` is a wrapper: `tokenize = 'porter unicode61 remove_diacritics 1'`.
  - `ascii` and `trigram` (substring matching) are also available.
- External-content tables (`content=t1, content_rowid=d`) must be kept in sync by the application, for example with triggers.
- FTS5 query syntax treats characters such as `"`, `-`, `*`, `:`, `(`, `)`, `AND`, `OR`, `NOT` and `NEAR` specially, so raw user or
  BEIR query text must be escaped or tokenised and quoted. This is from FTS5 docs knowledge; the query-syntax section was not
  captured in this fetch (**unverified**).
```sql
CREATE VIRTUAL TABLE docs_fts USING fts5(title, body, doc_id UNINDEXED,
                                         tokenize = 'porter unicode61 remove_diacritics 2');
SELECT doc_id, -bm25(docs_fts, 0.5, 1.0) AS score FROM docs_fts
WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts, 0.5, 1.0) LIMIT 100;
```

**`bm25s`** (https://github.com/xhluca/bm25s, 0.3.16, MIT):
- Pure NumPy/SciPy sparse BM25. Variants: `method="lucene"` (default), `robertson`, `atire`, `bm25l` and `bm25+`.
- Optional PyStemmer stemming. `save`/`load` persist the index.
- Release cadence and dates were not captured, but the project looks active (**unverified**).
```python
import bm25s, Stemmer
stem = Stemmer.Stemmer("english")
r = bm25s.BM25(); r.index(bm25s.tokenize(corpus, stopwords="en", stemmer=stem))
ids, scores = r.retrieve(bm25s.tokenize(queries, stemmer=stem), k=10)
```

**`rank_bm25`** (https://github.com/dorianbrown/rank_bm25, 0.2.2, Apache-2.0):
- The last PyPI release was 2022-02-16, with 17 open issues and 17 open PRs. Treat it as unmaintained.
- `BM25Okapi(tokenized_corpus).get_scores(q)` scores every document per query, which is slow for corpora of 50K or more documents.
- It is useful only as a cross-check.

---

## 7. Modern rerankers (CPU feasibility + licensing)

| Model | Size | License | CPU note |
|---|---|---|---|
| BAAI/bge-reranker-base | ~0.3B (XLM-R family) | MIT | Feasible for small top-k on CPU, about 10x MiniLM-L6 cost (estimate, **unverified**). https://huggingface.co/BAAI/bge-reranker-base |
| mixedbread-ai/mxbai-rerank-xsmall-v1 | ~70.8M | Apache-2.0 | Works with `CrossEncoder(...).rank(...)` and is CPU-friendly. Base/large v1 also exist. A v2 family was not confirmed (**unverified**). https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1 |
| colbert-ir/colbertv2.0 | ~0.1B (BERT) | MIT | The repo says "a GPU is required for training and indexing". CPU search is possible. Sentence Transformers v6 `MultiVectorEncoder` is an alternative ColBERT-style path whose CPU indexing cost we did not measure. https://huggingface.co/colbert-ir/colbertv2.0 |
| bge-reranker-v2-m3 | not fetched | **unverified** | — |

---

## 8. Public relevance datasets (BEIR)

Sources: https://github.com/beir-cellar/beir (code Apache-2.0) and https://github.com/beir-cellar/beir/wiki/Load-your-custom-dataset.
- Each zip contains `corpus.jsonl` (`{"_id", "title", "text"}`), `queries.jsonl` (`{"_id", "text"}`) and `qrels/<split>.tsv`.
- The qrels TSV has a header row `query-id	corpus-id	score`, confirmed at https://huggingface.co/datasets/BeIR/scifact-qrels.
- The README says users must check each dataset's own license themselves.

| Dataset | Test queries | Corpus | Rel/query | Download (md5) | License (upstream) |
|---|---|---|---|---|---|
| SciFact | 300 (339 test qrels rows) | 5,183 | 1.1 | `.../BEIR/datasets/scifact.zip` (`5f7d1de60b170fc8027bb7898e2efca1`) | Claims CC BY 4.0. Abstracts (corpus) ODC-By 1.0 (https://github.com/allenai/scifact/blob/master/LICENSE.md) |
| NFCorpus | 323 | 3,633 | 38.2 (graded) | `.../nfcorpus.zip` (`a89dba18a62ef92f7d323ec890a0d38d`) | "free to use for academic purposes". Other uses need permission (https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) |
| FiQA-2018 | 648 | 57,638 | 2.6 | `.../fiqa.zip` (`17918ed23cd04fb15047f73e6c3bd9d9`) | Train/test data "available only for non-commercial use" (https://sites.google.com/view/fiqa/) |
| ArguAna | 1,406 | 8.67K | 1.0 | `.../arguana.zip` | not checked |
| SCIDOCS | 1,000 | 25K | 4.9 | `.../scidocs.zip` | not checked |

- The base URL is `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/`.
- The HF mirrors (`BeIR/scifact`, `BeIR/nfcorpus`, `BeIR/fiqa`) are tagged cc-by-sa-4.0. That tag conflicts with the upstream
  terms, so SearchForge should record the **upstream** license and not redistribute data. Download it at run time and verify the md5.

---

## Implications for SearchForge

1. **Local default = SQLite FTS5 + NumPy dense.**
   - FTS5 ships with Python's `sqlite3`. Whether it is compiled in depends on the build, so check at startup with
     `pragma compile_options` or a test `CREATE VIRTUAL TABLE` (**unverified** for every platform).
   - FTS5 gives a real BM25 (fixed k1/b, column weights) with zero services and a single-file index.
   - Brute-force cosine over L2-normalised float32 vectors in NumPy is exact and fast enough for BEIR-small corpora:
     5K x 384 floats is about 7.5 MB, and FiQA's 57K x 384 is about 88 MB.
   - Exact search also removes ANN recall as a confounder in evaluation.
   - Normalise FTS5 scores (`-bm25()`) before any score-based fusion. RRF only needs ranks.
2. **Implement fusion client-side, engine-independently.**
   - RRF with k=60 by default. Note that Qdrant's default is k=2 and OpenSearch uses `rank_constant`.
   - Min-max and z-score normalisation with weighted sum, matching OpenSearch's normalization processor.
   - Elasticsearch RRF is Enterprise-gated, so client-side fusion is the portable path.
   - Engine-native fusion can be an optional adapter feature, compared against the client-side result.
3. **First external adapter: Qdrant.**
   - Apache-2.0 everywhere. `pip install qdrant-client` gives an in-process `:memory:`/`path=` mode, so CI needs no Docker.
     This holds up to the 20K-point local-mode warning; use Docker for FiQA.
   - The same API works against a server. Native prefetch + RRF/DBSF, payload filters and server-side BM25 `TextQuery` exist in 1.20.
   - Requirements: map string doc IDs to UUIDv5, pin `qdrant-client>=1.19` (the `search` deprecations), and use `query_points` only.
4. **Second adapter: OpenSearch** (Apache-2.0, full BM25 + k-NN + native hybrid/RRF).
   - It needs Docker, a JVM heap of 1–2 GB, `vm.max_map_count` and an admin password or disabled security. That makes it an
     integration-test target, not a unit-test one.
   - Use the Faiss or Lucene engine for filtered k-NN, and `cosinesimil` on 2.19+.
5. **pgvector** is a reasonable third adapter. Its full-text ranking is not BM25, so document that the lexical leg differs.
   **Elasticsearch** comes last because of license gating.
6. **Models.** Default to `bge-small-en-v1.5` or `all-MiniLM-L6-v2` (384-d, permissive) with `device="cpu"`, `revision=<sha>` and
   `normalize_embeddings=True`. Use `cross-encoder/ms-marco-MiniLM-L6-v2` as the optional reranker. Support `HF_HUB_OFFLINE=1`
   for reproducible offline runs. e5 needs prefixes, so make the query/passage prefix a model-config field.
7. **First study datasets.**
   - SciFact (5K docs, permissive CC BY / ODC-By) is the cleanest first benchmark.
   - NFCorpus (graded qrels) and FiQA (57K, non-commercial) are good second and third benchmarks.
   - Never vendor the data into the repo.
