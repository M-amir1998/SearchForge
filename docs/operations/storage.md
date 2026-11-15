# Storage and memory budgets

Safety rails that run before and during indexing, plus the commands that report and reclaim disk
space. Code: [storage/budget.py](../../src/searchforge/storage/budget.py),
[storage/manage.py](../../src/searchforge/storage/manage.py).

## Data profiles

`index.data_profile` (default `tiny`) states the intended corpus scale and caps the chunk count:

| Profile | Max chunks | Local NumPy dense backend |
|---|---|---|
| `tiny` | 5,000 | allowed |
| `small` | 50,000 | allowed |
| `research` | 1,000,000 | allowed |
| `large` | unlimited | refused |

- `large` is external-backend mode: with `dense.backend: numpy` the build fails with
  `StorageBudgetExceeded` (the check looks at the backend setting even if dense is disabled). Use
  `dense.backend: qdrant`. The chunk store and FTS5 index are still local SQLite files.
- The chunk limit is enforced after ingestion: a full build streams and chunks the whole corpus into
  the chunk store and FTS5 first, then fails if the count exceeds the cap (the partial directory is
  removed). An incremental update checks the resulting total.
- Raising the profile is a deliberate configuration change; it does not alter the build fingerprint.

## Disk estimation

`check_disk_budget` runs at the start of every full build (not for incremental updates). It estimates
the new version from the source size and configuration:

| Component | Heuristic |
|---|---|
| source bytes | total size of the configured source paths (symlinks not followed) |
| chunk text | source bytes x expansion, expansion = `1 + overlap_chars / max_chars` (`1 + overlap_tokens / max_tokens` for `fixed_token`) |
| estimated chunks | chunk text / `max(max_chars // 2, 64)` |
| chunk store | 1.3 x chunk text |
| FTS5 index | 1.5 x chunk text (if lexical is enabled) |
| dense vectors | estimated chunks x `dense.dimension` x 4 bytes (if dense is enabled) |
| embedding cache | 1.1 x dense vectors (if `cache_enabled`) |

The build fails with `StorageBudgetExceeded` when `free disk - estimated total < reserve`, or when the
estimated total exceeds `storage.max_index_gb`.

| Setting | Default |
|---|---|
| `storage.reserve_free_gb` | 5.0 |
| `storage.max_index_gb` | 20.0 |

Free space is measured on the filesystem holding the workspace. The estimate is a heuristic: source
bytes include markup (HTML tags, JSON keys, CSV columns) that never reaches chunk text, and the dense
term uses the configured `dense.dimension` even for providers whose real dimension differs (TF-IDF uses
its SVD component count, Sentence Transformers the model's dimension). The embedding cache term is
counted even when the vectors are already cached.

## storage status

```
searchforge storage status -c CONFIG [--results results]
searchforge storage status --workspace DIR
```

Reports, for the workspace: source data per source id (only with `-c`), each index name directory
(all versions, including partial builds), the embedding cache (`<workspace>/cache`), prepared corpus
exports (`<workspace>/prepared`, written by `searchforge corpus prepare`), partial builds, the results
directory and free disk.

## storage clean

```
searchforge storage clean -c CONFIG                              # dry run: list targets and sizes
searchforge storage clean -c CONFIG --yes                        # delete them
searchforge storage clean -c CONFIG --include-old-versions       # also list non-CURRENT versions
searchforge storage clean -c CONFIG --include-old-versions --yes
```

Dry run is the default. Targets, all inside the workspace:

| Target | Condition |
|---|---|
| `<workspace>/cache/` | exists (embedding cache; recomputable) |
| `<workspace>/prepared/` | exists (prepared exports; recomputable) |
| `<workspace>/indexes/*/*.partial/` | any partial build directory |
| `<workspace>/indexes/<name>/<version>/` | only with `--include-old-versions`, every complete version that is not `CURRENT` |

Every target is checked to resolve inside the workspace when the plan is made and again immediately
before deletion; read-only files of published versions are made writable for removal.

Cautions:

- A `.partial` directory of a build that is running right now is also a target. Do not clean while a
  build or update is in progress.
- If an index has no `CURRENT` pointer, `--include-old-versions` targets all of its versions.
- Old versions are rollback targets ([index lifecycle](index-lifecycle.md#rollback)).

**Never deleted:** source corpora, the results directory, configuration files, anything outside the
workspace, the `CURRENT` version of each index and the `CURRENT` pointer files. External Qdrant
collections and OpenSearch indexes are not touched either; remove them with their own tooling.

## Memory guidance for a 16 GB RAM machine

The profile caps are documented in code as local safety rails for a 16 GB RAM machine. Memory use is
driven by these terms:

| Term | Formula / behaviour |
|---|---|
| Dense vectors at search time | `chunks x dimension x 4` bytes, memory-mapped; the OS pages them in on demand |
| Dense vectors during a build or update | held in process memory and concatenated when saving, so plan for at least twice `chunks x dimension x 4` bytes |
| Row metadata | chunk id, document id and attribute dict per row, loaded from `rows.jsonl` into Python objects at open time |
| Filter masks | one byte per row per distinct filter, up to 64 memoized masks |
| TF-IDF fit | up to `fit_max_chunks` (200,000) chunk texts and their sparse matrix, `max_features` (100,000) terms |
| Search process | `max_workers` (4) worker threads plus the query cache (`max_bytes`, under-estimated) |

Measure rather than guess: every manifest records `build_seconds` and a sampled `peak_rss_mb`
(`searchforge index inspect NAME`), and benchmarks report `eval_peak_rss_mb` per system. Sampling can
miss short spikes, so treat these values as approximate. Prefer `tiny` or `small` for local iteration,
use `research` only with a measured build, and move to `large` with Qdrant when the dense vectors and
their row metadata no longer fit comfortably next to the rest of the workload. OpenSearch is the most
memory-hungry backend and runs only behind an explicit docker-compose profile.
