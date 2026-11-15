# Chunking

Documents are split into chunks, the unit of indexing and retrieval. Chunkers are in
[chunkers.py](../../src/searchforge/chunking/chunkers.py); configuration is the top-level `chunking`
block (`ChunkingConfig` in [models.py](../../src/searchforge/config/models.py)).

| Key | Default | Used by |
|---|---|---|
| `strategy` | `section` | `fixed_token`, `sentence` or `section` |
| `max_chars` / `overlap_chars` | 1200 / 150 | sentence, section |
| `max_tokens` / `overlap_tokens` | 200 / 30 | fixed_token |
| `min_chars` | 80 | all |

Validation requires `overlap_chars < max_chars`, `overlap_tokens < max_tokens` and
`min_chars < max_chars`.

## Strategies

### Fixed approximate-token windows (`fixed_token`)

Tokens are whitespace-separated words (`\S+`). Windows contain `max_tokens` words and start every
`max_tokens - overlap_tokens` words; a window spans from its first word's start to its last word's
end, so internal whitespace and punctuation are preserved. The last window may be short. Whitespace
words approximate model tokens without a tokenizer; subword counts for English prose are typically
higher. Boundaries ignore sentences and headings.

### Sentence packing (`sentence`)

1. The text is split into sentence units: a boundary is `.`, `!` or `?` (optionally followed by
   closing quotes or brackets) plus whitespace, or one or more newlines. Units are whitespace-trimmed.
2. A unit longer than `max_chars` is split at the last space between `max_chars / 2` and
   `max_chars`, or cut hard if there is none.
3. Units are packed greedily: a window grows while its span stays within `max_chars`.
4. Overlap: the next window starts at the earliest trailing whole sentences of the previous window
   whose span is at most `overlap_chars`. If the last sentence alone is longer than that, there is
   no overlap. Each window advances by at least one unit.

### Section-aware (`section`)

Markdown ATX headings outside code fences delimit sections. Each section is labelled with its
heading path, for example `Deployment > Rollback`; text before the first heading is an unlabelled
section. A section that fits in `max_chars` becomes one chunk; a longer section is sentence-packed
(with overlap) inside the section. Chunks never cross a heading boundary, except through the
`min_chars` rule below. Non-Markdown records without headings behave like a single section.

## The `min_chars` merge rule

After a chunker produces spans, any span shorter than `min_chars` is merged into the previous span
(the merged span runs from the previous start to the later end and keeps the previous section
label). If the first span is short, it is merged with the second. A document consisting of a single
short span keeps it. Nothing is dropped.

`min_chars` takes precedence over the maximum: a merged chunk can exceed `max_chars` (or
`max_tokens`). This avoids tiny fragments such as a lone heading or a one-line trailing window, which
rank poorly and add index noise.

## Invariants

- **Exact offsets.** Every chunk is a contiguous span of the normalized document text, and
  `document.text[start_char:end_char] == chunk.text` holds for every chunk (property-tested).
- **Document ids do not depend on chunking.** A document id is derived only from source identity
  ([ids.py](../../src/searchforge/domain/ids.py)).
- **Chunk ids depend on the chunker.** `chunk_id = <document_id>#<hash>` where the hash covers the
  document id, the chunker fingerprint, the span and the text. The fingerprint hashes the strategy
  name and its parameters (`max_tokens`, `overlap_tokens`, `min_chars` for `fixed_token`;
  `max_chars`, `overlap_chars`, `min_chars` otherwise). Changing the chunker changes chunk ids but
  not document ids, which is why relevance judgments are kept per document.

## Citation reconstruction

Each chunk carries `ChunkMetadata`: source id and key, URI, title, section path, `start_char`,
`end_char`, timestamps and filterable attributes ([chunk.py](../../src/searchforge/domain/chunk.py)).
Because offsets are exact, a citation can be resolved to the precise passage of the source document.
The context builder ([builder.py](../../src/searchforge/context/builder.py)) assigns each selected
chunk a stable citation id derived from its chunk id and suppresses overlapping windows and
near-duplicates when assembling context. The title and section path are also indexed: in the BM25
`title` column and as a prefix of the text that is embedded.

## Trade-offs

**Small chunks**

- Precise matches; BM25 length normalization and proximity favour focused passages.
- Less context per result; facts split across chunks can make every individual chunk a weak match
  (the failure taxonomy has a `chunking_failure` category for this).
- More chunks and vectors: larger index, more embedding work, and more chunks needed to fill `k`
  distinct documents after the chunk-to-document collapse.

**Large chunks**

- More context per hit and fewer chunks.
- Embeddings average over more topics and BM25 normalizes by a longer length, so a relevant passage
  inside a long chunk can be diluted.

**Overlap**

- Improves boundary recall: a passage that straddles a boundary appears whole in at least one chunk.
- Costs redundancy: near-identical chunks of one document compete for top-k slots and enlarge the
  index. Document-level metrics, per-document caps and context de-duplication mitigate the effect on
  results but not the index cost.

## No universal optimum

The best strategy depends on document structure (headed Markdown versus flat text), query style and
the downstream consumer. SearchForge does not assume one: the fixture benchmark
([benchmark.yaml](../../src/searchforge/datasets/fixture/benchmark.yaml)) builds separate indexes and
evaluates the `hybrid` profile on each:

| System | Configuration |
|---|---|
| C1-fixed-64tok | `fixed_token`, 64 tokens, no overlap, `min_chars` 20 |
| C2-fixed-128tok-ov32 | `fixed_token`, 128 tokens, 32 overlap, `min_chars` 20 |
| C3-sentence-400 | `sentence`, 400 chars, 80 overlap, `min_chars` 40 |
| C4-section-1200 | `section`, 1200 chars, 150 overlap, `min_chars` 80 (the default index) |

Use the measured results for the target corpus, not a default, to choose.
