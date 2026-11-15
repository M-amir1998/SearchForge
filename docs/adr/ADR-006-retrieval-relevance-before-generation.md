# ADR-006: Measure retrieval relevance before generation

## Status

Accepted

## Context

In retrieval-augmented generation, answer quality depends on the evidence that is retrieved. If
retrieval and generation are evaluated together, it is unclear whether a bad answer came from
missing evidence or from the generator. LLM serving also brings GPU requirements, model downloads,
API keys and per-token cost. All of these conflict with the CPU-only, offline design (ADR-009).

## Decision

SearchForge stops at evidence. It retrieves, ranks and packages citable context. It does not
generate text and does not serve LLMs.

- **Evidence API.** `integrations/rag.py::SearchForgeRetriever` returns `Evidence` records. Each
  record has a stable `citation_id`, chunk id, document id, text, score, rank, title and URI. It
  can also return a `ContextPackage`. No LLM or agent framework is imported.
- **Context packages.** `context/builder.py::CitationContextBuilder` assembles results within a
  character or estimated-token budget. It suppresses overlapping windows and near-duplicates
  (shingle Jaccard), caps chunks per document, and gives each item a citation id that maps back
  to exactly one chunk (`domain/ids.py::make_citation_id`).
- **Citation checks.** `integrations/reliability_lab.py::citation_validity` verifies that every
  citation in a package resolves to a chunk in the chunk store with matching text. The
  machine-readable evaluation export is validated against a JSON Schema.
- **Agent tool.** `integrations/agent_tool.py` exposes a JSON-Schema tool definition for
  `search(query, filters, top_k)`. Inputs go through the same typed validation as the API.
- **Retrieval evaluation first.** Quality is measured with IR metrics (nDCG, Recall, MRR, MAP,
  precision, hit rate) on document-level rankings against qrels, with per-query regression gates.
  These measure relevance of the evidence, not answer quality.
- **LLMs are optional collaborators only.** `query/planning.py::LLMQueryPlanner` accepts an
  injected `complete(prompt) -> str` callable. Its output is untrusted: it must name an existing,
  non-adaptive profile that passes validation. Otherwise the static plan is used and the fallback
  is recorded.

## Consequences

Positive:

- Retrieval quality can be measured, regressed and gated without a generator in the loop, at zero
  inference cost.
- Downstream systems can verify that generated answers cite real, retrievable chunks.
- The core package has no GPU, model-serving or API-key requirements.

Negative:

- SearchForge cannot say whether better retrieval produces better answers. End-to-end answer
  evaluation has to happen in the consuming system.
- Token budgets are estimates (`ceil(chars / chars_per_token)`) unless the caller injects a
  model-specific `token_counter`.
- Integrations have to adapt `Evidence` or `ContextPackage` to their own framework types.

## Alternatives considered

- **Built-in generation (local LLM or hosted API).** Rejected: it brings GPU or API-key
  requirements, non-deterministic outputs in CI, and evaluation that mixes retrieval with
  generation.
- **LLM-as-judge relevance labelling.** Not adopted in core. Relevance comes from curated qrels,
  public benchmarks, blinded human review (`evaluation/review.py`) and incident-derived cases.
  Click logs only produce weak, debiased label candidates.
- **Framework-specific retriever classes (LangChain, LlamaIndex).** Rejected for the core package.
  Thin adapters can live in consuming projects.
