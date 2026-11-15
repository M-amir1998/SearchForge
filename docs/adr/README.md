# Architecture Decision Records

Each record describes one significant design decision: its context, the decision, its positive and
negative consequences, and the alternatives that were rejected. Records are immutable once
accepted. A changed decision gets a new ADR that supersedes the old one, and the old record's
status is updated to point to it.

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](ADR-001-engine-independent-retrieval.md) | Engine-independent retrieval through ports and adapters | Accepted |
| [ADR-002](ADR-002-sqlite-fts5-local-lexical-backend.md) | SQLite FTS5 as the local lexical backend | Accepted |
| [ADR-003](ADR-003-separate-embedding-provider-from-vector-index.md) | Separate the embedding provider from the vector index | Accepted |
| [ADR-004](ADR-004-rrf-default-hybrid-fusion.md) | Reciprocal Rank Fusion as the default hybrid fusion | Accepted |
| [ADR-005](ADR-005-multi-stage-retrieval-and-reranking.md) | Multi-stage retrieval and reranking | Accepted |
| [ADR-006](ADR-006-retrieval-relevance-before-generation.md) | Measure retrieval relevance before generation | Accepted |
| [ADR-007](ADR-007-immutable-index-manifests.md) | Immutable, versioned indexes with manifests | Accepted |
| [ADR-008](ADR-008-jenkins-instead-of-github-actions.md) | Jenkins instead of GitHub Actions | Accepted |
| [ADR-009](ADR-009-cpu-first-local-architecture.md) | CPU-first, local architecture | Accepted |
| [ADR-010](ADR-010-raw-queries-not-persisted-by-default.md) | Raw queries are not persisted by default | Accepted |

## Template

```markdown
# ADR-NNN: Title

## Status
Proposed | Accepted | Superseded by ADR-MMM

## Context
The forces at play and the problem being solved.

## Decision
What we do, with pointers to the implementing code.

## Consequences
Positive and negative results, including known limitations.

## Alternatives considered
Options that were evaluated and why they were not chosen.
```
