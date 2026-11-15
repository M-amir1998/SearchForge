"""Exception model.

Every failure SearchForge raises on purpose derives from :class:`SearchForgeError`
so callers (CLI, API, adapters) can map errors to exit codes / HTTP statuses
without catching broad ``Exception``.
"""

from __future__ import annotations


class SearchForgeError(Exception):
    """Base class for all SearchForge errors."""


class ConfigurationError(SearchForgeError):
    """A configuration file or value is invalid."""


class CorpusNotFound(SearchForgeError):
    """A configured corpus source does not exist."""


class IngestionError(SearchForgeError):
    """A document could not be ingested (bad encoding, oversized, malformed)."""


class UnsafePathError(IngestionError):
    """A path escapes its permitted root (path traversal) or is otherwise unsafe."""


class IndexNotFound(SearchForgeError):
    """The requested index or index version does not exist."""


class IndexVersionMismatch(SearchForgeError):
    """An index artifact does not match the version/model it is used with."""


class IndexIntegrityError(SearchForgeError):
    """An index failed validation (checksum mismatch, missing files, bad counts)."""


class IndexImmutableError(SearchForgeError):
    """An attempt was made to modify a completed (immutable) index version."""


class EmbeddingFailure(SearchForgeError):
    """An embedding provider failed to produce vectors."""


class RetrievalFailure(SearchForgeError):
    """A retriever failed (backend error or timeout)."""


class RerankerFailure(SearchForgeError):
    """A reranker failed (model error or timeout)."""


class InvalidFilter(SearchForgeError):
    """A metadata filter references an unknown field, operator or a mistyped value."""


class InvalidQuery(SearchForgeError):
    """A search query is malformed or violates request limits."""


class InvalidPlan(SearchForgeError):
    """A query plan violates validation rules."""


class SearchTimeout(SearchForgeError):
    """The overall search deadline was exceeded."""


class CircuitOpenError(SearchForgeError):
    """A remote backend is short-circuited after repeated failures."""


class StorageBudgetExceeded(SearchForgeError):
    """An operation would exceed the configured disk/RAM budget."""


class EvaluationFailure(SearchForgeError):
    """An evaluation could not be completed (missing qrels, empty query set...)."""


class QualityGateFailure(SearchForgeError):
    """A retrieval regression gate failed."""
