"""Typed, validated configuration (Pydantic v2). Unknown keys are rejected."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from searchforge.domain.document import SourceFormat
from searchforge.domain.filters import FieldType, validate_field_name
from searchforge.domain.judgment import QuerySplit
from searchforge.domain.query import QueryClass
from searchforge.errors import InvalidFilter


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- corpus


class SourceConfig(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
    path: str
    format: SourceFormat | None = None
    glob: str = "**/*"
    id_field: str | None = "id"
    text_field: str = "text"
    title_field: str | None = "title"
    uri_field: str | None = "url"
    created_field: str | None = "created_at"
    updated_field: str | None = "updated_at"
    language_field: str | None = "language"
    metadata_fields: list[str] | None = None
    uri_prefix: str | None = None


class NormalizationConfig(StrictModel):
    unicode_form: Literal["NFC", "NFKC", "none"] = "NFC"
    collapse_whitespace: bool = True
    strip_control_chars: bool = True
    max_blank_lines: int = Field(default=2, ge=1, le=10)


class IngestionLimits(StrictModel):
    max_file_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    max_record_chars: int = Field(default=2_000_000, gt=0)
    max_documents: int | None = Field(default=None, gt=0)
    on_decode_error: Literal["replace", "skip", "fail"] = "replace"
    follow_symlinks: bool = False


class CorpusConfig(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
    sources: list[SourceConfig] = Field(min_length=1)
    normalization: NormalizationConfig = NormalizationConfig()
    limits: IngestionLimits = IngestionLimits()
    filterable_fields: dict[str, FieldType] = Field(default_factory=dict)

    @field_validator("filterable_fields")
    @classmethod
    def _valid_field_names(cls, value: dict[str, FieldType]) -> dict[str, FieldType]:
        for name in value:
            try:
                validate_field_name(name)
            except InvalidFilter as exc:
                raise ValueError(str(exc)) from exc
        return value


# --------------------------------------------------------------------------- chunking


class ChunkStrategy(StrEnum):
    FIXED_TOKEN = "fixed_token"
    SENTENCE = "sentence"
    SECTION = "section"


class ChunkingConfig(StrictModel):
    strategy: ChunkStrategy = ChunkStrategy.SECTION
    max_chars: int = Field(default=1200, ge=50, le=100_000)
    overlap_chars: int = Field(default=150, ge=0)
    min_chars: int = Field(default=80, ge=0)
    max_tokens: int = Field(default=200, ge=8, le=20_000)
    overlap_tokens: int = Field(default=30, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_window(self) -> ChunkingConfig:
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if self.min_chars >= self.max_chars:
            raise ValueError("min_chars must be smaller than max_chars")
        return self


# --------------------------------------------------------------------------- index


class DataProfile(StrEnum):
    TINY = "tiny"
    SMALL = "small"
    RESEARCH = "research"
    LARGE = "large"


class OpenSearchConfig(StrictModel):
    hosts: list[str] = Field(default_factory=lambda: ["http://localhost:9200"])
    index_prefix: str = Field(default="searchforge", pattern=r"^[a-z][a-z0-9_\-]{0,40}$")
    username_env: str | None = None
    password_env: str | None = None
    verify_certs: bool = True
    timeout_s: float = Field(default=10.0, gt=0)


class QdrantConfig(StrictModel):
    location: str | None = None
    url: str | None = None
    path: str | None = None
    api_key_env: str | None = None
    collection_prefix: str = Field(default="searchforge", pattern=r"^[A-Za-z][A-Za-z0-9_\-]{0,40}$")
    timeout_s: float = Field(default=10.0, gt=0)


class LexicalConfig(StrictModel):
    enabled: bool = True
    backend: Literal["sqlite_fts", "opensearch"] = "sqlite_fts"
    tokenizer: Literal["porter unicode61", "unicode61", "porter unicode61 remove_diacritics 2"] = (
        "porter unicode61 remove_diacritics 2"
    )
    text_weight: float = Field(default=1.0, gt=0)
    title_weight: float = Field(default=2.0, ge=0)
    opensearch: OpenSearchConfig | None = None


class TfidfConfig(StrictModel):
    svd_components: int | None = Field(default=256, ge=2)
    max_features: int = Field(default=100_000, ge=10)
    ngram_max: int = Field(default=2, ge=1, le=3)
    sublinear_tf: bool = True
    min_df: int = Field(default=1, ge=1)
    fit_max_chunks: int = Field(default=200_000, ge=1)
    random_state: int = 13


class RemoteEmbeddingConfig(StrictModel):
    url: str
    model: str
    api_key_env: str | None = None
    timeout_s: float = Field(default=30.0, gt=0)
    batch_size: int = Field(default=64, ge=1, le=2048)
    dimension: int | None = Field(default=None, ge=1)
    #: Corpus text is sent to an external service only when this is explicitly true.
    allow_corpus_transmission: bool = False


class EmbeddingProviderName(StrEnum):
    HASH = "hash"
    TFIDF = "tfidf"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    REMOTE = "remote"


class DenseConfig(StrictModel):
    enabled: bool = True
    provider: EmbeddingProviderName = EmbeddingProviderName.TFIDF
    model: str = ""
    revision: str | None = None
    dimension: int = Field(default=256, ge=2, le=8192)
    batch_size: int = Field(default=64, ge=1, le=4096)
    backend: Literal["numpy", "qdrant"] = "numpy"
    allow_download: bool = False
    cache_enabled: bool = True
    tfidf: TfidfConfig = TfidfConfig()
    remote: RemoteEmbeddingConfig | None = None
    qdrant: QdrantConfig | None = None

    @model_validator(mode="after")
    def _provider_settings(self) -> DenseConfig:
        if self.provider is EmbeddingProviderName.REMOTE and self.remote is None:
            raise ValueError("dense.provider=remote requires a dense.remote section")
        if self.provider is EmbeddingProviderName.SENTENCE_TRANSFORMER and not self.model:
            raise ValueError("dense.provider=sentence_transformer requires dense.model")
        return self


class IndexConfig(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
    data_profile: DataProfile = DataProfile.TINY
    lexical: LexicalConfig = LexicalConfig()
    dense: DenseConfig = DenseConfig()
    write_batch_size: int = Field(default=512, ge=1, le=100_000)

    @model_validator(mode="after")
    def _at_least_one_index(self) -> IndexConfig:
        if not self.lexical.enabled and not self.dense.enabled:
            raise ValueError("at least one of lexical/dense must be enabled")
        return self


# --------------------------------------------------------------------------- search


class FusionName(StrEnum):
    RRF = "rrf"
    WEIGHTED_RRF = "weighted_rrf"
    NORMALIZED = "normalized"


class FusionConfig(StrictModel):
    strategy: FusionName = FusionName.RRF
    k: int = Field(default=60, ge=1)
    weights: dict[str, float] = Field(default_factory=dict)
    normalization: Literal["minmax", "zscore"] = "minmax"

    @field_validator("weights")
    @classmethod
    def _non_negative(cls, value: dict[str, float]) -> dict[str, float]:
        if any(w < 0 for w in value.values()):
            raise ValueError("fusion weights must be non-negative")
        return value

    @model_validator(mode="after")
    def _weighted_needs_weights(self) -> FusionConfig:
        if self.strategy is FusionName.WEIGHTED_RRF and not self.weights:
            raise ValueError("fusion.strategy=weighted_rrf requires explicit fusion.weights")
        return self


class RerankerName(StrEnum):
    NONE = "none"
    LEXICAL_FEATURES = "lexical_features"
    CROSS_ENCODER = "cross_encoder"


class RerankerConfig(StrictModel):
    type: RerankerName = RerankerName.NONE
    top_n: int = Field(default=50, ge=1, le=1000)
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    revision: str | None = None
    batch_size: int = Field(default=32, ge=1)
    allow_download: bool = False
    #: Per-feature weights for the lexical_features reranker (prior, coverage, title_coverage, phrase,
    #: identifier, proximity); the prior is 1/(1+first-stage position).
    feature_weights: dict[str, float] = Field(default_factory=dict)


class DiversityConfig(StrictModel):
    max_per_document: int | None = Field(default=None, ge=1)
    mmr_lambda: float | None = Field(default=None, ge=0.0, le=1.0)


class RewriteConfig(StrictModel):
    enabled: bool = False
    synonyms: dict[str, list[str]] = Field(default_factory=dict)
    acronyms: dict[str, str] = Field(default_factory=dict)
    multi_query: bool = False
    max_variants: int = Field(default=3, ge=1, le=10)


class RetrieverName(StrEnum):
    LEXICAL = "lexical"
    DENSE = "dense"


class SearchProfile(StrictModel):
    description: str = ""
    retrievers: list[RetrieverName] = Field(default_factory=lambda: [RetrieverName.LEXICAL])
    candidate_k: int = Field(default=50, ge=1, le=1000)
    fusion: FusionConfig = FusionConfig()
    reranker: RerankerConfig = RerankerConfig()
    diversity: DiversityConfig = DiversityConfig()
    rewrite: RewriteConfig = RewriteConfig()
    adaptive: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> SearchProfile:
        if not self.adaptive and not self.retrievers:
            raise ValueError("a non-adaptive profile needs at least one retriever")
        if len(set(self.retrievers)) != len(self.retrievers):
            raise ValueError("retrievers must be unique")
        return self


class PlannerConfig(StrictModel):
    version: str = "static-v1"
    rules: dict[QueryClass, str] = Field(
        default_factory=lambda: {
            QueryClass.IDENTIFIER: "lexical",
            QueryClass.KEYWORD: "hybrid",
            QueryClass.NATURAL_LANGUAGE: "hybrid",
            QueryClass.COMPLEX_SEMANTIC: "hybrid_rerank",
            QueryClass.FILTER_HEAVY: "hybrid",
        }
    )
    default_profile: str = "hybrid"
    extract_inline_filters: bool = True


class TimeoutConfig(StrictModel):
    overall_ms: float = Field(default=3000.0, gt=0)
    retriever_ms: float = Field(default=1500.0, gt=0)
    reranker_ms: float = Field(default=1500.0, gt=0)
    reranker_fallback: bool = True
    allow_degraded: bool = True


class CacheConfig(StrictModel):
    enabled: bool = True
    max_entries: int = Field(default=2048, ge=1, le=10_000_000)
    max_bytes: int = Field(default=64 * 1024 * 1024, ge=1024)
    ttl_s: float = Field(default=600.0, gt=0)


class CapturePolicy(StrEnum):
    NONE = "none"
    HASH_ONLY = "hash_only"
    REDACTED = "redacted"
    FULL = "full"


class SearchConfig(StrictModel):
    default_profile: str = "hybrid"
    top_k: int = Field(default=10, ge=1, le=200)
    profiles: dict[str, SearchProfile] = Field(default_factory=dict)
    planner: PlannerConfig = PlannerConfig()
    timeouts: TimeoutConfig = TimeoutConfig()
    cache: CacheConfig = CacheConfig()
    capture_policy: CapturePolicy = CapturePolicy.HASH_ONLY
    max_workers: int = Field(default=4, ge=1, le=64)


# --------------------------------------------------------------------------- evaluation


class GateConfig(StrictModel):
    ndcg_at_10_min_delta: float | None = -0.005
    recall_at_20_min: float | None = None
    p95_ms_max: float | None = None
    max_query_regression_rate: float | None = 0.03
    block_on_critical_regression: bool = True
    regression_epsilon: float = Field(default=1e-9, ge=0)


class EvaluationConfig(StrictModel):
    queries: str
    qrels: str
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    profiles: list[str] = Field(default_factory=lambda: ["lexical", "dense", "hybrid"])
    splits: list[QuerySplit] | None = None
    latency_repeats: int = Field(default=3, ge=1, le=100)
    relevance_threshold: float = Field(default=1.0, gt=0)
    gates: GateConfig = GateConfig()

    @field_validator("k_values")
    @classmethod
    def _positive_k(cls, value: list[int]) -> list[int]:
        if not value or any(k < 1 for k in value):
            raise ValueError("k_values must be positive")
        return sorted(set(value))


class StorageConfig(StrictModel):
    reserve_free_gb: float = Field(default=5.0, ge=0)
    max_index_gb: float = Field(default=20.0, gt=0)


class SearchForgeConfig(StrictModel):
    workspace: str = ".searchforge"
    corpus: CorpusConfig
    chunking: ChunkingConfig = ChunkingConfig()
    index: IndexConfig
    search: SearchConfig = SearchConfig()
    evaluation: EvaluationConfig | None = None
    storage: StorageConfig = StorageConfig()
