"""Disk-budget estimation and data-profile safeguards (run before indexing)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from searchforge.config.loader import LoadedConfig
from searchforge.config.models import DataProfile
from searchforge.errors import StorageBudgetExceeded
from searchforge.ingestion.safety import directory_size

GB = 1024**3

#: Upper bound on chunks per data profile (local safety rails, 16 GB RAM machine).
PROFILE_MAX_CHUNKS: dict[DataProfile, int | None] = {
    DataProfile.TINY: 5_000,
    DataProfile.SMALL: 50_000,
    DataProfile.RESEARCH: 1_000_000,
    DataProfile.LARGE: None,
}


@dataclass(frozen=True, slots=True)
class DiskEstimate:
    source_bytes: int
    chunk_text_bytes: int
    chunk_store_bytes: int
    lexical_bytes: int
    dense_bytes: int
    embedding_cache_bytes: int
    total_bytes: int
    free_bytes: int
    reserve_bytes: int

    @property
    def fits(self) -> bool:
        return self.free_bytes - self.total_bytes >= self.reserve_bytes

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "source_bytes": self.source_bytes,
            "chunk_text_bytes": self.chunk_text_bytes,
            "chunk_store_bytes": self.chunk_store_bytes,
            "lexical_bytes": self.lexical_bytes,
            "dense_bytes": self.dense_bytes,
            "embedding_cache_bytes": self.embedding_cache_bytes,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "reserve_bytes": self.reserve_bytes,
            "fits": self.fits,
        }


def free_disk_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def estimate_index_disk(loaded: LoadedConfig) -> DiskEstimate:
    """Conservative estimate from source size and chunking/embedding configuration.

    Heuristics (documented in docs/operations/storage.md): chunk expansion from
    overlap ``1 + overlap/max``, chunk store ≈ 1.3x chunk text, FTS5 ≈ 1.5x chunk
    text, dense = chunks x dim x 4 bytes, embedding cache ≈ dense x 1.1.
    """
    cfg = loaded.config
    source_bytes = sum(directory_size(loaded.resolve(s.path)) for s in cfg.corpus.sources)
    ch = cfg.chunking
    expansion = 1.0 + (
        ch.overlap_chars / ch.max_chars if ch.strategy != "fixed_token" else ch.overlap_tokens / ch.max_tokens
    )
    chunk_text = int(source_bytes * expansion)
    avg_chunk_bytes = max(ch.max_chars // 2, 64)
    est_chunks = max(1, chunk_text // avg_chunk_bytes)
    dense = est_chunks * cfg.index.dense.dimension * 4 if cfg.index.dense.enabled else 0
    store = int(chunk_text * 1.3)
    lexical = int(chunk_text * 1.5) if cfg.index.lexical.enabled else 0
    emb_cache = int(dense * 1.1) if cfg.index.dense.cache_enabled else 0
    total = store + lexical + dense + emb_cache
    ws = loaded.workspace
    return DiskEstimate(
        source_bytes=source_bytes,
        chunk_text_bytes=chunk_text,
        chunk_store_bytes=store,
        lexical_bytes=lexical,
        dense_bytes=dense,
        embedding_cache_bytes=emb_cache,
        total_bytes=total,
        free_bytes=free_disk_bytes(ws),
        reserve_bytes=int(cfg.storage.reserve_free_gb * GB),
    )


def check_disk_budget(loaded: LoadedConfig) -> DiskEstimate:
    est = estimate_index_disk(loaded)
    if not est.fits:
        raise StorageBudgetExceeded(
            f"index build needs ~{est.total_bytes / GB:.2f} GB but only "
            f"{(est.free_bytes - est.reserve_bytes) / GB:.2f} GB is available above the "
            f"{loaded.config.storage.reserve_free_gb} GB reserve"
        )
    if est.total_bytes > loaded.config.storage.max_index_gb * GB:
        raise StorageBudgetExceeded(f"estimated index size {est.total_bytes / GB:.2f} GB exceeds storage.max_index_gb")
    return est


def check_profile_allows_local(loaded: LoadedConfig) -> None:
    profile = loaded.config.index.data_profile
    dense = loaded.config.index.dense
    if profile is DataProfile.LARGE and dense.enabled and dense.backend == "numpy":
        raise StorageBudgetExceeded(
            "data_profile=large is external-backend/cloud mode; it is never executed with the "
            "local NumPy index. Configure dense.backend=qdrant (and a remote lexical backend)."
        )


def enforce_chunk_limit(profile: DataProfile, chunk_count: int) -> None:
    limit = PROFILE_MAX_CHUNKS[profile]
    if limit is not None and chunk_count > limit:
        raise StorageBudgetExceeded(
            f"data_profile={profile.value} allows at most {limit} chunks (got {chunk_count}); "
            "choose a larger profile deliberately"
        )
