"""``searchforge storage status`` / ``storage clean``.

Cleanup only ever touches *project-owned, disposable* artifacts inside the
workspace: the embedding cache, abandoned ``.partial`` index builds, prepared
corpus exports, and (only with an explicit flag) index versions that are not
``CURRENT``. Source corpora, result reports and anything outside the workspace
are never deleted. ``dry_run`` is the default at the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from searchforge.indexes.lifecycle import IndexLayout, remove_tree
from searchforge.ingestion.safety import directory_size, ensure_within
from searchforge.storage.budget import free_disk_bytes


@dataclass(slots=True)
class StorageStatus:
    workspace: Path
    sources: dict[str, int]
    indexes: dict[str, int]
    embedding_cache: int
    prepared: int
    partial_builds: int
    results: int
    free_disk: int

    def total_workspace(self) -> int:
        return sum(self.indexes.values()) + self.embedding_cache + self.prepared + self.partial_builds


@dataclass(slots=True)
class CleanupPlan:
    targets: list[tuple[Path, int, str]] = field(default_factory=list)

    @property
    def bytes(self) -> int:
        return sum(t[1] for t in self.targets)


def storage_status(
    workspace: Path, sources: dict[str, Path] | None = None, results_dir: Path | None = None
) -> StorageStatus:
    layout = IndexLayout(workspace)
    return StorageStatus(
        workspace=workspace,
        sources={k: directory_size(v) for k, v in (sources or {}).items()},
        indexes={name: directory_size(layout.name_dir(name)) for name in layout.names()},
        embedding_cache=directory_size(workspace / "cache"),
        prepared=directory_size(workspace / "prepared"),
        partial_builds=sum(directory_size(p) for p in layout.partials()),
        results=directory_size(results_dir) if results_dir else 0,
        free_disk=free_disk_bytes(workspace),
    )


def plan_cleanup(workspace: Path, *, include_old_versions: bool = False) -> CleanupPlan:
    plan = CleanupPlan()
    candidates: list[tuple[Path, str]] = [
        (workspace / "cache", "embedding cache (recomputable)"),
        (workspace / "prepared", "prepared corpus exports (recomputable)"),
    ]
    layout = IndexLayout(workspace)
    candidates += [(p, "abandoned partial index build") for p in layout.partials()]
    if include_old_versions:
        for name in layout.names():
            current = layout.current_version(name)
            for manifest in layout.versions(name):
                if manifest.version != current:
                    candidates.append(
                        (layout.version_dir(name, manifest.version), f"non-current index version {manifest.index_id}")
                    )
    for path, reason in candidates:
        if path.exists():
            ensure_within(workspace, path)
            plan.targets.append((path, directory_size(path), reason))
    return plan


def execute_cleanup(workspace: Path, plan: CleanupPlan) -> int:
    freed = 0
    for path, size, _ in plan.targets:
        ensure_within(workspace, path)  # re-check right before deleting
        remove_tree(path)
        freed += size
    return freed
