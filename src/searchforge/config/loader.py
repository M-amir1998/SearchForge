"""YAML config loading with ``include:`` composition and safe path resolution.

* ``include: [a.yaml, b.yaml]`` deep-merges the listed files (later wins), then the
  including file is merged on top.
* Relative paths inside a config are resolved against the directory of the file
  that is being loaded (the top-level file).
* ``pkg:<relative>`` resolves to data shipped inside the ``searchforge`` package
  (the fixture corpus), so demos work from any working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from searchforge.config.models import SearchForgeConfig
from searchforge.errors import ConfigurationError

PKG_PREFIX = "pkg:"
MAX_INCLUDE_DEPTH = 8


def package_data_root() -> Path:
    return Path(str(resources.files("searchforge") / "datasets"))


def resolve_path(base_dir: Path, value: str) -> Path:
    """Resolve a config path: ``pkg:`` → package data, relative → ``base_dir``."""
    if value.startswith(PKG_PREFIX):
        rel = value[len(PKG_PREFIX) :].lstrip("/\\")
        root = package_data_root().resolve()
        target = (root / rel).resolve()
        if root != target and root not in target.parents:
            raise ConfigurationError(f"package path escapes package data: {value!r}")
        return target
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"config root must be a mapping: {path}")
    return raw


def load_raw(path: Path, depth: int = 0) -> dict[str, Any]:
    if depth > MAX_INCLUDE_DEPTH:
        raise ConfigurationError("config include depth exceeded (cycle?)")
    data = _read_yaml(path)
    includes = data.pop("include", []) or []
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for inc in includes:
        inc_path = resolve_path(path.parent, str(inc))
        merged = deep_merge(merged, load_raw(inc_path, depth + 1))
    return deep_merge(merged, data)


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    """A validated config plus the directory relative paths resolve against."""

    config: SearchForgeConfig
    base_dir: Path
    source_path: Path | None = None

    def resolve(self, value: str) -> Path:
        return resolve_path(self.base_dir, value)

    @property
    def workspace(self) -> Path:
        """The workspace (indexes, caches) resolves against the *current working directory*
        when relative — never against the config's directory, which may be inside the
        installed package for the bundled fixture configs."""
        path = Path(self.config.workspace).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()

    def raw_dict(self) -> dict[str, Any]:
        return self.config.model_dump(mode="json")


def validate_config(data: dict[str, Any], base_dir: Path, source: Path | None = None) -> LoadedConfig:
    try:
        cfg = SearchForgeConfig.model_validate(data)
    except ValidationError as exc:
        where = f" in {source}" if source else ""
        raise ConfigurationError(f"invalid configuration{where}:\n{exc}") from exc
    return LoadedConfig(config=cfg, base_dir=base_dir, source_path=source)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> LoadedConfig:
    cfg_path = Path(path).expanduser().resolve()
    data = load_raw(cfg_path)
    if overrides:
        data = deep_merge(data, overrides)
    return validate_config(data, cfg_path.parent, cfg_path)
