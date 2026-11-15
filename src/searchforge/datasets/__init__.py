"""Packaged data: the synthetic fixture collection used by tests, CI and demos."""

from __future__ import annotations

from pathlib import Path

from searchforge.config.loader import package_data_root


def fixture_dir() -> Path:
    return package_data_root() / "fixture"


def fixture_config_path() -> Path:
    return fixture_dir() / "fixture.yaml"
