from __future__ import annotations

import os
from importlib.resources import files as _res_files
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    # src/quantbox/plugin_manifest.py -> src/quantbox -> src -> repo root
    # Only valid when running from a git clone; not available in installed packages.
    return Path(__file__).resolve().parents[2]


def _bundled_manifest_path() -> Path:
    return Path(str(_res_files("quantbox.plugins").joinpath("manifest.yaml")))


def default_manifest_path() -> Path:
    env = os.environ.get("QUANTBOX_MANIFEST")
    if env:
        return Path(env)

    cwd_path = Path.cwd() / "plugins" / "manifest.yaml"
    if cwd_path.exists():
        return cwd_path

    return _bundled_manifest_path()


def load_manifest() -> dict[str, Any]:
    path = default_manifest_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_profile(profile: str, manifest: dict[str, Any]) -> dict[str, Any]:
    profiles = (manifest or {}).get("profiles", {}) or {}
    return profiles.get(profile, {}) or {}
