from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import os
import yaml


def _repo_root_from_here() -> Path:
    # quantbox/plugin_manifest.py -> quantbox -> src -> quantbox-core -> packages -> repo
    return Path(__file__).resolve().parents[4]

def repo_root() -> Path:
    return _repo_root_from_here()


def default_manifest_path() -> Path:
    env = os.environ.get("QUANTBOX_MANIFEST")
    if env:
        return Path(env)

    cwd_path = Path.cwd() / "plugins" / "manifest.yaml"
    if cwd_path.exists():
        return cwd_path

    repo_path = repo_root() / "plugins" / "manifest.yaml"
    return repo_path


def load_manifest() -> Dict[str, Any]:
    path = default_manifest_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_profile(profile: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    profiles = (manifest or {}).get("profiles", {}) or {}
    return profiles.get(profile, {}) or {}
