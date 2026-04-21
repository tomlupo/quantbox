"""Minimal yaml + override-profile loader + typed-schema validator.

Two entry points:

* :func:`load_yaml_merged` — returns the raw merged dict. Use this
  when the host wants to merge extra data (e.g. per-strategy yaml
  files) before validation.
* :func:`load_config` — convenience wrapper that additionally calls
  ``schema_cls.model_validate(data)`` and returns the validated
  instance.

Scope is deliberately narrow: load base yaml, optionally deep-merge a
profile override, return the dict or hand it to a schema validator.
Anything else (implicit auto-load of xlsx / csv, per-strategy merge,
runtime-field population) stays in the host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``.

    Lists are replaced, not extended — matches the legacy dict-merge
    behaviour used across host projects. Returns a new dict; inputs
    are not mutated.
    """
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml_merged(
    yaml_path: str | Path,
    overrides_dir: str | Path | None = None,
    override_profile: str | None = None,
) -> dict:
    """Load a yaml file and optionally deep-merge a profile override.

    Args:
        yaml_path: Path to the base yaml config file.
        overrides_dir: Directory holding ``{profile}.yaml`` files.
        override_profile: Name of the profile to merge. Missing files
            are silently ignored so dev machines without the profile
            file still boot.

    Returns:
        The merged dict (never validated — call
        :func:`load_config` if you want validation).
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path, encoding="utf-8") as f:
        data: dict = yaml.safe_load(f) or {}

    if overrides_dir and override_profile:
        override_path = Path(overrides_dir) / f"{override_profile}.yaml"
        if override_path.exists():
            with open(override_path, encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}
            data = _deep_merge(data, override)

    return data


def load_config(
    schema_cls: type[T],
    yaml_path: str | Path,
    overrides_dir: str | Path | None = None,
    override_profile: str | None = None,
    extra_data: dict[str, Any] | None = None,
) -> T:
    """Load yaml → (optional override) → (optional extra data) → validate.

    ``schema_cls`` must expose a ``model_validate(data) -> T``
    classmethod — the pydantic v2 interface. Duck-typed so the host
    can use any schema library that matches.

    Args:
        schema_cls: The schema class to validate into. Called as
            ``schema_cls.model_validate(merged_data)``.
        yaml_path: Path to the base yaml config file.
        overrides_dir: Directory holding ``{profile}.yaml`` files.
        override_profile: Profile name; missing files ignored.
        extra_data: Optional dict to deep-merge AFTER the profile
            override but BEFORE validation. Useful for host-specific
            pre-processing (e.g. merging a directory of per-strategy
            yaml files into a ``"strategies"`` key).

    Returns:
        Validated instance of ``schema_cls``.
    """
    data = load_yaml_merged(yaml_path, overrides_dir, override_profile)
    if extra_data:
        data = _deep_merge(data, extra_data)
    return schema_cls.model_validate(data)
