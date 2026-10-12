from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .contracts import PluginMeta

__all__ = [
    "CoverageReport",
    "DatasetManifest",
    "DatasetPlugin",
    "load_pinned_dataset",
]


def _resolve_pinned(name: str, lock: str | None = None) -> dict[str, Any]:
    """Where a by-name dataset will be read from, refused when its bytes are not the pin.

    :func:`quantbox.dataset_lock.resolve_dataset` — the root from ``$QUANTBOX_DATASETS_ROOT``,
    the pin from *lock* (default: the nearest ``datasets.lock``).
    """
    from quantbox.dataset_lock import require_match, resolve_dataset

    return require_match(resolve_dataset(name, lock=lock))


def load_pinned_dataset(
    name: str, lock: str | None = None, resolved: dict[str, Any] | None = None
) -> tuple[Any, dict[str, Any]]:
    """A quantbox-datasets Dataset and the resolution it was served from.

    Resolved by :func:`quantbox.dataset_lock.resolve_dataset` (unless *resolved*
    already is) and refused before any read when the bytes are not the pinned
    build. This was ``quantbox.plugins.datasources.local_file_data._load_pinned_dataset``
    (TOM-1449); since 0.13.0 that name raises ``ImportError`` naming this one (TOM-1457).
    """
    if resolved is None:
        resolved = _resolve_pinned(name, lock)
    try:
        from quantbox_datasets.lock import load
    except ImportError as exc:
        raise ImportError(
            f"dataset={name!r} needs quantbox-datasets installed (it carries quantbox_datasets.lock); "
            "quantbox does not depend on it — install it from its clone and point "
            "QUANTBOX_DATASETS_ROOT at <clone>/datasets"
        ) from exc
    return load(name, root=resolved["root"], sha256=resolved["sha256"], pinned=False), resolved


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    version: str
    date_range: Mapping[str, str]  # {"start": "...", "end": "..."}
    symbols_count: int
    data_fields: tuple[str, ...]
    source: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageReport:
    per_symbol: Mapping[str, Any]
    per_field: Mapping[str, Any]
    overall: Mapping[str, Any]
    extras: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetPlugin(Protocol):
    meta: PluginMeta
    dataset_id: str
    dataset_version: str
    capabilities: tuple[str, ...]

    def load_prices(self) -> pd.DataFrame: ...
    def load_universe(self) -> pd.DataFrame: ...
    def load_fx(self) -> pd.DataFrame | None: ...

    def manifest(self) -> DatasetManifest: ...
    def manifest_hash(self) -> str: ...
    def coverage_report(self) -> CoverageReport | None: ...
