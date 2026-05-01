from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .contracts import PluginMeta


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
