"""Incremental on-disk cache wrapping the stateless hyperliquid.data.v1 plugin.

Each run fetches only the missing tail per coin instead of the full 365-day
history, cutting a warm run from ~480 REST calls (~3 min, 429s) to ~70 (<1 min).

Cache layout (flat, NOT partitioned):
    cache_dir/prices.parquet
    cache_dir/volume.parquet
    cache_dir/funding_rates.parquet
each long-format with columns (date, symbol, value).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta

from .hyperliquid_data_plugin import HyperliquidDataPlugin

logger = logging.getLogger(__name__)

_SERIES = ("prices", "volume", "funding_rates")
_EMPTY_LONG_COLS = ["date", "symbol", "value"]


@dataclass
class HyperliquidCachedDataPlugin:
    """Incremental on-disk cache over ``hyperliquid.data.v1``.

    Config::

        plugins:
          data:
            name: hyperliquid.data.cached.v1
            params_init:
              cache_dir: ./cache/hyperliquid
              overlap_days: 3
    """

    cache_dir: str = "./cache/hyperliquid"
    overlap_days: int = 3

    meta = PluginMeta(
        name="hyperliquid.data.cached.v1",
        kind="data",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Incremental on-disk cache over hyperliquid.data.v1 (flat long-format parquet).",
        tags=("hyperliquid", "crypto", "futures", "live", "cache"),
        capabilities=("paper", "live", "crypto", "futures"),
        outputs=("universe", "prices", "volume", "funding_rates", "market_cap"),
        params_schema={
            "type": "object",
            "properties": {
                "cache_dir": {"type": "string"},
                "overlap_days": {"type": "integer"},
            },
        },
        examples=(
            "plugins:\n  data:\n    name: hyperliquid.data.cached.v1\n"
            "    params_init:\n      cache_dir: ./cache/hyperliquid",
        ),
    )

    def __post_init__(self) -> None:
        self._inner = HyperliquidDataPlugin()

    # ── delegation ───────────────────────────────────────────────
    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return self._inner.load_universe(params)

    def load_fx(self, asof: str, params: dict[str, Any]) -> pd.DataFrame | None:
        return self._inner.load_fx(asof, params)

    # ── cache I/O ────────────────────────────────────────────────
    def _cache_path(self, series: str) -> Path:
        return Path(self.cache_dir) / f"{series}.parquet"

    def _read_cache(self, series: str) -> pd.DataFrame:
        """Full read of the flat long-format cache file.

        Full (unfiltered) read is required: we rewrite the whole file and must
        preserve coins absent from today's universe. ~12k rows — negligible.
        """
        path = self._cache_path(series)
        if not path.exists():
            return pd.DataFrame(columns=_EMPTY_LONG_COLS)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    def _write_cache(self, series: str, long_df: pd.DataFrame) -> None:
        """Atomic write: temp file in cache_dir, then os.replace."""
        path = self._cache_path(series)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        long_df.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    # ── shape conversions ────────────────────────────────────────
    @staticmethod
    def _wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
        """Wide (date index x symbol) -> long (date, symbol, value), dropping NaN."""
        if wide.empty:
            return pd.DataFrame(columns=_EMPTY_LONG_COLS)
        long_df = wide.stack().rename("value").reset_index()
        if len(long_df.columns) != 3:
            raise ValueError(f"_wide_to_long: expected 3-column stack result, got {list(long_df.columns)}")
        long_df.columns = _EMPTY_LONG_COLS
        long_df["date"] = pd.to_datetime(long_df["date"], utc=True)
        return long_df.dropna(subset=["value"]).reset_index(drop=True)

    @staticmethod
    def _long_to_wide(
        long_df: pd.DataFrame,
        symbols: list[str],
        start: pd.Timestamp,
        asof: pd.Timestamp,
    ) -> pd.DataFrame:
        """Long -> wide, restricted to [start, asof] and reindexed to symbols."""
        if long_df.empty:
            return pd.DataFrame()
        mask = (long_df["date"] >= start) & (long_df["date"] <= asof)
        sub = long_df[mask]
        if sub.empty:
            return pd.DataFrame()
        wide = sub.pivot_table(index="date", columns="symbol", values="value", aggfunc="last")
        wide.columns.name = None
        cols = [s for s in symbols if s in wide.columns]
        return wide[cols].sort_index()

    # ── main entry (implemented in Task 2) ───────────────────────
    def load_market_data(
        self,
        universe: pd.DataFrame,
        asof: str,
        params: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError  # Task 2
