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

    Note:
        A given ``cache_dir`` must be used only by callers sharing the same
        ``lookback_days``. The cache is pruned to ``asof - (lookback_days +
        overlap_days)`` on every run, so a shorter-lookback caller sharing the
        directory would trim history a longer-lookback caller still needs,
        forcing a full re-fetch. The live config uses a single caller at
        ``lookback_days: 365``.
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
        if universe.empty or "symbol" not in universe.columns:
            empty = {s: pd.DataFrame() for s in _SERIES}
            empty["market_cap"] = pd.DataFrame()
            return empty

        requested = universe["symbol"].tolist()
        lookback = int(params.get("lookback_days", 365))
        asof_dt = pd.Timestamp(asof, tz="UTC").normalize()
        start = asof_dt - pd.Timedelta(days=lookback)

        # 1-2. read cache; last cached date per coin (prices = reference series)
        cache = {s: self._read_cache(s) for s in _SERIES}
        prices_cache = cache["prices"]
        last_date = {} if prices_cache.empty else prices_cache.groupby("symbol")["date"].max().to_dict()

        # 3. bucket coins
        new_coins = [c for c in requested if c not in last_date]
        cached_coins = [c for c in requested if c in last_date]
        fetched: dict[str, list[pd.DataFrame]] = {s: [] for s in _SERIES}

        # 4a. new coins -> full lookback
        if new_coins:
            res = self._inner.load_market_data(
                pd.DataFrame({"symbol": new_coins}), asof, {**params, "lookback_days": lookback}
            )
            for s in _SERIES:
                fetched[s].append(self._wide_to_long(res.get(s, pd.DataFrame())))

        # 4b. cached coins -> max gap + overlap (one batched call)
        if cached_coins:
            max_gap = max((asof_dt - last_date[c]).days for c in cached_coins) + self.overlap_days
            max_gap = max(max_gap, 1)
            res = self._inner.load_market_data(
                pd.DataFrame({"symbol": cached_coins}), asof, {**params, "lookback_days": max_gap}
            )
            for s in _SERIES:
                fetched[s].append(self._wide_to_long(res.get(s, pd.DataFrame())))

        # 5-7. merge -> dedup (fetched wins) -> prune -> atomic rewrite
        prune_floor = asof_dt - pd.Timedelta(days=lookback + self.overlap_days)
        merged: dict[str, pd.DataFrame] = {}
        for s in _SERIES:
            parts = [f for f in ([cache[s]] + fetched[s]) if not f.empty]
            combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_EMPTY_LONG_COLS)
            if not combined.empty:
                combined = combined.drop_duplicates(["date", "symbol"], keep="last")
                combined = combined[combined["date"] >= prune_floor]
                combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
            self._write_cache(s, combined)
            merged[s] = combined

        # 8. build return frames in memory (no second read)
        result = {s: self._long_to_wide(merged[s], requested, start, asof_dt) for s in _SERIES}
        result["market_cap"] = pd.DataFrame()
        logger.info(
            "Hyperliquid cached: %d new + %d cached coins, %d price date-rows returned",
            len(new_coins),
            len(cached_coins),
            len(result["prices"]),
        )
        return result
