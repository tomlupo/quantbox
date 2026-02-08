"""Shared universe selection for crypto strategies.

Provides vectorized and DuckDB-accelerated universe selection that works
with or without market-cap data.  When ``market_cap`` is ``None`` (e.g.
Hyperliquid), the mcap tier is skipped and assets are ranked directly by
dollar volume.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    DUCKDB_AVAILABLE = False

# Single source of truth for stablecoin / non-tradeable exclusion list.
DEFAULT_STABLECOINS = [
    # USD stablecoins
    "USDT", "USDC", "BUSD", "TUSD", "DAI", "MIM", "USTC", "FDUSD",
    "USDP", "GUSD", "FRAX", "LUSD", "USDD", "PYUSD", "USD1", "USDJ",
    # EUR stablecoins
    "EUR", "EURC", "EURT", "EURS", "EUROC",
    # Gold / commodity tokens
    "PAXG", "XAUT",
    # Wrapped tokens
    "WBTC", "WETH", "BETH", "ETHW", "CBBTC", "CBETH",
    # Other non-tradeable
    "BFUSD", "AEUR",
]


def select_universe(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    market_cap: pd.DataFrame | None = None,
    top_by_mcap: int = 30,
    top_by_volume: int = 10,
    exclude_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Select tradable universe – vectorized pandas implementation.

    When *market_cap* is provided and non-empty the two-stage filter from
    ``CryptoTrendStrategy`` is applied:

    1. Rank by market cap → keep top ``top_by_mcap``
    2. Within those, rank by dollar volume → keep top ``top_by_volume``

    When *market_cap* is ``None`` or an empty DataFrame the mcap tier is
    skipped entirely and assets are ranked directly by dollar volume, keeping
    the top ``top_by_volume``.

    Parameters
    ----------
    prices : DataFrame
        Price DataFrame (date index, ticker columns).
    volume : DataFrame
        Volume DataFrame aligned with *prices*.
    market_cap : DataFrame | None
        Market-cap DataFrame.  Pass ``None`` to skip the mcap tier.
    top_by_mcap : int
        How many assets to keep in the mcap tier (ignored when
        *market_cap* is ``None``).
    top_by_volume : int
        Final universe size (top N by dollar volume).
    exclude_tickers : list[str] | None
        Tickers to exclude.  Defaults to :data:`DEFAULT_STABLECOINS`.

    Returns
    -------
    DataFrame
        0/1 mask with same shape as *prices*.
    """
    if exclude_tickers is None:
        exclude_tickers = DEFAULT_STABLECOINS

    valid_tickers = [t for t in prices.columns if t not in exclude_tickers]

    if not valid_tickers:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    has_mcap = market_cap is not None and not market_cap.empty

    dollar_vol = prices[valid_tickers] * volume.reindex(
        index=prices.index, columns=valid_tickers
    ).fillna(0.0)

    if has_mcap:
        # Stage 1: market-cap rank
        mc = market_cap.reindex(  # type: ignore[union-attr]
            index=prices.index, columns=valid_tickers
        ).fillna(0.0)
        mc_rank = mc.rank(axis=1, ascending=False, method="min")
        mc_mask = mc_rank <= top_by_mcap

        # Stage 2: dollar-volume rank within mcap tier
        vol_masked = dollar_vol.where(mc_mask)
        vol_rank = vol_masked.rank(axis=1, ascending=False, method="min")
    else:
        # No mcap – rank directly by dollar volume
        vol_rank = dollar_vol.rank(axis=1, ascending=False, method="min")

    universe_valid = (vol_rank <= top_by_volume).astype(float)

    universe = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    universe[valid_tickers] = universe_valid
    return universe


# Keep the old name as an alias so existing callers keep working.
select_universe_vectorized = select_universe


def select_universe_duckdb(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    market_cap: pd.DataFrame | None = None,
    top_by_mcap: int = 30,
    top_by_volume: int = 10,
    exclude_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Select universe using DuckDB (faster for very large datasets).

    Falls back to :func:`select_universe` when DuckDB is not installed or
    when *market_cap* is ``None``.
    """
    has_mcap = market_cap is not None and not market_cap.empty

    if not DUCKDB_AVAILABLE or not has_mcap:
        return select_universe(
            prices, volume, market_cap, top_by_mcap, top_by_volume, exclude_tickers
        )

    if exclude_tickers is None:
        exclude_tickers = DEFAULT_STABLECOINS

    def _to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        return df.reset_index().melt(
            id_vars="date", var_name="ticker", value_name=value_name
        )

    prices_long = _to_long(prices, "price")
    volume_long = _to_long(volume, "volume")
    mc_long = _to_long(market_cap, "market_cap")  # type: ignore[arg-type]

    data = prices_long.merge(volume_long, on=["date", "ticker"])
    data = data.merge(mc_long, on=["date", "ticker"])
    data["dollar_volume"] = data["price"] * data["volume"]

    exclude_str = "', '".join(exclude_tickers)

    con = duckdb.connect()  # type: ignore[union-attr]
    con.register("data", data)

    query = f"""
    WITH ranked AS (
        SELECT
            date, ticker, price, market_cap, dollar_volume,
            RANK() OVER (PARTITION BY date ORDER BY market_cap DESC) AS mc_rank
        FROM data
        WHERE ticker NOT IN ('{exclude_str}')
    ),
    mc_filtered AS (
        SELECT *,
            RANK() OVER (PARTITION BY date ORDER BY dollar_volume DESC) AS vol_rank
        FROM ranked
        WHERE mc_rank <= {top_by_mcap}
    )
    SELECT
        date, ticker,
        CASE WHEN vol_rank <= {top_by_volume} THEN 1.0 ELSE 0.0 END AS in_universe
    FROM mc_filtered
    """

    result = con.execute(query).df()
    con.close()

    universe = result.pivot(index="date", columns="ticker", values="in_universe")
    universe = universe.reindex(columns=prices.columns, fill_value=0.0)
    universe = universe.reindex(index=prices.index, fill_value=0.0)
    return universe
