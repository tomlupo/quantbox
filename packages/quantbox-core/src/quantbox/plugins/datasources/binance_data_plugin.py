"""Binance live data plugin.

Wraps ``BinanceDataFetcher`` to implement the ``DataPlugin`` protocol so the
``TradingPipeline`` (and any other pipeline) can fetch live Binance market
data without pre-existing parquet files.

No API key required — uses public Binance endpoints.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from quantbox.contracts import PluginMeta
from .binance_data import BinanceDataFetcher

logger = logging.getLogger(__name__)


def _wide_to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert a wide DataFrame (date index, ticker columns) to long format.

    Uses ``melt`` instead of ``stack`` so the behaviour is identical on
    pandas 2.x **and** 3.0 (where ``stack(dropna=…)`` was removed).
    """
    df = wide.rename_axis("date").reset_index()
    long = df.melt(id_vars="date", var_name="symbol", value_name=value_name)
    return long


@dataclass
class BinanceDataPlugin:
    """DataPlugin adapter around BinanceDataFetcher.

    Usage in config::

        plugins:
          data:
            name: binance.live_data.v1
            params_init:
              quote_asset: USDT
    """

    meta = PluginMeta(
        name="binance.live_data.v1",
        kind="data",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Live market data from Binance public API (no API key needed).",
        tags=("binance", "crypto", "live"),
        capabilities=("backtest", "paper", "live", "crypto"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "quote_asset": {"type": "string", "default": "USDT"},
            },
        },
        outputs=("universe", "prices"),
        examples=(
            "plugins:\n  data:\n    name: binance.live_data.v1\n    params_init:\n      quote_asset: USDT",
        ),
    )

    quote_asset: str = "USDT"
    _fetcher: BinanceDataFetcher = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fetcher = BinanceDataFetcher(quote_asset=self.quote_asset)

    # ------------------------------------------------------------------
    # DataPlugin protocol
    # ------------------------------------------------------------------

    def load_universe(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Return the trading universe.

        Params:
            symbols (list[str]): Explicit list of ticker symbols.
            top_n (int): Auto-discover top N liquid tickers from Binance.
            min_volume_usd (float): Min 24h volume for auto-discovery (default 1M).
        """
        symbols: Optional[List[str]] = params.get("symbols")
        if symbols:
            return pd.DataFrame({"symbol": symbols})

        top_n = params.get("top_n")
        if top_n:
            min_vol = float(params.get("min_volume_usd", 1_000_000))
            tickers = self._fetcher.get_tradable_tickers(min_volume_usd=min_vol)
            tickers = tickers[: int(top_n)]
            return pd.DataFrame({"symbol": tickers})

        return pd.DataFrame(columns=["symbol"])

    def load_prices(
        self,
        universe: pd.DataFrame,
        asof: str,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Fetch historical OHLCV from Binance and return in long format.

        Returns DataFrame with columns: ``date``, ``symbol``, ``close``,
        ``volume`` (and optionally ``market_cap``).
        """
        if universe.empty or "symbol" not in universe.columns:
            return pd.DataFrame(columns=["date", "symbol", "close", "volume"])

        tickers = universe["symbol"].tolist()
        lookback = int(params.get("lookback_days", 365))

        data = self._fetcher.get_market_data(
            tickers=tickers,
            lookback_days=lookback,
            end_date=asof,
        )

        prices_wide: pd.DataFrame = data.get("prices", pd.DataFrame())
        volume_wide: pd.DataFrame = data.get("volume", pd.DataFrame())
        market_cap_wide: pd.DataFrame = data.get("market_cap", pd.DataFrame())

        if prices_wide.empty:
            return pd.DataFrame(columns=["date", "symbol", "close", "volume"])

        # Wide → long via melt (works identically across pandas 2.x and 3.0,
        # unlike stack() whose dropna semantics changed in pandas 3.0).
        long = _wide_to_long(prices_wide, "close")

        if not volume_wide.empty:
            vol_long = _wide_to_long(volume_wide, "volume")
            long = long.merge(vol_long, on=["date", "symbol"], how="left")
        else:
            long["volume"] = 0.0

        if not market_cap_wide.empty:
            mc_long = _wide_to_long(market_cap_wide, "market_cap")
            long = long.merge(mc_long, on=["date", "symbol"], how="left")

        long = long.dropna(subset=["close"]).reset_index(drop=True)
        logger.info(
            "Loaded %d price rows for %d symbols from Binance",
            len(long),
            long["symbol"].nunique(),
        )
        return long

    def load_fx(self, asof: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Not applicable for crypto (all USD-denominated)."""
        return None
