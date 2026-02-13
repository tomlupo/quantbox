"""Binance live data plugin.

Wraps ``BinanceDataFetcher`` to implement the ``DataPlugin`` protocol so the
``TradingPipeline`` (and any other pipeline) can fetch live Binance market
data without pre-existing parquet files.

No API key required — uses public Binance endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta

from .binance_data import BinanceDataFetcher

logger = logging.getLogger(__name__)


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
        outputs=("universe", "prices", "volume", "market_cap"),
        examples=("plugins:\n  data:\n    name: binance.live_data.v1\n    params_init:\n      quote_asset: USDT",),
    )

    quote_asset: str = "USDT"
    _fetcher: BinanceDataFetcher = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fetcher = BinanceDataFetcher(quote_asset=self.quote_asset)

    # ------------------------------------------------------------------
    # DataPlugin protocol
    # ------------------------------------------------------------------

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        """Return the trading universe.

        Params:
            symbols (list[str]): Explicit list of ticker symbols.
            top_n (int): Auto-discover top N liquid tickers from Binance.
            min_volume_usd (float): Min 24h volume for auto-discovery (default 1M).
        """
        symbols: list[str] | None = params.get("symbols")
        if symbols:
            return pd.DataFrame({"symbol": symbols})

        top_n = params.get("top_n")
        if top_n:
            min_vol = float(params.get("min_volume_usd", 1_000_000))
            tickers = self._fetcher.get_tradable_tickers(min_volume_usd=min_vol)
            tickers = tickers[: int(top_n)]
            return pd.DataFrame({"symbol": tickers})

        return pd.DataFrame(columns=["symbol"])

    def load_market_data(
        self,
        universe: pd.DataFrame,
        asof: str,
        params: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical OHLCV from Binance and return wide-format dict.

        Returns dict with keys: ``prices``, ``volume``, ``market_cap``
        (DataFrames with date index, ticker columns).
        """
        if universe.empty or "symbol" not in universe.columns:
            return {
                "prices": pd.DataFrame(),
                "volume": pd.DataFrame(),
                "market_cap": pd.DataFrame(),
            }

        tickers = universe["symbol"].tolist()
        lookback = int(params.get("lookback_days", 365))

        data = self._fetcher.get_market_data(
            tickers=tickers,
            lookback_days=lookback,
            end_date=asof,
        )

        logger.info(
            "Loaded market data for %d symbols from Binance",
            len(data.get("prices", pd.DataFrame()).columns),
        )
        return data

    def load_fx(self, asof: str, params: dict[str, Any]) -> pd.DataFrame | None:
        """Not applicable for crypto (all USD-denominated)."""
        return None
