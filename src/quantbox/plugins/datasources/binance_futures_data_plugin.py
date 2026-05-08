"""Binance USDM Futures data plugin.

Wraps ``BinanceFuturesDataFetcher`` to implement the ``DataPlugin`` protocol
for the ``TradingPipeline``.  Returns wide-format DataFrames including
funding rates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta

from .binance_futures_data import BinanceFuturesDataFetcher

logger = logging.getLogger(__name__)


@dataclass
class BinanceFuturesDataPlugin:
    """DataPlugin adapter for Binance USDM perpetual futures.

    Usage in config::

        plugins:
          data:
            name: binance.futures_data.v1
            params_init:
              quote_asset: USDT
              min_open_interest_usd: 5000000
    """

    meta = PluginMeta(
        name="binance.futures_data.v1",
        kind="data",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="USDM futures data from Binance (OHLCV + funding rates, no API key).",
        tags=("binance", "crypto", "futures", "live"),
        capabilities=("paper", "live", "crypto", "futures"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "quote_asset": {"type": "string", "default": "USDT"},
                "min_open_interest_usd": {"type": "number", "default": 0},
            },
        },
        outputs=("universe", "prices", "volume", "funding_rates", "market_cap"),
        examples=(
            "plugins:\n  data:\n    name: binance.futures_data.v1\n    params_init:\n"
            "      quote_asset: USDT\n      min_open_interest_usd: 5000000",
        ),
    )

    quote_asset: str = "USDT"
    min_open_interest_usd: float = 0.0
    _fetcher: BinanceFuturesDataFetcher = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fetcher = BinanceFuturesDataFetcher(quote_asset=self.quote_asset)

    # ------------------------------------------------------------------
    # DataPlugin protocol
    # ------------------------------------------------------------------

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        """Return the futures trading universe.

        Params:
            symbols (list[str]): Explicit ticker list.
            top_n (int): Auto-discover top N liquid futures.
            min_volume_usd (float): Min 24h volume for auto-discovery.
            min_open_interest_usd (float): Min OI filter (overrides instance field).
        """
        symbols: list[str] | None = params.get("symbols")
        if symbols:
            df = pd.DataFrame({"symbol": symbols})
        else:
            top_n = params.get("top_n")
            if top_n:
                min_vol = float(params.get("min_volume_usd", 1_000_000))
                tickers = self._fetcher.get_tradable_tickers(min_volume_usd=min_vol)
                tickers = tickers[: int(top_n)]
                df = pd.DataFrame({"symbol": tickers})
            else:
                return pd.DataFrame(columns=["symbol"])

        # Open-interest filter
        min_oi = float(params.get("min_open_interest_usd", self.min_open_interest_usd))
        if min_oi > 0 and not df.empty:
            syms = df["symbol"].tolist()
            oi = self._fetcher.get_open_interest(syms)
            before = len(df)
            df = df[df["symbol"].map(lambda s: oi.get(s, 0.0) >= min_oi)].reset_index(drop=True)
            filtered = before - len(df)
            if filtered:
                logger.info(
                    "OI filter: removed %d symbols (min $%.0f), %d remaining",
                    filtered,
                    min_oi,
                    len(df),
                )

        return df

    def load_market_data(
        self,
        universe: pd.DataFrame,
        asof: str,
        params: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV + funding rates and return wide-format dict.

        Returns dict with keys: ``prices``, ``volume``, ``funding_rates``,
        ``market_cap`` (DataFrames with date index, ticker columns).
        """
        if universe.empty or "symbol" not in universe.columns:
            return {
                "prices": pd.DataFrame(),
                "volume": pd.DataFrame(),
                "funding_rates": pd.DataFrame(),
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
            "Loaded %d futures price rows for %d symbols",
            len(data.get("prices", pd.DataFrame())),
            len(data.get("prices", pd.DataFrame()).columns),
        )
        return data

    def load_fx(self, asof: str, params: dict[str, Any]) -> pd.DataFrame | None:
        """Not applicable for crypto."""
        return None
