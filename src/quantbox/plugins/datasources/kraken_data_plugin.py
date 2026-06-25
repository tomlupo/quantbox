"""Kraken live data plugin.

Wraps :class:`KrakenDataFetcher` to implement the ``DataPlugin`` protocol so the
``TradingPipeline`` (and any other pipeline) can run a spot strategy on Kraken
without pre-existing parquet files. Public Kraken endpoints (via ccxt) need no
API key.

``quote_asset`` defaults to **USD** — the deep native large-cap books on Kraken
are ``*/USD``; USDC books are thin below the very top names (see the Kraken
adapter analysis). Set ``quote_asset: USDC`` only if USDC-denominated NAV is a
hard requirement.

NOTE — backtests: Kraken's public OHLC is capped at 720 recent candles, so this
live plugin is for live/paper only. Backtests must use the cached
``quantbox-datasets`` feeds (``local_file_data`` / ``dataset.curated.v1``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta

from ._utils import normalize_data_frequency
from .kraken_data import KrakenDataFetcher

logger = logging.getLogger(__name__)


@dataclass
class KrakenDataPlugin:
    """DataPlugin adapter around KrakenDataFetcher.

    Usage in config::

        plugins:
          data:
            name: kraken.live_data.v1
            params_init:
              quote_asset: USD
    """

    meta = PluginMeta(
        name="kraken.live_data.v1",
        kind="data",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Live spot market data from Kraken public API via ccxt (no API key needed).",
        tags=("kraken", "crypto", "live", "spot"),
        capabilities=("paper", "live", "crypto"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "quote_asset": {"type": "string", "default": "USD"},
                "mcap_source": {
                    "type": "string",
                    "enum": ["coingecko", "coinmarketcap", "cmc"],
                    "default": "coingecko",
                },
            },
        },
        outputs=("universe", "prices", "volume", "market_cap", "screen_volume"),
        examples=("plugins:\n  data:\n    name: kraken.live_data.v1\n    params_init:\n      quote_asset: USD",),
    )

    quote_asset: str = "USD"
    # Market-cap rankings source: "coingecko" (default) or "coinmarketcap"/"cmc".
    mcap_source: str = "coingecko"
    _fetcher: KrakenDataFetcher = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fetcher = KrakenDataFetcher(
            quote_asset=self.quote_asset,
            mcap_source=self.mcap_source,
        )

    # ------------------------------------------------------------------
    # DataPlugin protocol
    # ------------------------------------------------------------------

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        """Return the trading universe.

        Params:
            symbols (list[str]): Explicit list of ticker symbols.
            top_n (int): Auto-discover top N liquid tickers from Kraken.
            min_volume_usd (float): Min 24h volume for auto-discovery (default 1M).
            not_tradable (list[str]): On the CMC path, opt-out symbols removed
                from the CMC ranking BEFORE the top_n cut. Ignored on the
                coingecko path.
        """
        symbols: list[str] | None = params.get("symbols")
        if symbols:
            return pd.DataFrame({"symbol": symbols})

        top_n = params.get("top_n")
        if top_n:
            min_vol = float(params.get("min_volume_usd", 1_000_000))
            not_tradable: list[str] | None = params.get("not_tradable")
            tickers: list[str] | None = None

            # CMC path: rank candidates by GENUINE market cap (CMC top-N),
            # intersected with Kraken-tradable, ordered by CMC rank — mature
            # coins only (see KrakenDataFetcher.get_mcap_ranked_candidates).
            if str(self.mcap_source).lower() in ("coinmarketcap", "cmc"):
                tickers = self._fetcher.get_mcap_ranked_candidates(
                    top_n=int(top_n),
                    min_volume_usd=min_vol,
                    not_tradable=not_tradable,
                )
                if tickers is None:
                    logger.warning(
                        "CMC candidate ranking unavailable; falling back to "
                        "Kraken-volume ordering for the candidate universe."
                    )

            if tickers is None:
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
        """Fetch historical OHLCV from Kraken and return wide-format dict.

        Returns dict with keys ``prices``, ``volume``, ``market_cap``,
        ``screen_volume`` (DataFrames with date index, ticker columns). The
        universe-screen inputs are resolved mode-aware (see
        :func:`resolve_screen_inputs`); mode is read from ``params["mode"]``.

        Data is point-in-time up to ``asof`` (passed as ``end_date``). Note the
        Kraken 720-candle OHLC cap applies in live/paper.
        """
        if universe.empty or "symbol" not in universe.columns:
            return {
                "prices": pd.DataFrame(),
                "volume": pd.DataFrame(),
                "market_cap": pd.DataFrame(),
                "screen_volume": pd.DataFrame(),
            }

        tickers = universe["symbol"].tolist()
        lookback = int(params.get("lookback_days", 365))
        interval = normalize_data_frequency(params.get("frequency", "1d"))

        data = self._fetcher.get_market_data(
            tickers=tickers,
            lookback_days=lookback,
            end_date=asof,
            interval=interval,
            mode=params.get("mode"),
        )

        logger.info(
            "Loaded market data for %d symbols from Kraken",
            len(data.get("prices", pd.DataFrame()).columns),
        )
        return data

    def load_fx(self, asof: str, params: dict[str, Any]) -> pd.DataFrame | None:
        """FX not applicable for a single-quote spot book (USD-denominated)."""
        return None
