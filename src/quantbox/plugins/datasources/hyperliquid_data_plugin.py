"""Hyperliquid perpetual-futures data plugin.

Uses Hyperliquid's native REST API (no ccxt, no API key) to fetch
universe metadata, OHLCV candles, and funding-rate history.

All requests are POST to ``https://api.hyperliquid.xyz/info``.

## LLM Usage Guide

### Quick Start
```python
from quantbox.plugins.datasources import HyperliquidDataPlugin

data = HyperliquidDataPlugin()

# Fetch universe (top 100 by volume)
universe = data.load_universe({"top_n": 100})

# Fetch 365 days of prices + funding rates
market = data.load_market_data(universe, "2026-02-13", {"lookback_days": 365})
prices = market["prices"]       # DatetimeIndex × ticker wide-format
volume = market["volume"]       # same shape
funding = market["funding_rates"]  # daily funding rates
```

### Config Example
```yaml
plugins:
  data:
    name: hyperliquid.data.v1
    params_init: {}
```

### Key Features
- No API key required (public REST endpoint)
- 100ms rate limit between calls (built-in)
- Universe ranked by 24h notional volume
- Daily OHLCV candles and 8h funding rates (aggregated to daily)
- No market_cap available (returns empty DataFrame)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from quantbox.contracts import PluginMeta

from ._utils import MarketCapProvider

logger = logging.getLogger(__name__)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"


def _post(payload: dict, max_retries: int = 5) -> Any:
    """POST JSON to the Hyperliquid info endpoint with retry on 429."""
    delay = 1.0
    for attempt in range(max_retries):
        resp = requests.post(HL_INFO_URL, json=payload, timeout=30)
        if resp.status_code == 429:
            logger.warning("Rate limited (429), sleeping %.1fs before retry %d/%d", delay, attempt + 1, max_retries)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # raise on final failure


@dataclass
class HyperliquidDataPlugin:
    """DataPlugin for Hyperliquid USDC perpetual futures.

    Usage in config::

        plugins:
          data:
            name: hyperliquid.data.v1
            params_init: {}
    """

    # Injectable for testing; defaults to a live CoinGecko provider at use.
    mcap_provider: MarketCapProvider | None = None

    meta = PluginMeta(
        name="hyperliquid.data.v1",
        kind="data",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Perpetual-futures data from Hyperliquid REST API (no API key).",
        tags=("hyperliquid", "crypto", "futures", "live"),
        capabilities=("paper", "live", "crypto", "futures"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {},
        },
        outputs=("universe", "prices", "volume", "funding_rates", "market_cap", "screen_volume"),
        examples=("plugins:\n  data:\n    name: hyperliquid.data.v1\n    params_init: {}",),
    )

    # ------------------------------------------------------------------
    # DataPlugin protocol
    # ------------------------------------------------------------------

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        """Return the Hyperliquid perps universe sorted by 24h volume.

        Uses the ``metaAndAssetCtxs`` endpoint which returns both the
        static meta (symbol names) and live asset contexts (OI, mark
        price, 24h volume) in a single call.

        Filtering to tradeable tokens is handled downstream by
        ``TokenPolicy`` in the trading pipeline.

        Params
        ------
        symbols : list[str]
            Explicit ticker list (skips discovery).
        top_n : int
            Keep top N by 24h notional volume (upper bound).
        """
        symbols: list[str] | None = params.get("symbols")
        if symbols:
            return pd.DataFrame({"symbol": symbols})

        data = _post({"type": "metaAndAssetCtxs"})
        meta_info = data[0]["universe"]  # list of {name, szDecimals, ...}
        asset_ctxs = data[1]  # parallel list of live contexts

        rows: list[dict] = []
        for meta, ctx in zip(meta_info, asset_ctxs, strict=False):
            symbol = meta["name"]
            day_ntl_vlm = float(ctx["dayNtlVlm"])
            rows.append(
                {
                    "symbol": symbol,
                    "day_ntl_vlm": day_ntl_vlm,
                }
            )

        df = pd.DataFrame(rows)

        # Sort by daily notional volume descending
        df = df.sort_values("day_ntl_vlm", ascending=False).reset_index(drop=True)

        # top_n cap
        top_n = params.get("top_n")
        if top_n:
            df = df.head(int(top_n))

        logger.info(
            "Hyperliquid universe: %d symbols loaded",
            len(df),
        )

        return df[["symbol"]]

    def load_market_data(
        self,
        universe: pd.DataFrame,
        asof: str,
        params: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV candles and funding-rate history.

        Returns dict with keys: ``prices``, ``volume``, ``funding_rates``,
        ``market_cap`` and ``screen_volume``. Hyperliquid exposes neither market
        cap nor cross-venue volume, so both are sourced from CoinGecko (free, no
        key) via :class:`MarketCapProvider` — the same market-wide screen the
        Binance book uses. ``volume`` stays per-venue (Hyperliquid) for sizing.
        """
        if universe.empty or "symbol" not in universe.columns:
            return {
                "prices": pd.DataFrame(),
                "volume": pd.DataFrame(),
                "funding_rates": pd.DataFrame(),
                "market_cap": pd.DataFrame(),
                "screen_volume": pd.DataFrame(),
            }

        tickers = universe["symbol"].tolist()
        lookback = int(params.get("lookback_days", 365))

        asof_dt = datetime.strptime(asof, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_dt = asof_dt - timedelta(days=lookback)
        # Hyperliquid expects millisecond timestamps
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(asof_dt.timestamp() * 1000)

        all_prices: dict[str, pd.Series] = {}
        all_volume: dict[str, pd.Series] = {}
        all_funding: dict[str, pd.Series] = {}

        for i, ticker in enumerate(tickers):
            if i > 0:
                time.sleep(0.5)  # 500ms between tickers to stay under rate limit

            # --- OHLCV candles ---
            try:
                candles = _post(
                    {
                        "type": "candleSnapshot",
                        "req": {
                            "coin": ticker,
                            "interval": "1d",
                            "startTime": start_ms,
                            "endTime": end_ms,
                        },
                    }
                )
                if candles:
                    df_c = pd.DataFrame(candles)
                    df_c["date"] = pd.to_datetime(df_c["t"], unit="ms", utc=True).dt.normalize()
                    df_c["close"] = df_c["c"].astype(float)
                    df_c["vlm"] = df_c["v"].astype(float)
                    df_c = df_c.drop_duplicates(subset="date", keep="last").set_index("date")
                    all_prices[ticker] = df_c["close"]
                    all_volume[ticker] = df_c["vlm"]
            except Exception:
                logger.warning("Failed to fetch candles for %s", ticker, exc_info=True)

            time.sleep(0.5)

            # --- Funding rates (paginated — API returns ~20 days per call) ---
            try:
                _CHUNK_MS = 30 * 24 * 3600 * 1000  # 30-day chunks
                all_records: list[dict] = []
                chunk_start = start_ms
                while chunk_start < end_ms:
                    batch = _post(
                        {
                            "type": "fundingHistory",
                            "coin": ticker,
                            "startTime": chunk_start,
                            "endTime": end_ms,
                        }
                    )
                    if not batch:
                        break
                    all_records.extend(batch)
                    last_ms = max(r["time"] for r in batch)
                    if last_ms <= chunk_start:
                        break
                    chunk_start = last_ms + 1
                    time.sleep(0.3)

                if all_records:
                    df_f = pd.DataFrame(all_records)
                    df_f["date"] = pd.to_datetime(df_f["time"], unit="ms", utc=True).dt.normalize()
                    df_f["rate"] = df_f["fundingRate"].astype(float)
                    daily_rate = df_f.groupby("date")["rate"].sum()
                    all_funding[ticker] = daily_rate
            except Exception:
                logger.warning("Failed to fetch funding for %s", ticker, exc_info=True)

        prices = pd.DataFrame(all_prices)
        volume = pd.DataFrame(all_volume)
        funding_rates = pd.DataFrame(all_funding)

        # Ensure sorted date index
        for df in (prices, volume, funding_rates):
            df.sort_index(inplace=True)

        logger.info(
            "Loaded %d price rows for %d symbols from Hyperliquid",
            len(prices),
            len(prices.columns),
        )

        # Market-wide screen inputs from CoinGecko (Hyperliquid has neither
        # market cap nor cross-venue volume). This gives the HL book the same
        # two-stage market-wide screen as the Binance book; per-venue `volume`
        # above stays for sizing.
        if prices.empty:
            market_cap = pd.DataFrame()
            screen_volume = pd.DataFrame()
        else:
            provider = self.mcap_provider or MarketCapProvider()
            market_cap = provider.estimate_market_cap(prices, volume)
            screen_volume = provider.estimate_aggregate_volume(prices)

        return {
            "prices": prices,
            "volume": volume,
            "funding_rates": funding_rates,
            "market_cap": market_cap,
            "screen_volume": screen_volume,
        }

    def load_fx(self, asof: str, params: dict[str, Any]) -> pd.DataFrame | None:
        """Not applicable for crypto."""
        return None
