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
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from quantbox.contracts import PluginMeta

logger = logging.getLogger(__name__)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"


def _post(payload: dict) -> Any:
    """POST JSON to the Hyperliquid info endpoint."""
    resp = requests.post(HL_INFO_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


@dataclass
class HyperliquidDataPlugin:
    """DataPlugin for Hyperliquid USDC perpetual futures.

    Usage in config::

        plugins:
          data:
            name: hyperliquid.data.v1
            params_init: {}
    """

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
        outputs=("universe", "prices", "volume", "funding_rates", "market_cap"),
        examples=(
            "plugins:\n  data:\n    name: hyperliquid.data.v1\n    params_init: {}",
        ),
    )

    # ------------------------------------------------------------------
    # DataPlugin protocol
    # ------------------------------------------------------------------

    def load_universe(self, params: Dict[str, Any]) -> pd.DataFrame:
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
        symbols: Optional[List[str]] = params.get("symbols")
        if symbols:
            return pd.DataFrame({"symbol": symbols})

        data = _post({"type": "metaAndAssetCtxs"})
        meta_info = data[0]["universe"]  # list of {name, szDecimals, ...}
        asset_ctxs = data[1]             # parallel list of live contexts

        rows: list[dict] = []
        for meta, ctx in zip(meta_info, asset_ctxs):
            symbol = meta["name"]
            day_ntl_vlm = float(ctx["dayNtlVlm"])
            rows.append({
                "symbol": symbol,
                "day_ntl_vlm": day_ntl_vlm,
            })

        df = pd.DataFrame(rows)

        # Sort by daily notional volume descending
        df = df.sort_values("day_ntl_vlm", ascending=False).reset_index(drop=True)

        # top_n cap
        top_n = params.get("top_n")
        if top_n:
            df = df.head(int(top_n))

        logger.info(
            "Hyperliquid universe: %d symbols loaded", len(df),
        )

        return df[["symbol"]]

    def load_market_data(
        self,
        universe: pd.DataFrame,
        asof: str,
        params: Dict[str, Any],
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV candles and funding-rate history.

        Returns dict with keys: ``prices``, ``volume``, ``funding_rates``,
        ``market_cap`` (empty -- not available on Hyperliquid).
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
                time.sleep(0.1)  # 100ms between calls

            # --- OHLCV candles ---
            try:
                candles = _post({
                    "type": "candleSnapshot",
                    "req": {
                        "coin": ticker,
                        "interval": "1d",
                        "startTime": start_ms,
                        "endTime": end_ms,
                    },
                })
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

            time.sleep(0.1)

            # --- Funding rates ---
            try:
                funding = _post({
                    "type": "fundingHistory",
                    "coin": ticker,
                    "startTime": start_ms,
                    "endTime": end_ms,
                })
                if funding:
                    df_f = pd.DataFrame(funding)
                    df_f["date"] = pd.to_datetime(df_f["time"], unit="ms", utc=True).dt.normalize()
                    df_f["rate"] = df_f["fundingRate"].astype(float)
                    # Sum the 8h rates into a daily rate
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
            len(prices), len(prices.columns),
        )

        return {
            "prices": prices,
            "volume": volume,
            "funding_rates": funding_rates,
            "market_cap": pd.DataFrame(),  # Not available on Hyperliquid
        }

    def load_fx(self, asof: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Not applicable for crypto."""
        return None
