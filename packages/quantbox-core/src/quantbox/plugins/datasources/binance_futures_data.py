"""Binance USDM Futures Market Data Fetcher.

Mirrors ``BinanceDataFetcher`` but uses ``ccxt.binanceusdm`` for perpetual
futures data.  No API key required — uses public endpoints only.

Adds futures-specific methods:
- ``get_funding_rate_history()`` — historical 8-hourly rates
- ``get_open_interest()`` — current open interest per symbol
- ``get_position_limits()`` — exchange-imposed max notional limits
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    ccxt = None  # type: ignore[assignment]
    CCXT_AVAILABLE = False
    logger.warning("ccxt not installed — futures OHLCV/funding fetching unavailable")

# Stablecoins to exclude from universe discovery
DEFAULT_STABLECOINS = [
    "USDT", "USDC", "BUSD", "TUSD", "DAI", "MIM", "USTC", "FDUSD",
    "USDP", "GUSD", "FRAX", "LUSD", "USDD", "PYUSD", "EURC", "EURT",
]

FAPI_BASE = "https://fapi.binance.com"
DEFAULT_REQUEST_DELAY_MS = 100


@dataclass
class BinanceFuturesDataFetcher:
    """Production Binance USDM-futures data fetcher.

    Uses ``ccxt.binanceusdm`` for OHLCV and funding data, with direct REST
    fallback for open-interest and position limits.
    """

    quote_asset: str = "USDT"
    stablecoins: List[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())
    request_delay_ms: int = DEFAULT_REQUEST_DELAY_MS
    max_retries: int = 3

    _exchange: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if CCXT_AVAILABLE and self._exchange is None:
            self._exchange = ccxt.binanceusdm({
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        time.sleep(self.request_delay_ms / 1000)

    def _ccxt_symbol(self, ticker: str) -> str:
        """Convert bare ticker to ccxt futures symbol, e.g. ``BTC`` → ``BTC/USDT:USDT``."""
        return f"{ticker}/{self.quote_asset}:{self.quote_asset}"

    # ------------------------------------------------------------------
    # Universe discovery
    # ------------------------------------------------------------------

    def get_tradable_tickers(self, min_volume_usd: float = 1e6) -> List[str]:
        """Discover perpetual USDT futures pairs, sorted by 24h volume descending."""
        if not CCXT_AVAILABLE:
            return []

        try:
            self._exchange.load_markets(True)
            tickers_24h = self._exchange.fetch_tickers()
        except Exception as exc:
            logger.error("Failed to fetch futures tickers: %s", exc)
            return []

        candidates: List[Tuple[str, float]] = []
        for sym, info in tickers_24h.items():
            market = self._exchange.markets.get(sym, {})
            # Only perpetual USDT-margined
            if not market.get("swap") or market.get("quote") != self.quote_asset:
                continue
            base = market.get("base", "")
            if base.upper() in self.stablecoins:
                continue
            vol = float(info.get("quoteVolume") or 0)
            if vol >= min_volume_usd:
                candidates.append((base, vol))

        # Sort by volume descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates]

    def get_valid_pairs(self, tickers: List[str]) -> Dict[str, Optional[str]]:
        """Map bare tickers to ccxt futures symbols.

        Returns ``{ticker: ccxt_symbol}`` or ``{ticker: None}`` if not found.
        """
        if not CCXT_AVAILABLE:
            return {t: None for t in tickers}

        try:
            self._exchange.load_markets()
        except Exception as exc:
            logger.error("Failed to load futures markets: %s", exc)
            return {t: None for t in tickers}

        result: Dict[str, Optional[str]] = {}
        for ticker in tickers:
            sym = self._ccxt_symbol(ticker)
            if sym in self._exchange.markets:
                result[ticker] = sym
            else:
                result[ticker] = None
        return result

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Fetch futures OHLCV for a single ticker with pagination."""
        if not CCXT_AVAILABLE:
            return None

        symbol = self._ccxt_symbol(ticker)
        pairs = self.get_valid_pairs([ticker])
        if not pairs.get(ticker):
            logger.warning("No futures pair for %s", ticker)
            return None

        try:
            since = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

            all_ohlcv: List[list] = []
            limit = 1000

            while since < end_ts:
                self._rate_limit()
                ohlcv = self._exchange.fetch_ohlcv(
                    symbol, timeframe=interval, since=since, limit=limit,
                )
                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts == since:
                    break
                since = last_ts + 1
                if len(ohlcv) < limit:
                    break

            if not all_ohlcv:
                return None

            df = pd.DataFrame(
                all_ohlcv, columns=["date", "open", "high", "low", "close", "volume"],
            )
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            df["ticker"] = ticker
            return df.reset_index(drop=True)

        except Exception as exc:
            logger.error("Error fetching OHLCV for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Funding rates
    # ------------------------------------------------------------------

    def get_funding_rate_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch 8-hourly funding rates and forward-fill to daily.

        Returns DataFrame with columns ``date``, ``funding_rate`` (daily).
        """
        if not CCXT_AVAILABLE:
            return None

        symbol = self._ccxt_symbol(ticker)
        pairs = self.get_valid_pairs([ticker])
        if not pairs.get(ticker):
            return None

        try:
            since = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

            all_rates: List[Dict[str, Any]] = []
            limit = 1000

            while since < end_ts:
                self._rate_limit()
                rates = self._exchange.fetch_funding_rate_history(
                    symbol, since=since, limit=limit,
                )
                if not rates:
                    break

                all_rates.extend(rates)
                last_ts = rates[-1].get("timestamp", since)
                if last_ts == since:
                    break
                since = last_ts + 1
                if len(rates) < limit:
                    break

            if not all_rates:
                return None

            df = pd.DataFrame(all_rates)
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df[["date", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

            # Resample 8h→daily: sum the 3 daily funding payments
            df = df.set_index("date")
            daily = df["funding_rate"].resample("1D").sum()
            daily = daily.ffill()
            return daily.reset_index().rename(columns={"date": "date"})

        except Exception as exc:
            logger.error("Error fetching funding rates for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Open interest
    # ------------------------------------------------------------------

    def get_open_interest(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current open interest (in USD) for each symbol.

        Uses ``ccxt.fetch_open_interest()`` per symbol.
        """
        if not CCXT_AVAILABLE:
            return {}

        result: Dict[str, float] = {}
        for ticker in symbols:
            sym = self._ccxt_symbol(ticker)
            try:
                self._rate_limit()
                oi = self._exchange.fetch_open_interest(sym)
                # oi is dict with 'openInterestAmount' and 'openInterestValue'
                val = float(oi.get("openInterestValue") or oi.get("openInterestAmount", 0))
                result[ticker] = val
            except Exception as exc:
                logger.debug("OI fetch failed for %s: %s", ticker, exc)
                result[ticker] = 0.0
        return result

    # ------------------------------------------------------------------
    # Position limits
    # ------------------------------------------------------------------

    def get_position_limits(self, symbols: List[str]) -> Dict[str, float]:
        """Extract max notional limits from exchange market info.

        Returns ``{ticker: max_notional_usd}``.
        """
        if not CCXT_AVAILABLE:
            return {}

        try:
            self._exchange.load_markets()
        except Exception:
            return {}

        result: Dict[str, float] = {}
        for ticker in symbols:
            sym = self._ccxt_symbol(ticker)
            market = self._exchange.markets.get(sym, {})
            limits = market.get("limits", {})
            cost_limits = limits.get("cost", {})
            max_notional = cost_limits.get("max")
            if max_notional is not None:
                result[ticker] = float(max_notional)
            else:
                # Fallback: try info dict for MARKET_LOT_SIZE
                info = market.get("info", {})
                for f in info.get("filters", []):
                    if f.get("filterType") == "MARKET_LOT_SIZE":
                        max_qty = float(f.get("maxQty", 0))
                        price = float(info.get("markPrice", 0) or 0)
                        if max_qty and price:
                            result[ticker] = max_qty * price
                            break
        return result

    # ------------------------------------------------------------------
    # Full market data (main entry point)
    # ------------------------------------------------------------------

    def get_market_data(
        self,
        tickers: List[str],
        lookback_days: int = 400,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch prices, volume, funding rates, and estimated market cap.

        Returns dict with wide DataFrames (date index, ticker columns):
        ``prices``, ``volume``, ``funding_rates``, ``market_cap``.
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (pd.to_datetime(end_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        logger.info(
            "Fetching futures data for %d tickers from %s to %s",
            len(tickers), start_date, end_date,
        )

        prices_data: Dict[str, pd.Series] = {}
        volume_data: Dict[str, pd.Series] = {}
        funding_data: Dict[str, pd.Series] = {}

        for ticker in tickers:
            if ticker.upper() in self.stablecoins:
                continue

            # OHLCV
            df = self.get_ohlcv(ticker, start_date, end_date)
            if df is not None and not df.empty:
                df_idx = df.set_index("date")
                prices_data[ticker] = df_idx["close"]
                volume_data[ticker] = df_idx["volume"]

            # Funding rates
            fr = self.get_funding_rate_history(ticker, start_date, end_date)
            if fr is not None and not fr.empty:
                fr_idx = fr.set_index("date")["funding_rate"]
                funding_data[ticker] = fr_idx

        if not prices_data:
            logger.error("No futures data fetched for any ticker")
            return {
                "prices": pd.DataFrame(),
                "volume": pd.DataFrame(),
                "funding_rates": pd.DataFrame(),
                "market_cap": pd.DataFrame(),
            }

        prices = pd.DataFrame(prices_data)
        volume = pd.DataFrame(volume_data)
        funding_rates = pd.DataFrame(funding_data) if funding_data else pd.DataFrame()

        # Align funding to price index if present
        if not funding_rates.empty:
            funding_rates = funding_rates.reindex(prices.index).ffill().fillna(0.0)

        # Rough market cap estimate (same approach as spot fetcher)
        market_cap = prices * 1e9  # placeholder multiplier

        logger.info(
            "Fetched futures data: %d tickers, %d days", len(prices.columns), len(prices),
        )

        return {
            "prices": prices,
            "volume": volume,
            "funding_rates": funding_rates,
            "market_cap": market_cap,
        }
