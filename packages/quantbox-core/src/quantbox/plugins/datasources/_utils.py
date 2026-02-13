"""Shared utilities for data-source plugins.

Provides:
- OHLCV validation (ported from quantlab's ``basic_validate_ohlcv``)
- Transient-error classification for retry logic (used with ``tenacity``)
- DuckDB-backed Parquet OHLCV cache for incremental fetching
- Market-cap data via CoinGecko (``pycoingecko``) with hardcoded fallback
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# ============================================================================
# OHLCV Validation
# ============================================================================


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate and clean an OHLCV DataFrame.

    Checks:
    1. Required columns present (date, open, high, low, close, volume)
    2. No negative prices
    3. High >= Low consistency
    4. No duplicate dates
    5. Sorted by date ascending

    Returns the cleaned DataFrame. Logs warnings for issues found.
    Raises ``ValueError`` if data is fundamentally broken (missing columns,
    all-zero prices, or empty after cleaning).
    """
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{ticker}] Missing OHLCV columns: {missing}")

    if df.empty:
        raise ValueError(f"[{ticker}] Empty OHLCV DataFrame")

    n_before = len(df)

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

    # Drop rows where prices are zero or negative
    price_cols = ["open", "high", "low", "close"]
    mask_positive = (df[price_cols] > 0).all(axis=1)
    n_neg = (~mask_positive).sum()
    if n_neg > 0:
        logger.warning("[%s] Dropping %d rows with non-positive prices", ticker, n_neg)
        df = df[mask_positive].copy()

    # Fix high < low: swap them
    bad_hl = df["high"] < df["low"]
    n_hl = bad_hl.sum()
    if n_hl > 0:
        logger.warning("[%s] Swapping high/low on %d rows", ticker, n_hl)
        df = df.copy()
        df.loc[bad_hl, ["high", "low"]] = df.loc[bad_hl, ["low", "high"]].values

    # Drop duplicate dates, keep last
    n_dup = df.duplicated(subset=["date"], keep="last").sum()
    if n_dup > 0:
        logger.warning("[%s] Dropping %d duplicate dates", ticker, n_dup)
        df = df.drop_duplicates(subset=["date"], keep="last")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"[{ticker}] No valid rows after cleaning")

    n_after = len(df)
    if n_after < n_before:
        logger.info("[%s] Validation: %d -> %d rows", ticker, n_before, n_after)

    return df


# ============================================================================
# Transient Error Classification (used by tenacity)
# ============================================================================


def is_transient(exc: BaseException) -> bool:
    """Check if an exception is transient and worth retrying.

    Used as ``retry=retry_if_exception(is_transient)`` with tenacity.
    """
    if isinstance(exc, (ConnectionError, TimeoutError, OSError, httpx.TransportError)):
        return True
    # ccxt rate-limit / exchange-not-available errors
    exc_name = type(exc).__name__
    if exc_name in (
        "RateLimitExceeded",
        "ExchangeNotAvailable",
        "RequestTimeout",
        "NetworkError",
        "DDoSProtection",
    ):
        return True
    # Binance API transient codes (-1003 rate limit, -1001 disconnected)
    code = getattr(exc, "code", None)
    return code in (-1003, -1001, -1000, 503, 504)


# Pre-built tenacity retry decorator for data-fetching functions.
retry_transient = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=30),
    retry=retry_if_exception(is_transient),
    reraise=True,
)


# ============================================================================
# DuckDB-Backed OHLCV Cache
# ============================================================================


class OHLCVCache:
    """DuckDB-backed Parquet cache for OHLCV data.

    Stores OHLCV data as Parquet files on disk, queries them via DuckDB for
    incremental fetching (only fetch candles newer than what's cached).

    Parameters
    ----------
    cache_dir : str or Path
        Root directory for cached Parquet files.
    fresh_ttl_hours : float
        If the most recent cached candle for a ticker is younger than this,
        skip re-fetching entirely.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        fresh_ttl_hours: float = 4.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fresh_ttl_hours = fresh_ttl_hours

    def _ticker_dir(self, ticker: str) -> Path:
        return self.cache_dir / "ohlcv" / ticker.upper()

    def get_cached(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """Read cached OHLCV from Parquet via DuckDB.

        Returns DataFrame with columns [date, open, high, low, close, volume]
        or None if no cache exists.
        """
        tdir = self._ticker_dir(ticker)
        if not tdir.exists():
            return None

        glob_pattern = str(tdir / "*.parquet")
        try:
            query = f"""
                SELECT date, open, high, low, close, volume
                FROM read_parquet('{glob_pattern}')
                WHERE date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY date
            """
            df = duckdb.query(query).df()
            if df.empty:
                return None
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as exc:
            logger.debug("Cache read failed for %s: %s", ticker, exc)
            return None

    def get_last_date(self, ticker: str) -> pd.Timestamp | None:
        """Get the most recent cached date for a ticker."""
        tdir = self._ticker_dir(ticker)
        if not tdir.exists():
            return None

        glob_pattern = str(tdir / "*.parquet")
        try:
            query = f"SELECT MAX(date) AS last_date FROM read_parquet('{glob_pattern}')"
            result = duckdb.query(query).df()
            val = result["last_date"].iloc[0]
            if pd.isna(val):
                return None
            return pd.Timestamp(val)
        except Exception:
            return None

    def is_fresh(self, ticker: str, end_date: str) -> bool:
        """Check if cached data is fresh enough to skip re-fetching."""
        last = self.get_last_date(ticker)
        if last is None:
            return False
        target = pd.Timestamp(end_date)
        return (target - last) < timedelta(hours=self.fresh_ttl_hours)

    def store(self, ticker: str, df: pd.DataFrame) -> None:
        """Append new OHLCV data to the cache.

        Deduplicates by date before writing.
        """
        if df.empty:
            return

        tdir = self._ticker_dir(ticker)
        tdir.mkdir(parents=True, exist_ok=True)

        # Ensure date is datetime
        store_df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        store_df["date"] = pd.to_datetime(store_df["date"])

        # Deduplicate against existing cache
        existing = self.get_cached(
            ticker,
            store_df["date"].min().strftime("%Y-%m-%d"),
            store_df["date"].max().strftime("%Y-%m-%d"),
        )
        if existing is not None and not existing.empty:
            existing_dates = set(existing["date"].dt.normalize())
            store_df = store_df[~store_df["date"].dt.normalize().isin(existing_dates)]

        if store_df.empty:
            return

        # Write as a new shard
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = tdir / f"{ts}.parquet"
        store_df.to_parquet(path, engine="pyarrow", index=False)
        logger.debug("Cached %d rows for %s -> %s", len(store_df), ticker, path)

    def clear(self, ticker: str | None = None) -> None:
        """Clear cache for a ticker, or all tickers if None."""
        import shutil

        if ticker:
            tdir = self._ticker_dir(ticker)
            if tdir.exists():
                shutil.rmtree(tdir)
        else:
            ohlcv_dir = self.cache_dir / "ohlcv"
            if ohlcv_dir.exists():
                shutil.rmtree(ohlcv_dir)


# ============================================================================
# Market Cap Data (CoinGecko + hardcoded fallback)
# ============================================================================


class MarketCapProvider:
    """Fetch market cap rankings from CoinGecko (free, no API key).

    Falls back to hardcoded circulating-supply estimates when the API
    is unavailable.

    Parameters
    ----------
    cache_dir : str or Path or None
        Directory for caching responses as Parquet.
    fresh_ttl_hours : float
        Re-use cached rankings if younger than this (default 4h).
    fallback_ttl_hours : float
        Use stale cache up to this age when API fails (default 28h).
    limit : int
        Number of top coins to fetch.
    """

    # Hardcoded fallback circulating supplies
    _FALLBACK_SUPPLY: dict[str, float] = {
        "BTC": 19.6e6,
        "ETH": 120e6,
        "SOL": 440e6,
        "BNB": 150e6,
        "XRP": 55e9,
        "DOGE": 143e9,
        "ADA": 35e9,
        "AVAX": 390e6,
        "LINK": 600e6,
        "DOT": 1.4e9,
        "MATIC": 10e9,
        "SHIB": 589e12,
        "LTC": 74e6,
        "TRX": 89e9,
        "ATOM": 390e6,
        "UNI": 600e6,
        "APT": 470e6,
        "NEAR": 1.1e9,
        "INJ": 93e6,
        "FIL": 530e6,
    }

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        fresh_ttl_hours: float = 4.0,
        fallback_ttl_hours: float = 28.0,
        limit: int = 100,
        # Legacy params (ignored, kept for backwards compat)
        api_key: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fresh_ttl_hours = fresh_ttl_hours
        self.fallback_ttl_hours = fallback_ttl_hours
        self.limit = limit

        if self.cache_dir:
            (self.cache_dir / "market_cap").mkdir(parents=True, exist_ok=True)

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / "market_cap" / "rankings.parquet"

    def _read_cache(self) -> pd.DataFrame | None:
        """Read cached rankings if fresh or usable as fallback."""
        path = self._cache_path()
        if path is None or not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if "fetch_timestamp" not in df.columns:
                return None
            age_hours = (pd.Timestamp.now() - pd.Timestamp(df["fetch_timestamp"].iloc[0])).total_seconds() / 3600
            if age_hours < self.fresh_ttl_hours:
                logger.debug("Market cap cache is fresh (%.1fh old)", age_hours)
                return df
            if age_hours < self.fallback_ttl_hours:
                logger.info("Market cap cache is stale (%.1fh) but within fallback TTL", age_hours)
                return df
            logger.info("Market cap cache expired (%.1fh old)", age_hours)
            return None
        except Exception as exc:
            logger.warning("Failed to read market cap cache: %s", exc)
            return None

    def _write_cache(self, df: pd.DataFrame) -> None:
        path = self._cache_path()
        if path is None:
            return
        try:
            df.to_parquet(path, engine="pyarrow", index=False)
        except Exception as exc:
            logger.warning("Failed to write market cap cache: %s", exc)

    def fetch_rankings(self) -> pd.DataFrame | None:
        """Fetch top coin rankings from CoinGecko API.

        Returns DataFrame with columns: symbol, market_cap, circulating_supply,
        rank, fetch_timestamp.

        Returns None if the API call fails and no usable cache exists.
        """
        # Check cache first
        cached = self._read_cache()
        if cached is not None:
            age_hours = (pd.Timestamp.now() - pd.Timestamp(cached["fetch_timestamp"].iloc[0])).total_seconds() / 3600
            if age_hours < self.fresh_ttl_hours:
                return cached

        # Fetch from CoinGecko (free, no API key needed)
        try:
            from pycoingecko import CoinGeckoAPI

            cg = CoinGeckoAPI()
            coins = cg.get_coins_markets(
                vs_currency="usd",
                order="market_cap_desc",
                per_page=self.limit,
                page=1,
                sparkline=False,
            )

            if not coins:
                logger.warning("CoinGecko API returned empty data")
                return cached

            rows = []
            for coin in coins:
                rows.append(
                    {
                        "symbol": str(coin.get("symbol", "")).upper(),
                        "market_cap": float(coin.get("market_cap", 0) or 0),
                        "circulating_supply": float(coin.get("circulating_supply", 0) or 0),
                        "rank": int(coin.get("market_cap_rank", 0) or 0),
                        "fetch_timestamp": pd.Timestamp.now().isoformat(),
                    }
                )

            df = pd.DataFrame(rows)
            self._write_cache(df)
            logger.info("Fetched CoinGecko rankings for %d coins", len(df))
            return df

        except Exception as exc:
            logger.warning("CoinGecko API call failed: %s", exc)
            return cached  # fall back to stale cache

    def estimate_market_cap(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute market cap DataFrame using CoinGecko data with hardcoded fallback.

        Tries CoinGecko rankings first.  For any ticker not covered,
        falls back to hardcoded circulating supply estimates.

        Parameters
        ----------
        prices : DataFrame
            Wide DataFrame (date index, ticker columns) of close prices.
        volume : DataFrame
            Wide DataFrame of volumes (unused but kept for API compat).

        Returns
        -------
        DataFrame
            Same shape as *prices* with estimated market cap values.
        """
        rankings = self.fetch_rankings()
        supply_map: dict[str, float] = {}

        if rankings is not None and not rankings.empty:
            for _, row in rankings.iterrows():
                sym = str(row["symbol"]).upper()
                cs = float(row.get("circulating_supply", 0) or 0)
                if cs > 0:
                    supply_map[sym] = cs
            logger.info(
                "Using CoinGecko supply data for %d coins, fallback for remainder",
                len(supply_map),
            )

        market_cap = pd.DataFrame(index=prices.index)
        for ticker in prices.columns:
            t_upper = ticker.upper()
            if t_upper in supply_map and supply_map[t_upper] > 0:
                market_cap[ticker] = prices[ticker] * supply_map[t_upper]
            elif rankings is not None and not rankings.empty:
                # Use market_cap directly if supply is 0 but mc exists
                row = rankings[rankings["symbol"] == t_upper]
                if not row.empty:
                    mc_val = float(row["market_cap"].iloc[0])
                    if mc_val > 0:
                        latest_price = prices[ticker].dropna().iloc[-1] if not prices[ticker].dropna().empty else 1.0
                        market_cap[ticker] = (prices[ticker] / latest_price) * mc_val
                        continue

                # Final fallback: hardcoded supply
                supply = self._FALLBACK_SUPPLY.get(t_upper, 1e9)
                market_cap[ticker] = prices[ticker] * supply
                logger.debug("Using fallback supply for %s: %.0f", ticker, supply)
            else:
                supply = self._FALLBACK_SUPPLY.get(t_upper, 1e9)
                market_cap[ticker] = prices[ticker] * supply

        return market_cap


# Backwards compatibility alias
CMCMarketCapProvider = MarketCapProvider
