"""Shared utilities for data-source plugins.

Provides:
- OHLCV validation (ported from quantlab's ``basic_validate_ohlcv``)
- Transient-error classification for retry logic (used with ``tenacity``)
- DuckDB-backed Parquet OHLCV cache for incremental fetching
- Market-cap data via CoinGecko (``pycoingecko``) with hardcoded fallback
"""

from __future__ import annotations

import logging
import os
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
# Data Frequency Normalization
# ============================================================================

FREQUENCY_ALIASES: dict[str, str] = {
    "daily": "1d",
    "day": "1d",
    "d": "1d",
    "1day": "1d",
    "hourly": "1h",
    "hour": "1h",
    "h": "1h",
    "1hour": "1h",
    "4hourly": "4h",
    "4hour": "4h",
    "weekly": "1w",
    "week": "1w",
    "w": "1w",
    "monthly": "1M",
    "month": "1M",
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
}

_VALID_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}


def interval_step(interval: str) -> timedelta:
    """Return the duration of one bar for the given Binance interval string."""
    import re

    m = re.match(r"^(\d+)([mhdwM])$", interval)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        seconds = {"m": 60, "h": 3600, "d": 86400, "w": 604800, "M": 2592000}
        return timedelta(seconds=seconds.get(unit, 86400) * n)
    return timedelta(days=1)


def normalize_data_frequency(frequency: str) -> str:
    """Normalize frequency strings to Binance-compatible interval identifiers.

    Accepts semantic names ("daily", "hourly") and Binance-native intervals
    ("1d", "1h"). Returns the Binance interval string.

    Examples:
        >>> normalize_data_frequency("daily")
        '1d'
        >>> normalize_data_frequency("hourly")
        '1h'
        >>> normalize_data_frequency("4h")
        '4h'
    """
    freq_lower = frequency.lower().strip()
    if freq_lower in _VALID_INTERVALS:
        return freq_lower
    if freq_lower in FREQUENCY_ALIASES:
        return FREQUENCY_ALIASES[freq_lower]
    if frequency == "1M":
        return "1M"
    logger.warning("Unknown data frequency %r, passing through as-is", frequency)
    return frequency


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

    def _ticker_dir(self, ticker: str, interval: str = "1d") -> Path:
        return self.cache_dir / "ohlcv" / ticker.upper() / interval

    def get_cached(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Read cached OHLCV from Parquet via DuckDB.

        Returns DataFrame with columns [date, open, high, low, close, volume]
        or None if no cache exists.
        """
        tdir = self._ticker_dir(ticker, interval)
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

    def get_last_date(self, ticker: str, interval: str = "1d") -> pd.Timestamp | None:
        """Get the most recent cached date for a ticker."""
        tdir = self._ticker_dir(ticker, interval)
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

    def is_fresh(self, ticker: str, end_date: str, interval: str = "1d") -> bool:
        """Check if cached data is fresh enough to skip re-fetching."""
        last = self.get_last_date(ticker, interval)
        if last is None:
            return False
        target = pd.Timestamp(end_date)
        return (target - last) < timedelta(hours=self.fresh_ttl_hours)

    def store(self, ticker: str, df: pd.DataFrame, interval: str = "1d") -> None:
        """Append new OHLCV data to the cache.

        Deduplicates by exact timestamp before writing (handles intraday correctly).
        """
        if df.empty:
            return

        tdir = self._ticker_dir(ticker, interval)
        tdir.mkdir(parents=True, exist_ok=True)

        # Ensure date is datetime
        store_df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        store_df["date"] = pd.to_datetime(store_df["date"])

        # Deduplicate against existing cache using exact timestamps
        existing = self.get_cached(
            ticker,
            store_df["date"].min().strftime("%Y-%m-%d"),
            store_df["date"].max().strftime("%Y-%m-%d"),
            interval=interval,
        )
        if existing is not None and not existing.empty:
            existing_ts = set(existing["date"])
            store_df = store_df[~store_df["date"].isin(existing_ts)]

        if store_df.empty:
            return

        # Write as a new shard
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = tdir / f"{ts}.parquet"
        store_df.to_parquet(path, engine="pyarrow", index=False)
        logger.debug("Cached %d rows for %s -> %s", len(store_df), ticker, path)

    def clear(self, ticker: str | None = None, interval: str | None = None) -> None:
        """Clear cache for a ticker/interval, or all data if both None."""
        import shutil

        if ticker and interval:
            tdir = self._ticker_dir(ticker, interval)
            if tdir.exists():
                shutil.rmtree(tdir)
        elif ticker:
            # Remove all intervals for this ticker
            tdir = self.cache_dir / "ohlcv" / ticker.upper()
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
    """Fetch market cap rankings from CoinGecko (default) or CoinMarketCap.

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
    source : str
        Rankings source: ``"coingecko"`` (default; free, no key) or
        ``"coinmarketcap"``/``"cmc"`` (CMC pro-api, reads
        ``API_KEY_COINMARKETCAP``). CMC mirrors quantlab's universe screen.
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

    # Names that select the CoinMarketCap rankings source (case-insensitive).
    _CMC_ALIASES = frozenset({"cmc", "coinmarketcap", "coin_market_cap"})

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        fresh_ttl_hours: float = 4.0,
        fallback_ttl_hours: float = 28.0,
        limit: int = 250,
        source: str = "coingecko",
        # Legacy params (ignored, kept for backwards compat)
        api_key: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fresh_ttl_hours = fresh_ttl_hours
        self.fallback_ttl_hours = fallback_ttl_hours
        self.limit = limit
        # Rankings source: "coingecko" (default, free, no key) or "coinmarketcap"
        # (CMC pro-api, needs API_KEY_COINMARKETCAP — mirrors quantlab's screen).
        self.source = "coinmarketcap" if str(source).lower() in self._CMC_ALIASES else "coingecko"

        if self.cache_dir:
            (self.cache_dir / "market_cap").mkdir(parents=True, exist_ok=True)

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        # Keep the CoinGecko filename unchanged for back-compat; namespace CMC so
        # the two sources never overwrite each other's cached rankings.
        name = "rankings.parquet" if self.source == "coingecko" else f"rankings_{self.source}.parquet"
        return self.cache_dir / "market_cap" / name

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

        Returns DataFrame with columns: symbol, market_cap, total_volume,
        circulating_supply, rank, fetch_timestamp.

        ``total_volume`` is CoinGecko's market-wide 24h volume (USD, aggregated
        across all tracked pairs and exchanges) — used to screen the universe on
        true market liquidity rather than a single venue/quote-pair book.

        Returns None if the API call fails and no usable cache exists.
        """
        # Check cache first
        cached = self._read_cache()
        if cached is not None:
            age_hours = (pd.Timestamp.now() - pd.Timestamp(cached["fetch_timestamp"].iloc[0])).total_seconds() / 3600
            if age_hours < self.fresh_ttl_hours:
                return cached

        if self.source == "coinmarketcap":
            return self._fetch_cmc_rankings(cached)

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
                        "total_volume": float(coin.get("total_volume", 0) or 0),
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

    def _fetch_cmc_rankings(self, cached: pd.DataFrame | None) -> pd.DataFrame | None:
        """Fetch top coin rankings from CoinMarketCap (pro-api listings/latest).

        Returns the SAME schema as the CoinGecko path (symbol, market_cap,
        total_volume, circulating_supply, rank, fetch_timestamp) so every
        downstream consumer (estimate_market_cap / estimate_aggregate_volume) is
        source-agnostic. ``total_volume`` is CMC's 24h USD volume and
        ``market_cap`` its reported USD market cap — this is the same screen
        quantlab's crypto_trend_catcher uses, so a CMC-sourced book is a true
        mirror of quantlab's universe rather than the CoinGecko default.

        The API key is read from ``API_KEY_COINMARKETCAP`` (the var quantlab
        uses) with ``CMC_API_KEY`` as a fallback. Missing key or any API error
        falls back to the stale cache (then to hardcoded supplies upstream) —
        never raises, so the daily run degrades gracefully instead of aborting.
        """
        api_key = os.environ.get("API_KEY_COINMARKETCAP") or os.environ.get("CMC_API_KEY")
        if not api_key:
            logger.warning(
                "CoinMarketCap source requested but no API_KEY_COINMARKETCAP/CMC_API_KEY "
                "in env; falling back to cached/hardcoded market cap."
            )
            return cached

        try:
            resp = httpx.get(
                "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                headers={"X-CMC_PRO_API_KEY": api_key, "Accepts": "application/json"},
                params={"start": 1, "limit": self.limit, "convert": "USD"},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                logger.warning("CoinMarketCap API returned empty data")
                return cached

            rows = []
            for coin in data:
                quote = (coin.get("quote") or {}).get("USD") or {}
                rows.append(
                    {
                        "symbol": str(coin.get("symbol", "")).upper(),
                        "market_cap": float(quote.get("market_cap", 0) or 0),
                        "total_volume": float(quote.get("volume_24h", 0) or 0),
                        "circulating_supply": float(coin.get("circulating_supply", 0) or 0),
                        "rank": int(coin.get("cmc_rank", 0) or 0),
                        "fetch_timestamp": pd.Timestamp.now().isoformat(),
                    }
                )

            df = pd.DataFrame(rows)
            self._write_cache(df)
            logger.info("Fetched CoinMarketCap rankings for %d coins", len(df))
            return df

        except Exception as exc:
            logger.warning("CoinMarketCap API call failed: %s", exc)
            return cached  # fall back to stale cache

    def estimate_market_cap(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """Best-practice multi-layer, multi-venue market-cap estimate.

        For each ticker a single CURRENT market-cap *anchor* is resolved from
        the highest-quality source available, then projected across history by
        the coin's own price path::

            market_cap[t] = anchor * price[t] / price[latest]

        This pins the cross-sectional RANK (what the universe screen consumes)
        to the most accurate **multi-venue** value while still yielding a time
        series for backtests. Market cap moves slowly relative to price, so a
        price-path projection is a faithful proxy between vendor snapshots.

        Source layers, best → worst (per-layer coverage is logged):

        ====  =================================================  ============
        L1    CoinGecko reported ``market_cap``                  multi-venue
              (global VWAP price × circulating supply)           aggregate
        L2    CoinGecko ``circulating_supply`` × price           supply only
        L3    hardcoded circulating supply × price               off-vendor
        L4    no genuine source → NaN (dropped from screen)     excluded
        ====  =================================================  ============

        L4 emits ``NaN`` rather than fabricating a cap: a ticker with no
        genuine market-cap source is *excluded* from the mcap-ranked universe
        screen instead of being given a fake ``price * 1e9`` cap (which mis-ranked
        high- and low-unit-price coins). The dropped tickers are logged.

        L1 is preferred over the legacy single-venue ``price × supply`` because
        the reported market cap already aggregates price across venues, so the
        rank is not skewed by one exchange's quote. Adding a second vendor is a
        clean extension — merge its rankings into ``mc_map`` / ``cs_map`` ahead
        of the off-vendor fallbacks.

        Parameters
        ----------
        prices : DataFrame
            Wide DataFrame (date index, ticker columns) of close prices.
        volume : DataFrame
            Unused; kept for API compatibility.

        Returns
        -------
        DataFrame
            Same shape as *prices* with estimated market cap values.
        """
        rankings = self.fetch_rankings()
        mc_map: dict[str, float] = {}
        cs_map: dict[str, float] = {}
        if rankings is not None and not rankings.empty:
            for _, row in rankings.iterrows():
                sym = str(row["symbol"]).upper()
                mc = float(row.get("market_cap", 0) or 0)
                cs = float(row.get("circulating_supply", 0) or 0)
                if mc > 0:
                    mc_map[sym] = mc
                if cs > 0:
                    cs_map[sym] = cs

        market_cap = pd.DataFrame(index=prices.index)
        layers = {"L1_reported": 0, "L2_supply": 0, "L3_hardcoded": 0, "L4_dropped": 0}
        for ticker in prices.columns:
            t_upper = ticker.upper()
            col = prices[ticker]
            valid = col.dropna()
            latest_price = float(valid.iloc[-1]) if not valid.empty else 0.0

            if t_upper in mc_map and latest_price > 0:
                # L1: anchor to the multi-venue reported mcap, project by price
                market_cap[ticker] = (col / latest_price) * mc_map[t_upper]
                layers["L1_reported"] += 1
            elif t_upper in cs_map:
                # L2: CoinGecko circulating supply × price
                market_cap[ticker] = col * cs_map[t_upper]
                layers["L2_supply"] += 1
            elif t_upper in self._FALLBACK_SUPPLY:
                # L3: hardcoded circulating supply × price
                market_cap[ticker] = col * self._FALLBACK_SUPPLY[t_upper]
                layers["L3_hardcoded"] += 1
            else:
                # No genuine market-cap source covers this ticker. Emit NaN so it
                # is EXCLUDED from the mcap-ranked universe screen rather than
                # fabricating a cap (the old `price * 1e9` default gave high-unit-
                # price junk fake mega-caps and low-unit-price large-caps fake tiny
                # caps, corrupting any top_by_mcap selection).
                market_cap[ticker] = float("nan")
                layers["L4_dropped"] += 1

        layers_msg = (
            "Market-cap layers — L1 reported(multi-venue): %d, L2 supply: %d, "
            "L3 hardcoded: %d, L4 dropped(NaN, excluded): %d"
        )
        logger.info(
            layers_msg,
            layers["L1_reported"],
            layers["L2_supply"],
            layers["L3_hardcoded"],
            layers["L4_dropped"],
        )
        if layers["L4_dropped"]:
            dropped = [
                t
                for t in prices.columns
                if t.upper() not in mc_map and t.upper() not in cs_map and t.upper() not in self._FALLBACK_SUPPLY
            ]
            logger.warning(
                "Dropped %d uncovered ticker(s) from market-cap ranking (no genuine source — NaN, excluded): %s",
                layers["L4_dropped"],
                ", ".join(sorted(dropped)),
            )
        return market_cap

    def estimate_aggregate_volume(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Market-wide aggregate 24h volume per coin (USD), as a date×ticker frame.

        Sources CoinGecko ``total_volume`` (summed across all tracked pairs and
        exchanges) from the same rankings call as :meth:`estimate_market_cap`,
        so it adds no extra API request. This is the market-wide liquidity used
        to SCREEN the universe, as opposed to a single venue/quote-pair book.

        Like :meth:`estimate_market_cap`, the value is a current snapshot
        broadcast across the price index — CoinGecko's free endpoint is
        point-in-time. For the daily live screen this is exactly today's market
        state; for historical backtests prefer a dataset with a true aggregate-
        volume time series. Coins not covered by the rankings are left ``NaN`` so
        the caller (:func:`select_universe`) falls back to per-venue volume.

        Parameters
        ----------
        prices : DataFrame
            Wide DataFrame (date index, ticker columns) of close prices.

        Returns
        -------
        DataFrame
            Same shape as *prices*; each column is the coin's market-wide 24h
            volume broadcast across the index, or ``NaN`` if not covered.
        """
        rankings = self.fetch_rankings()
        vol_map: dict[str, float] = {}
        if rankings is not None and not rankings.empty and "total_volume" in rankings.columns:
            for _, row in rankings.iterrows():
                sym = str(row["symbol"]).upper()
                tv = float(row.get("total_volume", 0) or 0)
                if tv > 0:
                    vol_map[sym] = tv
            logger.info("Using CoinGecko aggregate volume for %d coins", len(vol_map))

        agg = pd.DataFrame(index=prices.index)
        for ticker in prices.columns:
            tv = vol_map.get(ticker.upper())
            agg[ticker] = float(tv) if tv else float("nan")
        return agg


# Backwards compatibility alias
CMCMarketCapProvider = MarketCapProvider


# ============================================================================
# Point-in-time screen inputs for BACKTEST (no look-ahead)
# ============================================================================
#
# The live data plugins (Hyperliquid, Binance) feed a two-stage universe screen
# (``select_universe``): Stage 1 ranks by market cap, Stage 2 by volume. For
# LIVE trading the screen inputs come from a CoinGecko *snapshot* — today's
# market cap and today's market-wide 24h volume — which is correct: "today" is
# the point of decision. In a BACKTEST that same snapshot, broadcast onto every
# historical row, is look-ahead + survivorship bias — a coin's PAST universe
# membership would be decided by its PRESENT size/liquidity.
#
# These helpers provide the point-in-time backtest replacement:
#   - market cap: curated daily PIT series from quantbox-datasets
#     (``crypto-spot-daily/market_cap.parquet``), carried forward causally.
#   - volume rank: NOT sourced here. The backtest Stage-2 rank uses the
#     per-venue point-in-time dollar volume the plugin already returns
#     (``select_universe`` ranks on it when ``screen_volume`` is empty). The
#     curated market-wide ``cmc_volume_usd`` series is monthly and ends mid-2025,
#     so for recent backtests it would forward-fill a many-month-stale value;
#     fresh per-venue PIT volume is the faithful liquidity record for a single-
#     venue book replica. See the fix report for the full tradeoff.

_CURATED_CRYPTO_DATASET = "crypto-spot-daily"


def load_pit_market_cap(prices: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time daily market cap aligned to *prices*, for backtests.

    Reads the curated, survivorship-augmented daily market-cap series from the
    optional ``quantbox-datasets`` package
    (``crypto-spot-daily/market_cap.parquet``) and aligns it to *prices* (date
    index x ticker columns). Reindexing uses a forward fill, which is **causal**
    — every date carries only the most recent *past* market-cap observation, so
    there is no look-ahead.

    Hyperliquid quotes some high-supply tokens with a ``k`` (1000x) prefix
    (``kPEPE``, ``kBONK`` ...). Market cap is notation-independent (it is the
    coin's total cap), so these map to the unprefixed base symbol.

    Returns an empty DataFrame when ``quantbox-datasets`` is not installed or the
    file is missing — callers then skip the market-cap tier and rank on
    point-in-time per-venue volume only (clean, no look-ahead). Tickers not
    covered by the curated dataset are excluded from the market-cap tier; this
    is logged.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()
    try:
        from quantbox_datasets.builtins import crypto_spot_daily
    except ImportError:
        logger.debug("quantbox-datasets not installed; skipping PIT market-cap tier in backtest")
        return pd.DataFrame()
    try:
        ds_path = Path(crypto_spot_daily()._ds_path())
        mc_path = ds_path / "market_cap.parquet"
        if not mc_path.exists():
            logger.debug("curated market_cap.parquet not found at %s; skipping PIT mcap tier", mc_path)
            return pd.DataFrame()
        cur = pd.read_parquet(mc_path)
    except Exception as exc:  # noqa: BLE001 — never break data loading on a soft dep
        logger.warning("Failed to read curated PIT market cap: %s; skipping mcap tier", exc)
        return pd.DataFrame()

    cur.index = pd.DatetimeIndex(cur.index)
    # Align tz to the price index (curated is tz-naive UTC dates).
    if prices.index.tz is not None and cur.index.tz is None:
        cur.index = cur.index.tz_localize("UTC")
    elif prices.index.tz is None and cur.index.tz is not None:
        cur.index = cur.index.tz_localize(None)
    cur = cur.sort_index()

    cur_cols = set(cur.columns)
    selected: dict[str, pd.Series] = {}
    missing: list[str] = []
    for t in prices.columns:
        src = None
        if t in cur_cols:
            src = t
        elif t.upper() in cur_cols:
            src = t.upper()
        elif len(t) > 1 and t[0] == "k" and t[1:].upper() in cur_cols:
            src = t[1:].upper()  # Hyperliquid 1000x notation -> base symbol
        if src is None:
            missing.append(t)
        else:
            selected[t] = cur[src]

    if not selected:
        logger.info(
            "PIT market cap: no curated coverage for any of %d tickers; skipping mcap tier", len(prices.columns)
        )
        return pd.DataFrame()
    if missing:
        logger.info(
            "PIT market cap: %d/%d tickers covered by curated dataset; uncovered (excluded from the mcap tier): %s",
            len(selected),
            len(prices.columns),
            missing,
        )
    mc = pd.DataFrame(selected)
    # Causal forward-fill onto the price index: each date gets the latest mcap
    # observation at or before it (and carries the last known value across the
    # short tail beyond the curated file's end). reindex(method="ffill") never
    # uses a future observation, so this is look-ahead-free.
    mc = mc.reindex(prices.index, method="ffill")
    return mc


def resolve_screen_inputs(
    mode: str | None,
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    provider: MarketCapProvider | None = None,
    screen_volume_source: str = "market",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve ``(market_cap, screen_volume)`` for the universe screen, mode-aware.

    LIVE / paper
        The CoinGecko snapshot *is* the point of decision ("today"), so use it:
        ``market_cap = provider.estimate_market_cap`` and
        ``screen_volume = provider.estimate_aggregate_volume`` (the market-wide,
        cross-exchange 24h volume that powers the live liquidity screen).

        ``screen_volume_source`` (default ``"market"``) selects the Stage-2
        liquidity ranker in live/paper — OPT-IN; the default is unchanged:

        - ``"market"`` (default) — market-wide aggregate volume. Right for an
          index/mirror book screened against a cross-venue reference.
        - ``"venue"`` — return an EMPTY ``screen_volume`` so :func:`select_universe`
          ranks Stage-2 on the plugin's per-venue dollar volume instead. Right for
          a single-venue *execution* book (e.g. a live Kraken-USD spot book), which
          must only hold names actually fillable on that venue. Market cap (Stage-1)
          is unaffected — still the genuine cross-venue snapshot.

    BACKTEST (and any non-live mode, including the unset default)
        The snapshot would be look-ahead. Source point-in-time market cap from
        the curated dataset (:func:`load_pit_market_cap`) and return an EMPTY
        ``screen_volume`` so :func:`select_universe` ranks Stage 2 on the
        per-venue, point-in-time dollar volume the plugin already returns. The
        flat today-anchored snapshot is NEVER used in a backtest.

    The default for an unset/unknown mode is the backtest (point-in-time) path —
    the conservative choice that can never silently introduce look-ahead.
    """
    empty = pd.DataFrame()
    if prices is None or prices.empty:
        return empty, empty
    if str(mode).lower() in ("live", "paper"):
        prov = provider or MarketCapProvider()
        market_cap = prov.estimate_market_cap(prices, volume)
        # OPT-IN venue-liquidity screen: empty screen_volume -> select_universe
        # ranks Stage-2 on per-venue dollar volume. Default keeps market-wide.
        if str(screen_volume_source).lower() == "venue":
            return market_cap, empty
        screen_volume = prov.estimate_aggregate_volume(prices)
        return market_cap, screen_volume
    # backtest / default: point-in-time market cap, per-venue volume for Stage 2.
    return load_pit_market_cap(prices), empty
