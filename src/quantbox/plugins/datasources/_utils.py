"""Shared utilities for data-source plugins.

Provides:
- OHLCV validation (ported from quantlab's ``basic_validate_ohlcv``)
- Transient-error classification for retry logic (re-exported from :mod:`quantbox.retry`)
- DuckDB-backed Parquet OHLCV cache for incremental fetching (duckdb: the ``[data]`` extra)
- The mode-aware universe-screen inputs (:func:`resolve_screen_inputs`)

Market cap moved to :mod:`quantbox.market_cap` (TOM-1449). Since 0.13.0,
importing ``MarketCapProvider``, ``CMCMarketCapProvider`` or ``load_pit_market_cap``
from here raises ``ImportError`` naming the new path (TOM-1457).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from quantbox._lazy import load
from quantbox._removed import removed_names
from quantbox.frequency import FREQUENCY_ALIASES
from quantbox.market_cap import MarketCapProvider as _MarketCapProvider
from quantbox.market_cap import load_pit_market_cap as _load_pit_market_cap

# Transient-error retry lives in the central quantbox.retry module now; keep the
# historical names importable here (is_transient / retry_transient) so the
# data-source plugins that ``from ._utils import ...`` them are unchanged.
from quantbox.retry import is_transient, retry_transient, with_retry  # noqa: F401

logger = logging.getLogger(__name__)

# The old market-cap names raise ImportError naming quantbox.market_cap (TOM-1457).
__getattr__ = removed_names(__name__)


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

# FREQUENCY_ALIASES lives in quantbox.frequency (imported above) — the pipelines'
# annualisation resolver reads the same table (TOM-1338).

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
# Transient Error Classification + retry decorator
# ============================================================================
#
# ``is_transient`` and the ``retry_transient`` decorator now live in
# ``quantbox.retry`` (re-imported at the top of this module for back-compat).
# They are the same 4-attempt, exponential-backoff, transient-only contract as
# before, shared with the broker plugins.


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
        duckdb = load("duckdb", extra="data")  # outside the try: a missing extra is not "no cache"
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
        duckdb = load("duckdb", extra="data")  # outside the try: a missing extra is not "no cache"
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
# Screen inputs, mode-aware (market cap itself lives in quantbox.market_cap)
# ============================================================================


def resolve_screen_inputs(
    mode: str | None,
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    provider: _MarketCapProvider | None = None,
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
    # Fail loud on an unknown source: a typo must NOT silently degrade a live
    # venue book to the market-wide screen. Validated regardless of mode.
    src = str(screen_volume_source).lower()
    if src not in ("market", "venue"):
        raise ValueError(f"screen_volume_source must be 'market' or 'venue', got {screen_volume_source!r}")
    empty = pd.DataFrame()
    if prices is None or prices.empty:
        return empty, empty
    if str(mode).lower() in ("live", "paper"):
        prov = provider or _MarketCapProvider()
        market_cap = prov.estimate_market_cap(prices, volume)
        # OPT-IN venue-liquidity screen: empty screen_volume -> select_universe
        # ranks Stage-2 on per-venue dollar volume. Default keeps market-wide.
        if src == "venue":
            return market_cap, empty
        screen_volume = prov.estimate_aggregate_volume(prices)
        return market_cap, screen_volume
    # backtest / default: point-in-time market cap, per-venue volume for Stage 2.
    return _load_pit_market_cap(prices), empty
