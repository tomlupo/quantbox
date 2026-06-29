"""
Kraken Market Data Fetcher - Live spot market data via ccxt.

Mirror of :class:`BinanceDataFetcher` for the Kraken spot venue. The universe
methodology is identical (top-N-by-mcap → dollar-volume filter); only the venue
plumbing differs:

- OHLCV / tradable-pair / 24h-volume data comes from **ccxt.kraken** (Kraken's
  public REST endpoints under the hood), which transparently handles Kraken's
  legacy asset codes (``XXBT``/``ZUSD``) and HMAC-SHA512 nonce signing for the
  private side. Public data needs no API key.
- ``quote_asset`` defaults to **USD** — on Kraken the deep native books for the
  large caps are ``*/USD``; USDC books are thin below the top few names (see the
  Kraken adapter analysis, Part B.6).

## 720-candle OHLC cap (IMPORTANT)

Kraken's public ``/0/public/OHLC`` (ccxt ``fetch_ohlcv``) only returns the **720
most-recent** candles per pair; older history is **not** retrievable. This is
fine for a live/paper daily-rebalance lookback (≈365 daily candles) but means
**backtests must use cached quantbox-datasets, never live Kraken OHLC**. The
fetcher logs a warning when a requested lookback approaches the cap.

## LLM Usage Guide

```python
from quantbox.plugins.datasources import KrakenDataFetcher

fetcher = KrakenDataFetcher()  # quote_asset="USD" by default
data = fetcher.get_market_data(["BTC", "ETH", "SOL"], lookback_days=365)
# Returns {"prices", "volume", "market_cap", "screen_volume"} DataFrames
```
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from quantbox.plugins.datasources._utils import (
    MarketCapProvider,
    OHLCVCache,
    interval_step,
    resolve_screen_inputs,
    retry_transient,
    validate_ohlcv,
)
from quantbox.plugins.strategies._universe import DEFAULT_STABLECOINS

logger = logging.getLogger(__name__)

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:  # pragma: no cover
    ccxt = None
    CCXT_AVAILABLE = False
    logger.warning("ccxt not installed - Kraken OHLCV fetching will be unavailable")


# ============================================================================
# Constants
# ============================================================================

DEFAULT_REQUEST_DELAY_MS = 600  # throttle the per-pair OHLC backfill loop
DEFAULT_MAX_RETRIES = 3

# Kraken's public OHLC endpoint returns at most this many recent candles per
# pair; older history is not retrievable. Backtests must use cached datasets.
KRAKEN_OHLC_MAX_CANDLES = 720

# Kraken legacy asset codes -> canonical ticker. ccxt already normalises most of
# these, but Balance/raw payloads can still surface legacy codes, so normalise
# defensively. ``X``/``Z`` are Kraken's old crypto/fiat namespace prefixes.
KRAKEN_ASSET_ALIASES = {
    "XBT": "BTC",
    "XXBT": "BTC",
    "XDG": "DOGE",
    "XXDG": "DOGE",
    "XETH": "ETH",
    "XETC": "ETC",
    "XLTC": "LTC",
    "XMLN": "MLN",
    "XREP": "REP",
    "XXLM": "XLM",
    "XXMR": "XMR",
    "XXRP": "XRP",
    "XZEC": "ZEC",
    "ZUSD": "USD",
    "ZEUR": "EUR",
    "ZGBP": "GBP",
    "ZCAD": "CAD",
    "ZJPY": "JPY",
    "ZAUD": "AUD",
}

# Earn/staking balance suffixes (e.g. ``DOT.S``, ``ETH.F``) — these are NOT the
# spot tradable balance and must be filtered out / folded into the spot asset.
KRAKEN_BALANCE_SUFFIXES = (".S", ".F", ".B", ".M", ".P")


def normalize_kraken_asset(code: str) -> str:
    """Normalise a Kraken asset code to a canonical ticker.

    Strips earn/staking suffixes (``.S``/``.F``/...) and maps legacy
    namespaced codes (``XXBT`` -> ``BTC``, ``ZUSD`` -> ``USD``). Idempotent for
    already-canonical codes (``BTC`` -> ``BTC``, ``USDC`` -> ``USDC``).
    """
    if not code:
        return code
    c = str(code).upper()
    for suffix in KRAKEN_BALANCE_SUFFIXES:
        if c.endswith(suffix):
            c = c[: -len(suffix)]
            break
    if c in KRAKEN_ASSET_ALIASES:
        return KRAKEN_ASSET_ALIASES[c]
    # Legacy single-prefix forms not in the table (e.g. a 4-char X-prefixed
    # crypto): drop a leading X/Z when the remainder is a plausible ticker.
    if len(c) >= 4 and c[0] in ("X", "Z") and c[1:] in KRAKEN_ASSET_ALIASES.values():
        return c[1:]
    return c


# ============================================================================
# Kraken Data Fetcher
# ============================================================================


@dataclass
class KrakenDataFetcher:
    """Production Kraken spot market-data fetcher (ccxt-backed).

    Public data only — no API key required. Shape and methods mirror
    :class:`BinanceDataFetcher` so the universe / strategy pipeline is venue
    agnostic.
    """

    # Configuration
    quote_asset: str = "USD"
    stablecoins: list[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())

    # Rate limiting
    request_delay_ms: int = DEFAULT_REQUEST_DELAY_MS
    max_retries: int = DEFAULT_MAX_RETRIES

    # Caching
    cache_ttl_seconds: int = 60
    cache_dir: str | None = None
    cache_fresh_ttl_hours: float = 4.0
    _ohlcv_cache: OHLCVCache | None = field(default=None, repr=False)

    # Market-cap rankings source: "coingecko" (default) or "coinmarketcap"/"cmc".
    mcap_source: str = "coingecko"
    cmc_api_key: str | None = None
    _cmc_provider: MarketCapProvider | None = field(default=None, repr=False)

    # CCXT exchange instance (injectable for tests)
    _exchange: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if CCXT_AVAILABLE and self._exchange is None:
            self._exchange = ccxt.kraken(
                {
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )

        if self.cache_dir and self._ohlcv_cache is None:
            self._ohlcv_cache = OHLCVCache(
                cache_dir=self.cache_dir,
                fresh_ttl_hours=self.cache_fresh_ttl_hours,
            )

        if self._cmc_provider is None:
            self._cmc_provider = MarketCapProvider(
                api_key=self.cmc_api_key,
                cache_dir=self.cache_dir,
                fresh_ttl_hours=self.cache_fresh_ttl_hours,
                source=self.mcap_source,
            )

    def describe(self) -> dict[str, Any]:
        """Describe fetcher capabilities for LLM introspection."""
        return {
            "purpose": "Fetch live and historical spot market data from Kraken (ccxt)",
            "api_key_required": False,
            "quote_asset": self.quote_asset,
            "capabilities": {
                "current_prices": "Real-time prices for any Kraken spot pair",
                "historical_ohlcv": f"Daily OHLCV (capped {KRAKEN_OHLC_MAX_CANDLES} most-recent candles)",
                "24h_stats": "24-hour quote volume for the dollar-volume universe screen",
                "market_cap": "Market-cap rankings via CoinGecko/CMC (not on Kraken)",
            },
            "ccxt_available": CCXT_AVAILABLE,
            "ohlc_candle_cap": KRAKEN_OHLC_MAX_CANDLES,
        }

    def _rate_limit(self) -> None:
        time.sleep(self.request_delay_ms / 1000)

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_tradable_tickers(self, min_volume_usd: float = 1e6) -> list[str]:
        """Tradable base tickers with sufficient 24h quote volume.

        Uses ccxt ``fetch_tickers`` (Kraken public Ticker), filters to pairs
        quoted in ``quote_asset``, drops stablecoin bases, and returns base
        symbols ordered by 24h quote-volume DESCENDING so a downstream
        ``top_n`` truncation keeps the most-liquid pairs.
        """
        if not CCXT_AVAILABLE or self._exchange is None:
            logger.error("ccxt not installed - cannot fetch Kraken tradable tickers")
            return []

        try:
            tickers = self._exchange.fetch_tickers()
        except Exception as e:  # pragma: no cover - network
            logger.error(f"Failed to fetch Kraken tickers: {e}")
            return []

        quote = self.quote_asset.upper()
        rows: list[tuple[str, float]] = []
        for symbol, t in tickers.items():
            # ccxt symbols are "BASE/QUOTE" (e.g. "BTC/USD"); skip synthetic /
            # non-spot or wrong-quote markets.
            if "/" not in symbol:
                continue
            base, _, sym_quote = symbol.partition("/")
            if sym_quote.upper() != quote:
                continue
            base = normalize_kraken_asset(base)
            if base in self.stablecoins:
                continue

            quote_vol = t.get("quoteVolume")
            if quote_vol is None:
                # Kraken ticker may omit quoteVolume; derive base_vol * last.
                base_vol = t.get("baseVolume") or 0.0
                last = t.get("last") or t.get("close") or 0.0
                quote_vol = float(base_vol) * float(last)
            quote_vol = float(quote_vol or 0.0)

            if quote_vol >= min_volume_usd:
                rows.append((base, quote_vol))

        rows.sort(key=lambda x: x[1], reverse=True)
        # Dedup preserving order (multiple legacy codes can map to one base).
        seen: set[str] = set()
        out: list[str] = []
        for base, _ in rows:
            if base not in seen:
                seen.add(base)
                out.append(base)
        return out

    def get_mcap_ranked_candidates(
        self,
        top_n: int,
        min_volume_usd: float = 1e6,
        not_tradable: list[str] | None = None,
    ) -> list[str] | None:
        """Candidate universe ranked by genuine market cap (CMC), Kraken-tradable.

        Identical methodology to :meth:`BinanceDataFetcher.get_mcap_ranked_candidates`
        (see that docstring): take the CMC market-cap ranking, drop the
        ``not_tradable`` opt-out symbols and stablecoins, cut to the first
        ``top_n`` by rank, then intersect with the Kraken-tradable set —
        preserving CMC rank order. Returns ``None`` if no CMC ranking is
        available so the caller can fall back to volume ordering.
        """
        not_tradable_set = {s.upper() for s in (not_tradable or [])}
        rankings = self._cmc_provider.fetch_rankings()
        if rankings is None or rankings.empty or "symbol" not in rankings.columns:
            logger.warning(
                "No CMC ranking available for Kraken candidate selection; caller should fall back to volume ordering."
            )
            return None

        tradable = set(self.get_tradable_tickers(min_volume_usd=min_volume_usd))
        if not tradable:
            logger.warning("No tradable Kraken tickers; cannot build mcap candidates.")
            return None

        ranked = rankings.copy()
        if "rank" in ranked.columns and ranked["rank"].gt(0).any():
            ranked = ranked.sort_values("rank", kind="stable")
        elif "market_cap" in ranked.columns:
            ranked = ranked.sort_values("market_cap", ascending=False, kind="stable")

        seen: set[str] = set()
        ranked_syms: list[str] = []
        for sym in ranked["symbol"].astype(str).str.upper():
            if sym in seen or sym in self.stablecoins or sym in not_tradable_set:
                continue
            seen.add(sym)
            ranked_syms.append(sym)

        top_by_rank = ranked_syms[: int(top_n)]
        candidates = [s for s in top_by_rank if s in tradable]
        logger.info(
            "Built CMC-mcap-ranked Kraken candidate universe: %d symbols "
            "(first top_n=%d by CMC rank after dropping %d not_tradable, then "
            "intersected with %d Kraken-tradable pairs).",
            len(candidates),
            int(top_n),
            len(not_tradable_set),
            len(tradable),
        )
        return candidates

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Fetch OHLCV for a single ticker with caching and validation.

        Honours the DuckDB Parquet cache for incremental fetching when
        ``cache_dir`` is set. Note the Kraken 720-candle cap: requesting a
        lookback longer than that will silently return only the most-recent
        720 candles from the live endpoint.
        """
        if not CCXT_AVAILABLE or self._exchange is None:
            logger.error("ccxt not installed - cannot fetch Kraken OHLCV")
            return None

        # --- Complete cache hit? ---
        if self._ohlcv_cache is not None:
            cached = self._ohlcv_cache.get_cached(ticker, start_date, end_date, interval=interval)
            if cached is not None and not cached.empty:
                last_cached = cached["date"].max()
                target_end = pd.Timestamp(end_date)
                if last_cached >= target_end - timedelta(hours=self.cache_fresh_ttl_hours * 24):
                    cached["ticker"] = ticker
                    try:
                        cached = validate_ohlcv(cached, ticker)
                    except ValueError as exc:
                        logger.warning("Cached OHLCV invalid for %s: %s", ticker, exc)
                    else:
                        cached["ticker"] = ticker
                        return cached.reset_index(drop=True)

        # Determine fetch start (incremental from cache).
        fetch_start = start_date
        if self._ohlcv_cache is not None:
            last_cached_date = self._ohlcv_cache.get_last_date(ticker, interval=interval)
            if last_cached_date is not None:
                next_bar = (last_cached_date + interval_step(interval)).strftime("%Y-%m-%d %H:%M:%S")
                if next_bar > start_date:
                    fetch_start = next_bar

        ccxt_symbol = f"{ticker}/{self.quote_asset}"

        try:
            df = self._fetch_ohlcv_with_retry(ccxt_symbol, fetch_start, end_date, interval)

            if df is not None and not df.empty and self._ohlcv_cache is not None:
                self._ohlcv_cache.store(ticker, df, interval=interval)

            if self._ohlcv_cache is not None:
                full = self._ohlcv_cache.get_cached(ticker, start_date, end_date, interval=interval)
                if full is not None and not full.empty:
                    df = full

            if df is None or df.empty:
                return None

            try:
                df = validate_ohlcv(df, ticker)
            except ValueError as exc:
                logger.error("OHLCV validation failed for %s: %s", ticker, exc)
                return None

            df["ticker"] = ticker
            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error fetching Kraken OHLCV for {ticker}: {e}")
            if self._ohlcv_cache is not None:
                cached = self._ohlcv_cache.get_cached(ticker, start_date, end_date, interval=interval)
                if cached is not None and not cached.empty:
                    logger.info("Serving stale cache for %s after fetch failure", ticker)
                    cached["ticker"] = ticker
                    return cached.reset_index(drop=True)
            return None

    @retry_transient
    def _fetch_ohlcv_with_retry(
        self,
        ccxt_symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV from Kraken with retry on transient errors.

        Kraken returns at most :data:`KRAKEN_OHLC_MAX_CANDLES` recent candles
        per pair regardless of ``since`` — pagination only helps within that
        window. We warn when the requested span clearly exceeds the cap.
        """
        since = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        if since >= end_ts:
            return None

        step = interval_step(interval)
        requested = (pd.Timestamp(end_date) - pd.Timestamp(start_date)) / step
        if requested > KRAKEN_OHLC_MAX_CANDLES:
            logger.warning(
                "Requested %.0f %s candles for %s exceeds Kraken's %d-candle OHLC cap; "
                "only the most-recent ~%d will be returned. Use cached datasets for backtests.",
                requested,
                interval,
                ccxt_symbol,
                KRAKEN_OHLC_MAX_CANDLES,
                KRAKEN_OHLC_MAX_CANDLES,
            )

        all_ohlcv: list[list[float]] = []
        limit = KRAKEN_OHLC_MAX_CANDLES

        while since < end_ts:
            self._rate_limit()
            ohlcv = self._exchange.fetch_ohlcv(
                ccxt_symbol,
                timeframe=interval,
                since=since,
                limit=limit,
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
            all_ohlcv,
            columns=["date", "open", "high", "low", "close", "volume"],
        )
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        df = df.drop_duplicates(subset="date")
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        return df.reset_index(drop=True)

    def get_market_data(
        self,
        tickers: list[str],
        lookback_days: int = 400,
        end_date: str | None = None,
        interval: str = "1d",
        mode: str | None = None,
        screen_volume_source: str = "market",
    ) -> dict[str, pd.DataFrame]:
        """Full market data for strategies (prices, volume, market_cap, screen_volume).

        Mirrors :meth:`BinanceDataFetcher.get_market_data`. The universe-screen
        inputs are resolved mode-aware via :func:`resolve_screen_inputs`
        (CoinGecko snapshot in live/paper; point-in-time in backtest).
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        start_date = (pd.to_datetime(end_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        logger.info(f"Fetching Kraken market data for {len(tickers)} tickers from {start_date} to {end_date}")

        prices_data: dict[str, pd.Series] = {}
        volume_data: dict[str, pd.Series] = {}

        for ticker in tickers:
            if ticker.upper() in self.stablecoins:
                logger.debug(f"Skipping stablecoin: {ticker}")
                continue
            df = self.get_ohlcv(ticker, start_date, end_date, interval=interval)
            if df is not None and not df.empty:
                df = df.set_index("date")
                prices_data[ticker] = df["close"]
                volume_data[ticker] = df["volume"]
            else:
                logger.warning(f"No data for {ticker}")

        if not prices_data:
            logger.error("No data fetched for any ticker")
            return {
                "prices": pd.DataFrame(),
                "volume": pd.DataFrame(),
                "market_cap": pd.DataFrame(),
                "screen_volume": pd.DataFrame(),
            }

        prices = pd.DataFrame(prices_data)
        volume = pd.DataFrame(volume_data)
        market_cap, screen_volume = resolve_screen_inputs(
            mode, prices, volume, self._cmc_provider, screen_volume_source=screen_volume_source
        )

        logger.info(f"Fetched Kraken data for {len(prices.columns)} tickers, {len(prices)} days")
        return {
            "prices": prices,
            "volume": volume,
            "market_cap": market_cap,
            "screen_volume": screen_volume,
        }

    def _estimate_market_cap(self, prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """Estimate market cap via CoinGecko/CMC (Kraken has no mcap endpoint)."""
        return self._cmc_provider.estimate_market_cap(prices, volume)
