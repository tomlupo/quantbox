"""Market cap for the universe screen: the live provider, the backtest series, one symbol mapping (TOM-1449).

- :class:`MarketCapProvider` — LIVE / paper. A CoinGecko (default) or
  CoinMarketCap snapshot of today's market cap and market-wide volume.
- :func:`load_pit_market_cap` — BACKTEST. The curated point-in-time daily
  series from ``quantbox-datasets``, carried forward causally.
- :func:`map_symbol` — the ONE mapping from a venue ticker to the symbol a
  market-cap source lists. Both paths above use it, so a Hyperliquid ``kPEPE``
  is ``PEPE`` live and in a backtest alike (TOM-1419).

These lived in ``quantbox.plugins.datasources._utils``; the old names there are
deprecation shims that resolve to the objects here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Container
from pathlib import Path

import httpx
import pandas as pd

from quantbox.parquet_io import read_parquet

__all__ = ["HL_K_PREFIX_UNITS", "MarketCapProvider", "load_pit_market_cap", "map_symbol"]

logger = logging.getLogger(__name__)


# ============================================================================
# Symbol mapping (shared by the live and the backtest path)
# ============================================================================

#: Base-coin units in one unit of a Hyperliquid ``k``-prefixed ticker:
#: ``kPEPE`` quotes 1000 PEPE.
HL_K_PREFIX_UNITS = 1000.0


def map_symbol(ticker: str, known: Container[str]) -> tuple[str, float] | None:
    """The symbol under which *known* lists *ticker*, and the base-coin units in one *ticker* unit.

    Tried in order: *ticker* itself, its upper case, then the Hyperliquid
    ``k`` (1000x) notation: ``kPEPE`` maps to ``PEPE`` with
    :data:`HL_K_PREFIX_UNITS` units. A market cap or a USD volume is the
    coin's own total, so it needs no scaling; a per-unit price times a
    circulating supply does (divide by the units). Returns ``None`` when
    *known* lists none of them.
    """
    if ticker in known:
        return ticker, 1.0
    upper = ticker.upper()
    if upper in known:
        return upper, 1.0
    if len(ticker) > 1 and ticker[0] == "k":
        base = ticker[1:].upper()
        if base in known:
            return base, HL_K_PREFIX_UNITS
    return None


# ============================================================================
# Market Cap Data (CoinGecko + hardcoded fallback) — LIVE / paper
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
        strict: bool = False,
        # Legacy params (ignored, kept for backwards compat)
        api_key: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fresh_ttl_hours = fresh_ttl_hours
        self.fallback_ttl_hours = fallback_ttl_hours
        self.limit = limit
        # Fail-closed (OPT-IN): when True and source is CoinMarketCap, a missing
        # key / empty / failed CMC fetch RAISES instead of silently degrading to
        # stale cache or hardcoded supplies. For a funded single-venue book whose
        # universe ranks on genuine CMC mcap, a degraded screen must never trade.
        # Default False keeps the graceful-degrade behaviour (paper/mirror books).
        self.strict = bool(strict)
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
            df = read_parquet(path)
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
            if self.strict:
                raise RuntimeError(
                    "strict CMC mcap: no API_KEY_COINMARKETCAP/CMC_API_KEY in env — refusing "
                    "to trade a degraded (cached/hardcoded) universe."
                )
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
                if self.strict:
                    raise RuntimeError("strict CMC mcap: CoinMarketCap returned empty data.")
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

        except RuntimeError:
            raise  # strict fail-closed already raised above — propagate
        except Exception as exc:
            if self.strict:
                raise RuntimeError(f"strict CMC mcap: CoinMarketCap fetch failed: {exc!r}") from exc
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

        Each ticker is matched to a rankings symbol by :func:`map_symbol`, the
        same mapping :func:`load_pit_market_cap` uses: a Hyperliquid ``kPEPE``
        takes PEPE's cap (TOM-1419). In L2/L3 its 1000x price is divided by
        :data:`HL_K_PREFIX_UNITS` before it meets PEPE's supply.

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
        dropped: list[str] = []
        for ticker in prices.columns:
            col = prices[ticker]
            valid = col.dropna()
            latest_price = float(valid.iloc[-1]) if not valid.empty else 0.0
            reported = map_symbol(ticker, mc_map)
            supply = map_symbol(ticker, cs_map)
            hardcoded = map_symbol(ticker, self._FALLBACK_SUPPLY)

            if reported is not None and latest_price > 0:
                # L1: anchor to the multi-venue reported mcap, project by price
                market_cap[ticker] = (col / latest_price) * mc_map[reported[0]]
                layers["L1_reported"] += 1
            elif supply is not None:
                # L2: CoinGecko circulating supply × price (per base-coin unit)
                market_cap[ticker] = (col / supply[1]) * cs_map[supply[0]]
                layers["L2_supply"] += 1
            elif hardcoded is not None:
                # L3: hardcoded circulating supply × price (per base-coin unit)
                market_cap[ticker] = (col / hardcoded[1]) * self._FALLBACK_SUPPLY[hardcoded[0]]
                layers["L3_hardcoded"] += 1
            else:
                # No genuine market-cap source covers this ticker. Emit NaN so it
                # is EXCLUDED from the mcap-ranked universe screen rather than
                # fabricating a cap (the old `price * 1e9` default gave high-unit-
                # price junk fake mega-caps and low-unit-price large-caps fake tiny
                # caps, corrupting any top_by_mcap selection).
                market_cap[ticker] = float("nan")
                layers["L4_dropped"] += 1
                if reported is None:
                    dropped.append(ticker)

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
        Tickers match rankings symbols by :func:`map_symbol`; a USD volume needs
        no unit scaling, so ``kPEPE`` takes PEPE's volume as is.

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
            hit = map_symbol(ticker, vol_map)
            tv = vol_map[hit[0]] if hit is not None else None
            agg[ticker] = float(tv) if tv else float("nan")
        return agg


# ============================================================================
# Point-in-time market cap for BACKTEST (no look-ahead)
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
# The point-in-time backtest replacement:
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

    Tickers match curated columns by :func:`map_symbol`. Hyperliquid quotes some
    high-supply tokens with a ``k`` (1000x) prefix (``kPEPE``, ``kBONK`` ...).
    Market cap is notation-independent (it is the coin's total cap), so these
    map to the unprefixed base symbol.

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
        cur = read_parquet(mc_path)
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
        hit = map_symbol(t, cur_cols)
        if hit is None:
            missing.append(t)
        else:
            selected[t] = cur[hit[0]]

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
