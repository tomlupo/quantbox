"""Shared universe selection for crypto strategies.

Provides vectorized and DuckDB-accelerated universe selection that works
with or without market-cap data.  When ``market_cap`` is ``None`` (e.g.
Hyperliquid), the mcap tier is skipped and assets are ranked directly by
dollar volume.
"""

from __future__ import annotations

import pandas as pd

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    DUCKDB_AVAILABLE = False


# Non-tradeable-crypto exclusion list. The authoritative source is
# ``catalog/asset_categories.yaml`` in the ``quantbox-datasets`` package; it's
# loaded lazily so quantbox itself stays domain-agnostic. If ``quantbox-datasets``
# is not installed the list is empty — quantbox does not carry an opinion about
# which tickers are stablecoins / wrapped / staked. Callers that need an
# exclusion list either install ``quantbox-datasets`` or pass their own
# ``exclude_tickers`` explicitly.
def _load_default_stablecoins() -> list[str]:
    """Resolve the default non-tradeable-crypto exclusion list.

    The authoritative source is ``catalog/asset_categories.yaml`` in
    ``quantbox-datasets`` (also packaged under ``quantbox_datasets/data/`` so
    it ships in the installed wheel). If the package is not installed or the
    YAML is missing the function returns ``[]`` — quantbox itself stays
    domain-agnostic. Callers in that situation must pass ``exclude_tickers``
    explicitly.
    """
    try:
        from quantbox_datasets.asset_categories import non_tradeable_crypto_symbols
    except ImportError:
        return []
    try:
        return list(non_tradeable_crypto_symbols())
    except Exception:  # noqa: BLE001 — never crash quantbox's import path
        return []


# Resolved at import time so existing consumers
# (``from ._universe import DEFAULT_STABLECOINS``) keep working unchanged.
DEFAULT_STABLECOINS: list[str] = _load_default_stablecoins()


def select_universe(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    market_cap: pd.DataFrame | None = None,
    top_by_mcap: int = 30,
    top_by_volume: int = 10,
    exclude_tickers: list[str] | None = None,
    volume_is_dollar: bool = False,
    volume_rolling_window: int = 1,
    min_listing_days: int = 0,
    hysteresis_rank_band: int = 0,
    screen_volume: pd.DataFrame | None = None,
    fine_lot_sz_decimals: dict[str, int] | None = None,
    fine_lot_min_notional: float = 0.0,
    fine_lot_max_lot_fraction: float = 0.0,
) -> pd.DataFrame:
    """Select tradable universe – vectorized pandas implementation.

    When *market_cap* is provided and non-empty the two-stage filter from
    ``CryptoTrendStrategy`` is applied:

    1. Rank by market cap → keep top ``top_by_mcap``
    2. Within those, rank by dollar volume → keep top ``top_by_volume``

    When *market_cap* is ``None`` or an empty DataFrame the mcap tier is
    skipped entirely and assets are ranked directly by dollar volume, keeping
    the top ``top_by_volume``.

    Parameters
    ----------
    prices : DataFrame
        Price DataFrame (date index, ticker columns).
    volume : DataFrame
        Volume DataFrame aligned with *prices*.  If *volume_is_dollar* is
        ``False`` (default) this must be base-asset quantity; dollar volume
        is computed as ``prices * volume``.  If *volume_is_dollar* is
        ``True`` the DataFrame is already in USD notional and is used
        directly — pass this when loading from a source that provides
        dollar volume (e.g. Binance ``quoteVolume`` / quantlab CSV).
    market_cap : DataFrame | None
        Market-cap DataFrame.  Pass ``None`` to skip the mcap tier.
    top_by_mcap : int
        How many assets to keep in the mcap tier (ignored when
        *market_cap* is ``None``).
    top_by_volume : int
        Final universe size (top N by dollar volume).
    exclude_tickers : list[str] | None
        Tickers to exclude.  Defaults to :data:`DEFAULT_STABLECOINS`.
    volume_is_dollar : bool
        Set ``True`` when *volume* is already in USD notional so the
        function does not multiply by *prices* again.
    volume_rolling_window : int
        Window for the rolling-mean smoother applied to dollar volume
        BEFORE ranking. Default ``1`` keeps the legacy "point-in-time daily
        volume" behaviour. Set ``>1`` (e.g., 30) to rank on the rolling
        average — drops listing-day spike noise and reduces boundary
        churn. Best-practice for slow-rotation strategies.
    min_listing_days : int
        Cool-off period after a coin's first valid price observation. The
        universe excludes any coin until it has at least this many days of
        price history. Default ``0`` disables. Recommended ``60``: most new
        Binance USDT listings spike then bleed, and trend-following needs
        enough history to fire reliable MA signals anyway.
    hysteresis_rank_band : int
        Rank band for sticky membership. When ``> 0``, a coin stays in the
        universe if its mcap rank ≤ ``top_by_mcap + band`` AND its volume
        rank ≤ ``top_by_volume + band`` AND it was in the universe on the
        previous day. Coins still enter via the strict ``top_by_*`` cuts.
        Reduces daily churn at the boundary at negligible alpha cost.
        Standard trick from index construction. Default ``0`` disables.
    screen_volume : DataFrame | None
        Market-wide liquidity used **only** for the volume RANK in screening,
        e.g. CoinGecko aggregate cross-exchange ``total_volume``. When provided
        it overrides *volume* for ranking, decoupling the question "is this coin
        liquid in the market?" (the SCREEN) from "how deep is our execution
        venue's book?" (per-venue *volume*, which stays available for sizing).
        This prevents a legitimately liquid coin from being dropped because of a
        thin single-quote-pair book (e.g. Binance USDC, 3.4–12.6× thinner than
        USDT non-uniformly per coin). Always interpreted as USD notional. Coins
        missing/zero in *screen_volume* fall back to per-venue dollar volume so a
        tradable coin the screen source doesn't cover is never silently zeroed.
        Default ``None`` keeps the legacy per-venue ranking unchanged.
    fine_lot_sz_decimals : dict[str, int] | None
        Per-coin Hyperliquid ``szDecimals`` (lot-size precision). When supplied
        together with the two thresholds below, activates the small-book fine-lot
        guard (see body). Default ``None`` disables it.
    fine_lot_min_notional : float
        Exchange minimum-notional floor (USD) used as the guard anchor (e.g.
        ``10.0`` on Hyperliquid). Default ``0.0`` disables the guard.
    fine_lot_max_lot_fraction : float
        Maximum allowed ratio of 1-lot notional to ``fine_lot_min_notional``. A
        coin is kept iff ``price * 10**-szDecimals <= fine_lot_max_lot_fraction *
        fine_lot_min_notional``. E.g. ``0.2`` => at the min-notional floor you get
        >= 5 lots of sizing granularity. Default ``0.0`` disables the guard.

    Returns
    -------
    DataFrame
        0/1 mask with same shape as *prices*.
    """
    if exclude_tickers is None:
        exclude_tickers = DEFAULT_STABLECOINS

    valid_tickers = [t for t in prices.columns if t not in exclude_tickers]

    if not valid_tickers:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Listing cool-off: drop tickers without enough history at each date.
    # We compute a per-(date, ticker) "age in days" mask and apply it to
    # both the mcap-mask and the volume-mask later.
    if min_listing_days > 0:
        first_valid = prices[valid_tickers].apply(lambda s: s.first_valid_index())
        # Broadcast per-column first-valid-date to a date×ticker matrix of days-since-listing
        days_since = pd.DataFrame(
            {
                t: (prices.index - first_valid[t]).days
                if first_valid[t] is not None
                else pd.Series(-1, index=prices.index)
                for t in valid_tickers
            },
            index=prices.index,
        )
        listing_mask = days_since >= min_listing_days
    else:
        listing_mask = pd.DataFrame(True, index=prices.index, columns=valid_tickers)

    # Fine-lot tradeability guard (small-book guardrail).
    #
    # At a small book the minimum tradeable increment (1 lot = 10**-szDecimals
    # base units) can be a large fraction of a leg, so a leg cannot be sized to
    # its target without coarse quantization error. A coin is kept on a date iff
    # its 1-lot notional (``price * 10**-szDecimals``) is <=
    # ``fine_lot_max_lot_fraction * fine_lot_min_notional`` — i.e. at the
    # exchange min-notional floor you still get >= ``1/max_lot_fraction`` lots of
    # granularity. ``min_notional`` is a hard exchange constant ($10 on
    # Hyperliquid), a far more stable anchor than the regime-dependent realized
    # per-leg notional. szDecimals is a (slowly varying) microstructure constant,
    # not a return series, so using the current snapshot against historical
    # prices is a tradeability approximation, not return look-ahead.
    #
    # Folded into ``listing_mask`` so it propagates to every downstream cut: an
    # excluded coin also frees its market-cap rank slot for the next tradeable
    # coin. Coins with no szDecimals entry fail closed (excluded) — the guard is
    # a tradeability *guarantee*. Disabled (no-op) unless both thresholds > 0 and
    # a szDecimals map is supplied.
    if fine_lot_sz_decimals and fine_lot_max_lot_fraction > 0 and fine_lot_min_notional > 0:
        threshold = fine_lot_max_lot_fraction * fine_lot_min_notional
        lot_step = pd.Series(
            {
                t: (10.0 ** (-int(fine_lot_sz_decimals[t])) if t in fine_lot_sz_decimals else float("nan"))
                for t in valid_tickers
            }
        )
        lot_notional = prices[valid_tickers].mul(lot_step, axis=1)
        # NaN lot_step (unknown szDecimals) -> NaN lot_notional -> excluded.
        fine_lot_mask = (lot_notional <= threshold) & lot_notional.notna()
        listing_mask = listing_mask & fine_lot_mask

    has_mcap = market_cap is not None and not market_cap.empty

    vol_aligned = volume.reindex(index=prices.index, columns=valid_tickers).fillna(0.0)
    if volume_is_dollar:
        dollar_vol = vol_aligned
    else:
        dollar_vol = prices[valid_tickers] * vol_aligned

    # Choose the volume series used for RANKING. By default this is the
    # per-venue dollar volume above. When *screen_volume* (market-wide
    # aggregate, USD notional) is supplied, rank on it instead — coalescing
    # to per-venue dollar volume wherever the screen source is missing/zero so
    # a tradable coin it doesn't cover is never silently zeroed. The per-venue
    # `volume`/`dollar_vol` remains the right input for downstream sizing.
    if screen_volume is not None and not screen_volume.empty:
        sv = screen_volume.reindex(index=prices.index, columns=valid_tickers)
        rank_vol = sv.where(sv.notna() & (sv > 0), other=dollar_vol).fillna(0.0)
    else:
        rank_vol = dollar_vol

    # Optional rolling-mean smoother on the ranking volume (best-practice). A
    # 30-day window cuts most listing-spike / single-day-anomaly noise in
    # the top-N-by-volume cut without materially changing which large-caps
    # dominate. (No-op when screen_volume is a flat snapshot.)
    if volume_rolling_window > 1:
        rank_vol = rank_vol.rolling(volume_rolling_window, min_periods=max(1, volume_rolling_window // 2)).mean()

    if has_mcap:
        # Stage 1: market-cap rank
        #
        # Two corrections vs the original implementation:
        #
        # 1. Forward-fill mcap before reindexing to daily — the curated
        #    crypto datasets store mcap (CoinCodex monthly snapshots) at
        #    month-end only (172 rows over 14 years). A plain
        #    reindex+fillna(0) would degenerate every non-month-end day's
        #    rank (all coins tie at 0 → all pass any top-N filter), silently
        #    disabling the mcap stage entirely. Month-end mcap is a fine
        #    rank reference for the rest of the month because mcap moves
        #    slowly relative to volume.
        #
        # 2. Rank across the FULL mcap universe (e.g., all ~707 CoinCodex
        #    tickers) BEFORE intersecting with the Binance-tradeable price
        #    columns. The notebook v2 replication source did this — coins
        #    that occupy top-30 mcap globally but aren't on Binance (CRO,
        #    LEO, OKB, MIOTA, HT, …) take up rank slots that effectively go
        #    unused once intersected with volume; ranking only within the
        #    Binance subset would silently promote different boundary
        #    coins into the top-30 (VET/FIL/ICP/MKR/ARB observed during
        #    the trend_catcher v2 audit) and change which names compete in
        #    the subsequent top-by-volume cut.
        mcap_full = (
            market_cap
            if exclude_tickers is None
            else market_cap.drop(  # type: ignore[union-attr]
                columns=[c for c in market_cap.columns if c in exclude_tickers],  # type: ignore[union-attr]
                errors="ignore",
            )
        )
        mc = mcap_full.reindex(index=prices.index, method="ffill").fillna(0.0)
        mc_rank_full = mc.rank(axis=1, ascending=False, method="min")
        # Intersect with the Binance-tradeable subset after ranking
        mc_rank = mc_rank_full.reindex(columns=valid_tickers)
        # Apply listing cool-off: pretend coins under cool-off don't have
        # a valid mcap rank, so they can't make either the strict or
        # relaxed cut.
        mc_rank = mc_rank.where(listing_mask)
        mc_mask_strict = mc_rank <= top_by_mcap
        mc_mask_relaxed = mc_rank <= top_by_mcap + hysteresis_rank_band

        # Stage 2: volume rank within strict mcap tier (for entry)
        vol_masked_strict = rank_vol.where(mc_mask_strict)
        vol_rank_strict = vol_masked_strict.rank(axis=1, ascending=False, method="min")
        strict_mask = (vol_rank_strict <= top_by_volume) & listing_mask

        if hysteresis_rank_band > 0:
            # For "stay-in" eligibility, allow up to top_by_mcap+band on
            # mcap and rank-by-volume within the RELAXED mcap pool.
            vol_masked_relaxed = rank_vol.where(mc_mask_relaxed)
            vol_rank_relaxed = vol_masked_relaxed.rank(axis=1, ascending=False, method="min")
            relaxed_mask = (vol_rank_relaxed <= top_by_volume + hysteresis_rank_band) & listing_mask
        else:
            relaxed_mask = strict_mask
    else:
        # No mcap – rank directly by volume (market-wide screen_volume if
        # supplied, else per-venue dollar volume)
        vol_rank = rank_vol.where(listing_mask).rank(axis=1, ascending=False, method="min")
        strict_mask = (vol_rank <= top_by_volume) & listing_mask
        if hysteresis_rank_band > 0:
            relaxed_mask = (vol_rank <= top_by_volume + hysteresis_rank_band) & listing_mask
        else:
            relaxed_mask = strict_mask

    if hysteresis_rank_band > 0:
        # State-machine pass over time: each day a coin is in iff it
        # passes the strict cut today OR (it was in yesterday AND passes
        # the relaxed cut today).
        strict_arr = strict_mask.values.astype(bool)
        relaxed_arr = relaxed_mask.values.astype(bool)
        sticky = strict_arr.copy()
        for i in range(1, len(sticky)):
            sticky[i] = strict_arr[i] | (sticky[i - 1] & relaxed_arr[i])
        universe_valid = sticky.astype(float)
    else:
        universe_valid = strict_mask.values.astype(float)

    universe = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    universe[valid_tickers] = universe_valid
    return universe


# Keep the old name as an alias so existing callers keep working.
select_universe_vectorized = select_universe


def select_universe_duckdb(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    market_cap: pd.DataFrame | None = None,
    top_by_mcap: int = 30,
    top_by_volume: int = 10,
    exclude_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Select universe using DuckDB (faster for very large datasets).

    Falls back to :func:`select_universe` when DuckDB is not installed or
    when *market_cap* is ``None``.
    """
    has_mcap = market_cap is not None and not market_cap.empty

    if not DUCKDB_AVAILABLE or not has_mcap:
        return select_universe(prices, volume, market_cap, top_by_mcap, top_by_volume, exclude_tickers)

    if exclude_tickers is None:
        exclude_tickers = DEFAULT_STABLECOINS

    def _to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        # Normalise the index name so reset_index() produces a column called "date"
        # regardless of how the caller's DataFrame is shaped.
        df = df.rename_axis("date", axis=0)
        return df.reset_index().melt(id_vars="date", var_name="ticker", value_name=value_name)

    prices_long = _to_long(prices, "price")
    volume_long = _to_long(volume, "volume")
    mc_long = _to_long(market_cap, "market_cap")  # type: ignore[arg-type]

    data = prices_long.merge(volume_long, on=["date", "ticker"])
    data = data.merge(mc_long, on=["date", "ticker"])
    data["dollar_volume"] = data["price"] * data["volume"]

    exclude_str = "', '".join(exclude_tickers)

    con = duckdb.connect()  # type: ignore[union-attr]
    con.register("data", data)

    query = f"""
    WITH ranked AS (
        SELECT
            date, ticker, price, market_cap, dollar_volume,
            RANK() OVER (PARTITION BY date ORDER BY market_cap DESC) AS mc_rank
        FROM data
        WHERE ticker NOT IN ('{exclude_str}')
    ),
    mc_filtered AS (
        SELECT *,
            RANK() OVER (PARTITION BY date ORDER BY dollar_volume DESC) AS vol_rank
        FROM ranked
        WHERE mc_rank <= {top_by_mcap}
    )
    SELECT
        date, ticker,
        CASE WHEN vol_rank <= {top_by_volume} THEN 1.0 ELSE 0.0 END AS in_universe
    FROM mc_filtered
    """

    result = con.execute(query).df()
    con.close()

    universe = result.pivot(index="date", columns="ticker", values="in_universe")
    universe = universe.reindex(columns=prices.columns, fill_value=0.0)
    universe = universe.reindex(index=prices.index, fill_value=0.0)
    # pivot() leaves NaN for every (date, ticker) that fell OUTSIDE the day's
    # top-30 mcap cut (no row in `mc_filtered`), and reindex(fill_value=0) only
    # fills wholly-absent columns — never these per-cell NaNs. A NaN here means
    # "not selected that day", so it MUST be 0.0. Leaving it as NaN let a
    # rank-31 near-stablecoin (e.g. "U") leak into the book: downstream
    # construct_weights does `signal.where(universe != 0, 0.0)`, and `NaN != 0`
    # is True, so the excluded coin was retained and — with an unclipped
    # vol-scaler — exploded to ~92% of the book. fillna(0.0) makes this path
    # byte-identical to select_universe() (0 cell mismatches on the live cache).
    universe = universe.fillna(0.0)
    return universe
