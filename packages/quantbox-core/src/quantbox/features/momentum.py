"""Momentum features: total returns, momentum returns, TSMOM signals.

All functions are pure, stateless, and operate on wide-format DataFrames
(DatetimeIndex x symbol columns). Suitable for both equities/fixed income
(trading_days=252) and crypto (trading_days=365).

The TSMOM (time-series momentum) implementation follows Moskowitz, Ooi &
Pedersen (2012) with extensions for fast/slow signal classification and
configurable crossover windows.

Where beneficial, vectorbt (free) is used for efficient moving-average
crossover computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Total & momentum returns
# ---------------------------------------------------------------------------


def compute_total_returns(
    prices: pd.DataFrame,
    windows: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Rolling total returns (log-return cumulative sum) over multiple windows.

    Args:
        prices: Wide-format (DatetimeIndex x symbol columns).
        windows: Lookback windows in trading days. Default ``[21, 63, 126, 189, 252]``.

    Returns:
        Dict keyed ``"total_return_{w}d"`` -> DataFrame.
    """
    if windows is None:
        windows = [21, 63, 126, 189, 252]
    result: dict[str, pd.DataFrame] = {}
    log_ret = np.log(prices / prices.shift(1))
    for w in windows:
        result[f"total_return_{w}d"] = np.exp(log_ret.rolling(window=w).sum()) - 1
    return result


def compute_momentum_returns(
    prices: pd.DataFrame,
    windows: list[int] | None = None,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Annualized momentum-based expected returns.

    Average of multi-window total returns, scaled to annual frequency.

    Args:
        prices: Wide-format (DatetimeIndex x symbol columns).
        windows: Lookback windows. Default ``[21, 63, 126, 189, 252]``.
        trading_days: Annualization factor (252 for equities, 365 for crypto).

    Returns:
        DataFrame of annualized expected returns (same shape as prices).
    """
    if windows is None:
        windows = [21, 63, 126, 189, 252]
    tr = compute_total_returns(prices, windows)
    stacked = pd.concat(tr.values(), axis=1, keys=tr.keys())
    # Mean across windows for each ticker
    result = stacked.T.groupby(stacked.columns.get_level_values(1)).mean().T
    return result * (trading_days / np.mean(windows))


# ---------------------------------------------------------------------------
# TSMOM indicator suite
# ---------------------------------------------------------------------------


def _try_vbt_crossovers(
    prices: pd.DataFrame,
    sma_windows: list[int],
    ewma_spans: list[int],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Attempt to use vectorbt for fast MA crossover computation.

    Returns (sma_cross_signals, ewma_cross_signals) or (None, None) if
    vectorbt is not available.
    """
    try:
        import vectorbt as vbt

        sma_cross = None
        if len(sma_windows) >= 2:
            mas = {w: vbt.MA.run(prices, window=w, ewm=False).ma for w in sma_windows}
            pairs = []
            names = []
            for i, w1 in enumerate(sma_windows):
                for w2 in sma_windows[i + 1 :]:
                    ratio = mas[w1] / mas[w2]
                    pairs.append((ratio > 1).astype(float))
                    names.append(f"sma_{w1}div{w2}")
            if pairs:
                sma_cross = pd.concat(pairs, axis=1, keys=names)

        ewma_cross = None
        if len(ewma_spans) >= 2:
            emas = {s: vbt.MA.run(prices, window=s, ewm=True).ma for s in ewma_spans}
            pairs = []
            names = []
            for i, s1 in enumerate(ewma_spans):
                for s2 in ewma_spans[i + 1 :]:
                    if s1 < s2:
                        ratio = emas[s1] / emas[s2]
                        pairs.append((ratio > 1).astype(float))
                        names.append(f"ewma_{s1}div{s2}")
            if pairs:
                ewma_cross = pd.concat(pairs, axis=1, keys=names)

        return sma_cross, ewma_cross
    except (ImportError, Exception):
        return None, None


def _pandas_crossovers(
    prices: pd.DataFrame,
    sma_windows: list[int],
    ewma_spans: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback SMA/EWMA crossover computation using pure pandas."""
    sma_dict = {w: prices.rolling(window=w).mean() for w in sma_windows}
    ewma_dict = {s: prices.ewm(span=s, adjust=False).mean() for s in ewma_spans}

    sma_pairs = []
    sma_names = []
    for i, w1 in enumerate(sma_windows):
        for w2 in sma_windows[i + 1 :]:
            ratio = sma_dict[w1] / sma_dict[w2]
            sma_pairs.append((ratio > 1).astype(float))
            sma_names.append(f"sma_{w1}div{w2}")

    ewma_pairs = []
    ewma_names = []
    for i, s1 in enumerate(ewma_spans):
        for s2 in ewma_spans[i + 1 :]:
            if s1 < s2:
                ratio = ewma_dict[s1] / ewma_dict[s2]
                ewma_pairs.append((ratio > 1).astype(float))
                ewma_names.append(f"ewma_{s1}div{s2}")

    sma_cross = pd.concat(sma_pairs, axis=1, keys=sma_names) if sma_pairs else pd.DataFrame()
    ewma_cross = pd.concat(ewma_pairs, axis=1, keys=ewma_names) if ewma_pairs else pd.DataFrame()
    return sma_cross, ewma_cross


def compute_tsmom(
    prices: pd.DataFrame,
    *,
    tr_windows: list[int] | None = None,
    sma_windows: list[int] | None = None,
    ewma_spans: list[int] | None = None,
    warmup_periods: int = 253,
    zscore_window: int = 756,
    winsorize_clip: float = 2.5,
    use_vectorbt: bool = False,
) -> dict[str, pd.DataFrame]:
    """Time-series momentum (TSMOM) indicator suite.

    Computes multi-window total returns, SMA/EWMA crossover signals,
    standardizes via rolling z-score, winsorizes, normalizes to [0, 1],
    and classifies signals into fast/slow components.

    Args:
        prices: Wide-format (DatetimeIndex x symbol columns).
        tr_windows: Total return lookback windows. Default ``[21, 63, 126, 189, 252]``.
        sma_windows: SMA windows for crossover signals. Default ``[1, 50, 100, 200]``.
        ewma_spans: EWMA spans for crossover signals. Default ``[8, 16, 32, 64, 128, 256]``.
        warmup_periods: Minimum observations before producing signals.
        zscore_window: Rolling window for z-score standardization (default 3y = 756).
        winsorize_clip: Clip z-scores to [-clip, clip].
        use_vectorbt: Try vectorbt for crossover computation (falls back to pandas).

    Returns:
        Dict with keys:

        - ``"total_returns"`` — multi-window total returns (wide, one col per window)
        - ``"signals_raw"`` — raw binary signals before standardization
        - ``"signals_standardized"`` — rolling z-scored signals
        - ``"signals_normalized"`` — winsorized + min-max normalized to [0, 1]
        - ``"composite"`` — composite signals: TF_fast, TF_slow, TF (all), TF_zscore
    """
    if tr_windows is None:
        tr_windows = [21, 63, 126, 189, 252]
    if sma_windows is None:
        sma_windows = [1, 50, 100, 200]
    if ewma_spans is None:
        ewma_spans = [8, 16, 32, 64, 128, 256]

    tickers = prices.columns.tolist()

    # --- Total returns → binary signals ---
    tr_dict = compute_total_returns(prices, tr_windows)
    tr_signals = pd.concat(
        [(df > 0).astype(float) for df in tr_dict.values()],
        axis=1,
        keys=[f"tr_{w}d_signal" for w in tr_windows],
    )

    # --- Crossover signals ---
    if use_vectorbt:
        sma_cross, ewma_cross = _try_vbt_crossovers(prices, sma_windows, ewma_spans)
    else:
        sma_cross, ewma_cross = None, None

    if sma_cross is None or ewma_cross is None:
        sma_cross_pd, ewma_cross_pd = _pandas_crossovers(prices, sma_windows, ewma_spans)
        if sma_cross is None:
            sma_cross = sma_cross_pd
        if ewma_cross is None:
            ewma_cross = ewma_cross_pd

    # Combine all raw signals
    parts = [tr_signals]
    if not sma_cross.empty:
        parts.append(sma_cross)
    if not ewma_cross.empty:
        parts.append(ewma_cross)
    signals_raw = pd.concat(parts, axis=1).sort_index()

    # --- Apply warmup filter ---
    start_dates = prices.shift(warmup_periods).apply(lambda x: x.first_valid_index())
    for ticker in tickers:
        if ticker in start_dates and start_dates[ticker] is not None:
            # For wide-format: mask rows before warmup for each ticker
            # signals_raw has multi-level columns (signal_name, ticker)
            pass
    # Simple warmup: drop rows where not enough data
    first_valid = prices.apply(lambda x: x.first_valid_index())
    warmup_cutoff = first_valid.max()
    if warmup_cutoff is not None:
        warmup_cutoff = prices.index[
            min(
                prices.index.get_loc(warmup_cutoff) + warmup_periods,
                len(prices.index) - 1,
            )
        ]
        signals_raw = signals_raw.loc[warmup_cutoff:]

    # --- Standardize (rolling z-score) ---
    signals_standardized = (
        (signals_raw - signals_raw.rolling(window=zscore_window).mean())
        / signals_raw.rolling(window=zscore_window).std().replace(0, np.nan)
    ).dropna(how="all")

    # --- Winsorize ---
    signals_winsorized = signals_standardized.clip(lower=-winsorize_clip, upper=winsorize_clip)

    # --- Normalize to [0, 1] (rolling min-max) ---
    rmin = signals_winsorized.rolling(window=zscore_window).min()
    rmax = signals_winsorized.rolling(window=zscore_window).max()
    denom = (rmax - rmin).replace(0, np.nan)
    signals_normalized = ((signals_winsorized - rmin) / denom).dropna(how="all")

    # --- Fast / slow classification ---
    signal_speed = (signals_standardized.diff().abs().rolling(window=zscore_window).std()).dropna(how="all")
    signal_speed_rank = signal_speed.rank(axis=1, method="min")
    n_signals = signal_speed_rank.shape[1]
    slow_flag = signal_speed_rank <= n_signals / 2
    fast_flag = signal_speed.rank(axis=1, method="max") >= n_signals / 2

    # --- Composite signals ---
    tf_fast = (
        fast_flag.mul(signals_raw.reindex(fast_flag.index))
        .sum(axis=1)
        .div(fast_flag.sum(axis=1).replace(0, np.nan))
        .rename("TF_fast")
    )
    tf_slow = (
        slow_flag.mul(signals_raw.reindex(slow_flag.index))
        .sum(axis=1)
        .div(slow_flag.sum(axis=1).replace(0, np.nan))
        .rename("TF_slow")
    )
    tf_all = signals_raw.mean(axis=1).rename("TF")
    tf_zscore = signals_standardized.mean(axis=1).rename("TF_zscore")
    tf_normalized = signals_normalized.mean(axis=1).rename("TF_normalized")

    composite = pd.concat(
        [tf_fast, tf_slow, tf_all, tf_zscore, tf_normalized],
        axis=1,
    ).dropna(how="all")

    # --- Pack total returns as wide format ---
    tr_wide = pd.concat(tr_dict.values(), axis=1, keys=tr_dict.keys())

    return {
        "total_returns": tr_wide,
        "signals_raw": signals_raw,
        "signals_standardized": signals_standardized,
        "signals_normalized": signals_normalized,
        "composite": composite,
    }
