"""Momentum features: total returns, momentum returns, TSMOM signals.

All functions accept **wide-format** prices (DatetimeIndex x symbol
columns). Suitable for both equities/fixed income (trading_days=252)
and crypto (trading_days=365).

The TSMOM (time-series momentum) implementation follows Moskowitz, Ooi &
Pedersen (2012) with extensions for fast/slow signal classification
and configurable crossover window selection.

``compute_tsmom`` returns a dict of long-format ``(date, ticker)``-indexed
DataFrames (composite + intermediate signal tables). Long format is the
natural shape for the per-``(date, ticker)`` composite aggregation and
avoids the ambiguity of ``(signal, ticker)`` MultiIndex-column pandas
operations. Callers that want wide can ``.unstack("ticker")`` at the
boundary.
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


# --- TSMOM helpers (long-format (date, ticker) operations) ---


def _sma_long(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average on a long-format ``(date, ticker)`` frame."""
    return df.unstack("ticker").apply(lambda x: x.dropna().rolling(window).mean()).stack("ticker")


def _ewma_long(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """EWMA on a long-format ``(date, ticker)`` frame."""
    return df.unstack("ticker").apply(lambda x: x.dropna().ewm(span=window, adjust=False).mean()).stack("ticker")


def _binary_signal(df: pd.DataFrame, threshold: float = 0) -> pd.DataFrame:
    """Binary signal on a frame: 1 if value > threshold, else 0."""
    s = (df > threshold).mask(pd.isna(df)) * 1
    return s.add_suffix("_signal")


def _crossover_pairs(
    df: pd.DataFrame,
    pair_indices: list[tuple[int, int]],
) -> pd.DataFrame:
    """Ratio crossovers between specified column pairs.

    ``df`` has top-level columns like ``"prefix<w>"`` (e.g. ``"prices_sma50"``).
    Returns a frame with one column per requested ``(i, j)`` pair named
    ``"{prefix}_{col_i_suffix}div{col_j_suffix}"``.
    """
    import os

    dfs = []
    cols = df.columns.get_level_values(0)
    prefix = os.path.commonprefix(list(cols))
    for i, j in pair_indices:
        col1 = cols[i].replace(prefix, "")
        col2 = cols[j].replace(prefix, "")
        name = prefix + "_" + col1 + "div" + col2
        res = df.iloc[:, i] / df.iloc[:, j]
        res.name = name
        dfs.append(res)
    return pd.concat(dfs, axis=1)


def _rolling_minmax(x: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    """Rolling ``(x - min) / (max - min)``; constant-window ⇒ 0, not NaN.

    Replacing the zero denominator with NaN silently drops the tail of
    any signal in a persistently-constant regime — see robo's
    ADR-0005 investigation. Replacing with 1 preserves warmup NaN (from
    the numerator) and emits 0 for constant windows, which is the
    semantically correct value.
    """
    rmin = x.rolling(window=window).min()
    rmax = x.rolling(window=window).max()
    denom = (rmax - rmin).replace(0, 1)
    return (x - rmin) / denom


# ---------------------------------------------------------------------------
# TSMOM indicator suite
# ---------------------------------------------------------------------------


def compute_tsmom(
    prices: pd.DataFrame,
    *,
    tr_windows: list[int] | None = None,
    sma_windows: list[int] | None = None,
    ewma_spans: list[int] | None = None,
    sma_pair_indices: list[tuple[int, int]] | None = None,
    ewma_pair_indices: list[tuple[int, int]] | None = None,
    warmup_periods: int = 253,
    zscore_window: int = 756,
    winsorize_clip: float = 2.5,
) -> dict[str, pd.DataFrame]:
    """Time-series momentum (TSMOM) indicator suite.

    Implements Moskowitz-Ooi-Pedersen time-series momentum with:

    * multi-window total-return binary signals,
    * SMA and EWMA ratio-crossover binary signals on configurable pair
      subsets (``sma_pair_indices`` / ``ewma_pair_indices``; ``None``
      defaults to all unordered pairs over the given windows/spans),
    * rolling z-score standardisation + winsorisation + [0, 1] min-max,
    * per-``(date, ticker)`` fast/slow composite aggregation using rolling
      std of per-signal absolute change to rank signals by reaction
      speed,
    * per-ticker warmup masking (each ticker's warmup is measured from
      its own first valid observation, not a single cohort cutoff).

    Args:
        prices: Wide-format (DatetimeIndex x symbol columns).
        tr_windows: Total-return lookback windows. Default ``[21, 63, 126, 189, 252]``.
        sma_windows: SMA windows. Default ``[1, 50, 100, 200]``.
        ewma_spans: EWMA spans. Default ``[8, 16, 32, 64, 128, 256]``.
        sma_pair_indices: List of (i, j) index tuples selecting which
            SMA window pairs to cross. ``None`` = every unordered pair.
        ewma_pair_indices: As above, for EWMA spans. ``None`` = every
            unordered pair.
        warmup_periods: Mask each ticker's signals until ``warmup_periods``
            business days after that ticker's first valid price.
        zscore_window: Rolling window for z-score / winsorize / min-max.
            Default 3y = 756.
        winsorize_clip: Clip z-scores to ``[-clip, clip]``.

    Returns:
        Dict with the following long-format ``(date, ticker)``-indexed
        DataFrames (callers wanting wide can ``.unstack("ticker")``):

        - ``"total_returns"`` — one column per window (``total_return_{w}``)
        - ``"signals_raw"`` — raw binary signals, pre-standardisation
        - ``"signals_standardized"`` — rolling z-scored signals
        - ``"signals_normalized"`` — winsorised + min-max normalised to [0, 1]
        - ``"composite"`` — per-``(date, ticker)`` composites:
          ``TF_fast``, ``TF_slow``, ``TF_zscore``,
          ``TF_zscore_winsorized``, ``TF_zscore_winsorized_minmax``, ``TF``
    """
    if tr_windows is None:
        tr_windows = [21, 63, 126, 189, 252]
    if sma_windows is None:
        sma_windows = [1, 50, 100, 200]
    if ewma_spans is None:
        ewma_spans = [8, 16, 32, 64, 128, 256]

    def _all_pairs(n: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    if sma_pair_indices is None:
        sma_pair_indices = _all_pairs(len(sma_windows))
    if ewma_pair_indices is None:
        ewma_pair_indices = _all_pairs(len(ewma_spans))

    # --- Total returns (long) → binary signals ---
    tr = pd.concat(
        [np.exp(np.log(prices / prices.shift(1)).rolling(window=w).sum()) - 1 for w in tr_windows],
        axis=1,
        keys=[f"total_return_{w}" for w in tr_windows],
    )
    tr = tr.stack(level=-1, future_stack=True).sort_index()
    tr.index.names = ["date", "ticker"]

    # --- Moving averages on a long-format price frame ---
    prices_long = prices.stack()
    prices_long.index.names = ["date", "ticker"]

    prices_sma = pd.concat(
        [_sma_long(prices_long, window=w) for w in sma_windows],
        axis=1,
        keys=[f"prices_sma{w}" for w in sma_windows],
    )
    prices_ewma = pd.concat(
        [_ewma_long(prices_long, window=s) for s in ewma_spans],
        axis=1,
        keys=[f"prices_ewma{s}" for s in ewma_spans],
    )

    # --- Crossover signals ---
    prices_sma_cross = _crossover_pairs(prices_sma, sma_pair_indices)
    prices_ewma_cross = _crossover_pairs(prices_ewma, ewma_pair_indices)

    # --- Binary signals ---
    tr_signals = _binary_signal(tr, 0)
    sma_signals = _binary_signal(prices_sma_cross, 1)
    ewma_signals = _binary_signal(prices_ewma_cross, 1)
    signals_raw = pd.concat([tr_signals, sma_signals, ewma_signals], axis=1).sort_index()

    # --- Per-ticker warmup filter ---
    start_dates = prices.shift(warmup_periods).apply(lambda x: x.first_valid_index())
    signals_raw = (
        signals_raw.groupby("ticker", group_keys=False)
        .apply(lambda x: x[x.index.get_level_values("date") >= pd.Timestamp(start_dates[x.name])])
        .sort_index()
        .dropna(how="all")
    )

    # --- Standardise (rolling z-score; std=0 → 1, not NaN; see ADR-0005) ---
    signals_standardized = (
        signals_raw.groupby("ticker", group_keys=False)
        .apply(
            lambda x: (x - x.rolling(window=zscore_window).mean()) / x.rolling(window=zscore_window).std().replace(0, 1)
        )
        .dropna(how="all")
        .sort_index()
    )

    # --- Winsorise z-scores ---
    signals_winsorized = (
        signals_standardized.groupby("ticker", group_keys=False)
        .apply(lambda x: x.clip(lower=-winsorize_clip, upper=winsorize_clip))
        .dropna(how="all")
        .sort_index()
    )

    # --- Min-max normalise to [0, 1] ---
    signals_normalized = (
        signals_winsorized.groupby("ticker", group_keys=False)
        .apply(lambda x: _rolling_minmax(x, zscore_window))
        .dropna(how="all")
        .sort_index()
    )

    # --- Signal speed → fast/slow masks ---
    signal_speed = (
        signals_standardized.groupby("ticker", group_keys=False)
        .apply(lambda x: x.diff().abs().rolling(window=zscore_window).std())
        .dropna(how="any")
    )
    n_signals = signal_speed.shape[1]
    signal_speed_rank_min = signal_speed.rank(axis=1, method="min")
    signal_speed_rank_max = signal_speed.rank(axis=1, method="max")
    slow_flag = signal_speed_rank_min <= n_signals / 2
    fast_flag = signal_speed_rank_max >= n_signals / 2

    # --- Composite per (date, ticker) ---
    tf_fast = fast_flag.mul(signals_raw).sum(axis=1).div(fast_flag.sum(axis=1)).rename("TF_fast")
    tf_slow = slow_flag.mul(signals_raw).sum(axis=1).div(slow_flag.sum(axis=1)).rename("TF_slow")
    tf_zscore = signals_standardized.mean(axis=1).rename("TF_zscore")
    tf_zscore_winsorized = signals_winsorized.mean(axis=1).rename("TF_zscore_winsorized")
    tf_zscore_winsorized_minmax = signals_normalized.mean(axis=1).rename("TF_zscore_winsorized_minmax")
    tf_all = signals_raw.mean(axis=1).rename("TF")

    composite = (
        pd.concat(
            [
                tf_fast,
                tf_slow,
                tf_zscore,
                tf_zscore_winsorized,
                tf_zscore_winsorized_minmax,
                tf_all,
            ],
            axis=1,
        )
        .dropna(how="all")
        .sort_index()
    )

    results = {
        "total_returns": tr,
        "signals_raw": signals_raw,
        "signals_standardized": signals_standardized,
        "signals_normalized": signals_normalized,
        "composite": composite,
    }
    return {k: v.dropna(how="all").astype(float) for k, v in results.items()}
