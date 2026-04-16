"""Signal transformation utilities.

Pure functions for transforming raw signals/indicators into
tradeable signals. Operates on wide-format DataFrames
(DatetimeIndex x symbol columns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def binary_signal(
    values: pd.DataFrame,
    threshold: float = 0,
) -> pd.DataFrame:
    """Binary signal: 1 where value > threshold, else 0.

    Args:
        values: Wide-format DataFrame of raw signal values.
        threshold: Cutoff level.

    Returns:
        DataFrame of 0/1 signals.
    """
    return (values > threshold).where(values.notna()).astype(float)


def rolling_zscore(
    values: pd.DataFrame,
    window: int = 756,
) -> pd.DataFrame:
    """Rolling z-score standardization (per-column, time-series).

    Args:
        values: Wide-format DataFrame.
        window: Rolling window size (default 3 years = 756 trading days).

    Returns:
        Z-scored DataFrame.
    """
    mean = values.rolling(window=window).mean()
    std = values.rolling(window=window).std().replace(0, np.nan)
    return (values - mean) / std


def cross_sectional_zscore(
    values: pd.DataFrame,
    clip: float = 3.0,
) -> pd.DataFrame:
    """Cross-sectional z-score (per-row, across symbols).

    Args:
        values: Wide-format DataFrame.
        clip: Clip z-scores to [-clip, clip]. Set to 0 to disable.

    Returns:
        Z-scored DataFrame.
    """
    mean = values.mean(axis=1)
    std = values.std(axis=1).replace(0, np.nan)
    z = values.sub(mean, axis=0).div(std, axis=0)
    if clip:
        z = z.clip(lower=-clip, upper=clip)
    return z


def winsorize(
    values: pd.DataFrame,
    lower: float = -2.5,
    upper: float = 2.5,
) -> pd.DataFrame:
    """Clip values to [lower, upper] bounds.

    Args:
        values: DataFrame to winsorize.
        lower: Lower bound.
        upper: Upper bound.

    Returns:
        Clipped DataFrame.
    """
    return values.clip(lower=lower, upper=upper)


def rolling_minmax_normalize(
    values: pd.DataFrame,
    window: int = 756,
) -> pd.DataFrame:
    """Normalize values to [0, 1] using rolling min-max scaling.

    Args:
        values: DataFrame to normalize.
        window: Rolling window for min/max computation.

    Returns:
        Normalized DataFrame with values in [0, 1].
    """
    rmin = values.rolling(window=window).min()
    rmax = values.rolling(window=window).max()
    denom = (rmax - rmin).replace(0, np.nan)
    return (values - rmin) / denom


def rank_select_top_n(
    scores: pd.DataFrame,
    top_n: int,
    *,
    mask: pd.DataFrame | None = None,
    equal_weight: bool = True,
) -> pd.DataFrame:
    """Select top-N assets per row by score, optionally filtered by mask.

    Args:
        scores: Wide-format DataFrame of scores (higher = better).
        top_n: Number of assets to select per row.
        mask: Optional 0/1 DataFrame — only consider assets where mask == 1.
        equal_weight: If True, return equal-weighted (sums to 1).
            If False, return 0/1 selection mask.

    Returns:
        DataFrame of weights or selection mask.
    """
    filtered = scores.copy()
    if mask is not None:
        filtered = filtered * mask
    filtered = filtered.where(filtered > 0, np.nan)

    rank = filtered.rank(axis=1, ascending=False, method="min")
    selection = (rank <= top_n).astype(float).fillna(0)

    if equal_weight:
        row_sum = selection.sum(axis=1).replace(0, np.nan)
        return selection.div(row_sum, axis=0).fillna(0)
    return selection


def inverse_volatility_weights(
    selection: pd.DataFrame,
    volatility: pd.DataFrame,
) -> pd.DataFrame:
    """Inverse-volatility weighting within selected assets.

    Args:
        selection: 0/1 selection mask (from ``rank_select_top_n``).
        volatility: Wide-format volatility DataFrame.

    Returns:
        Normalized weights (sums to 1 per row).
    """
    inv_vol = (1.0 / volatility.replace(0, np.nan)) * selection
    row_sum = inv_vol.sum(axis=1).replace(0, np.nan)
    return inv_vol.div(row_sum, axis=0).fillna(0)
