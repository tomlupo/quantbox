from __future__ import annotations
import pandas as pd


def compute_zscore_cross_sectional(
    signals: pd.DataFrame,
    *,
    clip: float = 3.0,
) -> pd.DataFrame:
    """Cross-sectional z-score normalization (row-wise).

    Args:
        signals: Wide-format DataFrame (DatetimeIndex x symbol columns).
        clip: Clip z-scores to [-clip, clip]. Set to 0 or None to disable.

    Returns:
        DataFrame of z-scored values.
    """
    mean = signals.mean(axis=1)
    std = signals.std(axis=1)
    zscore = signals.sub(mean, axis=0).div(std.replace(0, float("nan")), axis=0)
    if clip:
        zscore = zscore.clip(lower=-clip, upper=clip)
    return zscore


def compute_rank_cross_sectional(
    signals: pd.DataFrame,
    *,
    ascending: bool = True,
    pct: bool = True,
) -> pd.DataFrame:
    """Cross-sectional rank normalization (row-wise).

    Args:
        signals: Wide-format DataFrame (DatetimeIndex x symbol columns).
        ascending: Rank direction.
        pct: If True, return percentile ranks [0, 1].

    Returns:
        DataFrame of ranked values.
    """
    return signals.rank(axis=1, ascending=ascending, pct=pct)
