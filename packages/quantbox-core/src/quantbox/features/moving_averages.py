from __future__ import annotations
from typing import Dict, List
import pandas as pd


def compute_sma(
    prices: pd.DataFrame,
    windows: List[int],
) -> Dict[str, pd.DataFrame]:
    """Compute simple moving averages.

    Args:
        prices: Wide-format DataFrame (DatetimeIndex x symbol columns).
        windows: Rolling window sizes.

    Returns:
        Dict keyed ``"sma_{w}d"`` -> DataFrame.
    """
    return {f"sma_{w}d": prices.rolling(window=w).mean() for w in windows}


def compute_ema(
    prices: pd.DataFrame,
    spans: List[int],
) -> Dict[str, pd.DataFrame]:
    """Compute exponential moving averages.

    Args:
        prices: Wide-format DataFrame (DatetimeIndex x symbol columns).
        spans: EWM span values.

    Returns:
        Dict keyed ``"ema_{span}"`` -> DataFrame.
    """
    return {f"ema_{span}": prices.ewm(span=span).mean() for span in spans}
