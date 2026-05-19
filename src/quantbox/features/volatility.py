from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_vol(
    prices: pd.DataFrame,
    windows: list[int],
    *,
    annualize: bool = True,
    factor: float = 365.0,
) -> dict[str, pd.DataFrame]:
    """Compute rolling volatility of log returns.

    Args:
        prices: Wide-format DataFrame (DatetimeIndex x symbol columns).
        windows: Rolling window sizes.
        annualize: Whether to annualize (multiply by sqrt(factor)).
        factor: Annualization factor (365 for crypto, 252 for equities).

    Returns:
        Dict keyed ``"vol_{w}d"`` -> DataFrame of volatilities.
    """
    log_ret = np.log(prices / prices.shift(1))
    result: dict[str, pd.DataFrame] = {}
    for w in windows:
        vol = log_ret.rolling(window=w).std()
        if annualize:
            vol = vol * np.sqrt(factor)
        result[f"vol_{w}d"] = vol
    return result


def compute_ewm_vol(
    prices: pd.DataFrame,
    spans: list[int],
    *,
    annualize: bool = True,
    factor: float = 365.0,
) -> dict[str, pd.DataFrame]:
    """Compute exponentially weighted volatility of log returns.

    Args:
        prices: Wide-format DataFrame (DatetimeIndex x symbol columns).
        spans: EWM span values.
        annualize: Whether to annualize.
        factor: Annualization factor.

    Returns:
        Dict keyed ``"vol_ewm_{span}"`` -> DataFrame of volatilities.
    """
    log_ret = np.log(prices / prices.shift(1))
    result: dict[str, pd.DataFrame] = {}
    for span in spans:
        vol = log_ret.ewm(span=span).std()
        if annualize:
            vol = vol * np.sqrt(factor)
        result[f"vol_ewm_{span}"] = vol
    return result
