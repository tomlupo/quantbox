from __future__ import annotations

import numpy as np
import pandas as pd


def _returns(prices: pd.DataFrame, method: str) -> pd.DataFrame:
    """Compute returns from a price frame. ``method`` ∈ {``"log"``, ``"pct_change"``}."""
    if method == "log":
        return np.log(prices / prices.shift(1))
    if method == "pct_change":
        return prices.pct_change(fill_method=None)
    raise ValueError(f"returns_method must be 'log' or 'pct_change', got {method!r}")


def compute_rolling_vol(
    prices: pd.DataFrame,
    windows: list[int],
    *,
    returns_method: str = "log",
    annualize: bool = True,
    factor: float = 365.0,
) -> dict[str, pd.DataFrame]:
    """Compute rolling volatility.

    Args:
        prices: Wide-format DataFrame (DatetimeIndex x symbol columns).
        windows: Rolling window sizes.
        returns_method: ``"log"`` (default) or ``"pct_change"``.
            Equity-convention compute often prefers ``pct_change`` so
            the estimator matches the sample-volatility definition
            used in benchmarks and research tooling; crypto / risk
            frameworks typically use ``log``.
        annualize: Whether to annualize (multiply by sqrt(factor)).
        factor: Annualization factor (365 for crypto, 252 for equities).

    Returns:
        Dict keyed ``"vol_{w}d"`` -> DataFrame of volatilities.
    """
    ret = _returns(prices, returns_method)
    result: dict[str, pd.DataFrame] = {}
    for w in windows:
        vol = ret.rolling(window=w).std()
        if annualize:
            vol = vol * np.sqrt(factor)
        result[f"vol_{w}d"] = vol
    return result


def compute_ewm_vol(
    prices: pd.DataFrame,
    spans: list[int],
    *,
    returns_method: str = "log",
    annualize: bool = True,
    factor: float = 365.0,
) -> dict[str, pd.DataFrame]:
    """Exponentially weighted std of per-period returns.

    Computes ``returns.ewm(span=span).std()`` — an EWM over the
    standard deviation of returns. This is *not* the same as
    RiskMetrics volatility (see :func:`compute_riskmetrics_vol`),
    which smooths the squared-return *variance* instead and takes the
    square root only at the end.

    Args:
        prices: Wide-format DataFrame (DatetimeIndex x symbol columns).
        spans: EWM span values (pandas' ``span`` alpha-parameterisation).
        returns_method: ``"log"`` (default) or ``"pct_change"``.
        annualize: Whether to annualize.
        factor: Annualization factor.

    Returns:
        Dict keyed ``"vol_ewm_{span}"`` -> DataFrame of volatilities.
    """
    ret = _returns(prices, returns_method)
    result: dict[str, pd.DataFrame] = {}
    for span in spans:
        vol = ret.ewm(span=span).std()
        if annualize:
            vol = vol * np.sqrt(factor)
        result[f"vol_ewm_{span}"] = vol
    return result


def compute_riskmetrics_vol(
    prices: pd.DataFrame,
    alphas: list[float],
    *,
    returns_method: str = "pct_change",
    annualize: bool = True,
    factor: float = 252.0,
    warmup_trim: bool = True,
) -> dict[str, pd.DataFrame]:
    """RiskMetrics-style EWMA volatility.

    Computes ``sigma_t^2 = alpha * r_t^2 + (1 - alpha) * sigma_{t-1}^2``
    via ``squared_returns.ewm(alpha=alpha).mean()`` and returns the
    square root, optionally annualised.

    The semantics differ from :func:`compute_ewm_vol`:

    * RiskMetrics: EWMA over **variance** (squared returns), then sqrt.
    * ``compute_ewm_vol``: EWMA smoothing of the standard deviation.

    RiskMetrics is the industry-standard decay-based risk estimator.
    Parameterised by ``alpha`` directly (RiskMetrics conventions are
    usually expressed in terms of ``lambda = 1 - alpha``; pass
    ``alpha = 1 - lambda``, e.g. ``alpha = 0.06`` for ``lambda = 0.94``).

    Args:
        prices: Wide-format DataFrame (DatetimeIndex x symbol columns).
        alphas: List of alpha values (``= 1 - lambda``). Each produces
            one frame in the output dict.
        returns_method: ``"pct_change"`` (default — matches the
            RiskMetrics definition) or ``"log"``.
        annualize: Whether to annualize (multiply by sqrt(factor)).
        factor: Annualization factor (252 for equity/B-day, 365 for
            calendar / crypto).
        warmup_trim: Drop the first ``int(100 / alpha)`` rows per
            alpha to avoid under-converged initial values. Matches
            the conservative default used in RiskMetrics reference
            implementations.

    Returns:
        Dict keyed ``"vol_ewma_alpha{round(alpha*100)}"`` -> DataFrame.
    """
    ret = _returns(prices, returns_method)
    # RiskMetrics operates on the returns series from the first
    # observable return; drop the leading NaN introduced by shift(1).
    ret = ret.iloc[1:]
    squared = ret**2

    result: dict[str, pd.DataFrame] = {}
    scaling = np.sqrt(factor) if annualize else 1.0
    for alpha in alphas:
        ewma_var = squared.ewm(alpha=alpha).mean()
        vol = ewma_var.pow(0.5) * scaling
        if warmup_trim and alpha > 0:
            min_pts = int(100 / alpha)
            vol = vol.iloc[min_pts:]
        key = f"vol_ewma_alpha{round(alpha * 100)}"
        result[key] = vol
    return result
