"""Vectorized rolling covariance matrices with OAS shrinkage.

Computes covariance matrices for multiple dates and model configurations
in a single pass using numpy broadcasting, avoiding per-date Python loops.

The Oracle Approximating Shrinkage (OAS) estimator (Chen et al. 2010) has
a closed-form shrinkage coefficient that can be vectorized over a 3D array
of empirical covariance matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# --- Frequency helpers ---

_FREQ_TO_PERIODS: dict[str, int] = {
    "B": 252,
    "D": 365,
    "W": 52,
    "M": 12,
    "Q": 4,
    "Y": 1,
}


def _annualization_factor(freq: str) -> int:
    return _FREQ_TO_PERIODS.get(freq.upper(), 252)


def _resample_to_freq(
    prices: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """Resample prices to a lower frequency using last observation."""
    if freq.upper() in ("B", "D"):
        return prices
    return prices.resample(freq.upper()).last().dropna(how="all")


def _smoothed_returns(prices: pd.DataFrame, roll: int = 1) -> np.ndarray:
    """Average rolling returns. Returns numpy array (n_obs, n_assets)."""
    rets = prices.pct_change(periods=roll) / roll
    return rets.iloc[roll:].values


# --- OAS shrinkage (vectorized) ---


def _batch_oas_shrinkage(
    covs: np.ndarray,
    n_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply OAS shrinkage to a batch of empirical covariance matrices.

    Parameters
    ----------
    covs : ndarray, shape (n_dates, p, p)
        Empirical covariance matrices.
    n_samples : ndarray, shape (n_dates,)
        Number of observations used for each covariance estimate.

    Returns
    -------
    shrunk_covs : ndarray, shape (n_dates, p, p)
    shrinkage_coeffs : ndarray, shape (n_dates,)
    """
    n_dates, p, _ = covs.shape

    # OAS closed-form: Chen et al. 2010
    # alpha = mean(S_ij^2)  (Frobenius norm squared / p^2)
    alpha = np.mean(covs**2, axis=(1, 2))  # (n_dates,)
    mu = np.trace(covs, axis1=1, axis2=2) / p  # (n_dates,)
    mu_sq = mu**2

    num = alpha + mu_sq
    den = (n_samples + 1.0) * (alpha - mu_sq / p)
    # Avoid division by zero
    den = np.where(np.abs(den) < 1e-16, 1e-16, den)
    shrinkage = np.clip(num / den, 0.0, 1.0)

    # Apply shrinkage: (1 - delta) * S + delta * mu * I
    shrunk = (1.0 - shrinkage[:, None, None]) * covs
    # Add shrinkage * mu to diagonal
    diag_add = (shrinkage * mu)[:, None, None] * np.eye(p)[None, :, :]
    shrunk = shrunk + diag_add

    return shrunk, shrinkage


# --- Rolling covariance computation ---


@dataclass
class CovarianceModelConfig:
    """Configuration for one covariance model variant."""

    method: str = "oas"
    window: int = 52
    roll: int = 1
    freq: str = "W"

    @property
    def name(self) -> str:
        return f"{self.method}_{self.window}{self.freq}_roll{self.roll}"


def _compute_expanding_covs(
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
    freq: str,
    roll: int,
    window: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute empirical covariance matrices for each date.

    Uses an expanding window (all data up to each date) after optional
    resampling and return smoothing.

    Returns
    -------
    covs : ndarray, shape (n_dates, p, p)
    n_samples : ndarray, shape (n_dates,)
    tickers : list[str]
    """
    tickers = prices.columns.tolist()
    p = len(tickers)
    n_dates = len(dates)

    covs = np.zeros((n_dates, p, p))
    n_samples = np.zeros(n_dates)

    for i, dt in enumerate(dates):
        prices_slice = prices.loc[:dt]
        prices_resampled = _resample_to_freq(prices_slice, freq)
        rets = _smoothed_returns(prices_resampled, roll=roll)

        # Drop rows with any NaN (incomplete observations)
        nan_mask = ~np.isnan(rets).any(axis=1)
        rets = rets[nan_mask]

        if window is not None and len(rets) > window:
            rets = rets[-window:]

        n = rets.shape[0]
        n_samples[i] = n

        if n < 2:
            covs[i] = np.zeros((p, p))
            continue

        X_centered = rets - rets.mean(axis=0)
        covs[i] = X_centered.T @ X_centered / (n - 1)

    return covs, n_samples, tickers


def rolling_covariance_oas(
    prices: pd.DataFrame,
    models: list[CovarianceModelConfig] | None = None,
    dates: pd.DatetimeIndex | None = None,
    warmup_periods: int = 253,
    periods: int | None = None,
    annualize: bool = True,
    output_correlation: bool = False,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compute rolling covariance matrices with vectorized OAS shrinkage.

    Parameters
    ----------
    prices : DataFrame
        Wide-format prices (DatetimeIndex x ticker columns).
    models : list[CovarianceModelConfig], optional
        Model configurations. Defaults to two models matching the robo
        prod config: ``ledoit_wolf_52W_roll1`` and ``ledoit_wolf_63B_roll3``
        (renamed ``oas_*`` since we use OAS shrinkage).
    dates : DatetimeIndex, optional
        Dates to compute covariance for. Defaults to prices index after
        warmup_periods.
    warmup_periods : int
        Number of initial observations to skip (default 253).
    periods : int, optional
        If set, only compute for the last N dates.
    annualize : bool
        Annualize the covariance matrices (default True).
    output_correlation : bool
        If True, convert covariance to correlation (default False).
    show_progress : bool
        Log progress (default True).

    Returns
    -------
    dict[str, DataFrame]
        Keyed by model name. Each DataFrame has MultiIndex ``(date, ticker)``
        with ticker columns — same format as the robo ``covariance()``
        indicator.
    """
    import logging

    logger = logging.getLogger(__name__)

    if models is None:
        models = [
            CovarianceModelConfig(method="oas", window=52, roll=1, freq="W"),
            CovarianceModelConfig(method="oas", window=63, roll=3, freq="B"),
        ]

    if dates is None:
        clean = prices.dropna(how="any")
        dates = clean.index[warmup_periods:]
    if periods is not None:
        dates = dates[-periods:]

    results: dict[str, pd.DataFrame] = {}

    for model in models:
        if show_progress:
            logger.info("Computing covariance: %s (%d dates)", model.name, len(dates))

        # Compute empirical covariances for all dates at once
        covs, n_samples, tickers = _compute_expanding_covs(prices, dates, freq=model.freq, roll=model.roll)
        p = len(tickers)

        # Vectorized OAS shrinkage
        shrunk_covs, shrinkage_coeffs = _batch_oas_shrinkage(covs, n_samples)

        # Annualize
        if annualize:
            factor = _annualization_factor(model.freq)
            shrunk_covs = shrunk_covs * factor

        # Convert to correlation if requested
        if output_correlation:
            for i in range(len(dates)):
                diag = np.sqrt(np.diag(shrunk_covs[i]))
                diag = np.where(diag == 0, 1.0, diag)
                shrunk_covs[i] = shrunk_covs[i] / np.outer(diag, diag)

        # Reshape to DataFrame with (date, ticker) MultiIndex
        frames = []
        for i, dt in enumerate(dates):
            df = pd.DataFrame(
                shrunk_covs[i],
                index=pd.Index(tickers, name="ticker"),
                columns=tickers,
            )
            df.index = pd.MultiIndex.from_tuples([(dt, t) for t in tickers], names=["date", "ticker"])
            frames.append(df)

        if frames:
            results[model.name] = pd.concat(frames)
        else:
            results[model.name] = pd.DataFrame()

    return results


def rolling_covariance_oas_from_config(
    prices: pd.DataFrame,
    method: list[str] | str | None = None,
    window: list[int] | int | None = None,
    roll: list[int] | int | None = None,
    freq: list[str] | str | None = None,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper matching robo's indicator config format.

    Accepts the same parameter-list broadcasting as the original robo
    ``covariance()`` indicator and delegates to ``rolling_covariance_oas()``.
    """
    if method is None:
        method = ["oas"]
    if window is None:
        window = [52, 63]
    if roll is None:
        roll = [1, 3]
    if freq is None:
        freq = ["W", "B"]

    # Ensure lists
    if isinstance(method, str):
        method = [method]
    if isinstance(window, int):
        window = [window]
    if isinstance(roll, int):
        roll = [roll]
    if isinstance(freq, str):
        freq = [freq]

    # Broadcast: pad shorter lists to max length, then ffill NaNs
    max_len = max(len(method), len(window), len(roll), len(freq))

    def _pad(lst: list, n: int) -> list:
        return lst + [None] * (n - len(lst))

    params_df = pd.DataFrame(
        {
            "method": _pad(method, max_len),
            "window": _pad(window, max_len),
            "freq": _pad(freq, max_len),
            "roll": _pad(roll, max_len),
        }
    ).ffill()

    models = [
        CovarianceModelConfig(
            method=str(row.method),
            window=int(row.window),
            roll=int(row.roll),
            freq=str(row.freq),
        )
        for _, row in params_df.iterrows()
    ]

    return rolling_covariance_oas(prices, models=models, **kwargs)
