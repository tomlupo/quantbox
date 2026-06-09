"""Vectorized rolling covariance matrices with shrinkage estimators.

Computes covariance matrices for multiple dates and model configurations
in a single pass using numpy broadcasting, avoiding per-date Python loops.

Two shrinkage methods are supported, both targeting a scaled identity
matrix ``F = (tr(S)/p) * I``:

* **Ledoit-Wolf (2003/2004)** — the industry-standard shrinkage; the one
  used by ``sklearn.covariance.LedoitWolf`` and ``pypfopt``.  The
  shrinkage intensity is a closed-form ratio of second-moment statistics
  from the raw returns.  Implemented vectorized across dates.
* **Oracle Approximating Shrinkage (OAS)** (Chen et al. 2010) — a
  tighter closed-form shrinkage that depends only on moments of ``S``
  itself; asymptotically optimal under Gaussian assumptions.

Both shrinkage coefficients are evaluated in pure numpy over a
``(n_dates, p, p)`` stack of empirical covariance matrices.
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


# --- Ledoit-Wolf shrinkage (vectorized) ---


def _batch_ledoit_wolf_shrinkage(
    covs: np.ndarray,
    sum_quadratic: np.ndarray,
    n_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Ledoit-Wolf shrinkage to a batch of empirical covariance matrices.

    Shrinks each ``S`` toward ``F = (tr(S) / p) * I`` with the
    intensity derived by Ledoit & Wolf (2003) — the same estimator
    implemented in ``sklearn.covariance.LedoitWolf``.  The sklearn API
    is per-matrix; this function vectorizes the closed form across a
    ``(n_dates, p, p)`` stack using only numpy broadcasting.

    Assumes ``covs`` was computed with the biased divisor ``n``
    (matching the derivation in the paper and in sklearn).

    Parameters
    ----------
    covs : ndarray, shape (n_dates, p, p)
        Biased empirical covariance matrices (``S = X'X / n``).
    sum_quadratic : ndarray, shape (n_dates,)
        Per-date ``sum_t ||x_t||^4`` over the centered returns used to
        build each ``S``.  Produced by ``_compute_expanding_covs``.
    n_samples : ndarray, shape (n_dates,)
        Number of observations ``n`` used for each ``S``.

    Returns
    -------
    shrunk_covs : ndarray, shape (n_dates, p, p)
    shrinkage_coeffs : ndarray, shape (n_dates,)
    """
    n_dates, p, _ = covs.shape

    # mu = trace(S) / p  -> the common eigenvalue of the scaled-identity target
    mu = np.trace(covs, axis1=1, axis2=2) / p  # (n_dates,)

    # d^2 = ||S - mu*I||^2_F / p
    #       ||A||^2_F = sum of squared elements
    eye_p = np.eye(p)[None, :, :]
    shifted = covs - mu[:, None, None] * eye_p
    d_sq = np.sum(shifted**2, axis=(1, 2)) / p  # (n_dates,)

    # b_bar^2 = (1 / (p * n^2)) * sum_t ||x_t x_t' - S||^2_F
    # Expanding the inner term with biased S = (1/n) sum_t x_t x_t' gives:
    #   sum_t ||x_t x_t' - S||^2_F = sum_t ||x_t||^4 - n * ||S||^2_F
    s_fro_sq = np.sum(covs**2, axis=(1, 2))  # (n_dates,)
    b_bar_sq = (sum_quadratic - n_samples * s_fro_sq) / (p * n_samples**2)

    # b^2 = min(b_bar^2, d^2), a^2 = d^2 - b^2
    # Shrinkage intensity rho = b^2 / d^2 in [0, 1]
    b_sq = np.clip(b_bar_sq, 0.0, d_sq)
    d_sq_safe = np.where(d_sq < 1e-16, 1e-16, d_sq)
    shrinkage = np.clip(b_sq / d_sq_safe, 0.0, 1.0)

    # Shrunk = (1 - rho) * S + rho * mu * I
    shrunk = (1.0 - shrinkage[:, None, None]) * covs
    diag_add = (shrinkage * mu)[:, None, None] * eye_p
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
    biased: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Compute empirical covariance matrices for each date.

    Uses an expanding window (all data up to each date) after optional
    resampling and return smoothing.

    Parameters
    ----------
    biased : bool
        If True, divide by ``n`` (MLE / sklearn convention used by the
        Ledoit-Wolf derivation).  If False, divide by ``n-1`` (unbiased
        sample covariance, default for OAS).

    Returns
    -------
    covs : ndarray, shape (n_dates, p, p)
        Empirical covariance matrices.
    sum_quadratic : ndarray, shape (n_dates,)
        Sum over observations of ``||x_t||^4`` for the centered returns.
        Required by the Ledoit-Wolf shrinkage formula; always computed
        because the cost is negligible vs the covariance itself.
    n_samples : ndarray, shape (n_dates,)
        Number of observations used for each covariance estimate.
    tickers : list[str]
    """
    tickers = prices.columns.tolist()
    p = len(tickers)
    n_dates = len(dates)

    covs = np.zeros((n_dates, p, p))
    sum_quadratic = np.zeros(n_dates)
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
        denom = n if biased else (n - 1)
        covs[i] = X_centered.T @ X_centered / denom
        # ||x_t||^4 for each row; then sum over t — needed by LW.
        row_norm_sq = (X_centered**2).sum(axis=1)
        sum_quadratic[i] = (row_norm_sq**2).sum()

    return covs, sum_quadratic, n_samples, tickers


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
        covs, _sum_quad, n_samples, tickers = _compute_expanding_covs(prices, dates, freq=model.freq, roll=model.roll)
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


# --- Ledoit-Wolf rolling covariance ---


def rolling_covariance_lw(
    prices: pd.DataFrame,
    models: list[CovarianceModelConfig] | None = None,
    dates: pd.DatetimeIndex | None = None,
    warmup_periods: int = 253,
    periods: int | None = None,
    annualize: bool = True,
    output_correlation: bool = False,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compute rolling covariance matrices with vectorized Ledoit-Wolf shrinkage.

    Same signature and output format as :func:`rolling_covariance_oas`; the
    only difference is the shrinkage estimator.  Uses the biased empirical
    covariance (``S = X'X/n``) to match the derivation in Ledoit & Wolf
    (2003) and the implementation in ``sklearn.covariance.LedoitWolf``.

    Parameters
    ----------
    See :func:`rolling_covariance_oas`.

    Returns
    -------
    dict[str, DataFrame]
        Keyed by model name.  Each DataFrame has MultiIndex
        ``(date, ticker)`` with ticker columns.
    """
    import logging

    logger = logging.getLogger(__name__)

    if models is None:
        models = [
            CovarianceModelConfig(method="ledoit_wolf", window=52, roll=1, freq="W"),
            CovarianceModelConfig(method="ledoit_wolf", window=63, roll=3, freq="B"),
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

        # LW requires biased S and the per-date sum of ||x_t||^4
        covs, sum_quadratic, n_samples, tickers = _compute_expanding_covs(
            prices, dates, freq=model.freq, roll=model.roll, biased=True
        )

        # Vectorized Ledoit-Wolf shrinkage
        shrunk_covs, _shrinkage_coeffs = _batch_ledoit_wolf_shrinkage(covs, sum_quadratic, n_samples)

        if annualize:
            factor = _annualization_factor(model.freq)
            shrunk_covs = shrunk_covs * factor

        if output_correlation:
            for i in range(len(dates)):
                diag = np.sqrt(np.diag(shrunk_covs[i]))
                diag = np.where(diag == 0, 1.0, diag)
                shrunk_covs[i] = shrunk_covs[i] / np.outer(diag, diag)

        frames = []
        for i, dt in enumerate(dates):
            df = pd.DataFrame(
                shrunk_covs[i],
                index=pd.Index(tickers, name="ticker"),
                columns=tickers,
            )
            df.index = pd.MultiIndex.from_tuples([(dt, t) for t in tickers], names=["date", "ticker"])
            frames.append(df)

        results[model.name] = pd.concat(frames) if frames else pd.DataFrame()

    return results


def rolling_covariance_lw_from_config(
    prices: pd.DataFrame,
    method: list[str] | str | None = None,
    window: list[int] | int | None = None,
    roll: list[int] | int | None = None,
    freq: list[str] | str | None = None,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper matching robo's indicator config format.

    Parallel of :func:`rolling_covariance_oas_from_config`; delegates to
    :func:`rolling_covariance_lw`.
    """
    if method is None:
        method = ["ledoit_wolf"]
    if window is None:
        window = [52, 63]
    if roll is None:
        roll = [1, 3]
    if freq is None:
        freq = ["W", "B"]

    if isinstance(method, str):
        method = [method]
    if isinstance(window, int):
        window = [window]
    if isinstance(roll, int):
        roll = [roll]
    if isinstance(freq, str):
        freq = [freq]

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

    return rolling_covariance_lw(prices, models=models, **kwargs)


# ============================================================================
# EWMA + Ledoit-Wolf (constant-correlation target) + Bartlett-lag timing
# ============================================================================
#
# Methodology (ADR in the consumer repo):
#
# * Exponentially-weighted returns with configurable half-life — responsive
#   to recent regime while keeping older data at decaying weight (no abrupt
#   rolling-window boundary).
# * Rolling-sum returns (``roll`` > 1) absorb async end-of-day timing
#   across global markets.  A ``roll=3`` daily return is equivalent to a
#   Newey-West lag-2 covariance with Bartlett (triangular) kernel; this
#   form is PSD by construction.  Prod already uses ``roll=3`` on its
#   ``63B`` model, so the convention is preserved.
# * Ledoit-Wolf (2004) shrinkage toward a constant-correlation target
#   (``F_ij = r_bar * sqrt(s_ii * s_jj)``) rather than a scaled identity.
#   The constant-correlation target preserves the average correlation
#   structure, which matters more for mean-variance optimisation than
#   isotropy.  Effective sample size uses Kish's formula
#   ``n_eff = 1 / sum(w^2)`` to account for the EWMA weighting.


@dataclass
class EwmaLwModelConfig:
    """Configuration for one EWMA-LW-constant-correlation model variant."""

    half_life_obs: int = 63  # on observation scale (e.g. 63 business days)
    freq: str = "B"
    roll: int = 3  # rolling-sum span for Bartlett timing absorption
    method: str = "ewma_lw"

    @property
    def name(self) -> str:
        return f"ewma_lw_hl{self.half_life_obs}{self.freq}_roll{self.roll}"


def _ewma_weights(n: int, half_life_obs: float) -> np.ndarray:
    """Return EWMA weights for ``n`` observations, normalised to sum to 1.

    Weight of observation at age ``k`` (0 = newest, n-1 = oldest) is
    ``lambda^k`` with ``lambda = exp(-ln2 / half_life)``.
    """
    if n <= 0:
        return np.zeros(0)
    lam = np.exp(-np.log(2.0) / float(half_life_obs))
    ages = np.arange(n)[::-1].astype(float)
    w = lam**ages
    s = w.sum()
    return w / s if s > 0 else w


def _ewma_lw_const_corr_at_date(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """Vectorised EWMA weighted covariance with LW-constant-correlation shrinkage.

    Parameters
    ----------
    X : ndarray, shape (T, p)
        Centred returns (each column mean-zero under weights ``w``).
    w : ndarray, shape (T,)
        Weights summing to 1.

    Returns
    -------
    shrunk_cov : ndarray, shape (p, p)
    shrinkage : float in [0, 1]
    """
    T, p = X.shape
    if T < 2 or p < 2:
        return np.zeros((p, p)), 0.0

    # Weighted empirical covariance (biased-like; weights sum to 1).
    #   S_ij = sum_t w_t * X_ti * X_tj
    Xw = X * w[:, None]
    S = X.T @ Xw  # (p, p)

    s_diag = np.diag(S)
    # Guard against zero variance (constant column) to keep outer products safe
    s_diag_safe = np.where(s_diag > 1e-16, s_diag, 1e-16)
    D = np.sqrt(s_diag_safe)

    # Correlation matrix and mean off-diagonal correlation
    R = S / np.outer(D, D)
    mask_off = ~np.eye(p, dtype=bool)
    r_bar = R[mask_off].mean()

    # Constant-correlation target F
    F = r_bar * np.outer(D, D)
    np.fill_diagonal(F, s_diag)

    # pi_ij = E[(x_i * x_j)^2] - S_ij^2
    X_sq = X * X
    E_xi2_xj2 = X_sq.T @ (X_sq * w[:, None])  # (p, p)
    pi_mat = E_xi2_xj2 - S * S
    pi_sum = pi_mat.sum()
    rho_diag = np.trace(pi_mat)

    # A[i, j] = E[x_i * x_j^3]
    X_cubed = X_sq * X
    A = X.T @ (X_cubed * w[:, None])  # (p, p)

    # θ_ii,ij = E[(x_i^2 - s_ii)(x_i x_j - S_ij)] = A[j,i] - s_ii * S_ij
    # θ_jj,ij = E[(x_j^2 - s_jj)(x_i x_j - S_ij)] = A[i,j] - s_jj * S_ij
    theta_i = A.T - s_diag[:, None] * S
    theta_j = A - s_diag[None, :] * S

    # Off-diagonal rho contribution
    sqrt_j_over_i = np.sqrt(s_diag_safe[None, :] / s_diag_safe[:, None])
    sqrt_i_over_j = 1.0 / sqrt_j_over_i
    rho_off_mat = (r_bar / 2.0) * (sqrt_j_over_i * theta_i + sqrt_i_over_j * theta_j)
    np.fill_diagonal(rho_off_mat, 0.0)
    rho_off = rho_off_mat.sum()

    rho = rho_diag + rho_off

    # gamma = ||S - F||^2_F
    diff = S - F
    gamma = float(np.sum(diff * diff))
    gamma = max(gamma, 1e-16)

    # Effective sample size (Kish) and shrinkage intensity
    n_eff = 1.0 / float(np.sum(w * w))
    kappa = (pi_sum - rho) / gamma
    delta = float(np.clip(kappa / max(n_eff, 1.0), 0.0, 1.0))

    shrunk = delta * F + (1.0 - delta) * S
    return shrunk, delta


def _compute_ewma_lw_at_date(
    prices_slice: pd.DataFrame,
    freq: str,
    roll: int,
    half_life_obs: float,
) -> tuple[np.ndarray, float, int]:
    """Resample, build rolling-sum returns, compute EWMA-LW-const-corr covariance."""
    if freq.upper() not in ("B", "D"):
        prices_resampled = prices_slice.resample(freq.upper()).last().dropna(how="all")
    else:
        prices_resampled = prices_slice

    # ``smoothed_returns`` convention: pct_change(periods=roll) / roll.
    # For roll=1 this is the plain pct_change; for roll=3 it is the
    # Bartlett-lag-2 rolling-sum on daily-scale.
    rets = prices_resampled.pct_change(periods=roll) / roll
    rets = rets.iloc[roll:].dropna(how="any")

    if rets.empty:
        p = prices_resampled.shape[1]
        return np.zeros((p, p)), 0.0, 0

    X = rets.values.astype(float)
    n = X.shape[0]
    w = _ewma_weights(n, half_life_obs)

    # Centre about the weighted mean (matches the LW derivation assumption)
    w_mean = (w[:, None] * X).sum(axis=0)
    X_centred = X - w_mean
    shrunk, delta = _ewma_lw_const_corr_at_date(X_centred, w)
    return shrunk, delta, n


def rolling_covariance_ewma_lw(
    prices: pd.DataFrame,
    models: list[EwmaLwModelConfig] | None = None,
    dates: pd.DatetimeIndex | None = None,
    warmup_periods: int = 253,
    periods: int | None = None,
    annualize: bool = True,
    output_correlation: bool = False,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Rolling covariance matrices with EWMA weighting + LW-constant-correlation shrinkage.

    For each model in ``models``, compute a date-indexed stack of
    covariance matrices where each matrix is the EWMA-weighted
    empirical covariance of rolling-sum returns, shrunk toward a
    constant-correlation target by the Ledoit-Wolf (2004) closed-form
    intensity.

    Output shape matches :func:`rolling_covariance_lw`.
    """
    import logging

    logger = logging.getLogger(__name__)

    if models is None:
        models = [
            EwmaLwModelConfig(half_life_obs=63, freq="B", roll=3),
            EwmaLwModelConfig(half_life_obs=52, freq="W", roll=1),
        ]

    if dates is None:
        clean = prices.dropna(how="any")
        dates = clean.index[warmup_periods:]
    if periods is not None:
        dates = dates[-periods:]

    tickers = prices.columns.tolist()
    p = len(tickers)

    results: dict[str, pd.DataFrame] = {}

    for model in models:
        if show_progress:
            logger.info("Computing covariance: %s (%d dates)", model.name, len(dates))

        covs = np.zeros((len(dates), p, p))
        n_samples = np.zeros(len(dates))
        for i, dt in enumerate(dates):
            shrunk, _delta, n = _compute_ewma_lw_at_date(
                prices.loc[:dt],
                freq=model.freq,
                roll=model.roll,
                half_life_obs=model.half_life_obs,
            )
            covs[i] = shrunk
            n_samples[i] = n

        if annualize:
            factor = _annualization_factor(model.freq)
            covs = covs * factor

        if output_correlation:
            for i in range(len(dates)):
                diag = np.sqrt(np.diag(covs[i]))
                diag = np.where(diag == 0, 1.0, diag)
                covs[i] = covs[i] / np.outer(diag, diag)

        frames = []
        for i, dt in enumerate(dates):
            df = pd.DataFrame(
                covs[i],
                index=pd.Index(tickers, name="ticker"),
                columns=tickers,
            )
            df.index = pd.MultiIndex.from_tuples([(dt, t) for t in tickers], names=["date", "ticker"])
            frames.append(df)

        results[model.name] = pd.concat(frames) if frames else pd.DataFrame()

    return results


def rolling_covariance_ewma_lw_from_config(
    prices: pd.DataFrame,
    half_life_obs: list[int] | int | None = None,
    freq: list[str] | str | None = None,
    roll: list[int] | int | None = None,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper with the list-broadcast config format.

    Defaults to a responsive-daily + stable-weekly ensemble:
      - half_life_obs=63, freq=B, roll=3    (3-month half-life, Bartlett lag-2)
      - half_life_obs=52, freq=W, roll=1    (1-year half-life on weekly scale)
    """
    if half_life_obs is None:
        half_life_obs = [63, 52]
    if freq is None:
        freq = ["B", "W"]
    if roll is None:
        roll = [3, 1]

    if isinstance(half_life_obs, int):
        half_life_obs = [half_life_obs]
    if isinstance(freq, str):
        freq = [freq]
    if isinstance(roll, int):
        roll = [roll]

    max_len = max(len(half_life_obs), len(freq), len(roll))

    def _pad(lst: list, n: int) -> list:
        return lst + [None] * (n - len(lst))

    params_df = pd.DataFrame(
        {
            "half_life_obs": _pad(half_life_obs, max_len),
            "freq": _pad(freq, max_len),
            "roll": _pad(roll, max_len),
        }
    ).ffill()

    models = [
        EwmaLwModelConfig(
            half_life_obs=int(row.half_life_obs),
            freq=str(row.freq),
            roll=int(row.roll),
        )
        for _, row in params_df.iterrows()
    ]

    return rolling_covariance_ewma_lw(prices, models=models, **kwargs)
