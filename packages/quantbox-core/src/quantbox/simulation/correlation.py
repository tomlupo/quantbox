"""Correlation structures for multi-asset simulation.

Implements various correlation estimation methods:
- Static (sample) correlation
- Rolling window correlation
- Exponentially Weighted Moving Average (EWMA / RiskMetrics)
- DCC-GARCH (Engle 2002)
- Ledoit-Wolf shrinkage
- Stress correlation
- Regime detection

Ported from quantlabnew/src/market-simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

try:
    from scipy.optimize import minimize as _scipy_minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from arch import arch_model as _arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


@dataclass
class CorrelationResult:
    """Container for correlation analysis results."""
    current_correlation: np.ndarray
    correlation_history: Optional[np.ndarray] = None
    asset_names: Optional[list[str]] = None
    method: str = "static"

    def to_dataframe(self) -> pd.DataFrame:
        names = self.asset_names or [f"Asset_{i}" for i in range(len(self.current_correlation))]
        return pd.DataFrame(self.current_correlation, index=names, columns=names)

    def get_correlation_at(self, t: int) -> np.ndarray:
        if self.correlation_history is None:
            return self.current_correlation
        return self.correlation_history[t]


class CorrelationEngine:
    """Engine for computing and forecasting correlation structures."""

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.asset_names = list(returns.columns)
        self.n_assets = len(self.asset_names)

    def static_correlation(self) -> CorrelationResult:
        """Compute sample correlation matrix."""
        corr = self.returns.corr().values
        return CorrelationResult(
            current_correlation=corr,
            asset_names=self.asset_names,
            method="static",
        )

    def rolling_correlation(self, window: int = 60, min_periods: int = 30) -> CorrelationResult:
        """Compute rolling window correlation."""
        n_obs = len(self.returns)
        n_valid = n_obs - window + 1
        corr_history = np.zeros((n_valid, self.n_assets, self.n_assets))

        for t in range(n_valid):
            window_data = self.returns.iloc[t : t + window]
            corr_history[t] = window_data.corr().values

        return CorrelationResult(
            current_correlation=corr_history[-1],
            correlation_history=corr_history,
            asset_names=self.asset_names,
            method=f"rolling_{window}",
        )

    def ewma_correlation(self, lambda_param: float = 0.94, min_periods: int = 30) -> CorrelationResult:
        """Compute EWMA correlation (RiskMetrics style)."""
        returns_arr = self.returns.values
        n_obs, n_assets = returns_arr.shape

        cov = np.cov(returns_arr[:min_periods].T)
        corr_history = np.zeros((n_obs - min_periods, n_assets, n_assets))

        for t in range(min_periods, n_obs):
            r = returns_arr[t - 1]
            outer = np.outer(r, r)
            cov = lambda_param * cov + (1 - lambda_param) * outer

            std = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std, std)
            np.fill_diagonal(corr, 1.0)
            corr_history[t - min_periods] = corr

        return CorrelationResult(
            current_correlation=corr_history[-1],
            correlation_history=corr_history,
            asset_names=self.asset_names,
            method=f"ewma_{lambda_param}",
        )

    def dcc_correlation(self, a: float = 0.05, b: float = 0.93) -> CorrelationResult:
        """Compute DCC (Dynamic Conditional Correlation) model (Engle 2002).

        Requires the ``arch`` package.
        """
        if not HAS_ARCH:
            raise ImportError(
                "The 'arch' package is required for DCC. "
                "Install with: uv pip install 'quantbox[simulation]'"
            )
        returns_arr = self.returns.values
        n_obs, n_assets = returns_arr.shape

        standardized = np.zeros_like(returns_arr)
        for i in range(n_assets):
            model = _arch_model(returns_arr[:, i] * 100, vol="Garch", p=1, q=1)
            result = model.fit(disp="off")
            cond_vol = result.conditional_volatility / 100
            standardized[:, i] = returns_arr[:, i] / (cond_vol + 1e-8)

        Q_bar = np.corrcoef(standardized.T)
        Q = Q_bar.copy()
        corr_history = np.zeros((n_obs, n_assets, n_assets))

        for t in range(n_obs):
            eps = standardized[t]
            outer = np.outer(eps, eps)
            Q = (1 - a - b) * Q_bar + a * outer + b * Q

            Q_diag_sqrt = np.sqrt(np.diag(Q))
            R = Q / np.outer(Q_diag_sqrt, Q_diag_sqrt)
            np.fill_diagonal(R, 1.0)
            corr_history[t] = R

        return CorrelationResult(
            current_correlation=corr_history[-1],
            correlation_history=corr_history,
            asset_names=self.asset_names,
            method=f"dcc_a{a}_b{b}",
        )

    def fit_dcc(self) -> tuple[float, float, CorrelationResult]:
        """Fit DCC parameters via maximum likelihood.

        Requires ``arch`` and ``scipy``.
        """
        if not HAS_ARCH:
            raise ImportError("The 'arch' package is required for DCC fitting.")
        if not HAS_SCIPY:
            raise ImportError("scipy is required for DCC fitting.")

        returns_arr = self.returns.values
        n_obs, n_assets = returns_arr.shape

        standardized = np.zeros_like(returns_arr)
        for i in range(n_assets):
            model = _arch_model(returns_arr[:, i] * 100, vol="Garch", p=1, q=1)
            result = model.fit(disp="off")
            cond_vol = result.conditional_volatility / 100
            standardized[:, i] = returns_arr[:, i] / (cond_vol + 1e-8)

        Q_bar = np.corrcoef(standardized.T)

        def neg_log_likelihood(params):
            a, b = params
            if a < 0 or b < 0 or a + b >= 1:
                return 1e10
            Q = Q_bar.copy()
            ll = 0.0
            for t in range(n_obs):
                eps = standardized[t]
                outer = np.outer(eps, eps)
                Q = (1 - a - b) * Q_bar + a * outer + b * Q
                Q_diag_sqrt = np.sqrt(np.diag(Q))
                R = Q / np.outer(Q_diag_sqrt, Q_diag_sqrt)
                np.fill_diagonal(R, 1.0)
                try:
                    ll += -0.5 * (np.log(np.linalg.det(R)) + eps @ np.linalg.solve(R, eps))
                except np.linalg.LinAlgError:
                    return 1e10
            return -ll

        result = _scipy_minimize(
            neg_log_likelihood,
            x0=[0.05, 0.90],
            bounds=[(0.001, 0.3), (0.5, 0.99)],
            method="L-BFGS-B",
        )
        a_opt, b_opt = result.x
        corr_result = self.dcc_correlation(a_opt, b_opt)
        return a_opt, b_opt, corr_result

    def ledoit_wolf_shrinkage(self) -> CorrelationResult:
        """Ledoit-Wolf shrinkage toward identity matrix."""
        returns_arr = self.returns.values
        n_obs, n_assets = returns_arr.shape

        std_returns = (returns_arr - returns_arr.mean(axis=0)) / returns_arr.std(axis=0)
        sample_corr = np.corrcoef(std_returns.T)
        target = np.eye(n_assets)

        X = std_returns
        pi_sum = 0.0
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    wij = X[:, i] * X[:, j]
                    pi_sum += np.var(wij)

        gamma = np.sum((sample_corr - target) ** 2)
        kappa = pi_sum / (gamma * n_obs)
        shrinkage = max(0.0, min(1.0, kappa))
        shrunk_corr = shrinkage * target + (1 - shrinkage) * sample_corr

        return CorrelationResult(
            current_correlation=shrunk_corr,
            asset_names=self.asset_names,
            method=f"ledoit_wolf_{shrinkage:.3f}",
        )

    def correlation_stress(self, stress_factor: float = 1.5, floor: float = 0.0) -> np.ndarray:
        """Generate stressed correlation matrix (push off-diag toward 1)."""
        base_corr = self.static_correlation().current_correlation
        stressed = base_corr.copy()

        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if i != j:
                    stressed[i, j] = np.clip(stress_factor * base_corr[i, j], floor, 0.999)

        return self._nearest_positive_definite(stressed)

    def forecast_correlation(self, n_steps: int = 21, method: str = "ewma", **kwargs) -> np.ndarray:
        """Forecast future correlation matrices.

        Returns array of shape ``(n_steps, n_assets, n_assets)``.
        """
        if method == "static":
            current = self.static_correlation().current_correlation
            return np.tile(current, (n_steps, 1, 1))

        elif method == "ewma":
            ewma_result = self.ewma_correlation(**kwargs)
            current = ewma_result.current_correlation
            unconditional = self.static_correlation().current_correlation
            lambda_param = kwargs.get("lambda_param", 0.94)
            forecasts = np.zeros((n_steps, self.n_assets, self.n_assets))
            for t in range(n_steps):
                weight = lambda_param**t
                forecasts[t] = weight * current + (1 - weight) * unconditional
            return forecasts

        elif method == "dcc":
            _, b, dcc_result = self.fit_dcc()
            current = dcc_result.current_correlation
            unconditional = self.static_correlation().current_correlation
            forecasts = np.zeros((n_steps, self.n_assets, self.n_assets))
            for t in range(n_steps):
                weight = b**t
                forecasts[t] = weight * current + (1 - weight) * unconditional
            return forecasts

        raise ValueError(f"Unknown method: {method}")

    def correlation_regime_detection(self, window: int = 60, threshold: float = 0.3) -> pd.DataFrame:
        """Detect correlation regime changes."""
        rolling = self.rolling_correlation(window=window)
        avg_corr = []
        for t in range(len(rolling.correlation_history)):
            corr = rolling.correlation_history[t]
            mask = ~np.eye(self.n_assets, dtype=bool)
            avg_corr.append(np.mean(corr[mask]))

        avg_corr = np.array(avg_corr)
        regime_changes = np.abs(np.diff(avg_corr)) > threshold

        return pd.DataFrame({
            "average_correlation": avg_corr,
            "regime_change": np.concatenate([[False], regime_changes]),
        })

    # -- internal helpers --

    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """Nearest PSD matrix via Higham's algorithm."""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if self._is_positive_definite(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        return A3

    def _is_positive_definite(self, A: np.ndarray) -> bool:
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False


def generate_random_correlation_matrix(
    n: int,
    eigenvalue_concentration: float = 0.8,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate a random valid correlation matrix."""
    rng = np.random.default_rng(random_state)
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)

    eigenvalues = np.zeros(n)
    eigenvalues[0] = eigenvalue_concentration * n
    remaining = (1 - eigenvalue_concentration) * n
    eigenvalues[1:] = remaining / (n - 1)

    corr = Q @ np.diag(eigenvalues) @ Q.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


__all__ = [
    "CorrelationResult",
    "CorrelationEngine",
    "generate_random_correlation_matrix",
]
