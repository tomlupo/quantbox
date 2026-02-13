"""Stochastic price models for Monte Carlo simulation.

Implements various stochastic processes:
- Geometric Brownian Motion (GBM)
- Jump Diffusion (Merton model)
- Mean Reversion (Ornstein-Uhlenbeck)
- GARCH(1,1) with time-varying volatility
- Markov Regime Switching

Ported from quantlabnew/src/market-simulator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

try:
    from arch import arch_model as _arch_model

    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


@dataclass
class ModelParameters:
    """Base parameters for market models."""

    mu: float = 0.08  # Annual drift/expected return
    sigma: float = 0.20  # Annual volatility
    dt: float = 1 / 252  # Time step (daily by default)


@dataclass
class GBMParams(ModelParameters):
    """Parameters for Geometric Brownian Motion."""

    pass


@dataclass
class JumpDiffusionParams(ModelParameters):
    """Parameters for Jump Diffusion model."""

    jump_intensity: float = 0.1  # Expected jumps per year
    jump_mean: float = -0.05  # Mean jump size
    jump_std: float = 0.10  # Jump size volatility


@dataclass
class MeanReversionParams(ModelParameters):
    """Parameters for Ornstein-Uhlenbeck process."""

    theta: float = 0.5  # Mean reversion speed
    long_term_mean: float = 0.0  # Long-term mean level


@dataclass
class GARCHParams(ModelParameters):
    """Parameters for GARCH model."""

    omega: float = 0.00001  # Constant term
    alpha: float = 0.1  # ARCH coefficient
    beta: float = 0.85  # GARCH coefficient


class BaseModel(ABC):
    """Abstract base class for market models."""

    @abstractmethod
    def simulate(
        self,
        S0: float,
        n_steps: int,
        n_paths: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Simulate price paths.

        Returns array of shape ``(n_paths, n_steps + 1)``.
        """

    @abstractmethod
    def simulate_returns(
        self,
        n_steps: int,
        n_paths: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Simulate return paths.

        Returns array of shape ``(n_paths, n_steps)``.
        """


class GBM(BaseModel):
    """Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW."""

    def __init__(self, params: GBMParams | None = None):
        self.params = params or GBMParams()

    def simulate(self, S0, n_steps, n_paths, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = self.params.dt
        mu = self.params.mu
        sigma = self.params.sigma

        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        Z = rng.standard_normal((n_paths, n_steps))
        log_returns = drift + diffusion * Z

        log_prices = np.zeros((n_paths, n_steps + 1))
        log_prices[:, 0] = np.log(S0)
        log_prices[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)
        return np.exp(log_prices)

    def simulate_returns(self, n_steps, n_paths, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = self.params.dt
        mu = self.params.mu
        sigma = self.params.sigma

        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        Z = rng.standard_normal((n_paths, n_steps))
        log_returns = drift + diffusion * Z
        return np.exp(log_returns) - 1

    @classmethod
    def fit(cls, returns: np.ndarray, dt: float = 1 / 252) -> GBM:
        """Fit GBM parameters to historical returns."""
        mu = np.mean(returns) / dt + 0.5 * (np.std(returns) ** 2 / dt)
        sigma = np.std(returns) / np.sqrt(dt)
        return cls(GBMParams(mu=mu, sigma=sigma, dt=dt))


class JumpDiffusion(BaseModel):
    """Merton Jump Diffusion: dS = mu*S*dt + sigma*S*dW + S*dJ."""

    def __init__(self, params: JumpDiffusionParams | None = None):
        self.params = params or JumpDiffusionParams()

    def simulate(self, S0, n_steps, n_paths, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = self.params.dt
        mu = self.params.mu
        sigma = self.params.sigma
        lam = self.params.jump_intensity
        jump_mean = self.params.jump_mean
        jump_std = self.params.jump_std

        k = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        drift = (mu - 0.5 * sigma**2 - lam * k) * dt
        diffusion = sigma * np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps))
        N = rng.poisson(lam * dt, (n_paths, n_steps))
        J = rng.normal(jump_mean, jump_std, (n_paths, n_steps))
        log_returns = drift + diffusion * Z + N * J

        log_prices = np.zeros((n_paths, n_steps + 1))
        log_prices[:, 0] = np.log(S0)
        log_prices[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)
        return np.exp(log_prices)

    def simulate_returns(self, n_steps, n_paths, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = self.params.dt
        mu = self.params.mu
        sigma = self.params.sigma
        lam = self.params.jump_intensity
        jump_mean = self.params.jump_mean
        jump_std = self.params.jump_std

        k = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        drift = (mu - 0.5 * sigma**2 - lam * k) * dt
        diffusion = sigma * np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps))
        N = rng.poisson(lam * dt, (n_paths, n_steps))
        J = rng.normal(jump_mean, jump_std, (n_paths, n_steps))
        log_returns = drift + diffusion * Z + N * J
        return np.exp(log_returns) - 1


class MeanReversion(BaseModel):
    """Ornstein-Uhlenbeck process: dX = theta*(mu - X)*dt + sigma*dW."""

    def __init__(self, params: MeanReversionParams | None = None):
        self.params = params or MeanReversionParams()

    def simulate(self, S0, n_steps, n_paths, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = self.params.dt
        theta = self.params.theta
        mu_long = self.params.long_term_mean
        sigma = self.params.sigma

        X = np.zeros((n_paths, n_steps + 1))
        X[:, 0] = np.log(S0)
        Z = rng.standard_normal((n_paths, n_steps))

        for t in range(n_steps):
            X[:, t + 1] = X[:, t] + theta * (mu_long - X[:, t]) * dt + sigma * np.sqrt(dt) * Z[:, t]

        return np.exp(X)

    def simulate_returns(self, n_steps, n_paths, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = self.params.dt
        theta = self.params.theta
        mu_long = self.params.long_term_mean
        sigma = self.params.sigma

        X = np.zeros((n_paths, n_steps + 1))
        X[:, 0] = 0
        Z = rng.standard_normal((n_paths, n_steps))

        for t in range(n_steps):
            X[:, t + 1] = X[:, t] + theta * (mu_long - X[:, t]) * dt + sigma * np.sqrt(dt) * Z[:, t]

        return np.diff(X, axis=1)


class GARCH(BaseModel):
    """GARCH(1,1) with time-varying volatility.

    sigma^2(t) = omega + alpha * eps^2(t-1) + beta * sigma^2(t-1)
    """

    def __init__(self, params: GARCHParams | None = None):
        self.params = params or GARCHParams()
        self._fitted_model = None

    def simulate(self, S0, n_steps, n_paths, random_state=None):
        returns = self.simulate_returns(n_steps, n_paths, random_state)
        log_returns = np.log(1 + returns)
        log_prices = np.zeros((n_paths, n_steps + 1))
        log_prices[:, 0] = np.log(S0)
        log_prices[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)
        return np.exp(log_prices)

    def simulate_returns(self, n_steps, n_paths, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = self.params.dt
        mu = self.params.mu * dt
        omega = self.params.omega
        alpha = self.params.alpha
        beta = self.params.beta

        returns = np.zeros((n_paths, n_steps))
        sigma2 = np.full(n_paths, self.params.sigma**2 * dt)
        Z = rng.standard_normal((n_paths, n_steps))

        for t in range(n_steps):
            sigma = np.sqrt(sigma2)
            returns[:, t] = mu + sigma * Z[:, t]
            sigma2 = omega + alpha * returns[:, t] ** 2 + beta * sigma2

        return returns

    @classmethod
    def fit(cls, returns: np.ndarray, dt: float = 1 / 252) -> GARCH:
        """Fit GARCH(1,1) to historical returns via MLE (requires ``arch``)."""
        if not HAS_ARCH:
            raise ImportError(
                "The 'arch' package is required for GARCH fitting. "
                "Install it with: uv pip install 'quantbox[simulation]'"
            )
        model = _arch_model(returns * 100, vol="Garch", p=1, q=1)
        result = model.fit(disp="off")

        omega = result.params["omega"] / 10000 / dt
        alpha_val = result.params["alpha[1]"]
        beta_val = result.params["beta[1]"]
        mu = result.params["mu"] / 100 / dt
        sigma = np.sqrt(result.conditional_volatility[-1]) / 100 / np.sqrt(dt)

        params = GARCHParams(
            mu=mu,
            sigma=sigma,
            dt=dt,
            omega=omega,
            alpha=alpha_val,
            beta=beta_val,
        )
        instance = cls(params)
        instance._fitted_model = result
        return instance


class RegimeSwitching(BaseModel):
    """Markov Regime Switching between bull/bear GBM regimes."""

    def __init__(
        self,
        regimes: list[GBMParams] | None = None,
        transition_matrix: np.ndarray | None = None,
    ):
        if regimes is None:
            regimes = [
                GBMParams(mu=0.12, sigma=0.15),  # Bull
                GBMParams(mu=-0.05, sigma=0.35),  # Bear
            ]
        if transition_matrix is None:
            transition_matrix = np.array(
                [
                    [0.98, 0.02],
                    [0.05, 0.95],
                ]
            )
        self.regimes = regimes
        self.transition_matrix = transition_matrix
        self.n_regimes = len(regimes)

    def simulate(self, S0, n_steps, n_paths, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = self.regimes[0].dt

        prices = np.zeros((n_paths, n_steps + 1))
        prices[:, 0] = S0
        current_regime = np.zeros(n_paths, dtype=int)

        Z = rng.standard_normal((n_paths, n_steps))
        U = rng.random((n_paths, n_steps))

        for t in range(n_steps):
            mu = np.array([self.regimes[r].mu for r in current_regime])
            sigma = np.array([self.regimes[r].sigma for r in current_regime])

            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt)
            log_return = drift + diffusion * Z[:, t]
            prices[:, t + 1] = prices[:, t] * np.exp(log_return)

            for i in range(n_paths):
                cum_prob = np.cumsum(self.transition_matrix[current_regime[i]])
                current_regime[i] = np.searchsorted(cum_prob, U[i, t])

        return prices

    def simulate_returns(self, n_steps, n_paths, random_state=None):
        prices = self.simulate(100.0, n_steps, n_paths, random_state)
        return np.diff(prices, axis=1) / prices[:, :-1]


__all__ = [
    "BaseModel",
    "ModelParameters",
    "GBM",
    "GBMParams",
    "JumpDiffusion",
    "JumpDiffusionParams",
    "MeanReversion",
    "MeanReversionParams",
    "GARCH",
    "GARCHParams",
    "RegimeSwitching",
]
