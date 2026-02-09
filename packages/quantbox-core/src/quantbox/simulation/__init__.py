"""Monte Carlo simulation, correlation analysis, stress testing, and forecasting.

This package provides tools for generating synthetic market scenarios,
analyzing correlation structures, stress testing portfolios, and
forecasting returns across multiple horizons.

Quick start::

    from quantbox.simulation import MarketSimulator, GBM, GBMParams, SimulationConfig

    sim = MarketSimulator()
    sim.add_asset("SPY", GBM(GBMParams(mu=0.08, sigma=0.18)), initial_price=450)
    sim.add_asset("TLT", GBM(GBMParams(mu=0.03, sigma=0.10)), initial_price=100)
    result = sim.simulate(SimulationConfig(n_paths=10000, n_steps=252))
    print(result.get_path_statistics())

Optional dependencies:

- ``arch`` — required for GARCH fitting and DCC-GARCH correlation
- ``scipy`` — required for parametric VaR, Bayesian forecasting, DCC optimisation
- ``matplotlib`` + ``seaborn`` — required for visualisation (``SimulationPlotter``)
"""

from __future__ import annotations

# Models
from .models import (
    BaseModel,
    GBM,
    GBMParams,
    GARCH,
    GARCHParams,
    JumpDiffusion,
    JumpDiffusionParams,
    MeanReversion,
    MeanReversionParams,
    ModelParameters,
    RegimeSwitching,
)

# Engine
from .engine import (
    MarketSimulator,
    SimulationConfig,
    SimulationResult,
    generate_correlated_returns,
)

# Correlation
from .correlation import (
    CorrelationEngine,
    CorrelationResult,
    generate_random_correlation_matrix,
)

# Stress testing
from .stress_testing import (
    HistoricalScenario,
    StressScenario,
    StressTestEngine,
    StressTestResult,
    HISTORICAL_SCENARIOS,
)

# Forecasting
from .forecasting import (
    ForecastResult,
    Horizon,
    MultiHorizonForecast,
    ReturnForecaster,
)

# Visualization (optional — requires matplotlib/seaborn)
try:
    from .visualization import SimulationPlotter
except ImportError:
    pass

__all__ = [
    # Models
    "BaseModel",
    "GBM",
    "GBMParams",
    "GARCH",
    "GARCHParams",
    "JumpDiffusion",
    "JumpDiffusionParams",
    "MeanReversion",
    "MeanReversionParams",
    "ModelParameters",
    "RegimeSwitching",
    # Engine
    "MarketSimulator",
    "SimulationConfig",
    "SimulationResult",
    "generate_correlated_returns",
    # Correlation
    "CorrelationEngine",
    "CorrelationResult",
    "generate_random_correlation_matrix",
    # Stress testing
    "HistoricalScenario",
    "StressScenario",
    "StressTestEngine",
    "StressTestResult",
    "HISTORICAL_SCENARIOS",
    # Forecasting
    "ForecastResult",
    "Horizon",
    "MultiHorizonForecast",
    "ReturnForecaster",
    # Visualization
    "SimulationPlotter",
]
