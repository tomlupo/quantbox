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

The models, the simulator, correlation and stress testing live in
:mod:`quantbox.montecarlo` (plugins layer, TOM-1451): a builtin datasource and a
trade risk plugin use them, and neither may import research. This package
re-exports them unchanged. Since 0.13.0 the submodules
``quantbox.simulation.{models,engine,correlation,stress_testing}`` raise
``ImportError`` naming their ``quantbox.montecarlo`` home (TOM-1457).
"""

from __future__ import annotations

# Visualization (optional — requires matplotlib/seaborn)
import contextlib

from quantbox.montecarlo import (
    GARCH,
    GBM,
    HISTORICAL_SCENARIOS,
    BaseModel,
    CorrelationEngine,
    CorrelationResult,
    GARCHParams,
    GBMParams,
    HistoricalScenario,
    JumpDiffusion,
    JumpDiffusionParams,
    MarketSimulator,
    MeanReversion,
    MeanReversionParams,
    ModelParameters,
    RegimeSwitching,
    SimulationConfig,
    SimulationResult,
    StressScenario,
    StressTestEngine,
    StressTestResult,
    generate_correlated_returns,
    generate_random_correlation_matrix,
)

from .forecasting import (
    ForecastResult,
    Horizon,
    MultiHorizonForecast,
    ReturnForecaster,
)

with contextlib.suppress(ImportError):
    from .visualization import SimulationPlotter

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
