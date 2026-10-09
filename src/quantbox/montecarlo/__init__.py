"""Monte Carlo kernels: stochastic price models, the multi-asset simulator, correlation, stress tests.

Plugins layer (ADR-0010, TOM-1451). A builtin datasource (``data.synthetic.v1``)
and a trade risk plugin (``risk.stress_test.v1``) run on these kernels, so they
sit below research and trade. They import core only (numpy, pandas,
:mod:`quantbox.metrics`, :mod:`quantbox.inference`).

They lived in :mod:`quantbox.simulation` until TOM-1451. That package stays the
research home: it re-exports every name below and adds forecasting and the
plotter. ``quantbox.simulation.{models,engine,correlation,stress_testing}`` are
deprecation shims now.
"""

from __future__ import annotations

from .correlation import CorrelationEngine, CorrelationResult, generate_random_correlation_matrix
from .engine import MarketSimulator, SimulationConfig, SimulationResult, generate_correlated_returns
from .models import (
    GARCH,
    GBM,
    BaseModel,
    GARCHParams,
    GBMParams,
    JumpDiffusion,
    JumpDiffusionParams,
    MeanReversion,
    MeanReversionParams,
    ModelParameters,
    RegimeSwitching,
)
from .stress_testing import HISTORICAL_SCENARIOS, HistoricalScenario, StressScenario, StressTestEngine, StressTestResult

__all__ = [
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
    "MarketSimulator",
    "SimulationConfig",
    "SimulationResult",
    "generate_correlated_returns",
    "CorrelationEngine",
    "CorrelationResult",
    "generate_random_correlation_matrix",
    "HistoricalScenario",
    "StressScenario",
    "StressTestEngine",
    "StressTestResult",
    "HISTORICAL_SCENARIOS",
]
