"""Shared feature computation functions for strategies.

All functions are pure, stateless, and operate on wide-format DataFrames
(DatetimeIndex x symbol columns).
"""
from quantbox.features.returns import compute_returns
from quantbox.features.volatility import compute_rolling_vol, compute_ewm_vol
from quantbox.features.moving_averages import compute_sma, compute_ema
from quantbox.features.channels import compute_donchian
from quantbox.features.cross_sectional import (
    compute_zscore_cross_sectional,
    compute_rank_cross_sectional,
)
from quantbox.features.bundle import compute_features_bundle

__all__ = [
    "compute_returns",
    "compute_rolling_vol",
    "compute_ewm_vol",
    "compute_sma",
    "compute_ema",
    "compute_donchian",
    "compute_zscore_cross_sectional",
    "compute_rank_cross_sectional",
    "compute_features_bundle",
]
