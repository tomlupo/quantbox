"""Feature computation library for quantitative strategies.

All functions operate on **wide-format** DataFrames (DatetimeIndex x symbol
columns) and return either a DataFrame or a dict of DataFrames.

Modules:

- **returns** — period returns (pct_change, log)
- **momentum** — total returns, momentum returns, TSMOM indicator suite
- **volatility** — rolling and EWMA volatility
- **moving_averages** — SMA, EMA
- **cross_sectional** — z-score and rank normalization (row-wise)
- **signals** — signal transforms (binary, z-score, winsorize, top-N, inv-vol)
- **channels** — Donchian channels
- **bundle** — dispatch multiple features from a manifest dict
"""

from quantbox.features.bundle import compute_features_bundle
from quantbox.features.channels import compute_donchian
from quantbox.features.covariance import (
    _FREQ_TO_PERIODS as FREQ_TO_PERIODS,
)
from quantbox.features.covariance import (
    rolling_covariance_ewma_lw,
    rolling_covariance_ewma_lw_from_config,
    rolling_covariance_lw,
    rolling_covariance_lw_from_config,
    rolling_covariance_oas,
    rolling_covariance_oas_from_config,
)
from quantbox.features.cross_sectional import (
    compute_rank_cross_sectional,
    compute_zscore_cross_sectional,
)
from quantbox.features.momentum import (
    compute_momentum_returns,
    compute_total_returns,
    compute_tsmom,
)
from quantbox.features.moving_averages import compute_ema, compute_sma
from quantbox.features.returns import compute_returns
from quantbox.features.signals import (
    binary_signal,
    cross_sectional_zscore,
    inverse_volatility_weights,
    rank_select_top_n,
    rolling_minmax_normalize,
    rolling_zscore,
    winsorize,
)
from quantbox.features.simulations import parametric_mc, simulations_stats
from quantbox.features.volatility import compute_ewm_vol, compute_rolling_vol

__all__ = [
    # returns
    "compute_returns",
    # momentum
    "compute_total_returns",
    "compute_momentum_returns",
    "compute_tsmom",
    # volatility
    "compute_rolling_vol",
    "compute_ewm_vol",
    # moving averages
    "compute_sma",
    "compute_ema",
    # cross-sectional
    "compute_zscore_cross_sectional",
    "compute_rank_cross_sectional",
    # signals
    "binary_signal",
    "rolling_zscore",
    "cross_sectional_zscore",
    "winsorize",
    "rolling_minmax_normalize",
    "rank_select_top_n",
    "inverse_volatility_weights",
    # channels
    "compute_donchian",
    # covariance
    "rolling_covariance_oas",
    "rolling_covariance_oas_from_config",
    "rolling_covariance_lw",
    "rolling_covariance_lw_from_config",
    "rolling_covariance_ewma_lw",
    "rolling_covariance_ewma_lw_from_config",
    # simulations
    "parametric_mc",
    "simulations_stats",
    # frequency constants
    "FREQ_TO_PERIODS",
    # bundle
    "compute_features_bundle",
]
