"""Known-variance check for the rolling covariance estimators (TOM-1339).

On iid daily returns with per-period std ``sigma``, every estimator's
annualised variance must recover ``sigma**2 * 252`` for any ``roll``.
``roll`` > 1 overlaps ``roll``-period returns to absorb async closes
(Bartlett / Newey-West); it must not change the variance *scale*.
The previous ``pct_change(roll) / roll`` convention understated variance
by exactly ``1 / roll``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.features.covariance import (
    CovarianceModelConfig,
    EwmaLwModelConfig,
    rolling_covariance_ewma_lw,
    rolling_covariance_lw,
    rolling_covariance_oas,
)

SIGMA = 0.01
TRUE_ANN_VAR = SIGMA**2 * 252
# Sampling error of the mean diagonal over 4 assets and ~3000 obs is ~3%;
# the bias being guarded against is 3x (roll=3) and 5x (roll=5).
REL_TOL = 0.10


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(1339)
    n, p = 4000, 4
    idx = pd.bdate_range("2010-01-01", periods=n)
    rets = rng.normal(0.0, SIGMA, (n, p))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=list("ABCD"))


def _mean_diag_ratio(cov: pd.DataFrame) -> float:
    return float(np.diag(cov.values).mean() / TRUE_ANN_VAR)


@pytest.mark.parametrize("roll", [1, 3, 5])
def test_ewma_lw_recovers_known_variance(prices: pd.DataFrame, roll: int) -> None:
    model = EwmaLwModelConfig(half_life_obs=1000, freq="B", roll=roll)
    out = rolling_covariance_ewma_lw(prices, models=[model], dates=prices.index[-1:], show_progress=False)
    assert _mean_diag_ratio(out[model.name]) == pytest.approx(1.0, rel=REL_TOL)


@pytest.mark.parametrize("estimator", [rolling_covariance_lw, rolling_covariance_oas])
@pytest.mark.parametrize("roll", [1, 3, 5])
def test_shrunk_sample_covariance_recovers_known_variance(prices: pd.DataFrame, roll: int, estimator) -> None:
    model = CovarianceModelConfig(method="x", window=3000, roll=roll, freq="B")
    out = estimator(prices, models=[model], dates=prices.index[-1:], show_progress=False)
    assert _mean_diag_ratio(out[model.name]) == pytest.approx(1.0, rel=REL_TOL)
