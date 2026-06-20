"""FIX A: NaN-weight robustness in crypto_trend weight construction.

Regression for the freeze bug: a NON-selected asset whose signal is NaN (e.g. a
newly-listed coin with insufficient history) produced ``NaN * 0 = NaN`` in
``construct_weights`` (``sig_tranched * universe``). That NaN leaked into
``aggregated_weights`` and FROZE the rebalancer ("REBALANCER FROZEN ... Invalid
(NaN)") even though the asset was never held — so the book never traded.

The fix masks non-selected columns to exactly 0 via ``where`` (mask wins over
the signal value) and fills any residual NaN with 0, so no NaN can reach the
aggregator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbox.plugins.strategies.crypto_trend import construct_weights


def _frames():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    cols = ["BTC", "ETH", "NEWCOIN"]
    # NEWCOIN has insufficient history -> NaN signal throughout. It is NOT
    # selected (universe 0). BTC/ETH are selected with valid signals.
    signals = pd.DataFrame(
        {
            "BTC": [1.0, 1.0, 1.0, 1.0, 1.0],
            "ETH": [1.0, 1.0, 1.0, 1.0, 1.0],
            "NEWCOIN": [np.nan] * 5,
        },
        index=idx,
        columns=cols,
    )
    universe = pd.DataFrame(
        {
            "BTC": [1.0] * 5,
            "ETH": [1.0] * 5,
            "NEWCOIN": [0.0] * 5,  # never selected
        },
        index=idx,
        columns=cols,
    )
    scalers = {"50": pd.DataFrame(1.0, index=idx, columns=cols)}
    return signals, universe, scalers


def test_nonselected_nan_signal_yields_zero_not_nan():
    signals, universe, scalers = _frames()

    weights = construct_weights(signals, universe, scalers, tranches=[5], normalize=True)

    # No NaN anywhere in the constructed weights panel.
    assert not weights.isna().any().any(), "weights must contain no NaN"

    # The non-selected NaN-signal coin has weight exactly 0 (not NaN).
    newcoin = weights.xs("NEWCOIN", axis=1, level="ticker")
    assert (newcoin.fillna(-999) == 0.0).all().all()

    # Selected coins still carry positive weight (the fix didn't zero them).
    btc = weights.xs("BTC", axis=1, level="ticker")
    assert (btc.iloc[-1] > 0).all()


def test_aggregated_gross_is_finite_and_unit():
    """The latest-row sum across the selected names is finite and ~1.0 (two
    selected names, each 0.5 after normalize-by-n_positions)."""
    signals, universe, scalers = _frames()
    weights = construct_weights(signals, universe, scalers, tranches=[5], normalize=True)
    latest = weights.iloc[-1]
    assert np.isfinite(latest.to_numpy(dtype=float)).all()
    assert latest.sum() == 1.0
