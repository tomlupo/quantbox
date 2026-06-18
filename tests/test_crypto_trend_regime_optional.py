"""regime_ticker=None disables the donchian-overlay diagnostic without crashing,
and is weight-neutral (the overlay is diagnostic-only) — needed for a literal
1:1 port comparison vs quantlab's crypto_trend_catcher (no regime overlay)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbox.plugins.strategies.crypto_trend import run as crypto_trend_run


def _panel(n=120, k=8, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    cols = [f"C{i}" for i in range(k)]
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.001, 0.03, (n, k)), axis=0), index=idx, columns=cols)
    volume = pd.DataFrame(rng.uniform(1e6, 1e8, (n, k)), index=idx, columns=cols)
    mcap = pd.DataFrame(rng.uniform(1e8, 1e10, (n, k)), index=idx, columns=cols)
    return {"prices": prices, "volume": volume, "market_cap": mcap}


def _params(regime):
    return dict(
        lookback_windows=[5, 10, 20, 30],
        vol_targets=[0.5],
        tranches=[5],
        top_by_mcap=6,
        top_by_volume=4,
        vol_lookback=30,
        periods=20,
        output_track=["50", 5],
        volume_is_dollar=False,
        regime_ticker=regime,
    )


def test_regime_none_does_not_crash():
    out = crypto_trend_run(_panel(), _params(None))
    assert "weights" in out and not out["weights"].empty


def test_regime_is_weight_neutral():
    data = _panel()
    w_on = crypto_trend_run(data, _params("C0"))["weights"]
    w_off = crypto_trend_run(data, _params(None))["weights"]
    cols = w_on.columns.intersection(w_off.columns)
    np.testing.assert_allclose(w_on[cols].fillna(0).values, w_off[cols].fillna(0).values, atol=1e-9)
