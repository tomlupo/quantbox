"""Regression tests for the rsims engine margin/vol-invariance fix.

Background: the maintenance-margin/liquidation path (margin>0) spuriously
liquidated leveraged long/short books — scaling leverage flipped a positive
Sharpe to negative instead of leaving it (approximately) unchanged. The fix
defaults ``margin`` to 0.0 (exchange-liquidation modelling is opt-in). These
tests pin the new default and the Sharpe-invariance it restores.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from quantbox.plugins.backtesting.rsims_engine import fixed_commission_backtest_with_funding


def _sharpe(res: pd.DataFrame, initial: float) -> float:
    pnl = res.reset_index().groupby("date")["PeriodPnL"].sum()
    equity = initial + pnl.cumsum()
    r = equity.pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def _inputs():
    idx = pd.date_range("2026-01-01", periods=200, freq="D")
    rng = np.random.default_rng(7)
    px = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0.0015, 0.03, 200)),
            "B": 50 * np.cumprod(1 + rng.normal(-0.0010, 0.04, 200)),
        },
        index=idx,
    )
    fund = pd.DataFrame(0.0, index=idx, columns=px.columns)
    w = pd.DataFrame({"A": 0.15, "B": -0.15}, index=idx)
    return px, fund, w


def test_margin_default_is_off():
    """margin defaults to 0.0 (liquidation modelling is opt-in)."""
    assert inspect.signature(fixed_commission_backtest_with_funding).parameters["margin"].default == 0.0


def test_sharpe_invariant_to_leverage_without_margin():
    """With margin=0, scaling target weights leaves the Sharpe ~unchanged.

    This is the Kelly/vol-invariance property the bug broke: under the old
    margin=0.05 default the levered book liquidated and its Sharpe collapsed.
    """
    px, fund, w = _inputs()
    init = 1_000_000.0
    base = dict(funding_rates=fund, initial_cash=init, margin=0.0, capitalise_profits=False)
    s1 = _sharpe(fixed_commission_backtest_with_funding(px, w, **base), init)
    s2 = _sharpe(fixed_commission_backtest_with_funding(px, 2 * w, **base), init)
    assert abs(s1) > 0.1  # the fixture has a real signal to scale
    assert abs(s2 - s1) < 0.05 * max(1.0, abs(s1))  # Sharpe is leverage-invariant
