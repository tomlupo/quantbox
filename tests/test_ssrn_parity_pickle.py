"""Engine-isolation regression test for the SSRN "Catching Crypto Trends" port.

The notebook (``research/crypto/catching-crypto-trends-a-tactical-approach-for-
bitcoin-and-altcoins``) saves the final paper-chosen portfolio + weights to disk
as ``pf_bands5.pkl`` and ``weights_shifted_dc.pkl``. This test replays the
saved weights through quantbox's ``vectorbt_engine`` and asserts the resulting
metrics match the notebook's recorded numbers within tolerance.

Why this matters
- The strategy → weights logic and the engine logic are decoupled. A bug in
  ``vectorbt_engine`` could silently distort live PnL even if the strategy
  weights are correct.
- A bug in the strategy's weight construction would not be caught by this test
  — see ``test_crypto_trend.py`` for that side.
- Combined, the two cover both halves of the port.

The pickle files are not checked into the repo (they're outputs of a
quantlab-owned notebook). The test is skipped when they're not present, so CI
on machines without the lab still passes. Authors with the lab cloned will
catch engine regressions before they ship.

To run: drop the two pickles in either of:
  - ``$QUANTBOX_SSRN_NOTEBOOK_DIR`` (env override)
  - ``/mnt/c/Users/twilc/code/projects/quantlab/lab/research/crypto/catching-
    crypto-trends-a-tactical-approach-for-bitcoin-and-altcoins/output/``
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

# Default location on the author's workstation. Override via env var.
_DEFAULT_DIR = Path(
    "/mnt/c/Users/twilc/code/projects/quantlab/lab/research/crypto/"
    "catching-crypto-trends-a-tactical-approach-for-bitcoin-and-altcoins/output"
)


def _pickle_dir() -> Path:
    return Path(os.environ.get("QUANTBOX_SSRN_NOTEBOOK_DIR") or _DEFAULT_DIR)


def _pickle(name: str) -> Path:
    return _pickle_dir() / name


pytestmark = pytest.mark.skipif(
    not (_pickle("pf_bands5.pkl").exists() and _pickle("weights_shifted_dc.pkl").exists()),
    reason=(f"SSRN notebook pickles not present at {_pickle_dir()}. Set $QUANTBOX_SSRN_NOTEBOOK_DIR to override."),
)


# Notebook headline numbers for the paper-chosen variant
# (vol_target=50, tranches=5, fees=0.005, threshold=0.05) computed at runtime
# from ``pf_bands5.pkl``. The expected values below are recorded from a
# verified pre-merge run and serve as a tolerance check: if the engine drifts,
# the comparison fails.
_PAPER_TRACK = ("50", 5)
_EXPECTED = {
    "Total Return [%]": 161.482,
    "Annualized Return [%]": 18.043,
    "Annualized Volatility [%]": 22.671,
    "Sharpe Ratio": 0.846,
    "Max Drawdown [%]": 36.464,
    "Calmar Ratio": 0.495,
    "Sortino Ratio": 1.177,
}
_TOL_RELATIVE = 0.01  # 1% relative tolerance — engine should be bit-exact


@pytest.fixture(scope="module")
def notebook_portfolio():
    """The notebook's saved vbt Portfolio (vol_target=50, tranches=5)."""
    import vectorbt as vbt

    return vbt.Portfolio.load(_pickle("pf_bands5.pkl"))


@pytest.fixture(scope="module")
def notebook_weights() -> pd.DataFrame:
    """The notebook's saved weights DataFrame, MultiIndex columns."""
    import pickle

    with _pickle("weights_shifted_dc.pkl").open("rb") as f:
        w: pd.DataFrame = pickle.load(f)
    return w


def test_notebook_pickle_is_loadable(notebook_portfolio, notebook_weights):
    """Sanity: pickles deserialise to expected shapes/types."""
    assert notebook_portfolio is not None
    assert isinstance(notebook_weights, pd.DataFrame)
    assert isinstance(notebook_weights.columns, pd.MultiIndex)
    assert {"vol_target", "tranches", "ticker"}.issubset(set(notebook_weights.columns.names))


def test_notebook_metrics_match_recorded(notebook_portfolio):
    """The notebook itself should still produce the recorded metrics.

    This guards against silent pickle corruption / vbt version drift.
    """
    rs = notebook_portfolio.returns_stats(agg_func=None)
    paper_row = None
    for key in rs.index:
        if key[1] == _PAPER_TRACK[0] and key[2] == _PAPER_TRACK[1]:
            paper_row = rs.loc[key]
            break
    assert paper_row is not None, f"variant {_PAPER_TRACK} not in pickle index"
    for metric, expected in _EXPECTED.items():
        observed = float(paper_row[metric])
        if abs(expected) > 0:
            assert abs(observed - expected) / abs(expected) < _TOL_RELATIVE, (
                f"notebook pickle drifted on {metric!r}: recorded={expected:.4f}  loaded={observed:.4f}"
            )


def test_quantbox_engine_reproduces_notebook_metrics(notebook_portfolio, notebook_weights):
    """Feed the notebook's prices + weights through quantbox's engine.

    Asserts that the engine reproduces the notebook's headline metrics bit-exact
    (within 1% relative tolerance). If a future change to ``vectorbt_engine`` —
    fee model, rebalancing-band logic, order routing — drifts the numbers, this
    fails loudly.
    """
    from quantbox.plugins.backtesting.vectorbt_engine import run as run_vbt

    # Extract prices for the paper-chosen variant. PF.close has MultiIndex
    # columns (strategy, vol_target, tranches, ticker); slice down to (ticker).
    slice_close = notebook_portfolio.close.xs(_PAPER_TRACK, axis=1, level=("vol_target", "tranches"))
    if "strategy" in slice_close.columns.names:
        slice_close.columns = slice_close.columns.droplevel("strategy")
    prices = slice_close.ffill().bfill()

    # Extract weights for the same variant.
    w = notebook_weights.xs(_PAPER_TRACK, axis=1, level=("vol_target", "tranches"))
    common_idx = prices.index.intersection(w.index)
    common_cols = sorted(set(prices.columns) & set(w.columns))
    prices = prices.loc[common_idx, common_cols]
    weights = w.loc[common_idx, common_cols].fillna(0)

    # Same engine params the notebook used: daily rebalance, 5% bands, 50 bps fees.
    pf = run_vbt(prices, weights, rebalancing_freq=1, fees=0.005, threshold=0.05)
    rs = pf.returns_stats(agg_func=None)

    for metric, expected in _EXPECTED.items():
        observed = float(rs[metric])
        if abs(expected) > 0:
            rel = abs(observed - expected) / abs(expected)
            assert rel < _TOL_RELATIVE, (
                f"engine drift on {metric!r}: notebook={expected:.4f}  quantbox={observed:.4f}  rel-diff={rel:.4%}"
            )
