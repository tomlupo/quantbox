"""``regime_sharpe`` and ``regime_gate``: the Sharpe split over benchmark-quantile regimes (TOM-3716).

The card's proof: on a fixture, the bear / normal / bull Sharpe parts sum to
the full per-period Sharpe within 1e-9. The rest pins the labels (a row's
regime is its own benchmark return against the thresholds), the Gaussian null
(a part's null std is sqrt(n_k) / T for iid returns) and the gate's verdict on
a strategy that earns only in bull markets.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.gates import GateInputError, regime_gate
from quantbox.inference import REGIMES, InferenceInputError, regime_sharpe

RNG = np.random.default_rng(3716)
T = 1200
BENCH = RNG.normal(0.0004, 0.01, T)
MIXED = 0.3 * BENCH + RNG.normal(0.0003, 0.008, T)  # a beta book with a small drift
BULL_ONLY = np.where(np.quantile(BENCH, 5 / 6) <= BENCH, 0.004, 0.0) + RNG.normal(0, 0.004, T)
EVERYWHERE = RNG.normal(0.003, 0.004, T)  # independent of the benchmark, positive in every regime


@pytest.mark.parametrize("returns", [MIXED, BULL_ONLY, EVERYWHERE], ids=["mixed", "bull_only", "everywhere"])
def test_the_regime_parts_sum_to_the_full_sharpe(returns):
    out = regime_sharpe(returns, BENCH, draws=50)
    full = returns.mean() / returns.std(ddof=1)
    parts = [out["regimes"][k]["sharpe_part"] for k in REGIMES]
    assert out["sharpe"] == pytest.approx(full, rel=0, abs=1e-15)
    assert abs(sum(parts) - full) < 1e-9
    assert sum(out["regimes"][k]["n_obs"] for k in REGIMES) == T
    assert sum(out["regimes"][k]["share"] for k in REGIMES) == pytest.approx(1.0)
    for k in REGIMES:  # part = share * mean / full-sample std
        g = out["regimes"][k]
        assert g["sharpe_part"] == pytest.approx(g["share"] * g["mean"] / returns.std(ddof=1), rel=1e-12)


def test_thresholds_are_the_full_sample_quantiles_by_default():
    out = regime_sharpe(MIXED, BENCH, draws=10)
    assert out["thresholds"] == tuple(np.quantile(BENCH, [1 / 6, 5 / 6]))
    assert out["threshold_source"] == "full_sample_quantiles"
    assert out["quantiles"] == (1 / 6, 5 / 6)
    lo, hi = out["thresholds"]
    assert out["regimes"]["bear"]["n_obs"] == int((lo >= BENCH).sum())
    assert out["regimes"]["bull"]["n_obs"] == int((hi <= BENCH).sum())


def test_a_rows_label_is_its_own_benchmark_return_against_given_thresholds():
    r = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
    b = np.array([-0.03, -0.01, 0.0, 0.02, 0.01, -0.02])
    out = regime_sharpe(r, b, thresholds=(-0.02, 0.02), draws=10)
    s = r.std(ddof=1)
    # bear: b <= -0.02 -> rows 0, 5; bull: b >= 0.02 -> row 3; normal: rows 1, 2, 4
    assert out["regimes"]["bear"]["sharpe_part"] == pytest.approx((0.01 + 0.02) / (6 * s))
    assert out["regimes"]["bull"]["sharpe_part"] == pytest.approx(0.005 / (6 * s))
    assert out["regimes"]["normal"]["n_obs"] == 3
    assert out["threshold_source"] == "given" and out["quantiles"] is None
    # No look-ahead with given thresholds: changing a LATER benchmark row moves only that row's label.
    b2 = b.copy()
    b2[5] = 0.05  # row 5: bear -> bull
    moved = regime_sharpe(r, b2, thresholds=(-0.02, 0.02), draws=10)
    assert moved["regimes"]["bear"]["sharpe_part"] == pytest.approx(0.01 / (6 * s))
    assert moved["regimes"]["bull"]["sharpe_part"] == pytest.approx((0.005 + 0.02) / (6 * s))


def test_an_empty_regime_has_a_zero_part_and_no_test():
    out = regime_sharpe(MIXED, BENCH, thresholds=(-1.0, 0.0), draws=10)
    bear = out["regimes"]["bear"]
    assert bear["n_obs"] == 0 and bear["sharpe_part"] == 0.0 and bear["pvalue"] is None
    assert abs(sum(out["regimes"][k]["sharpe_part"] for k in REGIMES) - out["sharpe"]) < 1e-9


@pytest.mark.parametrize("bear_vol", [1.0, 3.0], ids=["homoskedastic", "wild_bear"])
def test_the_gaussian_null_is_per_regime_and_seeded(bear_vol):
    # A part's null std is sqrt(n_k) * s_k / (T * s): each regime's own volatility, not the full sample's.
    r = RNG.normal(0, 0.004, T) * np.where(np.quantile(BENCH, 1 / 6) >= BENCH, bear_vol, 1.0)
    out = regime_sharpe(r, BENCH, draws=4000, seed=7)
    s = r.std(ddof=1)
    for k in REGIMES:
        g = out["regimes"][k]
        assert g["null_std"] == pytest.approx(np.sqrt(g["n_obs"]) * g["std"] / (T * s), rel=0.06)
        assert 1 / 4001 <= g["pvalue"] <= 1
    assert regime_sharpe(EVERYWHERE, BENCH, draws=200, seed=1) == regime_sharpe(EVERYWHERE, BENCH, draws=200, seed=1)


def test_gate_sees_alpha_only_in_bull_markets():
    out = regime_gate(BULL_ONLY, BENCH)
    assert out["positive"] == ["bull"]
    assert out["gate_pass"] is False  # default: every regime required
    assert out["m"] == 3 and out["method"] == "holm"
    assert regime_gate(BULL_ONLY, BENCH, require=("bull",))["gate_pass"] is True
    assert regime_gate(BULL_ONLY, BENCH, require=("bear", "normal"))["gate_pass"] is False


def test_gate_passes_a_strategy_paid_in_every_regime():
    out = regime_gate(EVERYWHERE, BENCH, method="bhy")
    assert out["positive"] == list(REGIMES) and out["gate_pass"] is True
    for k in REGIMES:
        assert out["regimes"][k]["pvalue_adjusted"] >= out["regimes"][k]["pvalue"]


def test_gate_corrects_only_over_the_regimes_with_observations():
    out = regime_gate(EVERYWHERE, BENCH, thresholds=(-1.0, 0.0), require=("bull",))
    assert out["m"] == 2
    assert out["regimes"]["bear"]["pvalue_adjusted"] is None and out["regimes"]["bear"]["positive"] is False


def test_refusals():
    with pytest.raises(InferenceInputError, match="same rows"):
        regime_sharpe(MIXED, BENCH[:-1])
    with pytest.raises(InferenceInputError, match="at least 3"):
        regime_sharpe(MIXED[:2], BENCH[:2])
    with pytest.raises(InferenceInputError, match="zero-variance"):
        regime_sharpe(np.full(T, 0.001), BENCH)
    with pytest.raises(InferenceInputError, match="quantiles"):
        regime_sharpe(MIXED, BENCH, quantiles=(0.8, 0.2))
    with pytest.raises(InferenceInputError, match="below the bull"):
        regime_sharpe(MIXED, BENCH, thresholds=(0.01, 0.01))
    with pytest.raises(InferenceInputError, match="below the bull"):
        regime_sharpe(MIXED, np.zeros(T))  # a flat benchmark has no regimes
    for draws in (0, 1):
        with pytest.raises(InferenceInputError, match="draws"):
            regime_sharpe(MIXED, BENCH, draws=draws)
    for kw in ({"quantiles": (0.2,)}, {"quantiles": "ab"}, {"thresholds": (-0.01, 0.0, 0.01)}):
        with pytest.raises(InferenceInputError, match="pairs"):
            regime_sharpe(MIXED, BENCH, **kw)
    with pytest.raises(InferenceInputError, match="NaN|non-finite|finite"):
        regime_sharpe(np.r_[MIXED[:-1], np.nan], BENCH)
    dropped = regime_sharpe(np.r_[MIXED[:-1], np.nan], BENCH, allow_nonfinite_drop=True, draws=10)
    assert dropped["n_obs"] == T - 1 and dropped["n_nonfinite_dropped"] == 1 and dropped["n_obs_raw"] == T


def test_gate_refusals():
    for bad in ((), ("bull", "bull"), ("crash",)):
        with pytest.raises(GateInputError, match="require"):
            regime_gate(MIXED, BENCH, require=bad)
    with pytest.raises(GateInputError, match="no observations"):
        regime_gate(MIXED, BENCH, thresholds=(-1.0, 0.0))  # bear empty, but required by default
