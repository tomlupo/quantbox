"""Tests for FrozenWeightsStrategy — dated-weights replay plugin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.strategies.frozen_weights import FrozenWeightsStrategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def daily_prices() -> pd.DataFrame:
    """200 daily bars × 4 tickers, random-walk prices."""
    rng = np.random.RandomState(7)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    cols = ["AAA", "BBB", "CCC", "DDD"]
    data = {c: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 200))) for c in cols}
    return pd.DataFrame(data, index=dates)


@pytest.fixture()
def weekly_weights(tmp_path, daily_prices) -> str:
    """Weekly weight grid covering 3 of the 4 tickers; rows sum to 1.0."""
    weekly = pd.date_range("2024-01-05", "2024-07-15", freq="W-FRI")
    weights = pd.DataFrame(
        {
            "AAA": np.linspace(0.5, 0.2, len(weekly)),
            "BBB": np.linspace(0.3, 0.5, len(weekly)),
            "CCC": np.linspace(0.2, 0.3, len(weekly)),
        },
        index=weekly,
    )
    path = tmp_path / "weights.parquet"
    weights.to_parquet(path)
    return str(path)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_meta_registers_as_strategy():
    s = FrozenWeightsStrategy()
    assert s.meta.name == "strategy.frozen_weights.v1"
    assert s.meta.kind == "strategy"
    assert "frozen-weights" in s.meta.tags


def test_loads_weekly_grid_and_ffills_onto_daily_index(daily_prices, weekly_weights):
    s = FrozenWeightsStrategy()
    result = s.run({"prices": daily_prices}, {"weights_path": weekly_weights})
    weights = result["weights"]

    assert isinstance(weights, pd.DataFrame)
    assert weights.shape == daily_prices.shape
    assert list(weights.columns) == list(daily_prices.columns)
    # Last weight row should equal final weekly weight on the file (carried forward)
    last_used = weights.iloc[-1][["AAA", "BBB", "CCC"]]
    expected_last = pd.read_parquet(weekly_weights).iloc[-1]
    pd.testing.assert_series_equal(last_used.rename(None), expected_last.rename(None), check_names=False, atol=1e-10)
    # DDD never appears in the weights file → always 0
    assert (weights["DDD"] == 0).all()


def test_rebase_date_zeros_pre_period(daily_prices, weekly_weights):
    s = FrozenWeightsStrategy()
    rebase = "2024-04-01"
    result = s.run(
        {"prices": daily_prices},
        {"weights_path": weekly_weights, "rebase_date": rebase},
    )
    weights = result["weights"]
    pre = weights.loc[weights.index < pd.Timestamp(rebase)]
    post = weights.loc[weights.index >= pd.Timestamp(rebase)]
    assert (pre.values == 0).all()
    assert (post[["AAA", "BBB", "CCC"]].sum(axis=1) > 0).any()


def test_missing_ticker_fails_by_default(daily_prices, tmp_path):
    weights = pd.DataFrame(
        {"AAA": [0.5, 0.5], "ZZZ_not_in_prices": [0.5, 0.5]},
        index=pd.date_range("2024-01-05", periods=2, freq="W"),
    )
    p = tmp_path / "w.parquet"
    weights.to_parquet(p)
    s = FrozenWeightsStrategy()
    with pytest.raises(KeyError, match="ZZZ_not_in_prices"):
        s.run({"prices": daily_prices}, {"weights_path": str(p)})


def test_missing_ticker_zero_silently_drops(daily_prices, tmp_path):
    weights = pd.DataFrame(
        {"AAA": [0.5, 0.5], "ZZZ_not_in_prices": [0.5, 0.5]},
        index=pd.date_range("2024-01-05", periods=2, freq="W"),
    )
    p = tmp_path / "w.parquet"
    weights.to_parquet(p)
    s = FrozenWeightsStrategy()
    result = s.run(
        {"prices": daily_prices},
        {"weights_path": str(p), "missing_tickers": "zero"},
    )
    assert "ZZZ_not_in_prices" not in result["details"]["tickers"]
    assert result["details"]["tickers"] == ["AAA"]


def test_renormalize_false_preserves_source_sums(daily_prices, tmp_path):
    """A weights file with rows summing to 0.95 (5% cash residual) is kept as-is."""
    weekly = pd.date_range("2024-01-05", periods=4, freq="W")
    weights = pd.DataFrame(
        {"AAA": [0.5] * 4, "BBB": [0.45] * 4},  # sums to 0.95
        index=weekly,
    )
    p = tmp_path / "w.parquet"
    weights.to_parquet(p)
    s = FrozenWeightsStrategy()
    result = s.run({"prices": daily_prices}, {"weights_path": str(p)})
    w = result["weights"]
    # After ffill, post-rebase rows should sum to 0.95, not 1.0
    post_rebase = w.loc[w.index >= weekly[0]]
    sums = post_rebase[["AAA", "BBB"]].sum(axis=1)
    assert (np.isclose(sums, 0.95)).all(), f"expected 0.95 sum, got {sums.unique()}"


def test_renormalize_true_forces_sum_to_one(daily_prices, tmp_path):
    weekly = pd.date_range("2024-01-05", periods=4, freq="W")
    weights = pd.DataFrame(
        {"AAA": [0.5] * 4, "BBB": [0.45] * 4},  # sums to 0.95
        index=weekly,
    )
    p = tmp_path / "w.parquet"
    weights.to_parquet(p)
    s = FrozenWeightsStrategy()
    result = s.run(
        {"prices": daily_prices},
        {"weights_path": str(p), "renormalize": True},
    )
    w = result["weights"]
    post_rebase = w.loc[w.index >= weekly[0]]
    sums = post_rebase[["AAA", "BBB"]].sum(axis=1)
    assert (np.isclose(sums, 1.0)).all()


def test_missing_weights_path_raises(daily_prices):
    s = FrozenWeightsStrategy()
    with pytest.raises(ValueError, match="weights_path"):
        s.run({"prices": daily_prices}, {})


def test_nonexistent_weights_file_raises(daily_prices, tmp_path):
    s = FrozenWeightsStrategy()
    with pytest.raises(FileNotFoundError):
        s.run({"prices": daily_prices}, {"weights_path": str(tmp_path / "nope.parquet")})


def test_zero_weight_where_price_is_nan(tmp_path):
    """Tickers with NaN prices early in the window get zero weight even if
    the weights file says non-zero — matches the StaticWeights convention."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 110, 20),
            "BBB": [np.nan] * 10 + list(np.linspace(50, 55, 10)),  # late lister
        },
        index=dates,
    )
    weights = pd.DataFrame(
        {"AAA": [0.5, 0.5], "BBB": [0.5, 0.5]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-15"]),
    )
    p = tmp_path / "w.parquet"
    weights.to_parquet(p)
    s = FrozenWeightsStrategy()
    result = s.run({"prices": prices}, {"weights_path": str(p)})
    w = result["weights"]
    # BBB should be 0 before its first valid price
    assert (w["BBB"].iloc[:10] == 0).all()
    # BBB should be 0.5 once prices appear
    assert (w["BBB"].iloc[10:] == 0.5).all()
