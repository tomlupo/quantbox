"""Tests for MLPredictionStrategy plugin."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.strategies.ml_strategy import (
    MLPredictionStrategy,
    _create_target,
    _FeatureEngineer,
    _make_model,
    _predictions_to_weights_confidence,
    _predictions_to_weights_rank,
    _predictions_to_weights_threshold,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prices() -> pd.DataFrame:
    """Synthetic random-walk prices: 20 symbols, 600 days."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2023-01-01", periods=600)
    symbols = [f"SYM{i:02d}" for i in range(20)]
    log_returns = rng.normal(0.0002, 0.02, size=(600, 20))
    cumulative = np.exp(np.cumsum(log_returns, axis=0))
    base_prices = 100 * cumulative
    return pd.DataFrame(base_prices, index=dates, columns=symbols)


@pytest.fixture()
def volume(prices: pd.DataFrame) -> pd.DataFrame:
    """Synthetic volume aligned with prices."""
    rng = np.random.RandomState(99)
    return pd.DataFrame(
        rng.uniform(1e5, 1e7, size=prices.shape),
        index=prices.index,
        columns=prices.columns,
    )


@pytest.fixture()
def data(prices: pd.DataFrame, volume: pd.DataFrame) -> dict:
    return {"prices": prices, "volume": volume}


# ---------------------------------------------------------------------------
# PluginMeta
# ---------------------------------------------------------------------------


def test_meta_name():
    assert MLPredictionStrategy.meta.name == "strategy.ml_prediction.v1"
    assert MLPredictionStrategy.meta.kind == "strategy"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def test_feature_engineer_shape(prices: pd.DataFrame):
    eng = _FeatureEngineer([5, 10, 20, 60])
    features = eng.compute(prices["SYM00"])
    assert len(features) == len(prices)
    # returns(4) + vol(4) + momentum(4) + sma_ratio(4) + sma_slope(4)
    # + rsi(2) + bb(1) + macd(3) + atr(2) + day_sin(1) + day_cos(1)
    assert features.shape[1] == 30


def test_feature_engineer_with_volume(prices: pd.DataFrame, volume: pd.DataFrame):
    eng = _FeatureEngineer([5, 10, 20, 60])
    features = eng.compute(prices["SYM00"], volume["SYM00"])
    # 30 base features + 8 volume features (4 ratio + 4 trend)
    assert features.shape[1] == 38


def test_feature_engineer_no_nan_at_end(prices: pd.DataFrame):
    eng = _FeatureEngineer([5, 10, 20])
    features = eng.compute(prices["SYM00"])
    last_row = features.iloc[-1]
    assert not last_row.isna().all(), "Last row should have values"


# ---------------------------------------------------------------------------
# Target creation
# ---------------------------------------------------------------------------


def test_create_target_regression(prices: pd.DataFrame):
    target = _create_target(prices["SYM00"], horizon=5, task="regression")
    assert len(target) == len(prices)
    assert target.dtype == np.float64
    # Last 5 values should be NaN (forward-looking shift)
    assert target.iloc[-5:].isna().all()


def test_create_target_classification(prices: pd.DataFrame):
    target = _create_target(prices["SYM00"], horizon=5, task="classification")
    assert len(target) == len(prices)
    valid = target.dropna()
    assert set(valid.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def test_make_model_regression():
    model = _make_model("ridge", "regression")
    assert hasattr(model, "fit")


def test_make_model_classification():
    model = _make_model("gradient_boosting", "classification")
    assert hasattr(model, "fit")


def test_make_model_invalid():
    with pytest.raises(ValueError, match="Unknown model"):
        _make_model("nonexistent", "regression")


# ---------------------------------------------------------------------------
# Weight conversion
# ---------------------------------------------------------------------------


def test_weights_rank():
    preds = {"A": 0.5, "B": 0.3, "C": 0.8, "D": 0.1}
    weights = _predictions_to_weights_rank(preds, top_n=2)
    assert set(weights.keys()) == {"C", "A"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_weights_confidence_classification():
    preds = {"A": 1, "B": 1, "C": 0}
    probs = {"A": 0.9, "B": 0.7, "C": 0.3}
    weights = _predictions_to_weights_confidence(preds, probs, "classification")
    assert "C" not in weights
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_weights_threshold():
    preds = {"A": 0.5, "B": -0.2, "C": 0.3}
    weights = _predictions_to_weights_threshold(preds)
    assert "B" not in weights
    assert abs(sum(weights.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Full strategy run
# ---------------------------------------------------------------------------


def test_run_classification(data: dict):
    strategy = MLPredictionStrategy(
        train_lookback=200,
        retrain_frequency=50,
        prediction_horizon=5,
        lookback_periods=[5, 10, 20],
        top_n=5,
        max_symbols=10,
        output_periods=10,
    )
    result = strategy.run(data)

    assert "weights" in result
    assert "simple_weights" in result
    assert "details" in result

    weights = result["weights"]
    assert isinstance(weights, pd.DataFrame)
    assert len(weights) <= 10
    assert weights.shape[1] <= 10


def test_run_regression(data: dict):
    strategy = MLPredictionStrategy(
        task="regression",
        model_name="ridge",
        train_lookback=200,
        retrain_frequency=50,
        prediction_horizon=5,
        lookback_periods=[5, 10, 20],
        top_n=5,
        max_symbols=10,
        output_periods=10,
    )
    result = strategy.run(data)
    weights = result["weights"]
    assert isinstance(weights, pd.DataFrame)
    assert len(weights) > 0


def test_weights_sum_to_one(data: dict):
    strategy = MLPredictionStrategy(
        train_lookback=200,
        retrain_frequency=50,
        lookback_periods=[5, 10, 20],
        top_n=5,
        max_symbols=10,
        output_periods=10,
        weight_method="rank",
    )
    result = strategy.run(data)
    weights = result["weights"]
    row_sums = weights.sum(axis=1)
    nonzero = row_sums[row_sums > 0]
    if len(nonzero) > 0:
        assert (nonzero - 1.0).abs().max() < 0.05


def test_max_symbols_limiting(data: dict):
    strategy = MLPredictionStrategy(
        train_lookback=200,
        retrain_frequency=50,
        lookback_periods=[5, 10],
        max_symbols=5,
        output_periods=5,
    )
    result = strategy.run(data)
    assert result["weights"].shape[1] <= 5


def test_output_columns_match_symbols(data: dict):
    strategy = MLPredictionStrategy(
        train_lookback=200,
        retrain_frequency=50,
        lookback_periods=[5, 10],
        max_symbols=8,
        output_periods=5,
    )
    result = strategy.run(data)
    expected_symbols = list(data["prices"].columns[:8])
    assert list(result["weights"].columns) == expected_symbols


def test_params_override(data: dict):
    strategy = MLPredictionStrategy()
    result = strategy.run(
        data,
        params={
            "train_lookback": 200,
            "retrain_frequency": 50,
            "lookback_periods": [5, 10],
            "max_symbols": 6,
            "output_periods": 5,
            "top_n": 3,
        },
    )
    assert result["weights"].shape[1] <= 6


def test_sklearn_import_error(data: dict):
    strategy = MLPredictionStrategy()
    with (
        patch("quantbox.plugins.strategies.ml_strategy.HAS_SKLEARN", False),
        pytest.raises(ImportError, match="scikit-learn"),
    ):
        strategy.run(data)


def test_not_enough_data():
    rng = np.random.RandomState(7)
    dates = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame(
        rng.uniform(90, 110, size=(30, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    strategy = MLPredictionStrategy(
        train_lookback=504,
        lookback_periods=[5, 10, 20, 60],
        output_periods=10,
    )
    result = strategy.run({"prices": prices})
    weights = result["weights"]
    assert (weights == 0).all().all()
