"""ML Prediction Strategy — sklearn-based return/direction forecasting.

Wraps sklearn models (Ridge, Lasso, GBM, RandomForest, etc.) for
rolling train/predict on multi-asset close price data.  Features are
engineered from close prices (returns, volatility, momentum, SMA ratios,
RSI, MACD, ATR proxy, Bollinger bands, day-of-week encoding).  Volume
features are included when available.

Ported from quantlabnew ml_pipeline.py, adapted to the quantbox
StrategyPlugin interface (wide-format DataFrames, no OHLCV requirement).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.svm import SVC, SVR

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Feature engineering (private)
# ---------------------------------------------------------------------------


class _FeatureEngineer:
    """Build ML features from a single-asset close price series."""

    def __init__(self, lookback_periods: list[int]) -> None:
        self.lookback_periods = lookback_periods

    def compute(
        self,
        close: pd.Series,
        volume: pd.Series | None = None,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=close.index)

        # Returns and volatility at different horizons
        daily_ret = close.pct_change()
        for p in self.lookback_periods:
            features[f"return_{p}d"] = close.pct_change(p)
            features[f"volatility_{p}d"] = daily_ret.rolling(p).std()

        # Momentum
        for p in self.lookback_periods:
            features[f"momentum_{p}d"] = close / close.shift(p) - 1

        # SMA ratio and slope
        for p in self.lookback_periods:
            sma = close.rolling(p).mean()
            features[f"sma_ratio_{p}d"] = close / sma - 1
            features[f"sma_slope_{p}d"] = sma.pct_change(5)

        # RSI
        for period in (14, 28):
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            features[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # Bollinger band position (20d)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features["bb_position_20d"] = (close - sma20) / (2 * std20)

        # MACD (normalized)
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features["macd"] = macd / close
        features["macd_signal"] = signal / close
        features["macd_histogram"] = (macd - signal) / close

        # ATR proxy (no high/low, use abs(close.diff()) as true range)
        tr_proxy = close.diff().abs()
        for period in (14, 28):
            features[f"atr_{period}d"] = tr_proxy.rolling(period).mean() / close

        # Volume features (if available)
        if volume is not None:
            for p in self.lookback_periods:
                vol_sma = volume.rolling(p).mean()
                features[f"volume_ratio_{p}d"] = volume / vol_sma
                features[f"volume_trend_{p}d"] = vol_sma.pct_change(p)

        # Day-of-week cyclical encoding
        if hasattr(close.index, "dayofweek"):
            features["day_sin"] = np.sin(2 * np.pi * close.index.dayofweek / 5)
            features["day_cos"] = np.cos(2 * np.pi * close.index.dayofweek / 5)

        return features


def _create_target(
    close: pd.Series,
    horizon: int,
    task: str,
) -> pd.Series:
    """Create forward-looking target variable.

    For regression: forward percent return.
    For classification: binary direction (1 if positive, 0 otherwise).
    """
    forward_return = close.pct_change(horizon).shift(-horizon)
    if task == "regression":
        return forward_return
    return (forward_return > 0).astype(int)


# ---------------------------------------------------------------------------
# Model factory (private)
# ---------------------------------------------------------------------------


def _make_model(model_name: str, task: str) -> Any:
    """Instantiate an sklearn estimator with sensible defaults."""
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for MLPredictionStrategy")

    regression_models = {
        "ridge": (Ridge, {}),
        "lasso": (Lasso, {}),
        "elastic_net": (ElasticNet, {}),
        "random_forest": (RandomForestRegressor, {"n_estimators": 100, "random_state": 42}),
        "gradient_boosting": (GradientBoostingRegressor, {"n_estimators": 100, "random_state": 42}),
        "svr": (SVR, {"kernel": "rbf"}),
    }

    classification_models = {
        "logistic": (LogisticRegression, {"max_iter": 1000, "random_state": 42}),
        "random_forest": (RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
        "gradient_boosting": (GradientBoostingClassifier, {"n_estimators": 100, "random_state": 42}),
        "svc": (SVC, {"kernel": "rbf", "random_state": 42, "probability": True}),
    }

    models = regression_models if task == "regression" else classification_models
    if model_name not in models:
        available = sorted(models.keys())
        raise ValueError(f"Unknown model '{model_name}' for task '{task}'. Available: {available}")

    cls, defaults = models[model_name]
    return cls(**defaults)


def _make_scaler(scaler_name: str) -> Any:
    """Instantiate a sklearn scaler."""
    if scaler_name == "robust":
        return RobustScaler()
    return StandardScaler()


# ---------------------------------------------------------------------------
# Weight conversion helpers (private)
# ---------------------------------------------------------------------------


def _predictions_to_weights_rank(
    predictions: dict[str, float],
    top_n: int,
) -> dict[str, float]:
    """Top-N by predicted value, equal weight."""
    sorted_symbols = sorted(predictions, key=predictions.get, reverse=True)
    selected = sorted_symbols[:top_n]
    weight = 1.0 / len(selected) if selected else 0.0
    return {s: weight for s in selected}


def _predictions_to_weights_confidence(
    predictions: dict[str, float],
    probabilities: dict[str, float] | None,
    task: str,
) -> dict[str, float]:
    """Weight by prediction confidence (probability for classification,
    predicted return for regression)."""
    if task == "classification" and probabilities is not None:
        scores = {s: p for s, p in probabilities.items() if p > 0.5}
    else:
        scores = {s: v for s, v in predictions.items() if v > 0}

    if not scores:
        return {}
    total = sum(scores.values())
    if total == 0:
        return {}
    return {s: v / total for s, v in scores.items()}


def _predictions_to_weights_threshold(
    predictions: dict[str, float],
) -> dict[str, float]:
    """Positive predictions get proportional weight, normalized."""
    positive = {s: v for s, v in predictions.items() if v > 0}
    if not positive:
        return {}
    total = sum(positive.values())
    if total == 0:
        return {}
    return {s: v / total for s, v in positive.items()}


# ---------------------------------------------------------------------------
# Strategy plugin
# ---------------------------------------------------------------------------


@dataclass
class MLPredictionStrategy:
    """ML prediction strategy using sklearn models for return/direction forecasting.

    Trains a pooled cross-sectional model on engineered features from
    close prices, then converts predictions to portfolio weights.
    """

    meta = PluginMeta(
        name="strategy.ml_prediction.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="ML prediction strategy using sklearn models for return/direction forecasting",
        tags=("ml", "prediction", "sklearn"),
    )

    task: str = "classification"
    model_name: str = "gradient_boosting"
    prediction_horizon: int = 5
    lookback_periods: list[int] = field(default_factory=lambda: [5, 10, 20, 60])
    n_splits: int = 5
    scaler: str = "standard"
    train_lookback: int = 504
    retrain_frequency: int = 21
    weight_method: str = "rank"
    top_n: int = 10
    max_symbols: int = 50
    output_periods: int = 30

    def run(
        self,
        data: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for MLPredictionStrategy. Install with: uv add scikit-learn")

        if params:
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        prices: pd.DataFrame = data["prices"]
        volume_df: pd.DataFrame | None = data.get("volume")

        # Limit to max_symbols
        symbols = list(prices.columns[: self.max_symbols])
        prices = prices[symbols]
        if volume_df is not None:
            common = [s for s in symbols if s in volume_df.columns]
            volume_df = volume_df[common]

        engineer = _FeatureEngineer(self.lookback_periods)
        max_lb = max(self.lookback_periods)

        # Pre-compute features for each symbol
        symbol_features: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            vol_series = volume_df[sym] if (volume_df is not None and sym in volume_df.columns) else None
            symbol_features[sym] = engineer.compute(prices[sym], vol_series)

        n_features = symbol_features[symbols[0]].shape[1] if symbols else 0
        start_idx = self.train_lookback + max_lb
        dates = prices.index
        n_dates = len(dates)

        if start_idx >= n_dates:
            logger.warning(
                "Not enough data for training: need %d rows, have %d",
                start_idx,
                n_dates,
            )
            empty_weights = pd.DataFrame(0.0, index=dates[-self.output_periods :], columns=symbols)
            return {
                "weights": empty_weights,
                "simple_weights": {},
                "details": {"n_features": n_features, "model_name": self.model_name},
            }

        # Rolling train/predict
        weights_records: list[dict[str, float]] = []
        weight_dates: list[Any] = []

        current_model: Any | None = None
        last_train_idx = -self.retrain_frequency  # force initial train

        for i in range(start_idx, n_dates):
            need_retrain = (i - last_train_idx) >= self.retrain_frequency

            if need_retrain:
                current_model = self._train_pooled_model(
                    symbols,
                    symbol_features,
                    prices,
                    i,
                )
                last_train_idx = i

            if current_model is None:
                weights_records.append({s: 0.0 for s in symbols})
                weight_dates.append(dates[i])
                continue

            preds, probs = self._predict_symbols(
                symbols,
                symbol_features,
                current_model,
                i,
            )

            row_weights = self._convert_to_weights(preds, probs)
            weights_records.append(row_weights)
            weight_dates.append(dates[i])

        weights_df = pd.DataFrame(weights_records, index=weight_dates)
        weights_df = weights_df.reindex(columns=symbols, fill_value=0.0)
        output = weights_df.tail(self.output_periods)

        latest = output.iloc[-1] if len(output) > 0 else pd.Series(dtype=float)
        latest_dict = latest[latest > 1e-6].to_dict()

        return {
            "weights": output,
            "simple_weights": latest_dict,
            "details": {
                "n_features": n_features,
                "model_name": self.model_name,
            },
        }

    # -- private helpers --

    def _train_pooled_model(
        self,
        symbols: list[str],
        symbol_features: dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        current_idx: int,
    ) -> Any | None:
        """Train a pooled model on all symbols up to current_idx."""
        train_start = max(0, current_idx - self.train_lookback)
        x_parts: list[pd.DataFrame] = []
        y_parts: list[pd.Series] = []

        for sym in symbols:
            feat = symbol_features[sym].iloc[train_start:current_idx]
            target = _create_target(
                prices[sym],
                self.prediction_horizon,
                self.task,
            )
            target = target.iloc[train_start:current_idx]

            combined = pd.concat([feat, target.rename("_target")], axis=1).dropna()
            if len(combined) < 30:
                continue

            x_parts.append(combined.drop("_target", axis=1))
            y_parts.append(combined["_target"])

        if not x_parts:
            return None

        x_train = pd.concat(x_parts, axis=0)
        y_train = pd.concat(y_parts, axis=0)

        if len(x_train) < 50:
            return None

        model = _make_model(self.model_name, self.task)
        scaler_inst = _make_scaler(self.scaler)
        pipe = SKPipeline([("scaler", scaler_inst), ("model", model)])
        pipe.fit(x_train.values, y_train.values)
        return pipe

    def _predict_symbols(
        self,
        symbols: list[str],
        symbol_features: dict[str, pd.DataFrame],
        model: Any,
        idx: int,
    ) -> tuple[dict[str, float], dict[str, float] | None]:
        """Generate predictions for each symbol at a given index."""
        predictions: dict[str, float] = {}
        probabilities: dict[str, float] | None = None

        has_proba = hasattr(model, "predict_proba") or (
            hasattr(model, "named_steps") and hasattr(model.named_steps.get("model", None), "predict_proba")
        )
        if self.task == "classification" and has_proba:
            probabilities = {}

        for sym in symbols:
            feat = symbol_features[sym]
            if idx >= len(feat):
                continue
            row = feat.iloc[[idx]]
            if row.isna().all(axis=1).iloc[0]:
                continue

            row_filled = row.fillna(0.0)
            pred = model.predict(row_filled.values)[0]
            predictions[sym] = float(pred)

            if probabilities is not None:
                prob = model.predict_proba(row_filled.values)[0]
                # prob for positive class (index 1 if binary)
                probabilities[sym] = float(prob[-1])

        return predictions, probabilities

    def _convert_to_weights(
        self,
        predictions: dict[str, float],
        probabilities: dict[str, float] | None,
    ) -> dict[str, float]:
        """Convert predictions to portfolio weights based on weight_method."""
        if not predictions:
            return {}

        if self.weight_method == "rank":
            return _predictions_to_weights_rank(predictions, self.top_n)
        if self.weight_method == "confidence":
            return _predictions_to_weights_confidence(
                predictions,
                probabilities,
                self.task,
            )
        if self.weight_method == "threshold":
            return _predictions_to_weights_threshold(predictions)

        return _predictions_to_weights_rank(predictions, self.top_n)
