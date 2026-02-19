# Strategy Implementation Patterns

## Pattern 1: Signal-Based Strategy

Generate trading signals, scale by volatility, construct weights.

```python
@dataclass
class TrendStrategy:
    meta = PluginMeta(name="strategy.my_trend.v1", kind="strategy", ...)
    lookback: int = 60
    vol_target: float = 0.25

    def run(self, data, params=None):
        prices = data["prices"]
        if params:
            for k, v in params.items():
                if hasattr(self, k): setattr(self, k, v)

        # 1. Generate signals (0/1 or continuous)
        signals = (prices > prices.rolling(self.lookback).mean()).astype(float)

        # 2. Scale by inverse volatility
        returns = prices.pct_change()
        vol = returns.rolling(60).std() * np.sqrt(365)
        scaler = self.vol_target / vol.replace(0, np.nan)
        scaler = scaler.clip(0.1, 10.0)

        # 3. Construct weights
        raw_weights = signals * scaler
        weights = raw_weights.div(raw_weights.sum(axis=1).replace(0, np.nan), axis=0)

        return {"weights": weights, "simple_weights": weights.iloc[-1].dropna().to_dict()}
```

## Pattern 2: Optimization-Based Strategy

Use scipy or cvxpy for portfolio optimization.

```python
@dataclass
class OptimizerStrategy:
    meta = PluginMeta(name="strategy.my_optimizer.v1", kind="strategy", ...)
    method: str = "max_sharpe"  # max_sharpe | min_variance | risk_parity

    def run(self, data, params=None):
        prices = data["prices"]
        returns = prices.pct_change().dropna()

        mu = returns.mean() * 252
        cov = returns.cov() * 252

        # Optimize (simplified)
        n = len(mu)
        if self.method == "equal_weight":
            w = np.ones(n) / n
        elif self.method == "min_variance":
            from scipy.optimize import minimize
            result = minimize(
                lambda w: w @ cov.values @ w,
                np.ones(n) / n,
                constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                bounds=[(0, 1)] * n,
            )
            w = result.x
        # ... more methods

        weights = pd.DataFrame(
            {s: [w[i]] for i, s in enumerate(prices.columns)},
            index=[prices.index[-1]],
        )
        return {"weights": weights}
```

## Pattern 3: ML-Based Strategy

Use scikit-learn models for return prediction.

```python
@dataclass
class MLStrategy:
    meta = PluginMeta(name="strategy.my_ml.v1", kind="strategy", ...)
    model_type: str = "random_forest"
    n_estimators: int = 100

    def run(self, data, params=None):
        prices = data["prices"]
        returns = prices.pct_change()

        # Feature engineering
        features = pd.DataFrame({
            "mom_20": prices.pct_change(20).iloc[-1],
            "vol_60": returns.rolling(60).std().iloc[-1],
            "rsi_14": ...,
        })

        # Train on historical, predict forward
        # ... model training logic ...

        # Convert predictions to weights
        predictions = ...
        weights = predictions / predictions.sum()

        return {"weights": pd.DataFrame(weights, index=[prices.index[-1]]).T}
```

## Pattern 4: Universe Filtering

Filter tradeable universe before signal generation.

```python
def _filter_universe(self, data):
    """Select top assets by market cap and volume."""
    prices = data["prices"]
    volume = data.get("volume", pd.DataFrame())
    market_cap = data.get("market_cap", pd.DataFrame())

    if market_cap.empty:
        return prices.columns.tolist()

    # Top N by market cap, then filter by volume
    latest_mcap = market_cap.iloc[-1].dropna().nlargest(self.top_n)
    symbols = latest_mcap.index.tolist()

    # Exclude stablecoins
    exclude = {"USDT", "USDC", "DAI", "BUSD", "TUSD"}
    symbols = [s for s in symbols if s not in exclude]

    return symbols
```

## Pattern 5: Multi-Strategy Blending

Config-driven blending via the aggregator:

```yaml
plugins:
  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 0.5
      params: { lookback_days: 365 }
    - name: "strategy.portfolio_optimizer.v1"
      weight: 0.3
      params: { method: "risk_parity" }
    - name: "strategy.momentum_long_short.v1"
      weight: 0.2
      params: { lookback_days: 60 }

  aggregator:
    name: "strategy.weighted_avg.v1"
    params: {}
```

The `weighted_avg` aggregator normalizes config weights and computes:
`final_weights = sum(strategy_weight_i * strategy_output_i)`

## Testing Pattern

```python
import pandas as pd
import numpy as np
import pytest

def make_test_data(n_days=500, n_symbols=10):
    """Generate synthetic market data for testing."""
    dates = pd.bdate_range(end="2026-01-31", periods=n_days)
    symbols = [f"SYM{i}" for i in range(n_symbols)]

    prices = pd.DataFrame(
        np.cumprod(1 + np.random.randn(n_days, n_symbols) * 0.02, axis=0) * 100,
        index=dates, columns=symbols,
    )
    volume = pd.DataFrame(
        np.random.randint(1000, 100000, (n_days, n_symbols)),
        index=dates, columns=symbols,
    )
    market_cap = prices * volume

    return {"prices": prices, "volume": volume, "market_cap": market_cap}


def test_my_strategy():
    strategy = MyStrategy()
    data = make_test_data()
    result = strategy.run(data)

    assert "weights" in result
    weights = result["weights"]
    assert isinstance(weights, pd.DataFrame)
    assert not weights.empty
    # Weights should be reasonable
    assert weights.values.min() >= -1.0  # allow short positions if applicable
    assert weights.values.max() <= 5.0   # no extreme leverage
```
