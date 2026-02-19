# Strategies

Strategy plugins compute target portfolio weights from market data. They are the
core algorithmic component of any quantbox pipeline.

## When to Use

- You need to compute portfolio allocations from price/volume/market cap data
- You want to implement a new trading signal or alpha source
- You need to blend multiple strategies together

## Available Strategies

| ID | Type | Asset class | Description |
|---|---|---|---|
| `strategy.crypto_trend.v1` | Trend | Crypto | Donchian breakout + volatility targeting. Multi-window ensemble. |
| `strategy.carver_trend.v1` | Trend | Futures | Robert Carver-style trend following, multi-instrument. |
| `strategy.momentum_long_short.v1` | Momentum | Any | Cross-sectional: long winners, short losers. |
| `strategy.cross_asset_momentum.v1` | Momentum | Any | Time-series momentum across asset classes. |
| `strategy.crypto_regime_trend.v1` | Regime | Crypto | HMM regime detection combined with trend signals. |
| `strategy.beglobal.v1` | Multi-asset | Global | Equity, bonds, commodities diversified allocation. |
| `strategy.portfolio_optimizer.v1` | Optimization | Any | Mean-variance (max Sharpe, min variance, risk parity). |
| `strategy.ml_prediction.v1` | ML | Any | Scikit-learn classifier/regressor for return prediction. |
| `strategy.weighted_avg.v1` | Aggregator | - | Blends multiple strategy outputs by config weight. |

## Strategy Selection Guide

**Crypto-only portfolio?**
- Start with `strategy.crypto_trend.v1` (most battle-tested)
- Add `strategy.crypto_regime_trend.v1` for regime-aware sizing
- Blend via `strategy.weighted_avg.v1` aggregator

**Traditional assets (equities, bonds)?**
- `strategy.beglobal.v1` for diversified multi-asset
- `strategy.portfolio_optimizer.v1` for mean-variance optimization
- `strategy.cross_asset_momentum.v1` for momentum factor

**Futures / leveraged?**
- `strategy.carver_trend.v1` designed for futures
- Pair with `rebalancing.futures.v1` and `sim.futures_paper.v1`

**ML / custom signals?**
- `strategy.ml_prediction.v1` wraps scikit-learn models
- Or create a custom strategy (see [api.md](api.md))

**Multiple strategies?**
- List them in `plugins.strategies[]` with blend `weight`
- Always include `strategy.weighted_avg.v1` as `plugins.aggregator`

## Config Pattern

```yaml
plugins:
  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 0.6
      params:
        lookback_days: 365
    - name: "strategy.portfolio_optimizer.v1"
      weight: 0.4
      params:
        method: "max_sharpe"

  aggregator:
    name: "strategy.weighted_avg.v1"
    params: {}
```

## Next Steps

- **Implement a new strategy**: Load [api.md](api.md) for the protocol and template
- **See implementation recipes**: Load [patterns.md](patterns.md)
- **Debug strategy issues**: Load [gotchas.md](gotchas.md)
