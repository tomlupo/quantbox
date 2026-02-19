# Strategies

Strategy plugins compute target portfolio weights from market data. They are the
core algorithmic component of any quantbox pipeline.

## When to Use

- You need to compute portfolio allocations from price/volume/market cap data
- You want to implement a new trading signal or alpha source
- You need to blend multiple strategies together

## Available Strategies

<!-- BEGIN AUTO-GENERATED -->
| ID | Description | Tags |
|---|---|---|
| `strategy.beglobal.v1` | BeGlobal core-satellite multi-asset strategy with dual momentum and volatility targeting | multi-asset, etf, core-satellite, momentum |
| `strategy.carver_trend.v1` | Carver-style trend following with EWMAC and breakout rules | crypto, trend, carver |
| `strategy.cross_asset_momentum.v1` | Cross-asset momentum (XSMOM) with core-satellite portfolio construction | crypto, momentum, xsmom, core-satellite |
| `strategy.crypto_regime_trend.v1` | BTC regime-based long/short trend following with multi-window ensemble | crypto, trend, regime, long-short |
| `strategy.crypto_trend.v1` | Crypto trend catcher - multi-asset volatility-targeted trend following | crypto, trend, momentum |
| `strategy.ml_prediction.v1` | ML prediction strategy using sklearn models for return/direction forecasting | ml, prediction, sklearn |
| `strategy.momentum_long_short.v1` | Long-short momentum strategy - market-neutral crypto factor | crypto, momentum, long-short |
| `strategy.portfolio_optimizer.v1` | Mean-variance portfolio optimizer (max Sharpe, min variance, risk parity, equal weight) | optimization, mean-variance, multi-asset |
| `strategy.weighted_avg.v1` | Weighted-average meta-strategy aggregator | aggregator, meta-strategy |
<!-- END AUTO-GENERATED -->

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
