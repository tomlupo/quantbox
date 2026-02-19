# Config Patterns

Ready-to-use config examples for common use cases.

## Minimal Research / Fund Selection

```yaml
run:
  mode: backtest
  asof: "2026-01-31"
  pipeline: "fund_selection.simple.v1"

artifacts:
  root: "./artifacts"

plugins:
  pipeline:
    name: "fund_selection.simple.v1"
    params:
      top_n: 5
      universe:
        symbols: ["SPY", "QQQ", "IWM", "EEM", "TLT", "GLD"]
      prices:
        lookback_days: 365

  data:
    name: "local_file_data"
    params_init:
      prices_path: "./data/curated/prices.parquet"

  risk: []
  publishers: []
```

## Crypto Trend Backtest

```yaml
run:
  mode: backtest
  asof: "2026-02-01"
  pipeline: "backtest.pipeline.v1"

artifacts:
  root: "./artifacts"

plugins:
  pipeline:
    name: "backtest.pipeline.v1"
    params:
      engine: vectorbt
      rebalance_freq: daily
      fee_pct: 0.001
      trading_days: 365
      universe:
        top_n: 100
        lookback_days: 365

  data:
    name: "binance.live_data.v1"
    params_init:
      quote_currency: "USDT"

  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 1.0
      params:
        lookback_days: 365

  aggregator:
    name: "strategy.weighted_avg.v1"
    params: {}

  risk: []
  publishers: []
```

## Spot Paper Trading

```yaml
run:
  mode: paper
  asof: "2026-02-01"
  pipeline: "trade.full_pipeline.v1"

artifacts:
  root: "./artifacts"

plugins:
  pipeline:
    name: "trade.full_pipeline.v1"
    params:
      universe:
        symbols: ["BTC", "ETH", "SOL", "BNB"]
        lookback_days: 365

  data:
    name: "binance.live_data.v1"
    params_init:
      quote_currency: "USDT"

  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 1.0
      params:
        lookback_days: 365

  aggregator:
    name: "strategy.weighted_avg.v1"
    params: {}

  broker:
    name: "sim.paper.v1"
    params_init:
      cash: 100000
      quote_currency: "USDT"

  rebalancing:
    name: "rebalancing.standard.v1"
    params:
      min_trade_pct: 0.01

  risk:
    - name: "risk.trading_basic.v1"
      params:
        max_position_pct: 0.30
        max_leverage: 1.0

  publishers: []
```

## Futures Paper Trading

```yaml
run:
  mode: paper
  asof: "2026-02-01"
  pipeline: "trade.full_pipeline.v1"

artifacts:
  root: "./artifacts"

plugins:
  profile: "futures_paper"

  pipeline:
    name: "trade.full_pipeline.v1"
    params:
      universe:
        symbols: ["BTC", "ETH", "SOL"]
        lookback_days: 365

  strategies:
    - name: "strategy.carver_trend.v1"
      weight: 1.0
      params:
        lookback_days: 365

  aggregator:
    name: "strategy.weighted_avg.v1"
    params: {}

  broker:
    name: "sim.futures_paper.v1"
    params_init:
      cash: 100000
      quote_currency: "USDT"
      max_leverage: 3.0

  rebalancing:
    name: "rebalancing.futures.v1"
    params:
      target_leverage: 2.0

  risk:
    - name: "risk.trading_basic.v1"
      params:
        max_leverage: 3.0

  publishers: []
```

## Multi-Strategy Trading

```yaml
run:
  mode: paper
  asof: "2026-02-01"
  pipeline: "trade.full_pipeline.v1"

artifacts:
  root: "./artifacts"

plugins:
  pipeline:
    name: "trade.full_pipeline.v1"
    params:
      universe:
        symbols: ["BTC", "ETH", "SOL", "BNB", "AVAX"]
        lookback_days: 365

  data:
    name: "binance.live_data.v1"
    params_init:
      quote_currency: "USDT"

  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 0.5
      params:
        lookback_days: 365
    - name: "strategy.portfolio_optimizer.v1"
      weight: 0.3
      params:
        method: "risk_parity"
    - name: "strategy.momentum_long_short.v1"
      weight: 0.2
      params:
        lookback_days: 60

  aggregator:
    name: "strategy.weighted_avg.v1"
    params: {}

  broker:
    name: "sim.paper.v1"
    params_init:
      cash: 100000
      quote_currency: "USDT"

  rebalancing:
    name: "rebalancing.standard.v1"
    params:
      min_trade_pct: 0.01

  risk:
    - name: "risk.trading_basic.v1"
      params:
        max_position_pct: 0.25
        max_leverage: 1.0

  publishers: []
```

## Stress Test with Synthetic Data

```yaml
run:
  mode: backtest
  asof: "2026-02-01"
  pipeline: "backtest.pipeline.v1"

artifacts:
  root: "./artifacts"

plugins:
  profile: "stress_test"

  pipeline:
    name: "backtest.pipeline.v1"
    params:
      engine: vectorbt
      rebalance_freq: daily
      fee_pct: 0.001

  data:
    name: "data.synthetic.v1"
    params_init:
      n_symbols: 20
      n_days: 500

  strategies:
    - name: "strategy.portfolio_optimizer.v1"
      weight: 1.0
      params:
        method: "min_variance"

  aggregator:
    name: "strategy.weighted_avg.v1"
    params: {}

  risk:
    - name: "risk.trading_basic.v1"
      params:
        max_leverage: 1.0
    - name: "risk.stress_test.v1"
      params: {}

  publishers: []
```

## Existing Config Files

See `configs/` for 21 ready-to-use configs. Key ones:
- `configs/example_minimal.yaml` - annotated minimal example
- `configs/run_backtest_crypto_trend.yaml` - crypto trend backtest
- `configs/run_spot_paper_crypto_trend.yaml` - spot paper trading
- `configs/run_futures_paper.yaml` - futures paper trading
- `configs/run_trading_multi_strategy.yaml` - multi-strategy blending
- `configs/run_stress_test.yaml` - stress testing
