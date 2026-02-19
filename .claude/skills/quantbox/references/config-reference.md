# Config Reference

QuantBox pipelines are driven entirely by YAML config files. This reference covers the
complete config schema, the distinction between `params` and `params_init`, profiles,
and ready-to-use examples.

## Full Config Schema

```yaml
# --- Run settings (required) ---
run:
  mode: backtest            # "backtest" | "paper" | "live"
  asof: "2026-02-01"        # Reference date (strategies look back from here)
  pipeline: "pipeline.id"   # Pipeline plugin ID to execute

# --- Artifact storage (required) ---
artifacts:
  root: "./artifacts"       # Each run creates a timestamped subdirectory

# --- Plugin wiring (required) ---
plugins:
  # Optional preset from plugins/manifest.yaml
  profile: "research"

  # Pipeline: orchestrates the full run
  pipeline:
    name: "fund_selection.simple.v1"
    params:                 # Passed to pipeline.run() at runtime
      top_n: 5
      universe:
        symbols: ["SPY", "QQQ", "IWM"]
      prices:
        lookback_days: 365

  # Data source
  data:
    name: "local_file_data"
    params_init:            # Passed to dataclass constructor
      prices_path: "./data/curated/prices.parquet"

  # Strategies (list, each with blend weight)
  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 1.0           # Blend weight (normalized across strategies)
      params:
        lookback_days: 90

  # Strategy aggregator (blends multiple strategies)
  aggregator:
    name: "strategy.weighted_avg.v1"
    params: {}

  # Broker (required for paper/live modes)
  broker:
    name: "sim.paper.v1"
    params_init:
      cash: 100000
      quote_currency: "USDT"

  # Rebalancing (converts weights to orders)
  rebalancing:
    name: "rebalancing.standard.v1"
    params:
      min_trade_pct: 0.01   # Skip trades smaller than 1%

  # Risk (list; all must pass before execution)
  risk:
    - name: "risk.trading_basic.v1"
      params:
        max_position_pct: 0.25
        max_leverage: 1.0

  # Publishers (list; all run after pipeline completes)
  publishers:
    - name: "telegram.publisher.v1"
      params_init:
        token_env: "TELEGRAM_TOKEN"
        chat_id_env: "TELEGRAM_CHAT_ID"
```

## params vs params_init

| Field | When used | What it does |
|---|---|---|
| `params` | At runtime | Passed to `plugin.run()`, `plugin.check_targets()`, etc. |
| `params_init` | At construction | Passed as keyword arguments to the `@dataclass` constructor |

Use `params_init` for credentials, file paths, connection strings, and other constructor args.
Use `params` for runtime configuration like lookback windows, thresholds, and universe definitions.

## Profiles

Profiles in `plugins/manifest.yaml` provide preset defaults. Specify `plugins.profile` to
load a preset, then override individual sections as needed.

| Profile | Pipeline | Data | Broker | Use case |
|---|---|---|---|---|
| `research` | fund_selection.simple.v1 | local_file_data | - | Research with local data |
| `trading` | trade.allocations_to_orders.v1 | local_file_data | sim.paper.v1 | Spot paper trading |
| `trading_full` | trade.full_pipeline.v1 | binance.live_data.v1 | binance.live.v1 | Live crypto trading |
| `stress_test` | backtest.pipeline.v1 | data.synthetic.v1 | sim.paper.v1 | Risk analysis |
| `futures_paper` | trade.full_pipeline.v1 | binance.live_data.v1 | sim.futures_paper.v1 | Futures paper trading |

## Example: Minimal Backtest

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

## Example: Crypto Trend Backtest

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

## Example: Spot Paper Trading

```yaml
run:
  mode: paper
  asof: "2026-02-01"
  pipeline: "trade.full_pipeline.v1"

artifacts:
  root: "./artifacts"

plugins:
  profile: "trading"

  pipeline:
    name: "trade.full_pipeline.v1"
    params:
      universe:
        symbols: ["BTC", "ETH", "SOL", "BNB"]
        lookback_days: 365

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

## Example: Futures Paper Trading

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

## Validation

Always validate before running:

```bash
uv run quantbox validate -c configs/my_config.yaml
```

If validation fails, the error will include a `.findings` list describing each issue.
Common issues:
- Missing required plugin section (data is always required)
- Plugin name not found in registry (typo or not installed)
- `run.pipeline` doesn't match `plugins.pipeline.name`
- Invalid parameter types
