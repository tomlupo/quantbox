# Pipelines

Pipeline plugins orchestrate a full run: load data, run strategies, validate via risk,
execute trades, and store artifacts. They are the top-level entry point for any quantbox workflow.

## When to Use

- You need to run a backtest, paper trade, or live trade
- You want to create a custom workflow that coordinates multiple plugin types
- You need to understand the execution flow of an existing pipeline

## Available Pipelines

<!-- BEGIN AUTO-GENERATED -->
### Pipelines

| ID | Description | Kind |
|---|---|---|
| `backtest.pipeline.v1` | Backtesting pipeline: same config as TradingPipeline but routes weights through vectorbt or rsims backtesting engines instead of broker execution. | research |
| `fund_selection.simple.v1` | Simple ranking pipeline (research) | research |
| `trade.allocations_to_orders.v1` | Bridge allocations -> targets/orders (+ execute). Supports multipliers, lot/step rules, FX (USD base). | trading |
| `trade.full_pipeline.v1` | Full trading pipeline: strategy execution -> aggregation -> risk transforms -> order generation -> execution. Ported from quantlab trading workflow. | trading |

### Rebalancing

| ID | Description |
|---|---|
| `rebalancing.futures.v1` | Futures rebalancer: margin-based with signed position support |
| `rebalancing.standard.v1` | Standard rebalancer: risk transforms + order generation |

### Publishers

| ID | Description |
|---|---|
| `telegram.publisher.v1` | Send trading results to Telegram (execution summary, rebalancing, orders). |
<!-- END AUTO-GENERATED -->

## Execution Flow

```
Config YAML
  -> validate_config()
  -> run_from_config()
    -> Resolve profile (if specified)
    -> Instantiate plugins from registry
    -> Create FileArtifactStore(root, run_id)
    -> pipeline.run(mode, asof, params, data, store, broker, risk, strategies, ...)
      -> data.load_universe() + data.load_market_data()
      -> strategy.run() for each strategy
      -> aggregator blends weights
      -> risk.check_targets() for each risk plugin
      -> rebalancer.generate_orders()
      -> risk.check_orders()
      -> broker.place_orders() (paper/live only)
    -> Store run_manifest.json
    -> Publishers.publish()
```

## Artifact Output

Each run writes to `artifacts/<run_id>/`:
- `target_weights.parquet` - final portfolio weights
- `orders.parquet` - generated orders
- `fills.parquet` - execution fills (paper/live)
- `run_manifest.json` - full run metadata and artifact index
- `run_meta.json` - config + params snapshot
- `events.jsonl` - timestamped event stream

## Next Steps

- **Create a custom pipeline**: Load [api.md](api.md)
- **Debug pipeline issues**: Load [gotchas.md](gotchas.md)
