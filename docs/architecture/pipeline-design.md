# Pipeline architecture: why imperative orchestration?

## How pipelines work

Quantbox has three layers:

1. **Runner** (`runner.py`) — dependency injection container. Reads YAML config, resolves plugin classes from the registry, instantiates them with `params_init`, and calls `pipeline.run()` with all resolved dependencies.

2. **Pipeline** (e.g. `TradingPipeline`, `BacktestPipeline`) — imperative orchestrator. A Python class that defines the step sequence in its `run()` method: load data, run strategies, aggregate weights, check risk, generate orders, execute, store artifacts.

3. **Plugins** (Data, Strategy, Broker, Risk, Rebalancing, Publisher) — swappable implementations. Config picks *which* plugin class fills each slot; the pipeline calls them in order.

```
YAML config
  │
  ▼
Runner (DI container)
  ├── resolves plugin classes from registry
  ├── instantiates with params_init
  └── calls pipeline.run(mode, asof, params, data, store, broker, risk, ...)
        │
        ▼
      Pipeline.run()          ← imperative Python code
        ├── data.load_market_data()
        ├── for s in strategies: s.run()
        ├── aggregate weights
        ├── risk.check_targets()
        ├── rebalancer.generate_orders()
        ├── risk.check_orders()
        ├── broker.place_orders()
        └── store.put_parquet() / store.put_json()
```

The runner is generic. The pipeline is the decision-maker — it decides step order, error handling, and what to do with intermediate results.

## The design choice

The central question: should the pipeline be **imperative** (Python code defines the sequence) or **declarative** (YAML/DAG defines the sequence, framework executes it)?

Quantbox chose imperative.

### What declarative would look like

```yaml
# Hypothetical — NOT how quantbox works
pipeline:
  steps:
    - load_data: { plugin: binance.live_data.v1 }
    - run_strategies: { plugins: [crypto_trend.v1], aggregate: equal_weight }
    - check_risk: { plugins: [risk.trading_basic.v1] }
    - generate_orders: { plugin: rebalancing.futures.v1 }
    - execute: { plugin: hyperliquid.perps.v1 }
```

The framework would parse steps, resolve plugins, and execute them in order. Pipelines become data, not code.

### Why imperative was chosen

1. **Edge cases live in the gaps.** Trading pipelines have conditional logic between steps — "skip execution if risk check flagged critical", "use different aggregation for single vs. multi-strategy", "inject approval gate in live mode but not backtest." These are natural in Python, awkward in YAML.

2. **Debuggability.** When a pipeline fails, you step through Python with a debugger. With declarative pipelines, you debug the framework's step executor — an unnecessary abstraction layer between you and the problem.

3. **Small team, few pipelines.** Quantbox has three pipeline types (trading, backtest, research). A declarative framework pays off when you have dozens of pipeline variations defined by non-developers. Three pipelines written by developers doesn't justify the machinery.

4. **Full Python expressiveness.** Pipelines can do early returns, try/except around individual steps, log intermediate states, compute derived values between steps, and pass rich objects (not just artifact paths) between stages. No DSL limitations.

5. **No framework to maintain.** A declarative pipeline system is itself a significant piece of software — step resolution, dependency graphs, error propagation, retry policies, conditional execution. That's a framework you have to build, test, and document. Imperative pipelines are just Python.

## Trade-off analysis

| Dimension | Imperative (current) | Declarative |
|---|---|---|
| Adding a new pipeline | Write a Python class (~200 lines) | Write YAML (~20 lines) |
| Edge case handling | Native Python control flow | Escape hatches, custom step types, callbacks |
| Debugging | Standard Python debugger | Debug the step executor + your step logic |
| Non-developer users | Must read Python | Could compose pipelines from config |
| Pipeline count scaling | Fine for <10 pipelines | Pays off at 10+ similar pipelines |
| Testability | Test the class directly | Test framework + test each step + test composition |
| Reuse across pipelines | Extract shared functions | Framework handles composition |
| Onboarding | Read one file to understand a pipeline | Learn the DSL, then read the config |

## When this might change

Declarative pipelines would make sense if:

- **Pipeline count grows past ~10** with largely similar step sequences — the boilerplate of imperative classes starts to outweigh the flexibility.
- **Non-developers need to create pipelines** — researchers or ops people who shouldn't need to write Python.
- **The team grows significantly** — declarative pipelines enforce structure that prevents divergent implementations.
- **Cross-system orchestration** — if pipelines need to span multiple services or trigger external workflows, a DAG framework (like Airflow/Prefect) becomes appropriate.

None of these conditions currently apply. The framework has 3 pipeline types maintained by 1-2 developers.

## The composability you DO get

The imperative design doesn't mean zero composability. Config-driven plugin selection gives most of the benefit:

- **Swap data source**: change `data.name` from `local.parquet_data.v1` to `binance.live_data.v1` — same pipeline, different data.
- **Swap broker**: change `broker.name` from `sim.paper_broker.v1` to `hyperliquid.perps.v1` — same pipeline, paper vs. live.
- **Swap strategy**: change `strategies[0].name` — same pipeline, different alpha.
- **Add risk checks**: append to the `risk` list — pipeline calls them all.
- **Research to production**: change `pipeline.name` from `backtest.pipeline.v1` to `trade.full_pipeline.v1` with the same strategy params.

The pipeline is the fixed frame; plugins are the moving parts. This is dependency injection, not declarative composition, and it covers ~90% of the configuration use cases.

## Summary

Quantbox pipelines are imperative Python classes that receive plugin instances from a config-driven runner. This trades the elegance of declarative composition for debuggability, full control flow, and zero framework overhead — the right trade-off for a small team with few pipeline types and many edge cases.
