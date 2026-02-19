# Config Reference

QuantBox pipelines are driven entirely by YAML config files. This reference covers
the complete config schema and the key distinction between `params` and `params_init`.

## Config Structure

```yaml
run:
  mode: backtest|paper|live     # execution mode
  asof: "2026-02-01"            # reference date (strategies look back from here)
  pipeline: "pipeline.name.v1"  # pipeline to execute (must match plugins.pipeline.name)

artifacts:
  root: "./artifacts"           # output directory (each run gets a timestamped subdir)

plugins:
  profile: "research"           # optional preset from plugins/manifest.yaml

  pipeline:
    name: "pipeline.name.v1"
    params: { ... }             # runtime params -> pipeline.run()

  data:
    name: "data.source.v1"
    params_init: { ... }        # constructor params -> dataclass fields

  strategies:                   # list with blend weights
    - name: "strategy.name.v1"
      weight: 1.0
      params: { ... }

  aggregator:                   # blends multiple strategies
    name: "strategy.weighted_avg.v1"
    params: {}

  broker:
    name: "broker.name.v1"
    params_init: { ... }

  rebalancing:
    name: "rebalancing.type.v1"
    params: { ... }

  risk:                         # list - all must pass
    - name: "risk.type.v1"
      params: { ... }

  publishers:                   # list - all run after pipeline completes
    - name: "publisher.type.v1"
      params_init: { ... }
```

## params vs params_init

| Field | When used | Passed to | Use for |
|---|---|---|---|
| `params` | Runtime | `plugin.run()`, `plugin.check_targets()` | Lookback windows, thresholds, universe definitions |
| `params_init` | Construction | `@dataclass` constructor kwargs | API keys, file paths, connection strings, cash amount |

## Profiles

Profiles in `plugins/manifest.yaml` provide preset defaults. Override individual sections as needed.

| Profile | Pipeline | Data | Broker | Use case |
|---|---|---|---|---|
| `research` | fund_selection.simple.v1 | local_file_data | - | Research with local data |
| `trading` | trade.allocations_to_orders.v1 | local_file_data | sim.paper.v1 | Spot paper trading |
| `trading_full` | trade.full_pipeline.v1 | binance.live_data.v1 | binance.live.v1 | Live crypto trading |
| `stress_test` | backtest.pipeline.v1 | data.synthetic.v1 | sim.paper.v1 | Risk analysis |
| `futures_paper` | trade.full_pipeline.v1 | binance.live_data.v1 | sim.futures_paper.v1 | Futures paper trading |

## Validation

Always validate before running:
```bash
uv run quantbox validate -c configs/my_config.yaml
uv run quantbox run --dry-run -c configs/my_config.yaml
```

## Next Steps

- **See ready-to-use config examples**: Load [patterns.md](patterns.md)
- **Debug config issues**: Load [gotchas.md](gotchas.md)
