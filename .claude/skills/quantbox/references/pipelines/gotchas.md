# Pipeline Gotchas

## Pipeline crashes with "broker is None"

**Cause:** Running in `paper` or `live` mode without a broker configured.

**Solution:** Add a broker section to the config:
```yaml
run:
  mode: paper
plugins:
  broker:
    name: "sim.paper.v1"
    params_init:
      cash: 100000
```

Research pipelines (`fund_selection.simple.v1`) don't need a broker. Trading pipelines
(`trade.full_pipeline.v1`) require one in paper/live mode.

## Pipeline runs but produces no artifacts

**Cause:** Pipeline `run()` method doesn't call `store.put_parquet()` or `store.put_json()`.

**Solution:** Ensure your pipeline stores outputs:
```python
store.put_parquet("target_weights", weights_df)
store.put_json("run_summary", {"mode": mode, "asof": asof})
```

Check `artifacts/<run_id>/` for files after a run.

## "run.pipeline" doesn't match "plugins.pipeline.name"

**Cause:** Config has different pipeline IDs in the two sections.

**Solution:** They must match:
```yaml
run:
  pipeline: "trade.full_pipeline.v1"  # must match below
plugins:
  pipeline:
    name: "trade.full_pipeline.v1"    # must match above
```

## Missing strategies in trading pipeline

**Cause:** `trade.full_pipeline.v1` expects a `plugins.strategies` list but none configured.

**Solution:** Either add strategies or use a pipeline that doesn't require them:
```yaml
plugins:
  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 1.0
      params: { lookback_days: 365 }
  aggregator:
    name: "strategy.weighted_avg.v1"
    params: {}
```

## RunResult artifacts dict has wrong format

**Cause:** Putting DataFrame objects instead of file paths in `RunResult.artifacts`.

**Solution:** Use the artifact name (string), not the data:
```python
# CORRECT
return RunResult(artifacts={"target_weights": "target_weights.parquet"}, ...)

# WRONG
return RunResult(artifacts={"target_weights": weights_df}, ...)
```

## Rebalancer not invoked

**Cause:** Missing `plugins.rebalancing` section in config.

**Solution:** Add rebalancing config for trading pipelines:
```yaml
plugins:
  rebalancing:
    name: "rebalancing.standard.v1"
    params:
      min_trade_pct: 0.01
```

## Schema validation warnings after run

**Cause:** Artifact DataFrames don't match expected JSON schemas in `schemas/`.

**Solution:** Check the schema file for required columns. For example, `orders.schema.json`
expects: `symbol`, `side`, `qty`, `order_type`. These are best-effort warnings, not errors.
