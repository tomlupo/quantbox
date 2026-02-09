# Pipeline chaining

> **Status**: Not yet implemented. This page describes the planned design.

QuantBox pipelines currently run independently. To chain pipelines (e.g. research produces allocations, then trading consumes them), use manual two-step execution:

```bash
# Step 1: research pipeline produces allocations.parquet
uv run quantbox run -c configs/run_fund_selection.yaml

# Step 2: trading pipeline consumes the artifact path
uv run quantbox run -c configs/run_trade_from_allocations.yaml
```

The trading config references the research output via `allocations_run_id` or an explicit file path in its params.

## Resolving the latest run

Use `run_history.py` to find the most recent run for a given pipeline:

```python
from quantbox.run_history import find_latest_run

latest = find_latest_run("./artifacts", pipeline="fund_selection.simple.v1")
print(latest)  # run_id string
```

## Planned: meta-pipeline (v1)

A future `meta.pipeline.v1` plugin will orchestrate multiple pipelines in sequence, passing artifact paths between them automatically. Track progress in GitHub issues.
