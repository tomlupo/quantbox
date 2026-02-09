"""Quickstart: run a crypto trend backtest programmatically.

This example shows how to call the quantbox runner from Python
(instead of the CLI) and inspect the results.

Usage:
    uv run python examples/01_quickstart_backtest.py
"""

from quantbox.registry import PluginRegistry
from quantbox.runner import run_from_config
from quantbox.validate import validate_config

# 1. Discover all available plugins
registry = PluginRegistry.discover()
print("Available pipelines:", list(registry.pipelines.keys()))

# 2. Define config as a plain dict (same structure as YAML configs)
config = {
    "run": {
        "mode": "backtest",
        "asof": "2026-02-06",
        "pipeline": "backtest.pipeline.v1",
    },
    "artifacts": {"root": "./artifacts"},
    "plugins": {
        "pipeline": {
            "name": "backtest.pipeline.v1",
            "params": {
                "engine": "vectorbt",
                "fees": 0.001,
                "rebalancing_freq": 1,
                "trading_days": 365,
                "universe": {"top_n": 50},
                "prices": {"lookback_days": 180},
                "risk": {"tranches": 1, "max_leverage": 1, "allow_short": False},
            },
        },
        "strategies": [
            {
                "name": "strategy.crypto_trend.v1",
                "weight": 1.0,
                "params": {"lookback_days": 180},
            }
        ],
        "data": {
            "name": "binance.live_data.v1",
            "params_init": {"quote_asset": "USDT"},
        },
    },
}

# 3. Validate first (optional but recommended)
findings = validate_config(config)
for f in findings:
    print(f"{f.level.upper()}: {f.message}")

# 4. Run
result = run_from_config(config, registry)

# 5. Inspect results
print(f"\nRun ID:   {result.run_id}")
print(f"Pipeline: {result.pipeline_name}")
print(f"Mode:     {result.mode}")
print(f"Metrics:  {result.metrics}")
print("Artifacts:")
for name, path in result.artifacts.items():
    print(f"  {name}: {path}")
