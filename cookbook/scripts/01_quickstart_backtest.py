"""Quickstart: run a crypto trend backtest.

Shows two equivalent ways to invoke the runner:
  1. From a YAML config file (the normal way — what the CLI does).
  2. From a plain Python dict (useful for programmatic overrides).

Usage:
    uv run python cookbook/scripts/01_quickstart_backtest.py
"""

from pathlib import Path

from quantbox.registry import PluginRegistry
from quantbox.runner import run_from_config
from quantbox.validate import validate_config

# 1. Discover all available plugins
registry = PluginRegistry.discover()
print("Available pipelines:", list(registry.pipelines.keys()))

# -- Option A: load from YAML (mirrors `quantbox run -c cookbook/configs/run_backtest_crypto_trend.yaml`) --
config_path = Path(__file__).parent.parent / "configs" / "run_backtest_crypto_trend.yaml"

findings = validate_config(config_path)
for f in findings:
    print(f"{f.level.upper()}: {f.message}")

result = run_from_config(config_path, registry)

# -- Option B: pass a plain dict (same structure as the YAML above, useful for param sweeps) --
# import yaml
# config = yaml.safe_load(config_path.read_text())
# config["run"]["asof"] = "2025-12-31"   # override a single key
# result = run_from_config(config, registry)

print(f"\nRun ID:   {result.run_id}")
print(f"Pipeline: {result.pipeline_name}")
print(f"Mode:     {result.mode}")
print(f"Metrics:  {result.metrics}")
print("Artifacts:")
for name, path in result.artifacts.items():
    print(f"  {name}: {path}")
