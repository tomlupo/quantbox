# Quantbox Integration Guide

How to integrate quantbox into your Python project for quant workflows.

## Quick Start

### 1. Add Dependency

In your `pyproject.toml`:

```toml
[project]
dependencies = [
    "quantbox",
    # ... other deps
]

[tool.uv.sources]
quantbox = { git = "https://github.com/tomlupo/quantbox", subdirectory = "packages/quantbox-core" }
```

Then sync:
```bash
uv sync
```

### 2. Use Quantbox

```python
from quantbox.registry import PluginRegistry
from quantbox.contracts import PipelinePlugin, DataPlugin

# Discover all available plugins
registry = PluginRegistry.discover()

# List what's available
print("Pipelines:", list(registry.pipelines.keys()))
print("Data:", list(registry.data.keys()))
print("Brokers:", list(registry.brokers.keys()))

# Use a plugin
DataLoader = registry.data["duckdb_parquet"]
loader = DataLoader()
```

---

## Integration Layer (Optional)

Create a wrapper module for project-specific conventions:

```python
# shared/quantbox_integration.py
"""Quantbox integration for your project."""

from __future__ import annotations
from pathlib import Path
from datetime import datetime

from quantbox.registry import PluginRegistry
from quantbox.store import FileArtifactStore

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def get_registry() -> PluginRegistry:
    """Get plugin registry with all discovered plugins."""
    return PluginRegistry.discover()


def create_artifact_store(run_name: str) -> FileArtifactStore:
    """Create artifact store with project conventions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return FileArtifactStore(str(DEFAULT_OUTPUT_DIR), f"{run_name}_{timestamp}")


def list_plugins() -> dict[str, list[str]]:
    """List all available plugins by type."""
    registry = get_registry()
    return {
        "pipelines": list(registry.pipelines.keys()),
        "brokers": list(registry.brokers.keys()),
        "data": list(registry.data.keys()),
        "publishers": list(registry.publishers.keys()),
        "risk": list(registry.risk.keys()),
    }


def get_plugin(plugin_type: str, plugin_name: str) -> type:
    """Get a specific plugin by type and name."""
    registry = get_registry()
    plugins = getattr(registry, plugin_type)
    if plugin_name not in plugins:
        raise KeyError(f"Plugin '{plugin_name}' not found. Available: {list(plugins.keys())}")
    return plugins[plugin_name]
```

---

## Custom Plugins

When quantbox-core doesn't have what you need, create custom plugins:

### Directory Structure

```
your-project/
├── tools/
│   └── quantbox_plugins/          # Custom plugins
│       ├── __init__.py
│       ├── data/
│       │   └── my_data_source.py
│       └── strategies/
│           └── my_strategy.py
├── shared/
│   └── quantbox_integration.py
└── pyproject.toml
```

### Plugin Template

```python
# tools/quantbox_plugins/data/my_data_source.py
"""Custom data source plugin."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd

from quantbox.contracts import PluginMeta, DataPlugin


@dataclass
class MyDataSource:
    """Custom data source implementation."""

    meta: PluginMeta = PluginMeta(
        name="my_data_source",
        kind="data",
        version="0.1.0",
        core_compat=">=0.1.0",
        description="Custom data source for XYZ",
    )

    def load_universe(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Load trading universe."""
        # Your implementation
        pass

    def load_prices(
        self,
        universe: pd.DataFrame,
        asof: str,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Load price data."""
        # Your implementation
        pass

    def load_fx(self, asof: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load FX rates (optional)."""
        return None
```

### Register via Entry Points (Optional)

In `pyproject.toml`:

```toml
[project.entry-points."quantbox.data"]
my_data_source = "tools.quantbox_plugins.data.my_data_source:MyDataSource"
```

---

## Plugin Types

| Type | Entry Point | Protocol | Purpose |
|------|-------------|----------|---------|
| Pipeline | `quantbox.pipelines` | `PipelinePlugin` | Research/trading workflows |
| Data | `quantbox.data` | `DataPlugin` | Data sources |
| Broker | `quantbox.brokers` | `BrokerPlugin` | Order execution |
| Publisher | `quantbox.publishers` | `PublisherPlugin` | Output notifications |
| Risk | `quantbox.risk` | `RiskPlugin` | Risk checks |

---

## CLI Usage

```bash
# List plugins
uv run quantbox plugins list

# Get plugin info
uv run quantbox plugins info --name duckdb_parquet

# Run a pipeline
uv run quantbox run fund_selection.simple.v1 --mode backtest --asof 2024-01-01

# JSON output for automation
uv run quantbox plugins list --json
```

---

## Updating Quantbox

```bash
# Update to latest
uv sync --upgrade

# Pin to specific commit
[tool.uv.sources]
quantbox = { git = "https://github.com/tomlupo/quantbox", rev = "abc123", subdirectory = "packages/quantbox-core" }
```

---

## Best Practices

1. **Use quantbox-core first** - Check `list_plugins()` before building custom
2. **Build custom only when needed** - Keep custom plugins minimal
3. **Port mature plugins** - When a custom plugin is stable, contribute it back to quantbox
4. **Use the artifact store** - Consistent output management with manifests

---

## Example: Full Workflow

```python
from shared.quantbox_integration import get_registry, create_artifact_store

# Setup
registry = get_registry()
store = create_artifact_store("momentum-backtest")

# Get plugins
DataLoader = registry.data["duckdb_parquet"]
Pipeline = registry.pipelines["fund_selection.simple.v1"]

# Load data
data = DataLoader()
universe = data.load_universe({"path": "data/universe.parquet"})
prices = data.load_prices(universe, "2024-01-01", {"path": "data/prices.parquet"})

# Run pipeline
pipeline = Pipeline()
result = pipeline.run(
    mode="backtest",
    asof="2024-01-01",
    params={"top_n": 10},
    data=data,
    store=store,
    broker=None,
    risk=[],
)

print(f"Run ID: {result.run_id}")
print(f"Artifacts: {result.artifacts}")
print(f"Metrics: {result.metrics}")
```

---

## Links

- [Quantbox Repository](https://github.com/tomlupo/quantbox)
- [Plugin Contracts](https://github.com/tomlupo/quantbox/blob/main/packages/quantbox-core/src/quantbox/contracts.py)
- [Built-in Plugins](https://github.com/tomlupo/quantbox/tree/main/packages/quantbox-core/src/quantbox/plugins)
