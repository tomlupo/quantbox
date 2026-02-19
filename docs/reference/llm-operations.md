# LLM / Agent Operations

QuantBox provides multiple integration layers for LLM agents, from simple CLI commands to a full MCP server and Claude Agent SDK subagents.

## CLI commands (any agent)

```bash
quantbox plugins list --json                              # all plugins as JSON
quantbox plugins info --name strategy.crypto_trend.v1 --json  # plugin details
quantbox validate -c configs/run_backtest_crypto_trend.yaml   # validate config
quantbox run --dry-run -c configs/run_backtest_crypto_trend.yaml  # preview
quantbox run -c configs/run_backtest_crypto_trend.yaml    # execute
```

Artifacts per run are written to `artifacts/<run_id>/`:
- `run_manifest.json` — single-file summary for agents
- `events.jsonl` — structured event stream
- `target_weights.parquet`, `orders.parquet`, etc.

## Programmatic API (`quantbox.agents.QuantBoxAgent`)

All methods return plain dicts. Errors are returned as `{"error": ...}`, never raised.

```python
from quantbox.agents import QuantBoxAgent

agent = QuantBoxAgent()

# Discovery
agent.list_plugins()                          # {kind: [{name, description, tags}]}
agent.list_plugins(kind="strategy")           # filter by type
agent.plugin_info("strategy.crypto_trend.v1") # detailed info with params, methods
agent.search_plugins("trend")                 # search by keyword across name/desc/tags

# Config
config = agent.build_config(
    mode="backtest",
    pipeline="backtest.pipeline.v1",
    strategy="strategy.crypto_trend.v1",
    data="binance.live_data.v1",
    strategy_params={"lookback_days": 90},
)
agent.validate_config(config)                 # {"valid": True, "findings": [...]}

# Execution
result = agent.run(config)                    # {"run_id": ..., "metrics": ...}
result = agent.run(config, dry_run=True)      # preview without execution

# Inspection
agent.inspect_run("artifacts/<run_id>")       # manifest + artifact list
agent.read_artifact("artifacts/<run_id>", "target_weights.parquet")

# Profiles
agent.list_profiles()                         # research, trading, etc.
```

## Plugin introspection (`quantbox.introspect`)

Universal introspection that works with any plugin, whether or not it has a custom `describe()` method.

```python
from quantbox.introspect import describe_plugin, describe_plugin_class, describe_registry
from quantbox.registry import PluginRegistry

# Describe a plugin class (no instantiation needed)
from quantbox.plugins.builtins import builtins
plugins = builtins()
info = describe_plugin_class(plugins["strategy"]["strategy.crypto_trend.v1"])
# Returns: {name, kind, description, version, tags, capabilities,
#           params_schema, parameter_defaults, methods}

# Describe an instance (includes current parameter values)
instance = plugins["strategy"]["strategy.crypto_trend.v1"]()
info = describe_plugin(instance)
# Returns: same as above plus "parameters" with current values

# Full registry catalog
registry = PluginRegistry.discover()
catalog = describe_registry(registry)
# Returns: {kind: [plugin_info, ...]}
```

## MCP server

The MCP server exposes QuantBox operations as tools for Claude Code, Cursor, and other MCP-compatible clients.

### Setup

In `.claude/settings.json`:
```json
{
  "mcpServers": {
    "quantbox": {
      "command": "uv",
      "args": ["run", "--extra", "agents", "python", "-m", "quantbox.agents.mcp_server"]
    }
  }
}
```

Or run standalone: `uv run quantbox-mcp`

### Tools

| Tool | Description |
|---|---|
| `quantbox_list_plugins` | Browse all registered plugins, optionally filter by kind |
| `quantbox_plugin_info` | Get detailed info about a specific plugin |
| `quantbox_search_plugins` | Search by name, description, or tags |
| `quantbox_build_config` | Construct a YAML config from parameters |
| `quantbox_validate_config` | Validate a YAML config |
| `quantbox_run` | Execute a pipeline |
| `quantbox_dry_run` | Preview execution without side effects |
| `quantbox_inspect_run` | Read artifacts from a completed run |
| `quantbox_list_profiles` | List available plugin profiles |

### Typical MCP workflow

1. `quantbox_list_plugins` — browse what's available
2. `quantbox_plugin_info` — get details on chosen plugins
3. `quantbox_build_config` — construct a config
4. `quantbox_validate_config` — check for errors
5. `quantbox_run` — execute the pipeline
6. `quantbox_inspect_run` — read results

## Claude Agent SDK subagents

Pre-built agent configurations for common quant workflows. Requires `pip install claude-agent-sdk`.

### Available agents

| Agent | Function | Description |
|---|---|---|
| `quant-researcher` | `research_agent()` | Explores strategies, analyzes data, recommends configs |
| `backtest-runner` | `backtest_agent()` | Builds configs, runs backtests, summarizes results |
| `trading-monitor` | `monitor_agent()` | Inspects runs, checks positions, flags risk violations |
| `plugin-builder` | `plugin_builder_agent()` | Creates new plugins following framework conventions |

### Usage

```python
import asyncio
from quantbox.agents import research_agent, backtest_agent, monitor_agent

# Research: explore strategies
result = asyncio.run(research_agent(
    "What crypto strategies are available and which has the best risk-adjusted returns?"
))

# Backtest: run simulation
result = asyncio.run(backtest_agent(
    "Backtest crypto trend following on BTC, ETH, SOL for the last year"
))

# Monitor: check portfolio
result = asyncio.run(monitor_agent())
```

### Custom orchestration

Access the raw subagent definitions for custom orchestrators:

```python
from quantbox.agents import QUANTBOX_AGENTS

# Dict of agent name -> {description, prompt, tools}
for name, config in QUANTBOX_AGENTS.items():
    print(f"{name}: {config['description']}")
```

## Skill auto-refresh

Plugin catalog tables in the Claude Code skill (`.claude/skills/quantbox/`) are auto-generated from the live registry. Sections between `<!-- BEGIN AUTO-GENERATED -->` and `<!-- END AUTO-GENERATED -->` markers are regenerated by:

```bash
uv run python scripts/refresh_skill.py          # regenerate
uv run python scripts/refresh_skill.py --check  # CI mode (exit 1 if stale)
```

The `.githooks/pre-commit` hook runs this automatically when commits touch plugin source files.
