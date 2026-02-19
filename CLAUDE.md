# CLAUDE.md — Agent onboarding guide

## What is QuantBox?

Config-driven quant framework. Strategies, data, brokers, and risk are all plugins wired together via YAML configs. Same strategy params work across backtesting, paper trading, and live execution.

## Project layout

```
packages/quantbox-core/src/quantbox/   ← core library
  contracts.py       Protocol definitions (start here)
  runner.py          Config → plugin instantiation → pipeline.run()
  registry.py        Plugin discovery (builtins + entry points)
  introspect.py      Universal plugin introspection for LLM agents
  cli.py             CLI entry point (quantbox command)
  store.py           Artifact storage (Parquet + JSON)
  schemas.py         Runtime schema validation
  agents/            LLM agent integration layer
    tools.py         QuantBoxAgent programmatic API
    mcp_server.py    MCP server (9 tools for Claude Code / Cursor)
    claude_agents.py Claude Agent SDK subagent definitions
  plugins/
    builtins.py      Plugin registration map
    strategies/      Strategy plugins (compute target weights)
    pipeline/        Pipeline plugins (orchestrate full runs)
    datasources/     Data plugins (OHLCV, market cap, funding rates)
    broker/          Broker plugins (paper + live execution)
    rebalancing/     Rebalancing plugins (weights → orders)
    risk/            Risk plugins (pre-trade validation)
    publisher/       Publisher plugins (notifications)
    backtesting/     Backtest engines (vectorbt, rsims)
configs/             Example YAML configs for all pipeline types
schemas/             JSON schemas for artifact validation
plugins/manifest.yaml  Plugin profiles (research, trading, futures_paper)
scripts/
  refresh_skill.py   Auto-regenerate skill docs from live registry
.claude/skills/quantbox/  Claude Code agent skill (decision trees + references)
.githooks/pre-commit      Auto-refresh skill docs on plugin changes
```

## Key commands

```bash
uv run quantbox plugins list               # list all plugins
uv run quantbox plugins list --json         # JSON output
uv run quantbox plugins info --name <id>    # plugin details
uv run quantbox validate -c <config>        # validate config
uv run quantbox run -c <config>             # run pipeline
uv run quantbox run --dry-run -c <config>   # dry run
uv run pytest -q                            # run tests
uv run quantbox-mcp                        # start MCP server
```

## Plugin architecture

- All plugins implement Protocols defined in `contracts.py`
- Plugins are `@dataclass` classes with a class-level `meta = PluginMeta(...)` attribute
- Registration: `plugins/builtins.py` builds `{meta.name: class}` dict
- Discovery: `registry.py:PluginRegistry.discover()` merges builtins + entry points
- Runner: `runner.py:run_from_config()` instantiates via `params_init`, calls `pipeline.run()`

## Plugin types and key methods

| Type | Protocol | Key method |
|---|---|---|
| Pipeline | `PipelinePlugin` | `run(mode, asof, params, data, store, broker, risk)` |
| Strategy | `StrategyPlugin` | `compute_weights(market_data, universe, asof, params)` |
| Data | `DataPlugin` | `load_market_data(universe, asof, params) → Dict[str, DataFrame]` |
| Broker | `BrokerPlugin` | `execute_rebalancing(weights)`, `describe()` |
| Rebalancing | `RebalancingPlugin` | `rebalance(targets, positions, params)` |
| Risk | `RiskPlugin` | `check_targets(targets, params)`, `check_orders(orders, params)` |
| Publisher | `PublisherPlugin` | `publish(result, params)` |

## Data format

DataPlugin returns `Dict[str, pd.DataFrame]` of **wide-format** DataFrames:
- Index: date
- Columns: symbol names
- Keys: `"prices"` (required), `"volume"`, `"market_cap"`, `"funding_rates"` (optional)

## Config structure

```yaml
run:
  mode: backtest|paper|live
  asof: "2026-02-06"
  pipeline: "pipeline.name.v1"

plugins:
  pipeline:
    name: "trade.full_pipeline.v1"
    params: { ... }
  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 1.0
      params: { ... }
  data:
    name: "binance.live_data.v1"
    params_init: { ... }
  broker:
    name: "hyperliquid.perps.v1"
    params_init: { ... }
  rebalancing:
    name: "rebalancing.futures.v1"
    params: { ... }
  risk:
    - name: "risk.trading_basic.v1"
      params: { ... }
```

## Environment variables

| Variable | Required for | Default |
|---|---|---|
| `API_KEY_BINANCE` | Binance spot/futures brokers | — |
| `API_SECRET_BINANCE` | Binance spot/futures brokers | — |
| `HYPERLIQUID_WALLET` | Hyperliquid perps broker | — |
| `HYPERLIQUID_PRIVATE_KEY` | Hyperliquid perps broker | — |
| `TELEGRAM_TOKEN` | Telegram publisher | — |
| `TELEGRAM_CHAT_ID` | Telegram publisher | — |
| `QUANTBOX_MANIFEST` | Custom manifest path | `plugins/manifest.yaml` |

None are needed for backtesting or paper trading with simulated brokers.
Copy `.env.example` to `.env` and fill in only what you need.

## Error handling

Quantbox uses custom exceptions (see `quantbox.exceptions`):

| Exception | When | Recovery |
|---|---|---|
| `ConfigValidationError` | YAML config fails validation | Check `.findings` list for details |
| `PluginNotFoundError` | Plugin name not in registry | Check `.available` for valid names |
| `PluginLoadError` | Entry point import failed | Check dependencies (`uv sync --extra full`) |
| `DataLoadError` | Data plugin can't fetch data | Check API keys, network, date range |
| `BrokerExecutionError` | Order placement failed | Check broker credentials and balances |

## Agent infrastructure

QuantBox has a layered agent integration system:

### 1. Programmatic API (`quantbox.agents.QuantBoxAgent`)
```python
from quantbox.agents import QuantBoxAgent
agent = QuantBoxAgent()
agent.list_plugins()                          # browse registry
agent.plugin_info("strategy.crypto_trend.v1") # inspect a plugin
agent.build_config(mode="backtest", ...)      # construct config
agent.validate_config(config)                 # validate
agent.run(config)                             # execute pipeline
```

### 2. MCP Server (Claude Code, Cursor, etc.)
Configured in `.claude/settings.json`. Exposes 9 tools:
`quantbox_list_plugins`, `quantbox_plugin_info`, `quantbox_search_plugins`,
`quantbox_build_config`, `quantbox_validate_config`, `quantbox_run`,
`quantbox_dry_run`, `quantbox_inspect_run`, `quantbox_list_profiles`.

### 3. Claude Agent SDK subagents
Pre-built agents in `quantbox.agents.claude_agents`:
- `research_agent()` — explore strategies, analyze data, recommend configs
- `backtest_agent()` — build, validate, run backtests, summarize results
- `monitor_agent()` — inspect runs, check positions, flag risk violations
- `plugin_builder_agent()` — create new plugins following framework conventions

### 4. Universal introspection (`quantbox.introspect`)
`describe_plugin()` and `describe_plugin_class()` work with any plugin
without requiring per-plugin changes.

## Claude Code skill

The `.claude/skills/quantbox/` directory provides a structured agent skill with:
- Decision trees for intent-based routing
- Per-domain reference directories (strategies, pipelines, data, brokers, configs, risk)
- Progressive disclosure rules (load only what's needed)
- Auto-generated plugin catalogs (refreshed by `scripts/refresh_skill.py`)

The `.githooks/pre-commit` hook auto-refreshes skill docs when plugin source changes.

## Development rules

- Use `uv` as package manager, `uv run` to execute
- Don't use `requests` in core — use `urllib.request` or `httpx`
- `meta` is a class attribute, not an instance attribute
- Prefer additive changes and new plugin versions over breaking changes
- Don't rename existing entry-point IDs
- Add tests for new plugins or core behavior
- See `CONTRIBUTING_LLM.md` for full guidelines
