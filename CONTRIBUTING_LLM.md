# Contributing with LLMs (safe editing protocol)

Rules to keep QuantBox stable while allowing agentic development.

## Philosophy

QuantBox is designed as an **LLM-friendly framework** for AI-powered quantitative research and trading. Code should be:
- **Self-documenting**: Rich docstrings with examples
- **Type-safe**: Full type hints for IDE/LLM inference
- **Inspectable**: Methods like `describe()` for state snapshots
- **Testable**: Paper trading mode for safe experimentation

## Do
- Keep `quantbox` core **small** and avoid strategy logic in core.
- Write **LLM-friendly docstrings** with:
  - Clear description of what the method does
  - Type-annotated Args and Returns sections
  - Code examples showing typical usage
  - "LLM Note:" hints for AI agents
- When changing any artifact schema:
  - bump `schema_version` in the producing plugin meta
  - add/modify JSON schema under `/schemas`
  - add a contract test
- Add tests for any new plugin or core behavior.
- Prefer **additive** changes and new plugin versions over breaking changes.
- Use `describe()` methods to expose structured state snapshots.

## Don't
- Do not rename existing entry-point ids (e.g. `fund_selection.simple.v1`) unless you create a new version id.
- Do not embed secrets in YAML configs or artifacts. Use env vars / secret refs.
- Do not make pipelines depend on `print()` parsing. Use structured events.
- Do not use complex inheritance hierarchies - prefer composition.

## Plugin Types

### BrokerPlugin
Handles order execution and portfolio management.
```python
# Key methods for LLM agents:
broker.describe()                    # Get full state snapshot (JSON-friendly)
broker.get_positions()               # Current holdings DataFrame
broker.get_portfolio_value()         # Total value as float
broker.generate_rebalancing(weights) # Preview trades (no execution)
broker.execute_rebalancing(weights)  # Execute trades
```

### DataPlugin
Loads market data (prices, universe, FX). Returns **wide-format** DataFrames
(date index, symbol columns).
```python
data.load_universe(params)                      # Get tradeable assets → DataFrame[symbol]
market = data.load_market_data(universe, asof, params)
# Returns: {"prices": wide_df, "volume": wide_df, "market_cap": wide_df, ...}
# Only "prices" is required; others may be empty DataFrames
data.load_fx(asof, params)                      # FX rates (or None for crypto)
```

### PipelinePlugin
Orchestrates a full run: loads data, runs strategies, rebalances, executes.
```python
pipeline.run(mode, asof, params, data, store, broker, risk)
# Returns RunResult with artifacts and metrics
```

### StrategyPlugin
Computes target weights from market data.
```python
strategy.compute_weights(market_data, universe, asof, params)
# Returns: DataFrame with symbol columns and weight values
```

### RebalancingPlugin
Converts target weights into orders respecting leverage and position limits.
```python
rebalancer.rebalance(targets, current_positions, params)
# Returns: List[Dict] of orders
```

### RiskPlugin
Pre-trade validation of targets and orders.
```python
risk.check_targets(targets, params)   # Validate weight targets
risk.check_orders(orders, params)     # Validate generated orders
# Both return: List[Dict] of violations (empty = all clear)
```

### PublisherPlugin
Post-trade notifications and reporting.
```python
publisher.publish(result, params)
# Sends notifications (e.g. Telegram) after a run completes
```

## LLM Workflow Example

### Using QuantBoxAgent (recommended for agents)

```python
from quantbox.agents import QuantBoxAgent

agent = QuantBoxAgent()

# 1. Explore plugins
plugins = agent.list_plugins(kind="strategy")
info = agent.plugin_info("strategy.crypto_trend.v1")

# 2. Build and validate config
config = agent.build_config(
    mode="backtest",
    pipeline="backtest.pipeline.v1",
    strategy="strategy.crypto_trend.v1",
    data="binance.live_data.v1",
)
result = agent.validate_config(config)

# 3. Execute and inspect
run_result = agent.run(config)
artifacts = agent.inspect_run(run_result["run_id"])
```

### Using broker directly

```python
# 1. Initialize broker in paper mode
from quantbox.plugins.broker import BinanceLiveBroker
broker = BinanceLiveBroker(paper_trading=True)

# 2. Check current state
state = broker.describe()
print(f"Portfolio: {state['portfolio_value']} {state['stable_coin']}")

# 3. Define target allocation (research output)
target_weights = {'BTC': 0.40, 'ETH': 0.30, 'SOL': 0.15}

# 4. Preview what trades would happen
analysis = broker.generate_rebalancing(target_weights)
print(analysis[['Asset', 'Action', 'Delta_Qty']])

# 5. Execute (safe in paper mode)
result = broker.execute_rebalancing(target_weights)
print(f"Executed {result['summary']['total_executed']} trades")
```

## Agent infrastructure

When building agent integrations or modifying LLM-facing code:

- **Introspection**: Use `quantbox.introspect.describe_plugin_class()` for universal plugin descriptions. No per-plugin changes needed.
- **Agent API**: `quantbox.agents.QuantBoxAgent` returns plain dicts — errors in return values, never raised.
- **MCP server**: `quantbox.agents.mcp_server` exposes 9 tools. Test with `uv run quantbox-mcp`.
- **Claude Agent SDK**: Pre-built subagent definitions in `quantbox.agents.claude_agents.QUANTBOX_AGENTS`.
- **Skill docs**: Auto-generated sections in `.claude/skills/quantbox/` use `<!-- BEGIN AUTO-GENERATED -->` markers. Run `uv run python scripts/refresh_skill.py` after changing plugins.

## Claude Code skill

The `.claude/skills/quantbox/SKILL.md` provides decision trees for intent-based routing.
Domain-specific references live in `.claude/skills/quantbox/references/<domain>/`:
- `strategies/` — strategy selection, API, patterns, gotchas
- `pipelines/` — pipeline selection, API, gotchas
- `data/` — data source selection, API, gotchas
- `brokers/` — broker selection, API, gotchas
- `configs/` — config schema, patterns, gotchas
- `risk/` — risk plugin API

## Required checks after edits
1. `quantbox plugins list`
2. `quantbox validate -c <config>`
3. `quantbox run -c <config>`
4. `pytest -q` (if tests are installed)
