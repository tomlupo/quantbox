# Plugin Authoring

Plugins are how QuantBox extends behavior at L3+ (see [api-layers.md](api-layers.md)). They're optional — most casual work happens at L0/L1. Use a plugin when you want validated contracts, registry discoverability, or full-pipeline (L4) integration.

---

## When to write a plugin (vs not)

| Need | Use |
|---|---|
| Quick experiment, throwaway code | L0/L1 — just write a script |
| One-off comparison or notebook | L2 composable functions |
| Want validation + contracts but don't need YAML | L3 plugin instance, called directly |
| Need YAML config, manifest, and EXPERIMENTS.md entry | L3 plugin + L4 runner |
| Production cron, reproducibility-pinned | L3 plugin + L5 CLI + `--strict` |

If you can do the work at L1, do it at L1. Reach for plugins only when the contract or the runner adds value.

---

## Plugin types

| Type | Protocol | Key method | Returns |
|---|---|---|---|
| **Pipeline** | `PipelinePlugin` | `run(mode, asof, params, data, store, broker, risk)` | `RunResult` |
| **Strategy** | `StrategyPlugin` | `run(data, params)` | `dict` with `"weights"` DataFrame (date × symbol) plus optional details |
| **Data** | `DataPlugin` | `load_universe`, `load_market_data`, `load_fx` | `DataFrame` / `dict[str, DataFrame]` |
| **Broker** | `BrokerPlugin` | `get_positions`, `place_orders`, `fetch_fills` | `DataFrame` |
| **Rebalancing** | `RebalancingPlugin` | `generate_orders(weights, broker, params)` | `dict[str, DataFrame]` |
| **Risk** | `RiskPlugin` | `check_targets`, `check_orders` | `list[dict]` (findings) |
| **Publisher** | `PublisherPlugin` | `publish(result, params)` | `None` |
| **Feature** | `FeaturePlugin` | `compute(data, params)` | `DataFrame` |
| **Validation** | `ValidationPlugin` | `validate(returns, weights, benchmark, params)` | `dict` |
| **Monitor** | `MonitorPlugin` | `check(result, history, params)` | `list[dict]` |

**StrategyPlugin contract details:** `data` is a dict of wide-format DataFrames — required key `"prices"` (date index × symbol columns), optional `"volume"`, `"market_cap"`, `"universe"`, `"funding_rates"`. `params` overrides instance attributes. Return dict must contain `"weights"`; convention adds `"simple_weights"` (latest dict), `"details"`, `"exposure"`.

When in doubt about which type, prefer `FeaturePlugin` — it's the catch-all for derived signals/computations and accepts the broadest input.

Don't invent new plugin types unilaterally. New types require an ADR; see [lifecycle.md](lifecycle.md) and [decisions/README.md](../decisions/README.md).

---

## Plugin anatomy

```python
from dataclasses import dataclass
from typing import Any
import pandas as pd
from quantbox.contracts import PluginMeta, StrategyPlugin

@dataclass
class MyStrategy:
    """Brief description (one sentence) — what this strategy does."""

    meta = PluginMeta(
        name="my.strategy.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        status="research",          # see lifecycle.md
        description="Momentum strategy with vol scaling.",
        tags=("momentum", "research"),
        capabilities=("backtest", "paper", "live"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "lookback_days": {"type": "integer", "minimum": 30, "default": 60},
                "vol_target":    {"type": "number",  "minimum": 0.01, "default": 0.15},
            },
            "required": [],
        },
        outputs=("strategy_weights",),
    )

    def run(self, data, params=None):
        params = params or {}
        prices = data["prices"]
        lookback = params.get("lookback_days", 60)
        returns = prices.pct_change(lookback).iloc[-1]
        signal = (returns > 0).astype(float)
        weights = signal / signal.sum() if signal.sum() > 0 else signal
        weights_df = pd.DataFrame({"weight": weights}).T
        weights_df.index = [prices.index[-1]]
        return {
            "weights": weights_df,
            "simple_weights": weights[weights > 0].to_dict(),
        }
```

Key requirements:

- Class is a `@dataclass`.
- `meta` is a **class attribute**, not an instance attribute. The runner reads it before instantiation.
- `params_schema` is a JSON Schema — used by `quantbox validate` and the LLM-facing skill layer.
- Return shape matches `meta.outputs` and the schema in `src/quantbox/artifact_schemas/{output}.schema.json`.

---

## Status field — the single most important meta property

Every plugin declares a `status`:

| Status | Meaning | Where it lives | Promotability |
|---|---|---|---|
| `research` | Local-source or entry-pointed but unstable, no LOCKED methodology spec | `research/{study}/`, `src/{project}/plugins/` | LLM, freely |
| `locked` | Methodology spec frozen, validated, not yet prod-tagged | same file, status flipped + spec frontmatter `status: LOCKED` | human via `/promote-lock` |
| `production` | Has prod tag `prod-{subsystem}-vX.Y.Z-YYYYMMDD` | same file | human via `/promote` |

Optional later: `scratch` (local-source, single-study), `deprecated`, `retired`. See [lifecycle.md](lifecycle.md).

The runner's `--strict` mode rejects any plugin below `locked` for production runs. This is the safety net that catches "we shipped an experiment to prod."

---

## Registration

Two paths.

### Entry-point (preferred for stable plugins)

In `pyproject.toml`:

```toml
[project.entry-points."quantbox.plugins"]
"my.strategy.v1" = "my_pkg.strategy:MyStrategy"
```

Discoverable via `quantbox plugins list`.

### Local-source (for research scratch-plugins)

In a YAML config:

```yaml
plugins:
  strategies:
    - source: "research/regime-momentum/strategy.py:RegimeMomentum"
      weight: 1.0
      params: { ... }
```

The runner imports the file, validates `meta`, registers it for that one run only. No package release needed. Status defaults to `research`. The runner's `--strict` mode refuses `source:` entries entirely.

This is the escape hatch that makes LLM-authored plugins viable without packaging overhead.

---

## Naming conventions

| Element | Convention | Example |
|---|---|---|
| Plugin name | `{namespace}.{kind}.{slug}.v{N}` | `dm_evo.strategy.fund_scoring.v1` |
| Class name | PascalCase | `FundScoringStrategy` |
| File location | `src/{namespace}/plugins/{kind}/{slug}.py` | `src/dm_evo/plugins/strategy/fund_scoring.py` |
| Versioning | Bump the `v{N}` in the name on breaking changes | `dm_evo.strategy.fund_scoring.v2` |

Names are immutable post-promotion. If you change behavior, bump the suffix and keep the old version for reproducibility.

---

## Testing

Tests live next to the plugin: `tests/plugins/strategy/test_fund_scoring.py`.

Minimum tests:

1. **Smoke test** — instantiate, call key method with synthetic data, assert no crash.
2. **Schema test** — feed bogus params, assert `quantbox.validate_params` raises.
3. **Output schema test** — assert returned DataFrame matches `src/quantbox/artifact_schemas/{output}.schema.json`.

Beyond that, test domain logic the same way you'd test any function. Use plain pytest. Don't stand up the runner unless you're explicitly testing the runner.

---

## Versioning rules

| Change | Action |
|---|---|
| Bug fix, no behavior change | Patch bump (`0.1.0` → `0.1.1`); same `v1` name |
| New parameter with default | Minor bump (`0.1.x` → `0.2.0`); same `v1` name |
| Output schema change, polarity flip, breaking semantics | New plugin name (`v1` → `v2`); old version stays for reproducibility |

The rule: **the `v{N}` in the plugin name is the user-facing version.** Internal `meta.version` tracks patch/minor for engineering hygiene.

---

## Anti-patterns

| Anti-pattern | Fix |
|---|---|
| `meta` defined as instance attribute (`self.meta = ...`) | Class attribute. The runner reads it pre-instantiation. |
| Plugin reaches across files to mutate global state | Keep plugins pure; state lives in `data` argument. |
| Plugin imports vectorbt/mlflow directly instead of via adapter | Use the adapter so reuse propagates. |
| Plugin returns a dict-of-dicts where a DataFrame would do | Match the output schema exactly. |
| Renaming a plugin without bumping the `v{N}` suffix | Names are immutable. Bump and keep both. |
| Adding a new plugin type without an ADR | New protocols are architectural decisions. RFC first. |

---

## See also

- [api-layers.md](api-layers.md) — where plugins fit (L3) and how to use them at lower layers.
- [adapters.md](adapters.md) — how plugins compose external libs.
- [lifecycle.md](lifecycle.md) — the status state machine and promotion path.
- [skills.md](skills.md) — how LLMs author plugins via skills.
- [playbooks/add-a-plugin.md](../playbooks/add-a-plugin.md) — step-by-step.
