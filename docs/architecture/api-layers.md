# API Layers (L0–L5)

QuantBox exposes every capability at multiple layers so users (humans, scripts, LLMs) can pick the lightest one that solves their task. This is the operationalization of the [lowest-viable-abstraction principle](principles.md#2-lowest-viable-abstraction).

---

## The table

| Layer | API shape | When to use | Example |
|---|---|---|---|
| **L0** Re-exports | `from quantbox.adapters.vectorbt import vbt` | Quick experiment, throwaway script. Pure pass-through to the underlying lib. | `vbt.Portfolio.from_signals(prices, signals)` |
| **L1** Convenience helpers | `quantbox.bt.run(...)`, `quantbox.opt.max_sharpe(...)`, `quantbox.score.peer_z(...)` | Common idiom — one function call. No plugin/config layer. | `qbt.run(prices, signals, fees=0.001)` |
| **L2** Composable units (planned) | `quantbox.functions.run_strategy(strategy, data)` | Building a notebook, composing two ideas, no run_id ceremony. | `result = run_strategy(my_strat, data)` |
| **L3** Plugin instances | Instantiate `Strategy()`, `DataPlugin()`, call directly | You want validation and contracts but not the YAML/runner. | `MyStrat().run(data, params)` |
| **L4** Full pipeline | `quantbox.run_from_config(yaml_path)` | Logged experiment, ArtifactStore manifest, EXPERIMENTS.md entry. | `run_from_config("config/research.yaml")` |
| **L5** CLI | `quantbox run -c config.yaml` | Production cron, reproducibility-pinned, lifecycle-tracked. | scheduled job in agent-cron / systemd |

---

## Default layer per task type

| Task | Default layer | Rationale |
|---|---|---|
| "Try this idea" | L1 | Function call beats YAML for one-off work. |
| "Compare A vs B" | L2 | Composable units exist for this; runner overkill. |
| "Show me the chart" | L0 + L1 | Use vbt's plotting directly. |
| "Backtest with my dm-evo data" | L1 with project-specific helper, or L3 plugin | Depends on whether the data plugin is registered. |
| "Author a new strategy" | L3 (plugin instance) → L4 once registered | Build it without the runner first; promote later. |
| "Log this for EXPERIMENTS.md" | L4 | The runner produces the manifest you reference. |
| "Production methodology run" | L5 | CLI + `--strict` enforces reproducibility pins. |

When a skill is unsure, **start at L1.** Escalate only when the task demands it.

---

## Module map

| Module | Purpose | Layer |
|---|---|---|
| `quantbox.adapters.{lib}` | Re-exports + thin helpers (`vbt`, ...) — added when ≥2 consumers need same bridge | L0 |
| `quantbox.bt` | Convenience for backtesting (most common idiom) | L1 |
| `quantbox.opt` | Convenience for portfolio optimization | L1 |
| `quantbox.score` | Convenience for ranking/scoring | L1 |
| `quantbox.functions` (planned) | `run_strategy`, `validate_artifact` — composable units | L2 |
| `quantbox.contracts` | `Protocol`s, `PluginMeta`, `RunResult` | L3 |
| `quantbox.runner` | `run_from_config` | L4 |
| `quantbox.cli` | Typer-based CLI | L5 |

The L0–L2 surface is part of the public API stability contract — same as the plugin contracts. Don't break them lightly.

---

## L0 — Re-exports

The rule: an adapter re-exports the underlying library so users can drop down without import gymnastics.

```python
# adapters/vectorbt.py
import vectorbt as vbt
__all__ = ["vbt"]

# Optional: small convenience helpers, but vbt itself is the export.
def from_dm_evo(df): ...
```

Users:

```python
from quantbox.adapters.vectorbt import vbt
pf = vbt.Portfolio.from_signals(prices, entries, exits)
```

If a user has to write `import vectorbt as vbt` to bypass quantbox, the adapter has failed.

---

## L1 — Convenience helpers

The rule: one function call covers the most common idiom for that capability. No plugin, no config, no manifest.

```python
# bt.py
import pandas as pd
from .adapters.vectorbt import vbt

def run(prices: pd.DataFrame, signals: pd.DataFrame, *, fees: float = 0.001,
        slippage: float = 0.0005, freq: str = "1D") -> "Result":
    pf = vbt.Portfolio.from_signals(
        close=prices, entries=signals > 0, exits=signals <= 0,
        fees=fees, slippage=slippage, freq=freq,
    )
    return Result(portfolio=pf, metrics=pf.stats())
```

Users:

```python
import quantbox.bt as qbt
result = qbt.run(prices, signals, fees=0.001)
print(result.metrics)
```

L1 helpers should:
- Take primitive types (`DataFrame`, `Series`, `dict`) — not quantbox-specific objects.
- Have sensible defaults that match the most common use case.
- Return a result object that carries both the underlying-library object (`pf`) and a normalized view (metrics dict).

---

## L2 — Composable units

The rule: function-style API for users who want validation and contracts but want to compose things by hand.

```python
from quantbox.functions import run_strategy
weights = run_strategy(MyStrat(params=...), data=market_data, asof="2026-04-22")
```

L2 functions accept plugin instances and return validated artifacts. They don't write to ArtifactStore; that's L4's job.

---

## L3 — Plugin instances

Instantiate a plugin and call its methods directly. Useful when authoring a new plugin (test it before registering it) or when you want the contract but not the runner.

```python
strat = MyStrategy(target_vol=0.15)                 # dataclass attrs at construction
result = strat.run(data, params={"lookback_days": 60})  # params override at call
weights = result["weights"]                         # date × symbol DataFrame
```

`data` is a dict with required `"prices"` and optional `"volume"`, `"market_cap"`, `"universe"`, `"funding_rates"` (all wide-format DataFrames). `params` overrides instance attributes for that one call.

This layer is what scratch-plugins use during research (see [lifecycle.md](lifecycle.md)).

---

## L4 — Full pipeline

The runner. YAML config. Validated. Produces a `RunResult` with manifest, lineage, content-hashed datasets.

```python
from quantbox import run_from_config
result = run_from_config("config/research/regime_taa.yaml")
```

This is what the EXPERIMENTS.md log references. Use this when you want the run to be *part of the record*.

---

## L5 — CLI

Production. Combined with `--strict`, enforces reproducibility pins (uv.lock + dataset hashes + seeds).

```bash
quantbox run -c config/prod/dm_evo_fund_selection.yaml --strict
```

Used by cron jobs and agent-cron schedules. Not for interactive use.

---

## Skill frontmatter contract

Every capability skill declares its `default_layer` and `escalation_rules` in frontmatter:

```yaml
---
name: quantbox-backtest
description: ...
default_layer: L1
escalation_rules:
  - to: L4
    when: "task requires logged experiment / EXPERIMENTS.md entry"
  - to: L5
    when: "production run / reproducibility pinning required"
---
```

This makes layer choice auditable and consistent. Skills without this frontmatter are considered incomplete. See [skills.md](skills.md).

---

## When to add a new convenience helper

Add an L1 helper when:

- The same 3–10 line idiom appears in ≥2 places (your projects, examples, or skills).
- The helper hides no functionality — users can still drop to L0 if they need a knob the helper doesn't expose.
- The function has obvious sensible defaults.

Don't add an L1 helper:

- For a one-off project-specific need (put it in the project).
- That hides the underlying library's API in a way users would have to fight.
- That requires importing more than one external library — that's an L4 pipeline, not a helper.

---

## Versioning

Layers L0–L2 follow library semver: breaking changes only on majors.
Layers L3–L4 follow plugin semver: covered by `meta.version` on each plugin.
Layer L5 (CLI) follows the package semver.

Inner layers should be more stable than outer ones. Breaking L1 forces every L4 caller to revalidate. Breaking L4 only affects YAML configs.
