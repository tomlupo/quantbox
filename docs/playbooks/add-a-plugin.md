# Playbook — Add a Plugin

Use this when an adapter isn't enough — you need validated contracts, registry discoverability, or full-pipeline integration. Read [architecture/plugin-authoring.md](../architecture/plugin-authoring.md) first.

---

## Pre-flight

Decide what type:

| Need | Type |
|---|---|
| Compute target weights | `StrategyPlugin` |
| Load data from a new source | `DataPlugin` |
| Compute a derived signal/feature | `FeaturePlugin` (catch-all for novel ideas) |
| Send order to a new broker | `BrokerPlugin` |
| Validate weights/orders | `RiskPlugin` |
| Convert weights → orders | `RebalancingPlugin` |
| Send results to a new channel | `PublisherPlugin` |
| Orchestrate a workflow | `PipelinePlugin` |

When unsure, **prefer `FeaturePlugin`** — broadest input/output, easiest to refactor later.

If none fit, do **not** invent a new plugin type. New protocols require an ADR. See [decisions/](../decisions/).

---

## Steps

### 1. Decide where it lives

| Where | When |
|---|---|
| `research/{study}/strat.py` (scratch-plugin) | Single-study experiment, may be thrown away |
| `src/{project}/plugins/{kind}/{slug}.py` (project entry-pointed) | Project-specific, reusable within the project |
| `src/quantbox/plugins/{kind}/{slug}.py` | Cross-project utility, has earned upstream status |

Default: start in `research/` or in the project. Promote upstream only after the second project would benefit.

### 2. Author the file

```python
from dataclasses import dataclass
from typing import Any
import pandas as pd
from quantbox.contracts import PluginMeta, StrategyPlugin


@dataclass
class MyStrategy:
    """One-sentence description."""

    meta = PluginMeta(
        name="{namespace}.strategy.{slug}.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        status="research",  # see lifecycle.md
        description="...",
        tags=("...",),
        capabilities=("backtest", "paper", "live"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "lookback_days": {"type": "integer", "minimum": 30, "default": 60},
            },
            "required": [],
        },
        outputs=("strategy_weights",),
    )

    def run(self, data, params=None):
        params = params or {}
        prices = data["prices"]
        # implementation — produce a date × symbol DataFrame of weights
        return {"weights": weights_df, "simple_weights": latest_dict}
```

Key requirements:

- `@dataclass` decorator on the class.
- `meta` is a **class attribute**, not instance attribute.
- `params_schema` is JSON Schema — not just a dict — and it is **mandatory**: every
  key a config may set (constructor args via `params_init`, run-time `params`) is a
  property with a `description`. Constructor params get their `type` and `default`
  from the signature (`quantbox.params_schema.resolve_params_schema`); declare only
  what it cannot carry. `tests/test_params_schema.py` fails for a registered plugin
  without one, and `quantbox validate` refuses a key that is not a property.
- `meta.status="research"` for new code (you can't promote yourself).

### 3. Register

**Scratch-plugin** (research/local-source):

```yaml
# in your config YAML
plugins:
  strategies:
    - source: "research/{study}/strat.py:MyStrategy"
      params: { ... }
```

No package/install needed. Runner imports the file at runtime.

**Project entry-point** (stable plugin):

In project `pyproject.toml`:

```toml
[project.entry-points."quantbox.plugins"]
"{namespace}.strategy.{slug}.v1" = "{project_pkg}.plugins.strategy.{slug}:MyStrategy"
```

Then `uv sync` and verify:

```bash
quantbox plugins list | grep {slug}
```

**Upstream builtin** — three places, all required. The three names differ per kind
(directory `datasources`, `broker`, `strategies`…; manifest section `data`, `brokers`,
`strategies`…), so copy where a sibling plugin of the same kind sits rather than
deriving one name from another:

1. Export the class from the kind's directory, `src/quantbox/plugins/<dir>/__init__.py`.
2. Import it in `src/quantbox/plugins/builtins.py` and add it to that kind's map there.
3. List its `meta.name` under the kind's section in `src/quantbox/plugins/manifest.yaml`
   (`plugins.builtins.<section>`, the manifest's own key, e.g. `data` or `brokers`).

Then `uv run quantbox plugins doctor` and `uv run pytest -m pipeline_smoke` — a builtin
missing from any of the three is the silent break the smoke suite exists to catch.

### 4. Test

`tests/plugins/test_{slug}.py`:

```python
def test_smoke():
    """Plugin instantiates and runs without crashing."""
    strat = MyStrategy()
    data = {"prices": _synthetic_prices()}  # wide-format DataFrame
    result = strat.run(data, params={})
    assert "weights" in result
    assert isinstance(result["weights"], pd.DataFrame)


def test_params_schema_rejects_bad_input():
    """Schema validation catches bogus params."""
    from quantbox.params_schema import check_params, resolve_params_schema

    unknown, violations = check_params(resolve_params_schema(MyStrategy), {"lookback_days": -1})
    assert violations  # non-empty = the schema caught it


def test_output_shape():
    """Weights are wide: dates down the index, one column per symbol."""
    strat = MyStrategy()
    result = strat.run({"prices": _synthetic_prices()}, params={})
    assert isinstance(result["weights"].index, pd.DatetimeIndex)
```

### 5. Call it from Python before wiring YAML

```python
# python REPL or test
from {namespace}.plugins.strategy.{slug} import MyStrategy

strat = MyStrategy()
result = strat.run(data={"prices": prices_wide}, params={"lookback_days": 60})
print(result["weights"].tail())
print(result["simple_weights"])
```

If this works, the plugin is sound. Wiring it into a YAML config through the runner is a separate step.

### 6. Add a YAML config example

`cookbook/configs/{slug}_example.yaml`:

```yaml
run:
  mode: backtest
  asof: "2026-04-22"
  pipeline: "trade.full_pipeline.v1"

plugins:
  strategies:
    - name: "{namespace}.strategy.{slug}.v1"
      params: { lookback_days: 60 }
  data:
    name: "{your_data_plugin}"
  broker:
    name: "sim.paper.v1"
```

Test:

```bash
quantbox validate -c cookbook/configs/{slug}_example.yaml
quantbox run -c cookbook/configs/{slug}_example.yaml
```

### 7. Document

- Update the plugin table in the project `README.md`.
- If this plugin has a methodology, copy `quantbox/templates/methodology.md` into your project's `docs/methodology/{slug}.md` (DRAFT status).
- EXPERIMENTS.md entry referencing the plugin name and any backtest results.

---

## Validation checklist

- [ ] `meta` is a class attribute (not instance).
- [ ] `params_schema` is valid JSON Schema, describes every param, and `quantbox plugins schema --name <id> --json` shows it.
- [ ] `meta.status` is `research` (not auto-locked).
- [ ] Smoke test, schema test, output test all pass.
- [ ] Plugin appears in `quantbox plugins list`.
- [ ] Example YAML config validates and runs.
- [ ] EXPERIMENTS.md entry exists if this is research work.

---

## Common mistakes

| Mistake | Fix |
|---|---|
| `self.meta = ...` | Class attribute, not instance — runner reads it pre-instantiation |
| Renaming a registered plugin in place | Names are immutable post-registration; bump `v1` → `v2` |
| Auto-setting `meta.status="locked"` for your own plugin | Status flips are human-driven through `/promote-lock` |
| Using `import vectorbt as vbt` directly inside the plugin | Use `from quantbox.adapters.vectorbt import vbt` so reuse propagates |
| Output schema mismatch with declared `meta.outputs` | Match exactly; runtime validation will reject otherwise |
| Skipping the smoke test "because it's just research" | Smoke tests catch 80% of integration bugs; no excuse |

---

## When to bump version vs new plugin

| Change | Action |
|---|---|
| Bug fix | Patch (`0.1.0` → `0.1.1`); same name |
| New optional param | Minor (`0.1.x` → `0.2.0`); same name |
| Schema change, polarity flip, breaking semantics | New name with `v2` suffix; old `v1` stays |

---

## See also

- [architecture/plugin-authoring.md](../architecture/plugin-authoring.md) — full reference.
- [architecture/lifecycle.md](../architecture/lifecycle.md) — `meta.status` and promotion.
- [promote-a-methodology.md](promote-a-methodology.md) — when this plugin earns LOCKED.
- [add-a-skill.md](add-a-skill.md) — exposing the plugin to LLMs.
