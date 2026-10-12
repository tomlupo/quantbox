# API surface: audiences and entry styles

The decision is [ADR-0010](../adr/0010-audience-layers-one-distribution.md). It
supersedes the L0–L5 ladder of ADR-0002. This page is the operational rule:
what each layer is for, and how a caller reaches it.

---

## Four layers, three audiences

quantbox is one distribution. Its modules sit in four layers, in import order:
`core` < `plugins` < `research`, `trade`. A layer imports only the layers below
it. `research` and `trade` never import each other.

| Layer | Audience | Holds | Install |
|---|---|---|---|
| **core** | every caller, including a client install (robo) | the contracts, the registry, the runner and the CLI, the engine seam with rsims, metrics, inference and gates, datasets and the store | base |
| **plugins** | every caller that uses the builtin strategies and data | builtin strategies, datasources, features, overlays, monitors (ADR-0010 decision 6) | base, `[data]` for the data clients |
| **research** | the lab | backtest pipelines, sweeps, validation plugins, reports, the warehouse, the vectorbt adapter | base for an rsims backtest; `[research]` for reports and the warehouse; `[vectorbt]` |
| **trade** | live | brokers, the trading pipeline, rebalancing, reconciliation, risk, publishers | `[trade]` |

**The module map has one owner: the import-linter contracts in
`pyproject.toml` `[tool.importlinter]`.** Each layer is the `source_modules`
of the contract named after it. `uv run lint-imports` checks every import
against them (ci.yml job `lint`). `tests/test_layer_map.py` fails when a
module is in no layer or in two. To place a new module, add it to its layer's
contract and to `forbidden_modules` of each contract below it. Decide the
layer by who imports the module.

`pyproject.toml` also owns what each install carries.
`scripts/check_no_vectorbt.sh` proves the base install is the client install.

---

## Two entry styles

Every audience reaches the same plugins in one of two ways. Neither is "lower".

### Python API on the contracts

Instantiate a plugin and call it, or call a core module. No YAML, no run
directory. Use it to try an idea, to test a plugin before it is registered,
or to compose steps in a notebook.

```python
strat = MyStrategy(target_vol=0.15)  # dataclass attrs at construction
result = strat.run(data, params={"lookback_days": 60})  # params override at call
weights = result["weights"]  # date × symbol DataFrame
```

`data` is a dict with required `"prices"` and optional `"volume"`,
`"market_cap"`, `"universe"`, `"funding_rates"` (all wide-format DataFrames).
The book function is `quantbox.engine.simulate` ([ADR-0008](../adr/0008-engine-seam.md)).

Research also has the vectorbt re-export and one helper:

```python
from quantbox.adapters.vectorbt import vbt
import quantbox.bt as qbt

# vbt fills the bar it is handed: lag the signal yourself (ADR-0005).
pf = vbt.Portfolio.from_signals(prices, entries.shift(1, fill_value=False))
result = qbt.run(prices, signals, fees=0.001)  # next-bar always; lag_bars=0 raises
```

If a caller has to write `import vectorbt as vbt` to get past quantbox, the
adapter has failed ([adapters.md](adapters.md)).

### YAML through the runner

The runner validates the config, runs the pipeline and writes the run
manifest and the lineage. Use it when the run must be part of the record.

```python
from quantbox.runner import run_from_config

result = run_from_config("cookbook/configs/run_backtest_crypto_trend.yaml")
```

```bash
quantbox run -c cookbook/configs/run_backtest_crypto_trend.yaml  # the same, from the CLI
```

Strict mode is `run.strict: true` in the config (or a promotion run); the
runner reads it (`quantbox.runner.strict_refusal`). It is a property of the
run, not a layer. `uv run quantbox --help` is the authority for the commands.

---

## Skill frontmatter labels (L0–L5)

Skills still declare `default_layer` with the old labels. The labels now name
an entry style, not a module or a package:

| Label | Means |
|---|---|
| L0, L1 | Python API: the vectorbt re-export or `quantbox.bt` (research) |
| L3 | Python API: a plugin instance, called directly |
| L4 | YAML through `run_from_config` |
| L5 | YAML through the CLI, with `run.strict: true` for production |

L2 was never built. The field and its rules are owned by [skills.md](skills.md).

---

## Stability and moved names

A public module declares `__all__`. Since 0.13.0 a moved name has no import
shim. Its old import path raises `ImportError` and names the new path, from
one list, `quantbox._removed.REMOVED` (ADR-0010 decision 7, as amended by
TOM-1457). YAML plugin ids keep their aliases: a config written today keeps
running.
