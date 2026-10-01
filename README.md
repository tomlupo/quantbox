# QuantBox

Quant research and trading framework with a plugin architecture. Config-driven pipelines for backtesting, paper trading, and live execution.

**Not building:** HFT or real-time tick infrastructure, custom exchange gateways, or black-box LLM trading. Math and signals stay deterministic; LLM is for analysis and tool use only.

**Who uses it:**
- **Researcher** — runs research pipelines, adjusts plugins, inspects artifacts
- **Automation** — scheduled jobs producing allocations → orders → fills
- **AI assistant** — calls `validate`, `--dry-run`, `plugins list --json`; no direct trading authority

## Install

```bash
uv venv && source .venv/bin/activate
uv sync

# Optional extras:
uv sync --extra vectorbt  # vectorbt backtest engine (quantbox.bt, engine: vectorbt, sweeps)
uv sync --extra ccxt      # Binance, Hyperliquid (via ccxt)
uv sync --extra ibkr      # Interactive Brokers
uv sync --extra binance   # python-binance
uv sync --extra full      # all of the above
```

## Quick start

### Research (fund selection)

```bash
quantbox run -c cookbook/configs/run_fund_selection.yaml
```

### Backtest

```bash
quantbox run -c cookbook/configs/run_backtest_crypto_trend.yaml
```

See [backtesting guide](docs/playbooks/backtesting.md) for engine options and parameters.

### Paper trading

```bash
quantbox run -c cookbook/configs/run_futures_paper_crypto_trend.yaml
```

## Plugins

All plugins are config-driven and discovered automatically. List registered plugins:

```bash
quantbox plugins list
```

The catalogue is `src/quantbox/plugins/manifest.yaml` (`plugins.builtins`), kept
equal to the registry by `tests/pipeline/test_pipeline_smoke.py`; `quantbox plugins
info --name <id>` describes one plugin. No list is copied here — the last one
drifted.

## Plugin manifest and profiles

The bundled manifest defines available profiles:

```yaml
plugins:
  profile: research       # or: trading, trading_full, futures_paper, stress_test
```

Profiles bundle a set of plugins so you don't repeat them in every config.
Override the manifest via `QUANTBOX_MANIFEST=path/to/manifest.yaml`.

## CLI reference

```bash
quantbox plugins list              # list all registered plugins
quantbox plugins list --json       # JSON output
quantbox plugins info --name <id>  # plugin details
quantbox plugins schema --json     # every plugin: id, status, params JSON Schema
quantbox plugins doctor            # health check: schemas, entry points, config refs
quantbox validate -c <config>      # validate config without running
quantbox run -c <config>           # run a pipeline
quantbox run --dry-run -c <config> # dry run (no side effects)
quantbox run --json -c <config>    # print only the run manifest (quantbox/run@1); logs on stderr
quantbox report export <run-or-arms-dir> --format qute-research/finding-report@1 [-o out.json]
                                   # the data the qute-research finding page renders
quantbox approve --run-dir <path>  # write approval file for a run's orders
quantbox warehouse tables          # list warehouse tables
quantbox warehouse query -q <sql>  # run SQL against the artifact warehouse
```

## Artifacts

Each run writes to `artifacts/<run_id>/`:
- `run_manifest.json` — the run manifest, schema `quantbox/run@1`: run id, git sha, config sha256,
  engine, dataset, funding, execution timing, venue, `n_trials`, metrics, and `files` — the
  run-dir-relative paths of `returns`, `traded_weights` and `metrics`. Its contract, including the
  versioning rule (adding a field is minor, renaming or re-meaning one is major), is the schema
  [`run_manifest.schema.json`](src/quantbox/artifact_schemas/run_manifest.schema.json); check a
  manifest with `quantbox.run_manifest.validate_run_manifest`. A golden run directory for reader
  tests is [`tests/fixtures/golden_run/`](tests/fixtures/golden_run/).
- `finding_report.json` — every backtest's slim default report: returns as equity/drawdown,
  the engine's metrics, a per-arm calendar-year robustness grid and the manifest's provenance,
  as `qute-research/finding-report@1` data for the qute-research `/finding-report` page (which
  owns the page and the contract). `quantbox report export` writes the same for a run or a
  directory of arms (one run each); a variants run's arms come from `variant_returns.parquet`.
- `report.html` + `report_data.json` — the heavy HTML research report (tens of MB on a typical
  line), written only with `plugins.pipeline.params.full_report: true`.
- `events.jsonl` — structured event log
- Strategy-specific outputs (weights, orders, fills, metrics)

Artifact schemas are bundled at `src/quantbox/artifact_schemas/` and validated at runtime via `importlib.resources`.

## Development

```bash
make dev       # install dev deps
make dev-full  # install all extras + dev deps
pytest -q      # run tests
```

**Broker safety:** start with `readonly: true`, use `--dry-run` to inspect the plan, check `orders.parquet` before enabling live order placement.

See [CLAUDE.md](CLAUDE.md) for agent and LLM development guidelines.

Copy-paste scaffolds for methodology specs, dataset docs, and runbooks are in [`templates/`](templates/).

## Documentation

See [docs/](docs/) for full documentation:
- [Backtesting guide](docs/playbooks/backtesting.md)
- [Multi-repo workflow](docs/playbooks/multi-repo-workflow.md)
- [Trading bridge](docs/playbooks/trading-bridge.md)
- [Approval gate](docs/playbooks/approval-gate.md)
- [Integration guide](docs/playbooks/quantbox-integration-guide.md)
