# QuantBox — full working starter repo (core + plugins)

This repo is a **working minimal** QuantBox implementation:
- `quantbox` core (plugin registry + runner + artifacts)
- `eod.duckdb_parquet.v1` data plugin (DuckDB over Parquet)
- `fund_selection.simple.v1` pipeline plugin (research)
- `sim.paper.v1` broker plugin (paper simulator; not needed for research)

## Install (editable)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip

pip install -e packages/quantbox-core
pip install -e packages/quantbox-plugin-data-duckdb-parquet
pip install -e packages/quantbox-plugin-pipeline-fundselection
pip install -e packages/quantbox-plugin-broker-sim
pip install -e packages/quantbox-plugin-pipeline-alloc2orders
pip install -e packages/quantbox-plugin-broker-ibkr-stub
pip install -e packages/quantbox-plugin-broker-binance-stub
pip install -e packages/quantbox-plugin-broker-ibkr
pip install -e packages/quantbox-plugin-broker-binance
```

## Generate sample data

```bash
python scripts/make_sample_data.py
```

Creates: `data/curated/prices.parquet` with columns: `date, symbol, close`.

## List plugins

```bash
quantbox plugins list
```

## Run the example pipeline

```bash
quantbox run -c configs/run_fund_selection.yaml
```

Artifacts are written to `./artifacts/<run_id>/`.

## Next steps
- Add a **trading pipeline** plugin that consumes `allocations.parquet` and emits `targets/orders/fills`.
- Implement `broker.ibkr.*` and `broker.binance.*` plugins using the same `BrokerPlugin` interface.


## LLM-friendly additions
- `run_manifest.json` and `events.jsonl` written per run
- `quantbox validate` and `quantbox run --dry-run`
- `quantbox plugins list --json` and `quantbox plugins info --name <id> --json`
- artifact schema checks via `/schemas/*.schema.json`


## Trading bridge pipeline (research → paper)
1) Run fund selection to produce allocations:
```bash
quantbox run -c configs/run_fund_selection.yaml
```
2) Copy the produced RUN_ID and edit `configs/run_trade_from_allocations.yaml` to point `allocations_path` to that run.
3) Run trading bridge:
```bash
quantbox run -c configs/run_trade_from_allocations.yaml
```
This writes `targets/orders/fills/portfolio_daily`.


## Advanced sizing & FX
- Add `instrument_map: ./configs/instruments.yaml` to the trading pipeline params.
- Data plugin can load FX if you set `fx_path: ./data/curated/fx.parquet`.
- Trading pipeline writes extra debug artifact: `targets_ext.parquet` and `llm_notes.json`.


## Auto-resolve latest allocations
In the trading config you can auto-use the latest research run:

- `allocations_path: null`
- `allocations_ref: "latest:fund_selection.simple.v1"`

## Approval gate
To require approval before paper/live execution:

1) Run trade config once (fills will be empty until approved)
2) Create approval file:
```bash
python scripts/approve_orders.py --run-dir ./artifacts/<TRADE_RUN_ID>/ --who tom
```
3) Rerun the same trade config (now it can execute if `readonly: false` on the broker).
