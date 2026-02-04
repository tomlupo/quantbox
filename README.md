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
