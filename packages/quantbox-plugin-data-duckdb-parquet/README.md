# quantbox-plugin-data-duckdb-parquet

Local EOD prices from Parquet via DuckDB.

## Optional FX
Set `fx_path` in `params_init` to a Parquet with columns: `date, pair, rate` (e.g. `EURUSD`, `USDJPY`).
