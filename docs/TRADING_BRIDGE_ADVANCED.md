# Trading bridge advanced upgrade (v0.2)

`trade.allocations_to_orders.v1` now supports (best-effort) while keeping contracts stable:
- Futures multipliers (via `instrument_map`)
- Lot size + step size + min qty + min notional (via `instrument_map`)
- Multi-currency cash + FX conversion into USD base (via `fx_path`)

## Instrument map
Provide YAML/CSV. Example: `configs/instruments.yaml`

Fields:
- symbol (required)
- asset_type (optional)
- currency (default USD)
- multiplier (default 1)
- lot_size (default 1)
- min_qty (default 0)
- qty_step (default 0)
- min_notional (default 0) in USD

## FX data
Provide `data/curated/fx.parquet` with columns:
- date
- pair (e.g. EURUSD or USDJPY)
- rate

The pipeline uses latest rate <= asof.
