# Recipe: Add a broker plugin (IBKR/Binance)

Two options:

## Option A: Built-in broker (recommended for core plugins)
1) Add module under `packages/quantbox-core/src/quantbox/plugins/broker/`
2) Implement:
- `get_positions()`
- `get_cash()`
- `get_market_snapshot(symbols)`
- `place_orders(orders)` -> fills DataFrame
- `fetch_fills(since)`
3) Register in built-ins map:
- `packages/quantbox-core/src/quantbox/plugins/builtins.py`

## Option B: External broker (separate repo/package)
Entry point group:

```toml
[project.entry-points."quantbox.brokers"]
"ibkr.live.v1" = "quantbox_plugin_broker_ibkr.broker:IBKRBroker"
```

Implement:
- `get_positions()`
- `get_cash()`
- `get_market_snapshot(symbols)`
- `place_orders(orders)` -> fills DataFrame
- `fetch_fills(since)`

Keep secrets out of configs: read from env vars / secret refs.
