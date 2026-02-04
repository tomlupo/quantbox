# Recipe: Add a broker plugin (IBKR/Binance)

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
