# quantbox-plugin-broker-binance

Real broker adapter scaffold for Binance via `python-binance`.

## Secrets
Set env vars (recommended):
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`

Then config:

```yaml
plugins:
  broker:
    name: binance.live.v1
    params_init:
      api_key_env: BINANCE_API_KEY
      api_secret_env: BINANCE_API_SECRET
      testnet: false
```

## Notes
- This is a scaffold: you will likely want to add:
  - symbol mapping (e.g., BTCUSDT)
  - precision/lot-size filters
  - better fill reconciliation
"""