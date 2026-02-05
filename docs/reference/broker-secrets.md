# Broker secrets & safety

## Binance

Recommended: environment variables

- BINANCE_API_KEY
- BINANCE_API_SECRET

Config references env var names (`api_key_env`, `api_secret_env`).

## IBKR

No API keys needed; you connect locally to TWS/IB Gateway.
Use `readonly: true` for safety until you confirm everything.

## Safety checklist

- Start with `readonly: true`
- Use `quantbox run --dry-run` to inspect plan
- Inspect `orders.parquet` before enabling order placement
