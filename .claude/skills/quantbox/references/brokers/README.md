# Brokers

Broker plugins manage positions and execute orders. Required for paper and live trading
pipelines. Not needed for backtesting or research pipelines.

## When to Use

- You're running a trading pipeline (`trade.full_pipeline.v1`)
- You need to add support for a new exchange
- You need to inspect portfolio state (positions, cash, equity)

## Available Brokers

### Paper (simulated)

| ID | Type | Description |
|---|---|---|
| `sim.paper.v1` | Spot | Simple paper broker, tracks positions and cash |
| `sim.futures_paper.v1` | Futures | Futures paper broker with leverage, margin, and PnL |
| `ibkr.paper.stub.v1` | Spot | IBKR simulator stub |
| `binance.paper.stub.v1` | Spot | Binance simulator stub |

### Live (real money)

| ID | Type | Env vars required | Description |
|---|---|---|---|
| `ibkr.live.v1` | Spot | TWS/Gateway running | Interactive Brokers live trading |
| `binance.live.v1` | Spot | `API_KEY_BINANCE`, `API_SECRET_BINANCE` | Binance spot |
| `binance.futures.v1` | Futures | `API_KEY_BINANCE`, `API_SECRET_BINANCE` | Binance USDT-M futures |
| `hyperliquid.perps.v1` | Perps | `HYPERLIQUID_WALLET`, `HYPERLIQUID_PRIVATE_KEY` | Hyperliquid perps |

## Broker Selection Guide

**Testing / development?**
- Spot: `sim.paper.v1` (no credentials needed)
- Futures: `sim.futures_paper.v1`

**Crypto spot trading?**
- `binance.live.v1` for Binance
- Paper first with `binance.paper.stub.v1`

**Crypto futures / perps?**
- `binance.futures.v1` for Binance USDT-M
- `hyperliquid.perps.v1` for Hyperliquid
- Paper first with `sim.futures_paper.v1`

**Traditional assets?**
- `ibkr.live.v1` for IBKR (requires TWS/Gateway)
- Paper first with `ibkr.paper.stub.v1`

## Config Examples

### Paper broker (spot)
```yaml
plugins:
  broker:
    name: "sim.paper.v1"
    params_init:
      cash: 100000
      quote_currency: "USDT"
```

### Paper broker (futures)
```yaml
plugins:
  broker:
    name: "sim.futures_paper.v1"
    params_init:
      cash: 100000
      quote_currency: "USDT"
      max_leverage: 3.0
```

### Live Binance (spot)
```yaml
plugins:
  broker:
    name: "binance.live.v1"
    params_init: {}
    # Uses API_KEY_BINANCE and API_SECRET_BINANCE from environment
```

## Environment Variables

| Variable | Broker | Notes |
|---|---|---|
| `API_KEY_BINANCE` | binance.live.v1, binance.futures.v1 | Binance API key |
| `API_SECRET_BINANCE` | binance.live.v1, binance.futures.v1 | Binance API secret |
| `HYPERLIQUID_WALLET` | hyperliquid.perps.v1 | Wallet address |
| `HYPERLIQUID_PRIVATE_KEY` | hyperliquid.perps.v1 | Private key |

**Never put credentials in YAML configs.** Use env vars or `.env` file.

## Next Steps

- **Create a custom broker**: Load [api.md](api.md)
- **Debug broker issues**: Load [gotchas.md](gotchas.md)
