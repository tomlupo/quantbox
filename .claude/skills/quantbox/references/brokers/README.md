# Brokers

Broker plugins manage positions and execute orders. Required for paper and live trading
pipelines. Not needed for backtesting or research pipelines.

## When to Use

- You're running a trading pipeline (`trade.full_pipeline.v1`)
- You need to add support for a new exchange
- You need to inspect portfolio state (positions, cash, equity)

## Available Brokers

<!-- BEGIN AUTO-GENERATED -->
| ID | Description | Tags |
|---|---|---|
| `binance.futures.v1` | Binance USDM Futures broker (live) with leverage and short support | live, futures, binance |
| `binance.live.v1` | Binance broker adapter (python-binance) | binance, broker, crypto |
| `binance.paper.stub.v1` | Binance (paper stub) | paper, stub |
| `hyperliquid.perps.v1` | Hyperliquid DEX perpetuals broker (live) with short support | live, futures, hyperliquid, decentralized |
| `ibkr.live.v1` | Interactive Brokers broker adapter (ib_insync) | ibkr, broker |
| `ibkr.paper.stub.v1` | Interactive Brokers (paper stub) | paper, stub |
| `sim.futures_paper.v1` | Futures paper broker with margin accounting and short support | paper, futures |
| `sim.paper.v1` | Simple paper broker simulator for spot trading | paper |
<!-- END AUTO-GENERATED -->

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
