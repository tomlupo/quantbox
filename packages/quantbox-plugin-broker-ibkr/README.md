# quantbox-plugin-broker-ibkr

Real broker adapter scaffold for Interactive Brokers via `ib_insync`.

## Notes
- Requires IB Gateway or TWS running.
- `paper=True` simply means you connect to your paper trading port/account.
- Secrets are not stored in YAML; use env vars if needed (IBKR typically uses local session).

## Minimal params_init
```yaml
plugins:
  broker:
    name: ibkr.paper.v1
    params_init:
      host: 127.0.0.1
      port: 7497
      client_id: 7
      account: "DUXXXX"
```
