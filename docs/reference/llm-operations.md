# LLM / Agent Operations

Useful commands for an agent:

```bash
quantbox plugins list --json
quantbox plugins info --name fund_selection.simple.v1 --json
quantbox validate -c configs/run_fund_selection.yaml
quantbox run -c configs/run_fund_selection.yaml --dry-run
quantbox run -c configs/run_fund_selection.yaml
```

Artifacts per run include:
- `run_manifest.json` (single-file summary for agents)
- `events.jsonl` (structured event stream)

Trading bridge example:
```bash
quantbox plugins info --name trade.allocations_to_orders.v1 --json
```
