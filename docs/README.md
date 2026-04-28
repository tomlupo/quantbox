# QuantBox documentation

**Modifying QuantBox?** Read [architecture/principles.md](architecture/principles.md) and [architecture/api-layers.md](architecture/api-layers.md) first. Every change downstream is shaped by them.

| Section | Description |
|--------|-------------|
| [architecture/](architecture/) | **Read first when modifying.** Principles, API layers, plugin/adapter/skill conventions, lifecycle, templates, autoresearch. |
| [playbooks/](playbooks/) | Step-by-step how-tos: add-an-adapter, add-a-plugin, add-a-skill, promote-a-methodology, run-an-autoresearch-loop |
| [adr/](adr/) | Architecture Decision Records — *why* the design is the way it is |
| [methodology/](methodology/) | *Why* a model/algorithm works — research-side reasoning + LOCKED specs (one per locked plugin) |
| [runbooks/](runbooks/) | Operational procedures — cron jobs, deploys, brokers, recovery scenarios |
| [datasets/](datasets/) | Human-readable schema/semantics docs paired with `schemas/*.schema.json` |
| [PRD.md](PRD.md) | Product requirements — vision, scope, architecture |
| [specs/](specs/) | Feature and capability specs (active development) |
| [guides/](guides/) | How-to: backtesting, multi-repo workflow, trading bridge, approval gate |
| [reference/](reference/) | Operations: LLM/agent CLI, broker secrets and safety |

## Guides

| Guide | Description |
|-------|-------------|
| [Backtesting](guides/backtesting.md) | Vectorbt and rsims engines, configuration, outputs |
| [Multi-repo workflow](guides/multi-repo-workflow.md) | Versioning, branching, and promotion across quantbox / quantbox-live / quantbox-lab |
| [Trading bridge](guides/trading-bridge.md) | Research-to-trading pipeline, instrument maps, FX |
| [Pipeline chaining](guides/pipeline-chaining.md) | Manual pipeline chaining (planned: meta-pipeline) |
| [Approval gate](guides/approval-gate.md) | Pre-trade approval mechanism |
| [Integration guide](guides/quantbox-integration-guide.md) | Using quantbox in external projects |

Start with [PRD.md](PRD.md) for product context; use **guides** for workflows and **reference** for configuration and tooling.
