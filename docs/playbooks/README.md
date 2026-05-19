# Playbooks

How-tos for extending, operating, and integrating QuantBox. Each is self-contained and references the [architecture/](../architecture/) docs for principles.

## Extending the library

| Playbook | When to use |
|---|---|
| [add-an-adapter.md](add-an-adapter.md) | Wrapping an external library so it's usable at L0/L1 |
| [add-a-plugin.md](add-a-plugin.md) | Adding capability that doesn't fit an existing plugin |
| [add-a-broker-plugin.md](add-a-broker-plugin.md) | Adding a broker plugin (built-in or external entry point) |
| [add-a-pipeline-plugin.md](add-a-pipeline-plugin.md) | Adding a pipeline plugin (built-in or external entry point) |
| [add-a-skill.md](add-a-skill.md) | Authoring a new skill (foundation, capability, or generic-quant) |
| [promote-a-methodology.md](promote-a-methodology.md) | Walking research → locked → production |
| [run-an-autoresearch-loop.md](run-an-autoresearch-loop.md) | LLM-driven continuous improvement loop over an existing strategy |

## Using the library

| Playbook | When to use |
|---|---|
| [backtesting.md](backtesting.md) | Running backtests with vectorbt or rsims engines |
| [trading-bridge.md](trading-bridge.md) | Research-to-trading pipeline, instrument maps, FX |
| [approval-gate.md](approval-gate.md) | Pre-trade human approval mechanism |
| [multi-repo-workflow.md](multi-repo-workflow.md) | Versioning and promotion across quantbox / quantbox-live / quantbox-lab |
| [quantbox-integration-guide.md](quantbox-integration-guide.md) | Using quantbox as a dependency in an external project |

For the doctrine these playbooks operate under, read [architecture/principles.md](../architecture/principles.md) first.
