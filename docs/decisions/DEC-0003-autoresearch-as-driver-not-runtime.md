---
adr: 0003
title: Autoresearch as L4 driver, not runtime mode
status: accepted
date: 2026-04-25
---

# ADR-0003: Autoresearch as L4 driver, not runtime mode

## Context

LLM-driven continuous improvement loops are a real product need — improve client strategies over time without hand-tuning every parameter. The architectural question is *where* the loop lives.

Three candidate placements emerged:

1. **As a new mode in `run_from_config`** — alongside `backtest | paper | live`, add `autoresearch` mode.
2. **As a new pipeline kind** (`autoresearch.pipeline.v1`) — wraps a baseline pipeline with a proposer/evaluator.
3. **As a separate L4 driver** — `AutoResearchDriver`, sits alongside `run_from_config`, orchestrates standard pipeline runs.

This decision has consequences for: skill design, budget enforcement, cost tracking, lifecycle integration, and the `--strict` semantics.

## Decision

**Autoresearch is an L4 driver — `AutoResearchDriver` — that orchestrates standard `run_from_config` calls. It is not a new pipeline kind, not a runtime mode, and not embedded in the runner.**

A new `VariantProposerPlugin` Protocol is added for the proposal step. LLM and algorithmic proposers are adapters (anthropic, openai, optuna). The driver is owned by QuantBox; the proposers are adapter-shaped.

The driver exposes:

- `quantbox.autoresearch.AutoResearchDriver` (Python API, L4)
- `quantbox autoresearch run / tick / status` (CLI, L5)

Each iteration of the loop is a normal `run_from_config` call — same artifact store, same lineage, same reproducibility pins.

## Alternatives considered

### A. New runtime mode (`mode: autoresearch`)

Add autoresearch to the `Mode` literal in `contracts.py`; the runner branches on mode.

**Rejected because:**
- Conflates two different concerns: a *single run* (backtest/paper/live) vs *a sequence of runs* (autoresearch).
- Pollutes the `Mode` type with a non-orthogonal value — `mode=autoresearch` would still need an inner mode.
- Makes the runner stateful (it tracks the loop), violating the runner's current "one run, one result" contract.
- Forces every plugin to potentially handle autoresearch context, even those that don't care.
- Couples lifecycle of the loop to lifecycle of one pipeline run (e.g., `RunResult` semantics break).

### B. New pipeline kind (`autoresearch.pipeline.v1`)

A pipeline plugin that, in its `run()`, invokes a proposer + nested pipeline N times.

**Rejected because:**
- Pipelines are atomic units in the design — they produce one `RunResult`. An autoresearch loop produces N RunResults plus a meta-summary.
- Budget enforcement (cost, time, trials) is loop-level, not pipeline-level. Pipelines have no concept of budget.
- Schema/lineage become awkward: the "outputs" of an autoresearch pipeline are the *trials*, but each trial has its own outputs.
- Re-entrant problems: nesting an autoresearch pipeline inside another pipeline is undefined behavior.
- The skill layer becomes confused: `quantbox-research` (capability skill) vs `autoresearch.pipeline.v1` (pipeline kind) overlap conceptually.

### C. L4 driver alongside `run_from_config` (chosen)

`AutoResearchDriver` is a separate L4 entry point. Internally calls `run_from_config` once per trial. Owns the loop, the budget, the memory, the proposer integration.

**Accepted because:**
- Cleanly separates "run a thing" from "decide what to run next."
- Budget, cost tracking, convergence, persistence are loop-level concerns and live with the driver.
- Each trial is a normal pipeline run — reuses everything: ArtifactStore, lineage, schemas, validation, `--strict`.
- Plugins remain unaware of the loop; their semantics don't change.
- The L4 surface stays narrow: two drivers (`run_from_config`, `AutoResearchDriver`) instead of mode-bloating one.
- Skill layer is clean: `quantbox-autoresearch` is its own capability skill with `default_layer: L4`.
- Continuous mode (`tick`) and one-shot mode (`run`) are natural variants of the same driver.

## Consequences

### Intended

- New module: `quantbox.autoresearch` with `AutoResearchDriver`, `AutoResearchConfig`, `BudgetTracker`, `EvaluationGate`.
- New plugin Protocol: `VariantProposerPlugin`. Three built-in proposers (`optuna`, `llm`, `hybrid`) shipped via adapters.
- New CLI subcommand: `quantbox autoresearch {run,tick,status}`.
- New skill: `quantbox-autoresearch` at `quantbox/skills/`, `default_layer: L4`.
- New durable memory format: `EXPERIMENTS.jsonl` (machine-readable, append-only) alongside the existing `EXPERIMENTS.md`.
- Adapters used: `quantbox.adapters.optuna`, `quantbox.adapters.anthropic`, optionally `quantbox.adapters.openai`.

### Unintended (and accepted)

- A second L4 driver means the docs need to show both. `api-layers.md`'s L4 example needs a second row for `AutoResearchDriver`.
- The `VariantProposerPlugin` Protocol is the first new Protocol since the original architecture. Any future Protocol additions need similar ADR scrutiny.
- Per-client autoresearch state means a new convention for `research/clients/{client}/` directories. Templates may want to include this in the `client` template variant.
- Cost tracking (LLM usage, compute time) becomes part of the architecture — a new concern that hadn't existed before.

### Anti-patterns this rules out

- ❌ Embedding loop logic in `run_from_config` — runner stays single-run.
- ❌ Reimplementing search algorithms — Optuna is the wheel, used via adapter.
- ❌ Auto-promoting candidates to production — driver maxes at `research`; lifecycle gate is human.
- ❌ Loop without statistical gates — driver refuses to start without walk-forward + deflated Sharpe + drawdown caps.
- ❌ Unbounded budgets — driver requires explicit max_trials, max_wall_clock, max_cost.

## Notes

- Encoded in [`docs/architecture/autoresearch.md`](../architecture/autoresearch.md) — the full design.
- Operationalized in [`docs/playbooks/run-an-autoresearch-loop.md`](../playbooks/run-an-autoresearch-loop.md) — step-by-step.
- Skill at `quantbox/skills/quantbox-autoresearch/` — currently a stub, blocked on driver implementation.
- Design flows from [ADR-0001](DEC-0001-library-not-framework.md) (composing framework — proposers and search are adapters) and [ADR-0002](0002-layered-api.md) (driver placement at L4 follows the layered API rule).
- Inspired by patterns from LLM-driven autonomous research loops (Karpathy and others); QuantBox's contribution is the convention layer (memory format, lifecycle integration, budget primitives, statistical gates) — the heavy lifting is delegated to existing libraries.
