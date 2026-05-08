---
adr: 0001
title: Composing framework, not competing framework
status: accepted
date: 2026-04-25
status_changes:
  - 2026-04-25: title rephrased from "Library, not framework" to "Composing framework, not competing framework". The original framing overcorrected by avoiding the word "framework" — the actual decision was to refuse a *competing* framework while embracing a *composing* one. Substance unchanged. File slug retained for stable references.
---

# ADR-0001: Composing framework, not competing framework

## Context

QuantBox started as "config-driven quant framework" — plugins, registry, runner, mode separation, opinionated runtime. As the surface grew, we faced friction:

- Every casual research idea required YAML, plugin scaffolding, and validator ceremony.
- Adapter wrappers around vectorbt and others started reimplementing features that already existed in those libraries.
- Skills (LLM-facing API) implicitly required the full pipeline runner even for trivial backtests.
- Existing alternatives (Qlib, Kedro, NautilusTrader) cover the "competing framework" space well; competing with them on their turf was a losing race.

The choice was: keep growing as a framework that competes with Qlib/Nautilus, or restructure as a framework that *composes* those libraries instead.

## Decision

**QuantBox is a composing framework, not a competing framework.**

It is a framework — by the strict IoC sense (the runtime calls *your* code at L4/L5) and by the colloquial sense ("my framework for X"). What it refuses is the *kind* of framework that competes with the wheel by reimplementing it.

It owns three things:

1. Conventions (data layouts, run-artifact shape, lifecycle states, skill API).
2. Adapters (thin wrappers around best-of-breed external libraries).
3. Templates and skills (project bootstrap + LLM-facing API).

It does not own:

- Backtest engines (vectorbt does)
- Model registries (MLflow does)
- Data versioning (DVC does)
- Portfolio optimization algorithms (riskfolio, PyPortfolioOpt, skfolio do)
- Factor research pipelines (Qlib does, when used)
- Live broker abstractions (ccxt, IBKR APIs do)

The plugin runtime (`run_from_config`, CLI) is one of multiple entry points — see [ADR-0002](0002-layered-api.md) — not the only or default way to use QuantBox.

## Alternatives considered

### A. Competing framework

Continue growing the registry/runtime as the primary interface; build out backtesting, model lifecycle, factor analysis as first-party features that compete head-to-head with Qlib and NautilusTrader.

**Rejected because:** the framework race against Qlib, NautilusTrader, and Zipline-reloaded is unwinnable for a one-person shop. Framework maintenance tax compounds; QuantBox would always lag on ML, factor analysis, and live-trading integrations.

### B. Replace with Qlib + thin shim

Adopt Qlib as the underlying framework; make QuantBox a thin Polish-fund / domain-specific shim.

**Rejected because:**
- Qlib drags torch and other heavy deps incompatible with the VPS scale and Vercel-deployed projects.
- Qlib's data layer is opinionated and Chinese-market-shaped.
- Multi-consumer use (quantbox-live for crypto, dm-evo for advisory) doesn't fit one framework.
- Loses the LLM-skill integration that's QuantBox's unique value.

### C. Composing framework (chosen)

QuantBox owns conventions, plugin contracts, and the runtime; everything else is an adapter or a project concern.

**Accepted because:**
- Composing frameworks integrate; competing frameworks reinvent.
- Adapters mean the wheel does the wheel's work; QuantBox stays small.
- LLM-skill integration becomes the moat — nobody else has it, and it pays off across all consumers.
- Multi-consumer use is natural: each project picks the adapters it needs.

## Consequences

### Intended

- Adapter package (`quantbox.adapters.*`) becomes a first-class part of the library.
- Layered API (L0–L5, see [ADR-0002](0002-layered-api.md)) is required so users can drop down to underlying libs.
- Skills produce configs first, code only when capability is missing.
- The plugin runtime (L4) is one of six entry points, not the default.
- Methodology lifecycle (`meta.status`) becomes a convention, not a runtime gate.

### Unintended (and accepted)

- The "config-driven framework" framing in the README and docs needs nuance — "composing framework" or "template-driven SDK with adapters" depending on context.
- Some existing plugin code in `quantbox.backtesting.*` overlaps with what `adapters.vectorbt` will provide; will be deprecated and routed through the adapter.
- Future requests to add new plugin types or runtime features will face higher scrutiny — prefer composition (adapter) over framework growth (reimplementation).

### Anti-patterns this rules out

- Hiding a wrapped library behind opaque classes.
- Reimplementing features that exist in vectorbt/MLflow/etc.
- Forcing users up the abstraction stack (e.g., requiring YAML for a 5-line backtest).
- Building "framework features" without a concrete consumer use case.

## Notes

- Decision was made after a long iterative discussion comparing Qlib, Kedro, vectorbt, NautilusTrader, and the emerging dm-evo + quantlab use cases.
- Encoded in [architecture/principles.md](../architecture/principles.md) as principle #1.
- Operationalized in [architecture/api-layers.md](../architecture/api-layers.md) and [architecture/adapters.md](../architecture/adapters.md).
- Implications for skills and templates are captured in [architecture/skills.md](../architecture/skills.md) and [architecture/templates.md](../architecture/templates.md).
