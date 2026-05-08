---
adr: 0002
title: Layered API (L0–L5)
status: accepted
date: 2026-04-25
---

# ADR-0002: Layered API (L0–L5)

## Context

[ADR-0001](DEC-0001-library-not-framework.md) established that QuantBox is a composing framework — owned and opinionated, but composing external libraries rather than competing with them. That decision alone doesn't decide *how* the surface is shaped. We saw a recurring failure mode:

- A user wants to do something simple (try a backtest idea).
- The skill points at the runner with YAML.
- The user (or LLM) has to write a config, register a plugin, validate, run, parse manifest.
- Net effect: people bypass quantbox by `import vectorbt as vbt` directly.

If the library can be bypassed for casual use, the conventions don't propagate. If the library imposes ceremony for casual use, people work around it. Both fail.

## Decision

**QuantBox exposes every capability at multiple layers (L0–L5), and skills default to the lightest layer that solves the task.**

| Layer | Shape | Purpose |
|---|---|---|
| L0 | Re-exports (`from quantbox.adapters.vectorbt import vbt`) | Pure passthrough to the underlying lib |
| L1 | Convenience helpers (`quantbox.bt.run`, `quantbox.opt.max_sharpe`) | Common idiom in one function call |
| L2 | Composable units (`run_strategy(strategy, data)`, planned) | Building notebooks, comparing ideas |
| L3 | Plugin instances (call `Strategy()` directly) | Validated contracts without YAML |
| L4 | Full pipeline (`run_from_config`) | Logged experiment, manifest, lineage |
| L5 | CLI (`quantbox run -c config.yaml --strict`) | Production with reproducibility pins |

Inner layers (L0/L1/L2) are first-class — they get docs, tests, and skill defaults. They are not afterthoughts.

Skills declare a `default_layer` in frontmatter and escalation rules for when to climb the stack. Default is L1 unless the task demands more.

## Alternatives considered

### A. Single layer (runner-only)

Everything goes through `run_from_config`. Casual use requires YAML.

**Rejected because:** users bypass by importing the underlying library directly, which means QuantBox conventions don't reach them. The friction is paid every time, not amortized.

### B. Two layers (L0 re-exports + L4 pipeline)

Power users use the adapter; everyone else uses the runner.

**Rejected because:** this leaves a gap at "I want validation but not YAML" (L3) and "I want a one-liner that does the right thing" (L1). Forcing users to either bypass entirely or commit to the full runner is the original failure mode.

### C. Six-layer ladder (chosen)

Every step adds one level of structure; users pick the lowest one that meets their need.

**Accepted because:**
- Casual use stays at L0/L1 — conventions still propagate via the convenience helpers.
- Logged research fits L4 cleanly without dragging the CLI in.
- Production gets the full ceremony at L5 with `--strict`.
- Skills can teach the layer choice as a table, making decisions auditable.

## Consequences

### Intended

- `quantbox.bt`, `quantbox.opt`, `quantbox.score` become first-class L1 modules with their own docs and tests.
- Adapters at L0 must re-export the underlying library namespace; opaque wrappers are forbidden.
- Skill frontmatter gains `default_layer` and `escalation_rules`; skills without these are incomplete.
- Plugin instances must be callable directly without invoking the runner — no "must be invoked via run_from_config."
- The `--strict` mode is an L5 concern; L0–L4 do not require it.

### Unintended (and accepted)

- Some duplication: a capability available at L1 may also be exposed at L4 via plugin. This is fine — they serve different ergonomics.
- More surface area to maintain. Mitigated by inner layers being more stable than outer (breaking L1 forces L4 revalidation; breaking L4 only affects YAML configs).
- Users may write code that mixes layers (e.g., a script that uses L1 for prep and L4 for the run). Acceptable — we're not policing layer purity within user code.

### Anti-patterns this rules out

- Skills that default to L4/L5 for "test this idea" tasks.
- Adapters that hide the underlying lib behind opaque wrappers.
- Plugins that error if called outside the runner.
- "Convenience helpers" that hide functionality the underlying library exposes.

## Notes

- Operationalized in [architecture/api-layers.md](../architecture/api-layers.md) — the canonical layer table.
- Adapter rule that supports L0 is in [architecture/adapters.md](../architecture/adapters.md) and [ADR-0001](DEC-0001-library-not-framework.md).
- Skill frontmatter contract is in [architecture/skills.md](../architecture/skills.md).
- Pre-existing `quantbox.backtesting.*` modules will be reorganized to fit this layering.
