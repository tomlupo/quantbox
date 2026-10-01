---
adr: 0004
title: Overlays are a plugin kind, applied as one stage before execution
status: accepted
date: 2026-10-01
---

# ADR-0004: Overlays are a plugin kind, applied as one stage before execution

## Context

Research keeps needing modifiers that sit between a base strategy's decision
and its execution: de-risk an instrument after its trend flips (H22), scale the
book by a volatility regime (H12, H15), cap gross exposure when the book's
holdings move together. Quantbox had no place for them, so three of Quark's
last five dev-loop runs blocked on "needs new strategy code, not config"
(TOM-1364). In quantbox-lab, eight hand-written overlay sites each shifted
their own multiplier, which mixed execution timings and made H08 and H32
incomparable.

`plugin-authoring.md` says a new plugin type needs an ADR. The quantbox 1.0
spec (TOM-1325, user story 46, phase P2) asks for "an overlay stage on a base
strategy, configurable from YAML".

## Decision

**`overlay` is a plugin kind** (`OverlayPlugin` in `contracts.py`,
`apply(weights, data, params) -> DataFrame`, entry-point group
`quantbox.overlays`), configured as an ordered list under `plugins.overlays`
and listed by `quantbox plugins schema --json` like every other kind.

**The backtest pipeline applies the chain as ONE stage**: after the strategies
are aggregated into the decided book, before the venue constraint, the risk
transforms and the execution lag. Overlays chain in config order
(`quantbox.overlays.apply_overlays`).

**An overlay never shifts its output.** Row `t` may read data up to and
including bar `t` and stays on row `t`; `execution.lag_bars` moves the whole
overlaid book to the fill bar once. The chain refuses an overlay whose result
changes the index or columns.

**The run records it.** `RunResult.notes["overlays"]` and the run manifest's
`overlays` field (run@1, minor 1) list what was applied, in order, with params;
`traded_weights` is the book after the chain; `weights_history` is the
overlaid decided book and `base_weights_history` the book before it.

**Only a pipeline that applies overlays accepts them.** A pipeline declares
`accepts_overlays = True`; `validate` and the runner refuse `plugins.overlays`
on any other (today: the trading pipelines), rather than dropping it.

## Consequences

- H22 is a config change (`cookbook/configs/run_backtest_overlay_reversal_derisk.yaml`).
- Every overlay inherits one execution timing; overlay results stay comparable.
- The engine's NaN policy is materialised before the chain, so an overlay
  multiplying a "hold" cell is not lost; idempotent, so engine numbers are unchanged.
- Live trading does not apply overlays yet. Adding them there means the
  trading pipeline declaring `accepts_overlays` and applying the same chain
  to its latest decided row; until then the refusal is the contract.
- Overlays are research-status plugins; `source:` (local file) works for them
  as for strategies.
