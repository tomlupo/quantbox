---
adr: 0005
title: Next-bar execution is mandatory — lag_bars 0 is refused
status: amended by ADR-0006
date: 2026-10-02
amended_by: 0006-same-bar-explicit-override.md
---

> **Amended by [ADR-0006](0006-same-bar-explicit-override.md) (2026-10-02).**
> Next-bar stays mandatory. But "There is no opt-out flag" and the rejected
> `allow_same_bar` alternative below are superseded: `lag_bars: 0` runs under
> the explicit `execution.same_bar: {allow: true, reason}` override, and such a
> run is RESEARCH, not a backtest. ADR-0006 is the authority on same-bar; the
> superseded text is kept below, struck through, as the record.

# ADR-0005: Next-bar execution is mandatory — `lag_bars: 0` is refused

## Context

TOM-1337 made next-bar the default (`execution.lag_bars: 1`, v0.8.0) but kept
`lag_bars: 0` reachable on purpose, with a warning, so historical same-bar
numbers could be reproduced. A warning above a result enforces nothing: a
same-bar number still comes out the other end looking like a backtest, and the
weekly research review on 2026-10-02 filed "re-run the same-bar findings" as a
research topic — treating the timing as an open question.

It is not one. Tom, 2026-10-02: execution on the next bar is the standard, not
a research question; one can never assume a fill on the same bar the signal
was computed from, and that holds for quick calculations too.

## Decision

**`execution.lag_bars` must be an integer >= 1.** `0` raises `ValueError`
from the shared resolver (`quantbox.execution.resolve_lag_bars` /
`resolve_sweep_lag_bars`), so every entry point refuses it before any data is
loaded: `quantbox run -c` (`backtest.pipeline.v1`), `backtest()`,
`optimize()`, `analysis.parameter_grid.sweep`, `quantbox sweep`, arms, the
L1 helpers `quantbox.bt.run` and `adapters.vectorbt.from_signals_with_costs`
(which used to fill signals same-bar and now take `lag_bars=`, default 1), and
`quantbox validate` (an error finding, not a warning). `apply_execution_lag`
refuses it too, so a caller that skips the resolver cannot hand an engine a
same-bar book. The deprecated `shift_signal=0` alias is refused the same way.

The engine primitives — `vectorbt_engine.run` and
`rsims_engine.fixed_commission_backtest_with_funding` — fill row t at close[t]
by contract and take weights already lagged. They are no longer exported from
`quantbox.plugins.backtesting` (asking for `run_vectorbt` there raises an
`ImportError` that points at `backtest()`); they stay importable from their
own modules for the pipeline and the engine tests.

~~There is no opt-out flag.~~ *(Superseded by ADR-0006: the explicit
`execution.same_bar` override.)* The same-bar warning (`warn_if_same_bar`), the
frozen same-bar canonical goldens (`cookbook/canonical/expected_same_bar/`)
and the test that reproduced them are deleted.

`run_manifest.json` keeps `execution.same_bar` (always `false` under this
ADR; `true` only under ADR-0006's override) so the manifest schema and readers
of old manifests do not change.

## Consequences

- Any config, script or lab that sets `lag_bars: 0` fails loudly on upgrade;
  the fix is to delete the line.
- A same-bar number can only be reproduced by pinning a release older than
  this one, deliberately — and it is then evidence of a look-ahead, not a
  result to compare against.
- Labs pinned below this release still run same-bar wherever they set it;
  bumping the pin is how a lab inherits the refusal.
- Code that imported `run_vectorbt` or `fixed_commission_backtest_with_funding`
  from `quantbox.plugins.backtesting` breaks at import. The fix is `backtest()`,
  or the module path plus an explicit `apply_execution_lag`.
- `quantbox.bt.run` and `from_signals_with_costs` now fill next-bar, so the
  same inputs return different numbers than before — one bar later.
- Two bypasses are left, both documented:
  a caller that imports an engine primitive from its module and hands it raw
  weights, and the L0 `vbt` re-export (`quantbox.adapters.vectorbt.vbt`),
  which fills whatever it is handed. Every L0 example lags its signals one bar
  first.
- `run_manifest.schema.json` keeps `lag_bars` minimum 0 so manifests written
  before this decision still validate.

## Alternatives considered

- ~~**An opt-in flag (`allow_same_bar: true`)** to keep historical numbers
  reproducible on the current release. Rejected: it is the v0.8.0 warning
  with one more key, and a same-bar number is not a result worth reproducing.~~
  *(Superseded by ADR-0006. An opt-in now exists, for data where same-bar is
  closer to reality, not for reproducing history. It labels the run research,
  which the v0.8.0 warning never did.)*
- **Lagging inside the engine primitives.** Rejected: every caller that
  already lags (the pipeline, `backtest()`, labs) would silently trade two
  bars late. Removing the public name fails loudly instead.
