---
adr: 0006
title: Same-bar runs only under an explicit override, and are research — not backtests
status: accepted
date: 2026-10-02
supersedes: "ADR-0005 § Decision, the sentence \"There is no opt-out flag\"; ADR-0005 § Alternatives considered, the rejected opt-in flag (allow_same_bar)"
---

# ADR-0006: Same-bar runs only under an explicit override, and are research — not backtests

## Context

[ADR-0005](0005-next-bar-is-mandatory.md) made next-bar mandatory: `lag_bars: 0`
is refused everywhere, with no opt-out. Next-bar stays the rule. Tom,
2026-10-02: *"zawsze musi być lag plus 1, nigdy ten sam bar"* — always lag +1,
never the same bar.

The same day he named the one case the rule does not fit: same-bar is
**an explicit allowance against best practice**, granted by the user beyond
the config value. Example: with monthly-only data, the month-end close is the
only price there is. Filling a month-end decision at the NEXT month-end can be
further from what a real order achieved (a few days into the month) than
filling it at the same one. A number made that way is still not a backtest:
*"to bardziej nie backtest, tylko ogólny research"* — it is general research
rather than a backtest.

ADR-0005 rejected an `allow_same_bar` flag as "the v0.8.0 warning with one
more key". The difference here is what happens to the number: v0.8.0 let a
same-bar number come out looking like a backtest. Under this decision it
comes out labelled research, everywhere it is read back.

## Decision

**`execution.lag_bars: 0` runs only when the same block carries the
override:**

```yaml
execution:
  lag_bars: 0
  same_bar: {allow: true, reason: "monthly-only data: the month-end close is the only price"}
```

The Python helpers `backtest()` and `optimize()` take the same pair as
keywords: `lag_bars=0, allow_same_bar=True, same_bar_reason="..."`.

- **One gate.** The override is checked in the place ADR-0005 put the refusal,
  `quantbox.execution._check_lag`. `lag == 0` passes only with a
  `SameBarOverride`, and only `resolve_execution` builds one, after checking
  the block. `apply_execution_lag` asks for that object, so a bare `0` never
  fills a book. Every entry point that resolves an `execution:` block inherits
  the gate: `quantbox validate`, `quantbox config explain`, `quantbox run -c`,
  `backtest()`, `optimize()` and arms.
- **Refused:** `lag_bars: 0` without the block; `allow: false`; a missing,
  empty or blank `reason`; an unknown key in the block; an override next to
  `lag_bars >= 1` (it would label a next-bar run research); and a negative lag
  in every case. The error message names the override.
- **Recorded.** The run@1 manifest's `execution` block carries
  `same_bar: true` and `same_bar_reason`. A new `run: {kind}` block says
  `research` (same-bar) or `backtest` (next-bar). This is minor 2 of run@1, and
  the schema refuses a manifest whose reason, `same_bar` and `run.kind`
  disagree. `quantbox config explain` plans the same `run` block (explain@1
  minor 1).
- **Never presented as a backtest.** The finding-report export
  (`finding_report.json`) gives a research run hero cards that do NOT report
  the finding's `backtest_sharpe`. Those cards are toned `bad`, the chart title
  says RESEARCH, the provenance table carries `run.kind` and the reason, and
  `audit.axes` holds a failed "Execution timing" axis. `quantbox gates` reads
  the `run_manifest.json` beside a returns file. For a research run's returns
  it adds `run_kind: research` and the reason to the verdict and prints a
  RESEARCH line under it. `run.strict` and `promotion` mode refuse a research
  run.
- **Not opened:** `quantbox sweep` refuses the `same_bar` block by name,
  because a grid of same-bar numbers is the multiple-testing search the override
  must never feed. The L1 signal helpers `quantbox.bt.run` and
  `adapters.vectorbt.from_signals_with_costs` keep refusing `0`.

## Consequences

- A lab that needs same-bar (monthly-only data) writes the override and its
  reason in the config. Nothing else in the run changes, and the reason
  travels with every number the run produces.
- A config that set `lag_bars: 0` without the override still fails, as under
  ADR-0005. The error now tells the user how to ask for it deliberately.
- Labs must not use the override to keep old same-bar research alive. The
  reason must say why same-bar is closer to reality for THAT data. Every
  existing same-bar finding stays a record that needs re-measurement next-bar
  (TOM-1399).
- `run_manifest.schema.json` keeps `lag_bars` minimum 0. Manifests written
  before minor 2 have no `run` block and still validate. A v0.8.0 manifest that
  recorded `same_bar: true` without a reason is read as research by
  `quantbox.run_manifest.run_kind`.
- `EXECUTION_SCHEMA` (what `quantbox plugins schema` and `validate` read)
  declares `same_bar` and `lag_bars` minimum 0. The rule "0 only with the
  override" is enforced by the resolver, not by the params schema.

## Alternatives considered

- **Keep ADR-0005 as is (no override).** Rejected by Tom on 2026-10-02. Data
  that has only one price per period has no honest next-bar fill. Forcing one
  makes a different error rather than removing one.
- **A CLI flag (`--allow-same-bar`) instead of a config key.** Rejected. The
  config is what the manifest hashes and records, and a flag would leave the
  config claiming a timing the run did not use.
- **Mark research only in the manifest.** Rejected. A reader of the finding
  page or a gate verdict never opens the manifest, so the label has to be where
  the number is read.
