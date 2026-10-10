---
adr: 0009
title: Metrics describe, inference tests, gates decide — three modules, one implementation per statistic
status: accepted
date: 2026-10-08
supersedes:
superseded_by:
amends: "ADR-0008 in sense only: its 'one seam, one function per job' rule now covers the statistics too."
status_changes:
  - 2026-10-08: accepted with TOM-1618 — Tom, 2026-10-08: codebase-design pass over metrics, analysis and the statistics code ("Karta + buduj teraz")
  - 2026-10-08: amended with TOM-1644 — decision 7, the Newey-West lag floor for overlapping observations (Tom: "Tak + do quantbox")
  - 2026-10-08: amended with TOM-1646 — decision 8, multiple-testing corrections (Tom: "Bonferroni do quantbox")
---

# ADR-0009: Metrics describe, inference tests, gates decide

## Context

Before TOM-1618 the statistics code had three homes and no rule for which one a
function belongs to.

- `quantbox.metrics` described a run. Since #254 (TOM-1596) it also held
  inference: `hac_ols`, `newey_west_tstat`, `require_finite`.
- `quantbox.analysis` held the DSR (`analysis.dsr`), HAC (`analysis.hac`), the
  gates (`analysis.gates`) and the parameter-grid sweep (`analysis.parameter_grid`).
- Newey-West had three import paths.

The duplication guard of #236 matched metric NAMES. It did not see these copies:

- two NaN refusals (`analysis.gates._finite` raised `GateInputError`,
  `metrics.require_finite` raised `ValueError`), plus three more inline copies;
- a second skew/kurtosis in `validation.deflated_sharpe_blp`, and three more
  in `simulation` and the report blocks;
- three bootstraps and a Monte-Carlo null outside the gates;
- `gates.max_drawdown` (positive) next to `metrics.max_drawdown` (negative).

## Decision

1. **Three modules, three verbs.**
   - `quantbox.metrics` DESCRIBES a run: Sharpe, IR, tracking error, CAGR,
     drawdowns, turnover, VaR, IC/ICIR, beta, hit rate, active share, risk
     contributions. The one aggregate is `compute_backtest_metrics`. A
     statistic with no answer is NaN (Sharpe keeps its 0.0).
   - `quantbox.inference` TESTS a claim: `require_finite`, `moments`,
     `return_moments`, `hac_ols`, `newey_west_tstat`, `factor_regression`, the
     DSR family, `bootstrap` (iid or paired stationary block),
     `moving_block_indices`, `gaussian_null`, `largest_drawdown_episode`. An
     input that cannot give a statistic raises `InferenceInputError`.
   - `quantbox.gates` DECIDES: trial range, thresholds, windows, the claim leg,
     `gate_pass`. It computes no statistic. `quantbox.gates_cli` is its CLI.
   Dependencies point one way: gates → inference → metrics.
2. **One implementation per statistic.** One NaN refusal (`require_finite`,
   1-D per observation or 2-D per row), one "constant to floating-point
   noise" test (`quantbox._numerics.flat`, threshold `DEGENERATE_RTOL`), one
   HAC fit, one `moments`, one resampling loop. `tests/test_metrics_module.py`
   guards each class by its MECHANICS (counting non-finite rows, a HAC fit,
   `**3`, a `replace=True` resample, ...) outside `inference.py` and
   `_numerics.py`. On origin/dev `bd5865d` it found 22 (file, class) pairs.
3. **One refusal type.** `InferenceInputError` subclasses `ValueError`, so
   every `except ValueError` written before still catches it.
   `quantbox.gates.GateInputError` IS that class.
4. **One drawdown sign: negative.** `metrics.max_drawdown`,
   `metrics.top_drawdowns` and `inference.largest_drawdown_episode` report
   negative fractions. `quantbox.gates` has no `max_drawdown`. The gates'
   JSON keeps its positive depth (the `max_drawdown` leg value and
   `episode.depth`), because the qute-research acceptance-gates contract and
   every threshold written against it use that sign; one helper
   (`gates._depth`) converts it.
   *Amended by TOM-1627 (2026-10-08):* the sign is now also in the NAME.
   Every drawdown output carries the pair `max_drawdown` (signed, <= 0) and
   `max_drawdown_abs` (the positive depth, >= 0), from one helper,
   `metrics.drawdown_fields`: the metrics dict, run@1 `metrics` (minor 8),
   the gates' JSON (`episode`, and `drawdowns` on a `max_drawdown` leg) and
   the finding-report export. No existing key changed value: the leg value
   and `episode.depth` stay positive.
5. **The sweep is not analysis.** `analysis.parameter_grid` moves to
   `quantbox.sweep`, the library half of `quantbox sweep`.
6. **Old paths are shims.** `quantbox.analysis` and its four submodules, and
   `quantbox.metrics.{newey_west_tstat, newey_west_auto_lags, hac_ols,
   require_finite}`, resolve each name to the SAME object in its new home and
   emit a `DeprecationWarning` that names it (`quantbox._deprecation.moved`).
   `analysis.gates.max_drawdown` and `analysis.gates.largest_drawdown_episode`
   keep their old positive sign on the old path only.
7. **Overlapping observations set a lag floor.** *Added by TOM-1644
   (2026-10-08), Tom: "Tak + do quantbox"; from robo-lab#16.* A series of
   overlapping h-period observations (an IC on h-day forward returns, a
   rolling h-day spread) is an MA(h-1): neighbours share h-1 periods. Its
   Newey-West lag count is **max(auto lags, h-1)** (Hansen and Hodrick
   1980). The automatic rule alone leaves the long-horizon t-stat
   overstated: in robo-lab#16 the floor moves bg6's 252d IC t from 7.93
   to 0.83.
   - `inference.newey_west_tstat(returns, lags=None, *, overlap=None)` holds
     the rule. `overlap=h` counts periods of the series passed; converting a
     calendar window to periods (21 trading days in a monthly series) is the
     caller's job.
   - Each horizon is tested on its own, with only that horizon's overlap.
   - An explicit `lags` still wins. A `lags` below h-1 is refused, and so is
     an h longer than the sample: the floor could only be met by clamping,
     which would hide the overstatement.
   - The result dict reports `nw_overlap` (`None` when not given). Without
     `overlap` no number moves.
   - `tests/test_inference_nw_overlap.py` holds the known answer: an MA(62)
     series where auto lags recover an SE ratio near 0.44 of the true SE,
     and the floor recovers near 0.82.
   - The `nw` gate and `factor_regression` do not take `overlap` yet: a gate
     tests a strategy's per-period returns, which do not overlap.
8. **Multiple-testing corrections are inference.** *Added by TOM-1646 (2026-10-08).* `inference.multiple_testing(pvalues, *, alpha, method="bonferroni"|"holm")` and `inference.bonferroni_alpha(n_tests, *, alpha)` (the per-test level `alpha/m`, for a CI) are the one home; the guard's `multiple_testing` class refuses a second one.

## Consequences

- Numbers do not move. A before/after snapshot of every gate, every metric
  and every validation plugin on fixed fixtures (5664 values) is bit-identical,
  except sample skew/kurtosis in the BLP DSR plugin and in the simulation path
  statistics, which move in the last bits (relative difference at most 4.2e-14):
  scipy computes `m3 / m2**1.5`, the deleted copies computed `mean(z**3)`. No
  verdict changes.
- A new statistic goes into `inference` and nowhere else; the guard fails the
  build otherwise. A new pattern must come with a copy that trips it.
- Removing the shims is a later, breaking change. quantbox-lab imports
  `quantbox.analysis` (three files); robo-lab and quantbox-live import only
  paths that did not move.
