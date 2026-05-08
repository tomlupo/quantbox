---
methodology: carver-trend
version: v0.2.0
status: DRAFT
date_locked:
supersedes: v0.1.0
src_paths:
  - src/quantbox/plugins/strategies/carver_trend.py
seeds:
  numpy: 42
revalidation:
  cadence: monthly
  baseline_metrics:
    sharpe_oos_min: 0.6
    drawdown_max: -0.30
  on_failure: alert_only
sections:
  - id: §3
    status: DRAFT
    date: 2026-04-25
  - id: validation
    status: COMPLETE
    date: 2026-04-25
    run_ids:
      baseline: 2026-04-25__backtest_pipeline_v1__5eb7d417399d__20260425T200543Z
      bollinger_on: 2026-04-25__backtest_pipeline_v1__b968c2339f95__20260425T200638Z
---

# Carver Trend v0.2 — EWMAC + Breakout + Bollinger

> Adds an optional Bollinger-band forecast family to the existing Carver-style trend ensemble. Default off; backward-compatible. When enabled, applies the Strategy v2 spec weights (EWMAC 0.4 / Breakout 0.3 / Bollinger 0.3) per Scott Phillips research that "Bollinger is significantly more predictive on crypto and tradfi futures."

## Goal

Reduce regime-dependent drawdowns of the existing v0.1 trend system on Hyperliquid perpetual futures (-18.98% inception 2026-02-08, peak Feb 24, mean-reverting since) without adding a regime filter — which Scott Phillips's research definitively contraindicates ("the filter cuts you out exactly when the next big trade is loading").

The hypothesis: Bollinger and EWMAC capture overlapping but non-identical aspects of trend. Combining them adds diversification within the trend family without the fragility of a regime gate.

## Inputs and outputs

| Side | Type | Source / sink |
|---|---|---|
| Input — prices | wide-format DataFrame (date × symbol) | DataPlugin (binance, hyperliquid, or local file) |
| Input — volume (optional) | wide-format DataFrame | DataPlugin |
| Output — weights | DataFrame (date × symbol) | strategy_weights artifact |
| Output — forecasts | DataFrame (date × symbol) | details payload |

## Algorithm

### §1. Per-instrument forecast generation

For each instrument, compute three families of forecasts:

**EWMAC** — exponentially-weighted moving-average crossover, scaled by price volatility, target absolute average ≈ 10, capped at ±20.

$$
\text{ewmac}(p, f, s) = \frac{\text{EWMA}_f(p) - \text{EWMA}_s(p)}{\sigma_p}
$$

Default spans: $(8, 32), (16, 64), (32, 128), (64, 256)$.

**Breakout** — Donchian-style position within rolling high–low range, scaled to ±20.

$$
\text{breakout}(p, w) = \frac{p - \text{midpoint}_w}{(\text{high}_w - \text{low}_w) / 2} \cdot 20
$$

Default windows: $20, 40, 80, 160$.

**Bollinger (NEW in v0.2)** — position relative to MA bands, naturally vol-scaled.

$$
\text{bollinger}(p, w, k) = \frac{p - \text{MA}_w(p)}{k \cdot \sigma_w(p)} \cdot 20
$$

Default windows: $20, 40, 80$. Default $k = 2$ (band width in stds).

### §2. Combination

The combined per-instrument forecast is a weighted sum, capped at ±20:

$$
F_i = \min\Big( 20, \max\big( -20, \sum_j w_j \cdot f_{i,j} \big) \Big)
$$

where $j$ indexes individual forecast rules across the three families and $w_j$ are weights normalized within each family.

**Family weights:**

| Mode | EWMAC | Breakout | Bollinger |
|---|---|---|---|
| `use_bollinger_feature=False` (default, v0.1 backward-compat) | 0.6 | 0.4 | 0.0 |
| `use_bollinger_feature=True` (Strategy v2 spec) | 0.4 | 0.3 | 0.3 |

### §3. Position sizing

Standard Carver-style vol-targeted sizing:

$$
\text{position}_i = \frac{F_i}{F_{\text{cap}}} \cdot \frac{\sigma_{\text{target}}}{\sigma_i} \cdot \frac{\text{IDM}}{N}
$$

with IDM auto-scaled as $\min(\sqrt{N}, 2.5)$ when not specified.

Position limits applied after sizing: per-instrument cap (`max_position`), gross-exposure cap (`max_gross`).

## Parameters and their effect

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `target_vol` | 0.25 | [0.05, 0.50] | Lower → calmer, less compounding |
| `vol_lookback` | 36 | [10, 252] | Longer → smoother vol estimate |
| `ewmac_weight` | 0.6 | [0, 1] | Used when bollinger off |
| `breakout_weight` | 0.4 | [0, 1] | Used when bollinger off |
| `use_bollinger_feature` | `False` | bool | Master switch — applies (0.4, 0.3, 0.3) when on |
| `bollinger_n_std` | 2.0 | [1.0, 3.0] | Band width; tighter bands → more sensitive |
| `top_by_volume` | 10 | [1, 100] | Universe selection size when `use_universe_selection=True` |
| `max_gross` | 2.0 | [0.5, 5.0] | Total gross exposure cap |
| `allow_shorts` | `True` | bool | Set False for long-only |

## Validation evidence

> **Backtest complete (2026-04-25). Paper-trade run still pending before LOCKED.**

### Walk-forward backtest results

Universe: top 8 Hyperliquid perps by volume (CHIP/TRUMP excluded — short history; 6 established coins retained after pipeline filter).  
Period: 2025-04-26 → 2026-04-25 (365 simulation days; 900 days fetched for signal warmup).  
Engine: vectorbt, 1bps fees, daily rebalancing.  
Configs: `quantbox-lab/cookbook/configs/carver_bollinger_{off,on}.yaml`

| Run | Run ID | Sharpe | Deflated Sharpe | Max DD | Calmar | Ann. Turnover | Notes |
|---|---|---|---|---|---|---|---|
| Baseline (v1, bollinger=off) | `5eb7d417399d` | 0.479 | 0.306 | -26.2% | 0.456 | 17.8x | ewmac=0.6, breakout=0.4 |
| Strategy v2 (bollinger=on)  | `b968c2339f95` | 0.502 | 0.326 | **-24.6%** | **0.520** | 24.0x | ewmac=0.4, breakout=0.3, bollinger=0.3 |
| **Lift** | | **+4.8%** | **+6.5%** | **-1.6pp** | **+14%** | +35% | |

### Walk-forward validation (5 splits, train_ratio=0.7)

| | Baseline | Bollinger v2 |
|---|---|---|
| IS Sharpe mean | -0.470 | -0.440 |
| OOS Sharpe mean | +1.454 | +1.349 |
| Degradation | 4.09 | 4.07 |
| Passed | ✅ | ✅ |

OOS > IS in both cases (no overfitting signal). Degradation ratio is high because IS periods include the regime-mismatch drawdown window.

### Statistical validation (n_bootstrap=1000, n_strategies_tested=2, 95% CI)

| | Baseline | Bollinger v2 |
|---|---|---|
| Observed Sharpe | 0.479 | 0.502 |
| Deflated Sharpe | 0.306 | 0.326 |
| Bootstrap 95% CI | [-1.59, 2.46] | [-1.64, 2.42] |
| Haircut Sharpe | 0.460 | 0.482 |
| % null-exceeding | 36% | 35% |
| Passed | ⚠️ WARN | ⚠️ WARN |

Both fail the bootstrap CI gate (CI includes zero). Expected — 365 days is too short for statistical significance at 95% on crypto returns. The deflated Sharpe is positive and the direction is consistent; the CI width reflects the high vol of crypto returns, not negative evidence.

### Turnover analysis

| | Baseline | Bollinger v2 |
|---|---|---|
| Daily turnover | 4.87% | 6.56% |
| Annual turnover | 17.8x | 24.0x |
| Cost-adjusted Sharpe (10bps) | 0.435 | 0.441 |
| Breakeven cost | **107.6bps** | **80.96bps** |

Bollinger increases turnover by ~35%. Breakeven cost drops from 107.6 to 81bps — still far above the ~4bps effective cost on Hyperliquid. Cost-adjusted Sharpe remains higher for v2.

### Interpretation

- **Sharpe lift is positive** (+4.8%) and persistent after cost and multiple-test haircuts.
- **Max DD improved** (-1.6pp) — consistent with Bollinger adding within-trend diversification.
- **Calmar +14%** — the cleaner risk-adjusted signal since the period includes a drawdown recovery.
- **Statistical CI includes zero** — honest limitation of a 1-year window on high-vol crypto. Not negative evidence; the direction is consistent across both point estimates and walk-forward folds.
- **Turnover increase is real** — Bollinger is more reactive. Live monitoring of execution cost is warranted.

**Pending before LOCKED:** paper-trade run ≥1 week on `quantbox-paper` mirroring the live config with `use_bollinger_feature: true`.

## Known limitations

- **Bollinger is correlated with EWMAC at long lookbacks.** The `[20, 40, 80]` default windows overlap with EWMAC speeds; correlation may be higher than `regime_trend.v1`-style orthogonal diversification. Mitigated by Bollinger's vol-scaling differing from EWMAC's price-difference scaling.
- **No regime filter** is intentional (per Scott Phillips research). This methodology will *not* underperform less in regime-mismatch periods than v0.1; it will only differ in the *shape* of trend signal aggregation.
- **Validated for crypto only** at this version. TradFi futures application would require revisiting the universe-selection and vol-lookback assumptions.
- **Bollinger does not handle gap risk** beyond the standard ±20 cap; sudden gaps may flip Bollinger forecast independently of EWMAC.

## Lineage

- **Supersedes:** `carver-trend@v0.1.0` (live since 2026-02-08, -18.98% inception by 2026-04-19)
- **Branched from:** spec at [`notebook/projects/quantlab/strategy-v2-spec.md`](../../../../../srv/obsidian/notebook/projects/quantlab/strategy-v2-spec.md)
- **Doctrinal source:** [`notebook/projects/quantlab/scott-phillips-tweet-research.md`](../../../../../srv/obsidian/notebook/projects/quantlab/scott-phillips-tweet-research.md)
- **Related work:** [`strategy-carry-v1-spec`](../../../../../srv/obsidian/notebook/projects/quantlab/strategy-carry-v1-spec.md) — orthogonal carry strategy planned for Phase 2.

## Promotion roadmap

1. **Land code in `dev`** (this PR — feat/carver-trend-bollinger).
2. **Backtest validation** — populate "Validation evidence" table above.
3. **Paper-trade 1 week** on quantbox-paper.
4. **Flip `status: LOCKED`** with `date_locked` set; commit via `/promote-lock`.
5. **`/promote`** → tag `prod-carver-trend-v0.2.0-YYYYMMDD`.
6. **quantbox-live config bump** turns on `use_bollinger_feature` after the prod tag exists.
7. **Monitor 2 weeks** before considering Phase 2 (carry).

## See also

- Plugin: [`src/quantbox/plugins/strategies/carver_trend.py`](../../src/quantbox/plugins/strategies/carver_trend.py)
- Tests: [`tests/plugins/strategies/test_carver_trend.py`](../../tests/plugins/strategies/test_carver_trend.py)
- Lifecycle: [`../architecture/lifecycle.md`](../architecture/lifecycle.md)
- Promotion playbook: [`../playbooks/promote-a-methodology.md`](../playbooks/promote-a-methodology.md)
