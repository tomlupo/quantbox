---
methodology: carry
version: v0.1.0
status: DRAFT
date_locked:
supersedes:
superseded_by:
src_paths:
  - packages/quantbox-core/src/quantbox/plugins/strategies/carry.py
  - packages/quantbox-core/src/quantbox/plugins/datasources/hyperliquid_data_plugin.py
seeds:
  numpy: 42
revalidation:
  cadence: monthly
  baseline_metrics:
    sharpe_oos_min: 0.3
    drawdown_max: -0.30
  on_failure: alert_only
---

# Carry v1 — Funding-Rate Momentum on Hyperliquid Perps

> Directional carry strategy for Hyperliquid perpetual futures. Goes long the assets paying the highest annualized funding (market is bullish, paying to be long) and short the assets with the most negative funding (market is bearish, paying to be short). Vol-targeted at 20%. Designed as the orthogonal diversifier alongside `strategy.carver_trend.v1` in a 70/30 trend+carry portfolio.

## Goal

Exploit the persistency of funding rate regimes: when a perp pays high funding, momentum and crowd positioning tend to sustain it for days to weeks. The signal is not purely mechanical carry (collecting funding directly) — it is directional carry momentum, a bet on which way the crowd is leaning.

Success = low correlation to trend signals while contributing consistent positive returns via funding income and directional P&L. Per Scott Phillips's doctrine: trend (~1.7 Sharpe) + carry (~1.7 Sharpe) = combined ~2.0 Sharpe due to correlation < 1.

## Inputs and outputs

| Side | Type | Source |
|---|---|---|
| Input — prices | wide-format DataFrame (date × ticker) | `hyperliquid.data.v1` or `binance.futures_data.v1` |
| Input — funding_rates | wide-format DataFrame (date × ticker, daily sum of 3×8h rates) | same data plugin |
| Output — weights | wide-format DataFrame (date × ticker, signed weights) | `strategy.carry.v1` → `strategy.weighted_avg.v1` |

## Algorithm

### §1. Funding signal construction

Funding rates arrive as daily sums of the three 8-hour funding payments. Apply a 3-day EWM to smooth noise:

```
smoothed[t] = EWM(funding[t], span=3)
annualized[t] = smoothed[t] * 365
```

The annualized signal ranges from roughly -100% to +200% in extreme regimes.

### §2. Rank and select

On each date:
1. Sort all assets by `annualized` descending.
2. **Long leg**: take the top `top_n_long` assets (default: 3). Filter: only include if `annualized >= min_signal_annualized` (default: 5%).
3. **Short leg**: take the bottom `top_n_short` assets (default: 3). Filter: only include if `annualized <= -min_signal_annualized`.
4. Assign equal weight per leg: `w = min(0.5 / n, max_concentration)`. Each leg targets 50% gross exposure. If a leg is empty (no assets pass the filter), that leg is flat.

### §3. Vol targeting

Scale the full position book so realized portfolio volatility tracks `target_vol` (default: 20%):

```
port_rets[t] = sum(weights[t-1] * returns[t])
realized_vol[t] = EWM(port_rets, span=vol_lookback).std() * sqrt(365)
scale[t] = clip(target_vol / realized_vol[t], 0.1, 3.0)
final_weights[t] = weights[t] * scale[t]
```

## Parameters and their effect

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `signal_span_days` | 3 | [1, 30] | Smoothing window; larger → slower to react to regime shifts |
| `top_n_long` | 3 | [1, 10] | Long positions; more → diversified but weaker signal |
| `top_n_short` | 3 | [1, 10] | Short positions; symmetric with long |
| `min_signal_annualized` | 0.05 | [0.0, 1.0] | Noise floor; 5% annual prevents tiny-signal trades |
| `target_vol` | 0.20 | [0.05, 0.50] | Lower than trend (0.50); carry is income-like, steadier |
| `max_concentration` | 0.20 | [0.05, 0.50] | Per-position cap; prevents crowded short trap |
| `vol_lookback` | 20 | [5, 60] | Rolling vol window for scale estimator |

## Validation evidence

All runs use the `backtest.pipeline.v1` engine with `fees: 0.001`, daily rebalancing.

**Hyperliquid top-10 universe (1-year window, 2025-04 → 2026-04):**

| Run | Config | Assets | Dates | Sharpe | WF OOS Sharpe | Max DD | Notes |
|---|---|---|---|---|---|---|---|
| `2026-04-25__backtest_pipeline_v1__73f58c697104` | mega_tc_tight | 10 | 366 | 0.88 | 1.63 | -17.0% | Trend 70% + Carry 30% |
| `2026-04-25__backtest_pipeline_v1__60d7f5bff4f5` | mega_trend_carry | 20 | 200 | 0.69 | 0.09 | -15.8% | Same, wider universe |

**Binance USDM perpetuals — 5-year window (2021-2026):**

| Run | Config | Assets | Dates | Sharpe | WF OOS Sharpe | Deflated | Max DD | Notes |
|---|---|---|---|---|---|---|---|---|
| `2026-04-25__backtest_pipeline_v1__6d38847af81d` | mega_5yr_binance | 10 | 2000 | 0.67 | 0.88 | 0.62 | -52.5% | 7yr lookback, proper warmup |

**Annual Sharpe breakdown (Binance 5yr run):**

| Year | Sharpe | Return |
|---|---|---|
| 2020 | 4.69 | +38.0% |
| 2021 | 2.20 | +88.0% |
| 2022 | 0.10 | -5.7% |
| 2023 | 0.48 | +10.6% |
| 2024 | 0.88 | +29.1% |
| 2025 | -0.20 | -13.2% |
| 2026 | -0.37 | -10.1% |

**Standalone carry contribution:** Trend-only baseline Sharpe ~0.50 on the HL 1yr window. Adding carry (30% weight) raised portfolio Sharpe to 0.88 (+76% improvement). The max drawdown also improved (-25% → -17%) due to carry providing a partially uncorrelated signal.

**Current market regime (2026-04):** All major perps have positive funding. Carry is fully long-only (no assets below the -5% threshold for shorts). This reduces diversification vs a mixed-regime environment.

## Known limitations

- **Regime dependency**: When all assets have positive funding (bull regime), carry adds no diversification — it becomes a long-only overlay. Currently the case as of 2026-04.
- **Single-venue**: Mode A only (directional). Mode B (delta-neutral basis carry: long spot / short perp) is deferred — requires spot venue routing.
- **Crowded short trap**: High positive funding → crowded longs → squeeze risk. Mitigated by `max_concentration=0.20` but not eliminated.
- **Funding API pagination**: Hyperliquid `fundingHistory` endpoint returns ~20 days per call. The data plugin now paginates (fixed 2026-04-25) but each full-year fetch requires ~18 API calls per ticker.
- **Backtest warmup**: This strategy requires prices to compute vol targeting. The first `vol_lookback` dates have `scale=1` (no scaling). Use `lookback_days` >> `output_periods` (at least 500 extra days) to ensure the vol estimator is warmed up.
- **2025-2026 underperformance**: Choppy, mean-reverting crypto regime. The strategy lost -13% in 2025. Not unusual for trend/carry strategies in choppy markets — revalidate monthly.

## Lineage

- Specification: `notebook/projects/quantlab/strategy-carry-v1-spec.md`
- Doctrine: Scott Phillips funding-momentum approach (3-day EMA, equal weight, vol target)
- Related: `carry-in-crypto`, `signal-stacking-in-tactical-allocation` (Atlas notes)
- Implementation date: 2026-04-25

## Portfolio integration

Best combined with `strategy.carver_trend.v1`:

```yaml
strategies:
  - name: "strategy.carver_trend.v1"
    weight: 0.70
    params:
      target_vol: 0.50
      use_bollinger_feature: true
      top_by_volume: 8
      output_periods: 365
  - name: "strategy.carry.v1"
    weight: 0.30
    params:
      target_vol: 0.20
      output_periods: 365
```

Use `mega_tc_tight.yaml` in quantbox-lab as the reference config.

## See also

- Plugin: `packages/quantbox-core/src/quantbox/plugins/strategies/carry.py`
- Reference configs: `quantbox-lab/config/mega_tc_tight.yaml`, `mega_5yr_binance.yaml`
- Lifecycle: `docs/architecture/lifecycle.md`
