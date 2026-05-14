"""Tests for the per-variant report layout in generate_report_data().

Covers the multi-variant branch added to backtest.pipeline.v1's report builder:
- variant_metrics is populated with each variant's metrics + strategy name
- framework charts are namespaced as <variant>__<chart_name>
- single-asset variants don't emit a contrib bar (1-bar = uninformative)
- single-asset variants don't emit a position_stack (1-line = uninformative)
- multi-variant runs skip the unnamespaced top-level framework charts
- single-strategy runs (legacy path) keep the unnamespaced top-level charts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.pipeline._report import generate_report_data


def _make_variant(
    name: str,
    *,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    strategy_name: str,
    seed: int,
) -> dict:
    """Build a synthetic variant_results entry."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    n_sym = len(symbols)

    # Synthetic close prices: random walk per symbol, start at 100.
    rets = rng.normal(0.0005, 0.02, size=(n, n_sym))
    closes = 100 * np.cumprod(1 + rets, axis=0)
    bt_prices = pd.DataFrame(closes, index=dates, columns=symbols)

    # Weights: if single-asset (BAH-like), 100% in symbols[0]; else equal weight.
    if len(symbols) == 1:
        weights = pd.DataFrame(1.0, index=dates, columns=symbols)
    else:
        weights = pd.DataFrame(
            np.tile([1.0 / n_sym], (n, n_sym)),
            index=dates,
            columns=symbols,
        )

    # Daily returns from a static portfolio of the (synthetic) underlying.
    port_rets = (weights.shift(1) * bt_prices.pct_change()).sum(axis=1)
    port_rets.iloc[0] = 0.0
    nav = 100 * (1 + port_rets).cumprod()
    portfolio_daily = nav.to_frame("portfolio_value")
    portfolio_daily.index.name = "date"
    portfolio_daily = portfolio_daily.reset_index()

    metrics = {
        "total_return": float(nav.iloc[-1] / 100 - 1),
        "cagr": 0.10,
        "sharpe": 1.0,
        "sortino": 1.2,
        "max_drawdown": -0.20,
        "max_drawdown_duration_days": 30,
        "annual_volatility": 0.30,
        "calmar": 0.5,
        "win_rate": 0.55,
        "profit_factor": 1.3,
        "var_95": -0.04,
        "cvar_95": -0.07,
    }

    return {
        "name": name,
        "strategy_name": strategy_name,
        "returns": port_rets,
        "metrics": metrics,
        "portfolio_daily": portfolio_daily,
        "vbt_portfolio": None,
        "weights_history": weights,
        "bt_prices": bt_prices,
        "strategy_details": {strategy_name: {}},
        "config": {"strategy_params": {}, "fees": 0.005, "rebalancing_freq": "1D"},
    }


@pytest.fixture
def two_variant_results() -> dict:
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    bah = _make_variant(
        "BTC_BAH",
        dates=dates,
        symbols=["BTC"],
        strategy_name="strategy.static_weights.v1",
        seed=1,
    )
    trend = _make_variant(
        "Trend_long_EW",
        dates=dates,
        symbols=["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOT", "MATIC", "LINK"],
        strategy_name="strategy.trend_catcher.v1",
        seed=2,
    )
    return {"BTC_BAH": bah, "Trend_long_EW": trend}


def _common_kwargs(variant_results: dict) -> dict:
    """Top-level args mirror what _run_variants_flow passes (primary = first)."""
    primary = next(iter(variant_results.values()))
    return dict(
        run_id="test_run",
        asof="2026-05-14",
        metrics=primary["metrics"],
        portfolio_daily=primary["portfolio_daily"],
        returns=primary["returns"],
        weights_history=primary["weights_history"],
        bt_prices=primary["bt_prices"],
        strategy_names=list(variant_results.keys()),
        period_start="2020-01-01",
        period_end="2020-07-18",
        strategy_details={v["name"]: list(v["strategy_details"].values())[0] for v in variant_results.values()},
        variant_results=variant_results,
    )


class TestMultiVariantReport:
    def test_variant_metrics_populated_per_variant(self, two_variant_results):
        rd = generate_report_data(**_common_kwargs(two_variant_results))
        vm = rd.get("variant_metrics") or {}
        assert set(vm.keys()) == {"BTC_BAH", "Trend_long_EW"}
        for name, m in vm.items():
            assert "total_return" in m
            assert "sharpe" in m
            assert "max_drawdown" in m
            assert m.get("_strategy_name"), f"{name} missing _strategy_name"
            assert m.get("_fees") == 0.005

    def test_framework_charts_are_namespaced(self, two_variant_results):
        rd = generate_report_data(**_common_kwargs(two_variant_results))
        charts = rd["charts"]
        for vname in ("BTC_BAH", "Trend_long_EW"):
            assert f"{vname}__portfolio" in charts, f"missing {vname}__portfolio"
            assert f"{vname}__monthly" in charts, f"missing {vname}__monthly"
            assert f"{vname}__weights" in charts, f"missing {vname}__weights"

    def test_multi_variant_skips_unnamespaced_top_level_charts(self, two_variant_results):
        rd = generate_report_data(**_common_kwargs(two_variant_results))
        charts = rd["charts"]
        # The top-level framework charts must not exist in multi-variant mode —
        # each variant has its own namespaced copy instead.
        for k in ("portfolio", "monthly", "contrib", "weights", "position_stack"):
            assert k not in charts, f"unexpected unnamespaced {k!r} chart in multi-variant report"

    def test_single_asset_variant_omits_contrib_and_position_stack(self, two_variant_results):
        # BTC_BAH only holds BTC → contrib and position_stack are uninformative.
        rd = generate_report_data(**_common_kwargs(two_variant_results))
        charts = rd["charts"]
        assert "BTC_BAH__contrib" not in charts, "single-asset variant should not emit a contrib chart"
        assert "BTC_BAH__position_stack" not in charts, "single-asset variant should not emit a position_stack chart"

    def test_multi_asset_variant_emits_contrib_and_position_stack(self, two_variant_results):
        rd = generate_report_data(**_common_kwargs(two_variant_results))
        charts = rd["charts"]
        assert "Trend_long_EW__contrib" in charts
        assert "Trend_long_EW__position_stack" in charts


class TestSingleStrategyReportUnchanged:
    """Legacy single-strategy flow must keep the unnamespaced top-level charts."""

    def test_single_strategy_emits_top_level_framework_charts(self, two_variant_results):
        # Drop to a single variant — exercises the legacy code path.
        single = {"Trend_long_EW": two_variant_results["Trend_long_EW"]}
        rd = generate_report_data(**_common_kwargs(single))
        charts = rd["charts"]
        # Single-strategy flow: unnamespaced framework charts at top level.
        assert "portfolio" in charts
        assert "monthly" in charts
        assert "weights" in charts
        # variant_metrics is empty in single-strategy flow.
        assert rd.get("variant_metrics") == {}
