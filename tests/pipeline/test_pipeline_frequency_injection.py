"""Tests for pipeline-level Frequency resolution + annualize injection (issue #20)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quantbox.frequency import Frequency
from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.plugins.strategies import CrossAssetMomentumStrategy

# ---------------------------------------------------------------------------
# _resolve_frequency
# ---------------------------------------------------------------------------


def test_resolve_frequency_default_preserves_pre_pr_b_crypto_behaviour():
    """Empty config → 1d 24/7 → 365 bars/yr, preserving the old implicit default."""
    f = BacktestPipeline._resolve_frequency({}, {})
    assert f == Frequency(pd.Timedelta("1d"), "24/7")
    assert f.bars_per_year() == 365.0


def test_resolve_frequency_explicit_frequency_dict_wins():
    f = BacktestPipeline._resolve_frequency({"frequency": {"bar_size": "1h", "calendar": "NYSE"}}, {})
    assert f.bar_size == pd.Timedelta("1h")
    assert f.calendar == "NYSE"


def test_resolve_frequency_explicit_string_uses_24_7_default():
    f = BacktestPipeline._resolve_frequency({"frequency": "4h"}, {})
    assert f.bar_size == pd.Timedelta("4h")
    assert f.calendar == "24/7"


def test_resolve_frequency_shorthand_combines_prices_and_market_calendar():
    """Legacy shorthand: prices.frequency for bar_size, top-level market_calendar."""
    f = BacktestPipeline._resolve_frequency({"market_calendar": "NYSE"}, {"frequency": "1h"})
    assert f.bar_size == pd.Timedelta("1h")
    assert f.calendar == "NYSE"


def test_resolve_frequency_explicit_frequency_overrides_prices_frequency():
    """If both are set, explicit `frequency:` wins."""
    f = BacktestPipeline._resolve_frequency(
        {"frequency": "1d", "market_calendar": "NYSE"},
        {"frequency": "1h"},  # ignored
    )
    assert f.bar_size == pd.Timedelta("1d")


# ---------------------------------------------------------------------------
# Drift warning when pipeline-derived vs explicit values disagree
# ---------------------------------------------------------------------------


def _make_prices(n: int = 400) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for t in ["BTC", "ETH", "BNB", "XRP", "USDT"]},
        index=dates,
    )


def test_strategy_default_annualize_is_now_none():
    """Pipeline-injection compatibility check: default must be None, not 252.0."""
    assert CrossAssetMomentumStrategy().annualize is None


def test_pipeline_injected_annualize_used_when_strategy_default(caplog):
    """Strategy with annualize=None + _pipeline_annualize=365.0 → uses 365.0, no warning."""
    s = CrossAssetMomentumStrategy()
    prices = _make_prices()
    with caplog.at_level(logging.WARNING, logger="quantbox.plugins.strategies.cross_asset_momentum"):
        result = s.run({"prices": prices}, {"_pipeline_annualize": 365.0})
    assert "overrides pipeline-derived" not in caplog.text
    assert "weights" in result


def test_explicit_annualize_disagreeing_with_injected_logs_warning(caplog):
    """annualize=252 + _pipeline_annualize=365.0 → warning fires (no exception)."""
    s = CrossAssetMomentumStrategy(annualize=252.0)
    prices = _make_prices()
    with caplog.at_level(logging.WARNING, logger="quantbox.plugins.strategies.cross_asset_momentum"):
        result = s.run({"prices": prices}, {"_pipeline_annualize": 365.0})
    assert "overrides pipeline-derived" in caplog.text
    assert "252" in caplog.text
    assert "365" in caplog.text
    assert "weights" in result  # still runs successfully


def test_explicit_annualize_matching_injected_no_warning(caplog):
    """annualize=365 + _pipeline_annualize=365.0 → no warning."""
    s = CrossAssetMomentumStrategy(annualize=365.0)
    prices = _make_prices()
    with caplog.at_level(logging.WARNING, logger="quantbox.plugins.strategies.cross_asset_momentum"):
        s.run({"prices": prices}, {"_pipeline_annualize": 365.0})
    assert "overrides pipeline-derived" not in caplog.text


def test_strategy_without_pipeline_injection_falls_back_to_252():
    """Backwards-compat: when called outside the pipeline, default is 252.0."""
    from quantbox.plugins.strategies.cross_asset_momentum import compute_ewma_volatility

    s = CrossAssetMomentumStrategy()
    assert s.annualize is None
    prices = _make_prices()
    s.run({"prices": prices}, {})  # no _pipeline_annualize
    # Function-level default also confirms the new 252.0 baseline
    import inspect

    assert inspect.signature(compute_ewma_volatility).parameters["annualize"].default == 252.0
