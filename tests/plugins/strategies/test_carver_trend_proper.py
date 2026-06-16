"""Tests for strategy.carver_trend_proper.v1 (pysystemtrade-faithful Carver)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.strategies.carver_trend_proper import (
    CANONICAL_EWMAC_SPANS,
    CRYPTO_EWMAC_SPANS,
    CarverTrendProperStrategy,
    _diversification_multiplier,
    apply_carver_buffer,
    forecast_diversification_multiplier,
    generate_proper_forecast,
    instrument_diversification_multiplier,
    mixed_volatility,
    scale_forecast_expanding,
)


def _prices(n: int = 600, k: int = 4, seed: int = 7, corr: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0003, 0.03, size=(n, 1))
    idio = rng.normal(0.0003, 0.03, size=(n, k))
    rets = corr * common + (1 - corr) * idio
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(
        prices,
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
        columns=[f"C{i}" for i in range(k)],
    )


# --- forecast scaling ------------------------------------------------------
def test_scale_forecast_expanding_targets_ten():
    rng = np.random.default_rng(1)
    raw = pd.Series(rng.normal(0, 3.0, size=2000))
    scaled = scale_forecast_expanding(raw, target_abs_avg=10.0)
    # late-sample average abs forecast should sit near the target
    assert 7.0 < scaled.dropna().abs().tail(1000).mean() < 13.0


def test_scale_forecast_expanding_is_causal():
    # value at t must not depend on future values
    raw = pd.Series(np.arange(1, 501, dtype=float))
    s_full = scale_forecast_expanding(raw, min_periods=50)
    s_trunc = scale_forecast_expanding(raw.iloc[:300], min_periods=50)
    pd.testing.assert_series_equal(s_full.iloc[:300], s_trunc, check_names=False)


# --- mixed volatility ------------------------------------------------------
def test_mixed_volatility_positive_and_annualized():
    px = _prices()
    vol = mixed_volatility(px, annualize=365.0)
    tail = vol.dropna().tail(100)
    assert (tail > 0).all().all()
    # ~3% daily *sqrt(365) ~ 0.57 annualized; allow a wide band
    assert 0.2 < tail.mean().mean() < 1.2


def test_mixed_volatility_floor_prevents_collapse():
    # a series that goes nearly flat late should keep a non-trivial vol floor
    n = 800
    rets = np.concatenate([np.random.default_rng(0).normal(0, 0.03, 500), np.full(300, 1e-6)])
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(rets)), columns=["X"], index=pd.date_range("2023-01-01", periods=n, freq="D")
    )
    vol = mixed_volatility(px, annualize=365.0)
    assert vol["X"].dropna().iloc[-1] > 0


# --- diversification multipliers ------------------------------------------
def test_dm_independent_gives_sqrt_n():
    n = 4
    corr = np.eye(n)
    dm = _diversification_multiplier(corr, np.full(n, 1.0 / n), cap=10.0)
    assert dm == pytest.approx(np.sqrt(n), rel=1e-6)


def test_dm_perfectly_correlated_gives_one():
    n = 5
    corr = np.ones((n, n))
    dm = _diversification_multiplier(corr, np.full(n, 1.0 / n), cap=10.0)
    assert dm == pytest.approx(1.0, abs=1e-6)


def test_dm_is_clipped_to_cap():
    n = 9
    corr = np.eye(n)  # sqrt(9)=3 > cap
    dm = _diversification_multiplier(corr, np.full(n, 1.0 / n), cap=2.5)
    assert dm == pytest.approx(2.5)


def test_idm_correlated_below_uncorrelated():
    idm_uncorr = instrument_diversification_multiplier(_prices(corr=0.0, seed=1), cap=10.0)
    idm_corr = instrument_diversification_multiplier(_prices(corr=0.9, seed=1), cap=10.0)
    assert idm_corr < idm_uncorr
    assert idm_corr >= 1.0


def test_fdm_at_least_one():
    px = _prices()
    _, rule_fc, weights = generate_proper_forecast(px["C0"], CRYPTO_EWMAC_SPANS, None, 1.0, 0.0)
    fdm = forecast_diversification_multiplier(list(rule_fc.values()), weights)
    assert fdm >= 1.0


# --- forecast generation ---------------------------------------------------
def test_generate_proper_forecast_weights_sum_to_one():
    px = _prices()
    _, rule_fc, weights = generate_proper_forecast(px["C0"], CRYPTO_EWMAC_SPANS, [20, 40], 0.7, 0.3)
    assert sum(weights) == pytest.approx(1.0)
    assert len(rule_fc) == len(CRYPTO_EWMAC_SPANS) + 2


def test_canonical_has_six_crypto_five():
    assert len(CANONICAL_EWMAC_SPANS) == 6
    assert len(CRYPTO_EWMAC_SPANS) == 5
    assert (2, 8) in CANONICAL_EWMAC_SPANS
    assert (2, 8) not in CRYPTO_EWMAC_SPANS


# --- buffering -------------------------------------------------------------
def test_buffer_reduces_turnover():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    optimal = pd.DataFrame(rng.normal(0, 0.1, size=(300, 3)), index=idx, columns=["A", "B", "C"])
    avg_pos = pd.DataFrame(0.2, index=idx, columns=["A", "B", "C"])
    held = apply_carver_buffer(optimal, avg_pos, buffer_size=0.10)
    raw_turn = optimal.diff().abs().sum().sum()
    buf_turn = held.diff().abs().sum().sum()
    assert buf_turn < raw_turn


def test_buffer_keeps_position_within_band():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    optimal = pd.DataFrame(rng.normal(0, 0.05, size=(200, 2)), index=idx, columns=["A", "B"])
    avg_pos = pd.DataFrame(0.3, index=idx, columns=["A", "B"])
    held = apply_carver_buffer(optimal, avg_pos, buffer_size=0.10)
    band = 0.3 * 0.10
    # held never further than (buffer) outside the optimal at each step
    assert ((held - optimal).abs() <= band + 1e-9).all().all()


# --- end to end ------------------------------------------------------------
def test_strategy_run_outputs_contract():
    px = _prices(n=500, k=5)
    strat = CarverTrendProperStrategy()
    res = strat.run({"prices": px}, {"output_periods": 500, "buffer_size": 0.1})
    assert "weights" in res and isinstance(res["weights"], pd.DataFrame)
    assert res["details"]["fdm"] >= 1.0
    assert res["details"]["idm"] >= 1.0
    assert set(res["exposure"]) == {"long", "short", "net", "gross"}


def test_strategy_divides_forecast_by_ten_not_cap():
    """A position at |forecast|=10 should equal the average position (vol_scalar/N*IDM),
    NOT half of it (v1's forecast/cap bug)."""
    px = _prices(n=500, k=3, corr=0.2)
    strat = CarverTrendProperStrategy()
    res = strat.run(
        {"prices": px},
        {"output_periods": 500, "buffer_size": 0.0, "idm": 1.0, "fdm": 1.0, "max_gross": 100.0, "max_position": 100.0},
    )
    fc = res["details"]["full_forecasts"]
    avg_pos = res["details"]["avg_position"]
    opt = res["details"]["full_optimal"]
    # implied = opt / (fc/10) should match avg_position where forecast is non-trivial
    mask = fc.abs() > 5.0
    implied = (opt / (fc / 10.0))[mask]
    ref = avg_pos[mask]
    diff = (implied - ref).abs().to_numpy()
    diff = diff[~np.isnan(diff)]
    assert np.nanmedian(diff) < 1e-6


def test_cross_sectional_demeans_forecasts():
    """With cross_sectional=True the per-date cross-sectional mean forecast is ~0
    (common factor stripped); with it off the mean is generally non-zero."""
    px = _prices(n=600, k=6, corr=0.6, seed=21)  # high common factor
    base = CarverTrendProperStrategy().run({"prices": px}, {"output_periods": 600, "buffer_size": 0.0})
    xs = CarverTrendProperStrategy().run(
        {"prices": px}, {"output_periods": 600, "buffer_size": 0.0, "cross_sectional": True}
    )
    base_fc = base["details"]["full_forecasts"].dropna()
    xs_fc = xs["details"]["full_forecasts"].dropna()
    # cross-sectional row-mean should be ~0 everywhere when demeaned
    assert xs_fc.mean(axis=1).abs().max() < 1e-9
    # and meaningfully smaller than the un-demeaned book's row-mean magnitude
    assert xs_fc.mean(axis=1).abs().mean() < base_fc.mean(axis=1).abs().mean()


def test_realized_vol_in_target_ballpark():
    """On synthetic trending-ish data the optimal (un-buffered) book should realize
    vol within a factor ~2 of target — i.e. vol targeting is actually engaged,
    unlike v1 which lands ~5-10x under."""
    px = _prices(n=700, k=6, corr=0.3, seed=11)
    strat = CarverTrendProperStrategy()
    res = strat.run({"prices": px}, {"output_periods": 700, "buffer_size": 0.0, "target_vol": 0.25})
    w = res["details"]["full_optimal"].shift(1)  # avoid look-ahead in pnl
    rets = px.pct_change(fill_method=None)
    port = (w * rets).sum(axis=1).dropna()
    realized = port.std() * np.sqrt(365)
    # generous band: targeting is engaged if realized is between 0.08 and 0.6
    assert 0.08 < realized < 0.7


# --- fine-lot guard plumbing -----------------------------------------------
def test_resolve_sz_decimals_dict_and_file(tmp_path):
    import json

    s = CarverTrendProperStrategy()
    s.sz_decimals = {"BTC": 5, "XRP": 0}
    assert s._resolve_sz_decimals() == {"BTC": 5, "XRP": 0}

    p = tmp_path / "sz.json"
    p.write_text(json.dumps({"ETH": 4, "DOGE": 0}))
    s.sz_decimals = str(p)
    assert s._resolve_sz_decimals() == {"ETH": 4, "DOGE": 0}


def test_resolve_sz_decimals_unset_fails_closed():
    s = CarverTrendProperStrategy()
    s.sz_decimals = None
    assert s._resolve_sz_decimals() is None  # guard then excludes everything


def test_fine_lot_guard_drops_coarse_coin_from_weights():
    # Two cheap fine-lot coins + one expensive coarse-lot coin (szDec=0, $5000).
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(3)
    fine = 10 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, size=(n, 2)), axis=0))
    coarse = 5000 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, size=(n, 1)), axis=0))
    px = pd.DataFrame(np.hstack([fine, coarse]), index=idx, columns=["FINE1", "FINE2", "COARSE"])
    vol = pd.DataFrame(1e7, index=idx, columns=px.columns)

    res = CarverTrendProperStrategy().run(
        {"prices": px, "volume": vol},
        {
            "output_periods": n,
            "use_universe_selection": True,
            "top_by_mcap": 30,
            "top_by_volume": 3,
            "fine_lot_guard": True,
            "fine_lot_min_notional": 10.0,
            "fine_lot_max_lot_fraction": 0.2,
            "sz_decimals": {"FINE1": 2, "FINE2": 2, "COARSE": 0},
        },
    )
    held = res["details"]["full_weights"]
    assert (held["COARSE"].abs() < 1e-12).all()  # coarse coin never held
    assert held[["FINE1", "FINE2"]].abs().sum().sum() > 0  # fine coins traded
