"""The research statistics live in `quantbox.metrics` (TOM-1596).

One known answer per statistic, on a fixture small enough to check by hand:
Newey-West t-stat of the mean, IC and ICIR, beta, hit rate, active share and
risk decomposition. robo-lab carried its own copies (TOM-1453, robo-lab #11);
where a definition differs from the lab's, the test names the difference.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantbox import inference, metrics

# ---------------------------------------------------------------------------
# Newey-West t-stat of the mean: ONE implementation for metrics and gates
# ---------------------------------------------------------------------------


def test_newey_west_tstat_known_answer_lag_1():
    """r = [1, 3, -1, 5]%: mean 2%, residuals u = [-1, 1, -3, 3]%.

    gamma0 = sum(u^2)/n = 5e-4; gamma1 = sum(u_t u_{t-1})/n = -13e-4/4.
    Bartlett weight at lag 1 of 1: 1 - 1/2. Long-run variance
    S = gamma0 + 2 * 0.5 * gamma1 = 1.75e-4, var(mean) = S / n.
    """
    r = np.array([0.01, 0.03, -0.01, 0.05])
    out = inference.newey_west_tstat(r, lags=1)
    se = math.sqrt(1.75e-4 / 4)
    assert out["nw_lags"] == 1
    assert out["mean_return"] == pytest.approx(0.02, rel=1e-12)
    assert out["nw_se"] == pytest.approx(se, rel=1e-10)
    assert out["nw_tstat"] == pytest.approx(0.02 / se, rel=1e-10)


def test_newey_west_tstat_lag_0_is_the_iid_tstat_with_population_std():
    r = np.array([0.01, 0.03, -0.01, 0.05])
    expected = r.mean() / (r.std(ddof=0) / math.sqrt(len(r)))
    assert inference.newey_west_tstat(r, lags=0)["nw_tstat"] == pytest.approx(expected, rel=1e-10)


def test_newey_west_matches_robo_labs_bartlett_sandwich():
    """robo-lab's `active_etfs._ols_nw` (fixed 3 lags, no small-sample correction) on a constant regressor."""
    rng = np.random.default_rng(3)
    y = rng.normal(0.001, 0.01, 120)
    lags = 3
    x = np.ones((len(y), 1))
    xtx_inv = np.linalg.pinv(x.T @ x)
    b = xtx_inv @ x.T @ y
    u = y - x @ b
    s = (x * u[:, None]).T @ (x * u[:, None])
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = (x[lag:] * u[lag:, None]).T @ (x[:-lag] * u[:-lag, None])
        s += w * (g + g.T)
    se = float(np.sqrt((xtx_inv @ s @ xtx_inv)[0, 0]))
    out = inference.newey_west_tstat(y, lags=lags)
    assert out["nw_se"] == pytest.approx(se, rel=1e-10)
    assert out["nw_tstat"] == pytest.approx(b[0] / se, rel=1e-10)


# ---------------------------------------------------------------------------
# IC and ICIR
# ---------------------------------------------------------------------------


@pytest.fixture
def ic_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-01-31", periods=4, freq="ME")
    signal = pd.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, np.nan, np.nan, np.nan]],
        index=idx,
        columns=list("ABCD"),
        dtype=float,
    )
    fwd = pd.DataFrame(
        [
            [0.1, 0.2, 0.4, 0.3],  # ranks 1 2 4 3 -> 1 - 6*2/60 = 0.8
            [0.4, 0.3, 0.2, 0.1],  # reversed -> -1
            [0.1, 0.3, 0.2, 0.4],  # ranks 1 3 2 4 -> 0.8
            [0.1, 0.2, 0.3, 0.4],  # one asset with a signal: no IC that date
        ],
        index=idx,
        columns=list("ABCD"),
    )
    return signal, fwd


def test_information_coefficient_is_the_per_date_spearman(ic_fixture):
    signal, fwd = ic_fixture
    ic = metrics.information_coefficient(signal, fwd)
    assert list(ic.index) == list(signal.index[:3])  # the one-asset date has no IC
    assert ic.tolist() == pytest.approx([0.8, -1.0, 0.8], abs=1e-12)


def test_information_coefficient_min_assets(ic_fixture):
    signal, fwd = ic_fixture
    assert metrics.information_coefficient(signal, fwd, min_assets=5).empty


def test_information_coefficient_ties_take_average_ranks():
    """Same as scipy.stats.spearmanr / pandas' spearman: tied values share their mean rank."""
    from scipy.stats import spearmanr

    idx = pd.date_range("2024-01-31", periods=1, freq="ME")
    s = pd.DataFrame([[1.0, 1.0, 2.0, 3.0, 5.0]], index=idx)
    f = pd.DataFrame([[0.3, 0.1, 0.1, 0.2, 0.5]], index=idx)
    expected = spearmanr(s.iloc[0], f.iloc[0]).statistic
    assert metrics.information_coefficient(s, f).iloc[0] == pytest.approx(expected, rel=1e-12)


def test_icir_is_mean_over_std_unannualised(ic_fixture):
    """IC = [0.8, -1, 0.8]: mean 0.2, std (ddof=1) sqrt(2.16 / 2)."""
    ic = metrics.information_coefficient(*ic_fixture)
    assert metrics.icir(ic) == pytest.approx(0.2 / math.sqrt(1.08), rel=1e-12)
    assert metrics.hit_rate(ic) == pytest.approx(2 / 3)


def test_icir_degenerate_is_nan():
    assert math.isnan(metrics.icir(pd.Series([0.1])))
    assert math.isnan(metrics.icir(pd.Series([0.1, 0.1, 0.1])))


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------


def test_beta_known_answer():
    b = pd.Series([0.01, 0.02, -0.01, 0.0, 0.03])
    r = 1.5 * b + 0.001
    assert metrics.beta(r, b) == pytest.approx(1.5, rel=1e-12)


def test_beta_aligns_by_date_and_drops_missing_rows():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    b = pd.Series([0.01, 0.02, -0.01, 0.0, 0.03], index=idx)
    r = (2.0 * b).where(b != 0.0)  # one NaN row
    assert metrics.beta(r.iloc[1:], b) == pytest.approx(2.0, rel=1e-12)


def test_beta_degenerate_is_nan():
    assert math.isnan(metrics.beta(pd.Series([0.01, 0.02]), pd.Series([0.01, 0.01])))
    assert math.isnan(metrics.beta(pd.Series([0.01]), pd.Series([0.02])))


def test_benchmark_validation_beta_is_metrics_beta():
    from quantbox.plugins.validation.benchmark import BenchmarkValidation

    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    r = pd.DataFrame({"returns": rng.normal(0, 0.01, 50)}, index=idx)
    b = pd.DataFrame({"benchmark": rng.normal(0, 0.01, 50)}, index=idx)
    out = BenchmarkValidation().validate(r, pd.DataFrame(), b, {})
    assert out["metrics"]["beta"] == pytest.approx(metrics.beta(r["returns"], b["benchmark"]), rel=1e-12)


# ---------------------------------------------------------------------------
# hit rate
# ---------------------------------------------------------------------------


def test_hit_rate_counts_strictly_positive_and_skips_missing():
    r = pd.Series([0.01, -0.02, 0.0, 0.03, np.nan])
    assert metrics.hit_rate(r) == pytest.approx(0.5)  # 2 of 4; a flat bar is not a hit


def test_hit_rate_vs_benchmark():
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    r = pd.Series([0.01, 0.02, 0.03], index=idx)
    b = pd.Series([0.0, 0.03, 0.01], index=idx)
    assert metrics.hit_rate(r, b) == pytest.approx(2 / 3)


def test_hit_rate_empty_is_nan():
    assert math.isnan(metrics.hit_rate(pd.Series([], dtype=float)))


def test_win_rate_in_the_run_metrics_is_hit_rate():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0, 0.01, 60), index=pd.date_range("2024-01-01", periods=60, freq="D"))
    assert metrics.compute_backtest_metrics(r)["win_rate"] == metrics.hit_rate(r)


# ---------------------------------------------------------------------------
# active share
# ---------------------------------------------------------------------------


def test_active_share_known_answer_vs_a_static_reference():
    """Date 1: 0.5 * (0.1 + 0.15 + 0.25) = 0.25; date 2 equals the reference: 0. Mean 0.125."""
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    w = pd.DataFrame({"A": [0.6, 0.5], "B": [0.4, 0.25], "C": [0.0, 0.25]}, index=idx)
    ref = {"A": 0.5, "B": 0.25, "C": 0.25}
    assert metrics.active_share(w, ref) == pytest.approx(0.125)
    assert metrics.active_share(w, pd.Series(ref)) == pytest.approx(0.125)


def test_active_share_vs_a_reference_book_counts_names_held_by_one_side():
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    w = pd.DataFrame({"A": [1.0, 0.5]}, index=idx)
    ref = pd.DataFrame({"B": [1.0, 0.5], "A": [np.nan, 0.5]}, index=idx)
    # date 1: |1 - 0| + |0 - 1| -> 1.0; date 2: 0 + 0.5 -> 0.25
    assert metrics.active_share(w, ref) == pytest.approx((1.0 + 0.25) / 2)


def test_active_share_skips_dates_either_book_is_empty():
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    w = pd.DataFrame({"A": [np.nan, 1.0]}, index=idx)
    assert metrics.active_share(w, {"A": 0.5, "B": 0.5}) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# risk decomposition
# ---------------------------------------------------------------------------


def test_risk_contributions_known_answer_from_a_covariance():
    """Uncorrelated, vols 20% and 10%, 50/50: variances 0.01 and 0.0025 of 0.0125."""
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    rc = metrics.risk_contributions(pd.Series({"A": 0.5, "B": 0.5}), cov=cov)
    assert rc.to_dict() == pytest.approx({"A": 0.8, "B": 0.2})


def test_risk_contributions_from_returns_match_robo_labs_formula():
    """robo-lab strategy_map: w * cov(r_i, r_p) / var(r_p), on the rows where every sleeve has a return."""
    rng = np.random.default_rng(9)
    r = pd.DataFrame(rng.normal(0, 0.01, (200, 3)), columns=["x", "y", "z"])
    r["y"] += 0.5 * r["x"]
    r.iloc[5, 2] = np.nan
    w = pd.Series({"x": 0.5, "y": 0.3, "z": 0.2})
    rd = r.dropna()
    acct = rd @ w.values
    lab = w * rd.apply(lambda col: col.cov(acct)) / acct.var()
    rc = metrics.risk_contributions(w, r)
    assert rc.to_dict() == pytest.approx(lab.to_dict(), rel=1e-12)
    assert rc.sum() == pytest.approx(1.0, rel=1e-12)


def test_risk_contributions_needs_exactly_one_input():
    w = pd.Series({"A": 1.0})
    with pytest.raises(ValueError):
        metrics.risk_contributions(w)
    with pytest.raises(ValueError):
        metrics.risk_contributions(w, pd.DataFrame({"A": [0.1, 0.2]}), cov=pd.DataFrame([[1.0]]))
