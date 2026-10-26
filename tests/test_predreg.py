"""The predictive-regression test (TOM-2411): Stambaugh's bias, Lewellen's conditional test, the ``predreg`` gate.

Three kinds of evidence, each named by what it reads:

- a property test on simulated AR(1) predictors with a KNOWN slope (seeded), where
  the OLS bias, the bias the correction removes and the size of each test are known;
- a reference check against a PUBLISHED number, Lewellen (2004) JFE 74(2) Table 2
  (log D/P, NYSE value-weighted, 1946-2000), on the SBBI large-cap series;
- the refusals and the CLI contract shared with the other gates.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quantbox import gates as g
from quantbox import inference
from quantbox.cli import app

# ── simulation ─────────────────────────────────────────────────────────────


def _simulate(rng, T, beta, rho, corr, sigma_e=1.0, sigma_u=0.05):
    """``(r, x)`` aligned: r[t] = beta * x[t-1] + e[t], x[t] = rho * x[t-1] + u[t], corr(e, u) = corr."""
    e = rng.normal(size=T + 1)
    u = corr * e + np.sqrt(1 - corr**2) * rng.normal(size=T + 1)
    e, u = sigma_e * e, sigma_u * u
    x = np.empty(T + 1)
    x[0] = rng.normal(scale=sigma_u / np.sqrt(1 - rho**2))  # a draw from the stationary distribution
    for t in range(1, T + 1):
        x[t] = rho * x[t - 1] + u[t]
    r = np.empty(T + 1)
    r[0] = 0.0  # carries no pair: the first return has no lagged predictor
    r[1:] = beta * x[:-1] + e[1:]
    return r, x


T, RHO, CORR = 240, 0.98, -0.9
GAMMA = CORR * 1.0 / 0.05  # cov(e, u) / var(u) = -18
STAMBAUGH_BIAS = GAMMA * -(1 + 3 * RHO) / T  # Stambaugh (1999): gamma * E[rho_hat - rho], about +0.30


def _monte_carlo(beta, reps, seed):
    rng = np.random.default_rng(seed)
    runs = [inference.predictive_regression(*_simulate(rng, T, beta, RHO, CORR), rho_bound=RHO) for _ in range(reps)]
    return {k: np.array([run[k] for run in runs]) for k in runs[0]}


@pytest.fixture(scope="module")
def mc_slope():
    return _monte_carlo(beta=0.5, reps=1000, seed=20261025)


def test_ols_carries_the_stambaugh_bias_and_the_correction_removes_it(mc_slope):
    beta = 0.5
    ols_bias = mc_slope["beta_ols"].mean() - beta
    # the bias is real (about 25 Monte-Carlo standard errors) and has Stambaugh's sign and size
    assert ols_bias > 0.25
    assert mc_slope["stambaugh_bias"].mean() == pytest.approx(STAMBAUGH_BIAS, abs=0.02)
    # the corrected slope keeps at most a quarter of it (first-order correction: ~10% is left)
    assert abs(mc_slope["beta_stambaugh"].mean() - beta) < 0.25 * ols_bias
    # the estimated gamma is the simulated one
    assert mc_slope["gamma"].mean() == pytest.approx(GAMMA, rel=0.01)


def test_the_conditional_slope_is_unbiased_when_the_bound_is_the_true_rho(mc_slope):
    se = mc_slope["beta_conditional"].std(ddof=1) / np.sqrt(mc_slope["beta_conditional"].size)
    assert abs(mc_slope["beta_conditional"].mean() - 0.5) < 4 * se
    assert mc_slope["conditional_conservative"].all()  # gamma < 0 in every draw


def test_only_the_bias_adjusted_tests_hold_their_size_under_the_null():
    mc = _monte_carlo(beta=0.0, reps=500, seed=7)
    assert (mc["beta_ols_pvalue_onesided"] < 0.05).mean() > 0.12  # OLS over-rejects: its slope is biased up
    assert (mc["beta_stambaugh_pvalue_onesided"] < 0.05).mean() < 0.09
    assert (mc["beta_conditional_pvalue_onesided"] < 0.05).mean() < 0.09


def test_both_corrections_are_the_ols_slope_minus_gamma_times_the_rho_gap():
    r, x = _simulate(np.random.default_rng(1), 300, 0.2, 0.95, -0.8)
    out = inference.predictive_regression(r, x, rho_bound=0.999)
    gap = out["gamma"] * (out["rho_hat"] - out["rho_corrected"])
    assert out["beta_stambaugh"] == pytest.approx(out["beta_ols"] - gap, rel=1e-10)
    assert out["stambaugh_bias"] == pytest.approx(gap, rel=1e-10)
    n = out["n_obs"]
    assert out["rho_corrected"] == pytest.approx(out["rho_hat"] + (1 + 3 * out["rho_hat"]) / n, rel=1e-14)
    cond = out["beta_ols"] - out["gamma"] * (out["rho_hat"] - 0.999)
    assert out["beta_conditional"] == pytest.approx(cond, rel=1e-10)
    assert out["conditional_df"] == n - 3 == 297


# ── the published number ───────────────────────────────────────────────────
#
# Lewellen (2004), Table 2, log D/P, 1946-2000, 660 months, NYSE value-weighted,
# returns in percent per month (as quoted, with the paper's column, by the DEFR
# replication report of the paper, August 2026):
#   OLS 0.917 (p .027), Stambaugh 0.196 (p .308), rho ~ 1: 0.663 (t 4.67);
#   excess returns, rho ~ 1: 0.661. Table 1: mean D/P 3.80%.
# Our data is NOT the paper's: SBBI large cap is the S&P 500 (Lewellen: CRSP NYSE
# VW). The OLS slope is the estimator most sensitive to the index (DEFR's own CRSP
# rebuild gives 0.795), so it is held loosely; the conditional slope is the
# paper's claim that the correction is robust, and is held to 5% (t to 10%).
# Measured: conditional 0.647 / t 4.34 (2.4% / 7% off), excess 0.668 (1.1%),
# Stambaugh 0.189 / p 0.330, OLS 0.715, mean D/P 3.80%.

SBBI_SHA256 = "75cbe64c4494bd7fc4c2f6981a26c8a1692fa0804640f1dca42fb32374b34b1e"  # sbbi manifest, monthly.parquet


@pytest.fixture(scope="module")
def sbbi():
    from quantbox.dataset_lock import datasets_root

    try:
        path = datasets_root() / "sbbi" / "monthly.parquet"
    except Exception as exc:  # noqa: BLE001 — no quantbox-datasets clone on this machine
        pytest.skip(f"sbbi not readable here ({exc}): the Lewellen reference check DID NOT RUN")
    if not path.is_file():
        pytest.skip(f"{path} absent: the Lewellen reference check DID NOT RUN")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SBBI_SHA256, "sbbi monthly is not the pinned build"
    m = pd.read_parquet(path)
    price = (1 + m["LargeCap_CAR"]).cumprod()
    dividends = m["LargeCap_IR"] * price.shift(1)
    dy = dividends.rolling(12).sum() / price  # trailing-12m dividends over the current price, as Lewellen
    window = slice("1945-12-31", "2000-12-31")  # x for Dec 1945 predicts Jan 1946
    return {
        "x": np.log(dy).loc[window].to_numpy(),
        "nominal": (100 * m["LargeCap_TR"]).loc[window].to_numpy(),
        "excess": (100 * (m["LargeCap_TR"] - m["TBill_30d"])).loc[window].to_numpy(),
        "dy_mean": float(100 * dy.loc["1946":"2000"].mean()),
    }


def test_lewellen_2004_table_2_dp_1946_2000_on_sbbi(sbbi):
    out = inference.predictive_regression(sbbi["nominal"], sbbi["x"])
    assert out["n_obs"] == 660
    assert sbbi["dy_mean"] == pytest.approx(3.80, abs=0.05)
    assert out["beta_conditional"] == pytest.approx(0.663, rel=0.05)
    assert out["beta_conditional_tstat"] == pytest.approx(4.67, rel=0.10)
    assert out["beta_conditional_pvalue_onesided"] < 0.001
    assert out["beta_stambaugh"] == pytest.approx(0.196, abs=0.05)
    assert out["beta_stambaugh_pvalue_onesided"] == pytest.approx(0.308, abs=0.05)
    assert out["beta_ols"] == pytest.approx(0.917, abs=0.25) and out["beta_ols_pvalue_onesided"] < 0.10
    assert out["rho_corrected"] > 1  # the Stambaugh correction assumes a non-stationary D/P: Lewellen's point
    assert out["conditional_conservative"]
    excess = inference.predictive_regression(sbbi["excess"], sbbi["x"])
    assert excess["beta_conditional"] == pytest.approx(0.661, rel=0.05)


def test_the_gate_passes_on_the_conditional_test_alone_for_lewellen(sbbi):
    out = g.predreg_gate(sbbi["nominal"], sbbi["x"])
    assert out["family_tests"] == ["stambaugh", "conditional"]
    assert out["pvalues_adjusted"]["stambaugh"] > 0.05
    assert out["pvalues_adjusted"]["conditional"] < 0.05
    assert out["gate_pass"] and out["predict_pass"] and out["obs_pass"]


# ── refusals ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "call",
    [
        lambda r, x: inference.predictive_regression(r[:-1], x),  # misaligned lengths
        lambda r, x: inference.predictive_regression(np.where(np.arange(r.size) == 50, np.nan, r), x),
        lambda r, x: inference.predictive_regression(r, np.where(np.arange(x.size) == 9, np.inf, x)),
        lambda r, x: inference.predictive_regression(r, np.full_like(x, 0.03)),  # constant predictor
        lambda r, x: inference.predictive_regression(np.full_like(r, 0.01), x),  # constant returns
        lambda r, x: inference.predictive_regression(r[:5], x[:5]),  # 4 pairs < 5
        lambda r, x: inference.predictive_regression(r, 0.5 ** np.arange(x.size)),  # deterministic AR(1)
        lambda r, x: inference.predictive_regression(np.r_[0.0, 2 * x[:-1] + 0.1], x),  # r exact in x_t-1
        lambda r, x: inference.predictive_regression(r, x, rho_bound=1.01),
        lambda r, x: inference.predictive_regression(r, x, rho_bound=0.0),
        lambda r, x: inference.predictive_regression(r, x, rho_bound=True),
        lambda r, x: g.predreg_gate(r, x, min_obs=0),
        lambda r, x: g.predreg_gate(r, x, alpha=1.0),
    ],
)
def test_inputs_that_cannot_compute_are_refused(call):
    r, x = _simulate(np.random.default_rng(3), 120, 0.1, 0.9, -0.5)
    with pytest.raises(inference.InferenceInputError):
        call(r, x)


def test_an_opted_in_drop_removes_every_pair_the_bad_row_touches():
    r, x = _simulate(np.random.default_rng(4), 120, 0.1, 0.9, -0.5)
    x[60] = np.nan  # x_60 is x_t for pair 59 and x_t-1 for pair 60
    r[30] = np.nan
    out = inference.predictive_regression(r, x, allow_nonfinite_drop=True)
    assert (out["n_obs_raw"], out["n_nonfinite_dropped"], out["n_obs"]) == (120, 3, 117)


def test_a_positive_gamma_keeps_the_conditional_bound_out_of_the_verdict():
    r, x = _simulate(np.random.default_rng(5), 400, 0.0, 0.98, 0.9)
    out = g.predreg_gate(r, x)
    assert not out["conditional_conservative"]
    assert out["family_tests"] == ["stambaugh"]
    assert out["pvalues_adjusted"] == {"stambaugh": out["beta_stambaugh_pvalue_onesided"]}


def test_too_few_pairs_fail_the_gate_even_with_a_strong_slope():
    r, x = _simulate(np.random.default_rng(6), 40, 20.0, 0.9, -0.5)
    out = g.predreg_gate(r, x)
    assert out["predict_pass"] and not out["obs_pass"] and not out["gate_pass"]


# ── CLI ────────────────────────────────────────────────────────────────────


@pytest.fixture()
def files(tmp_path):
    r, x = _simulate(np.random.default_rng(8), 300, 1.0, 0.95, -0.7)
    dates = pd.date_range("1990-01-31", periods=r.size, freq="ME")
    both = tmp_path / "both.csv"
    pd.DataFrame({"date": dates, "ret": r, "dp": x}).to_csv(both, index=False)
    return {"both": both, "r": r, "x": x, "dates": dates, "dir": tmp_path}


def _run(*argv):
    return CliRunner().invoke(app, ["gates", "predreg", *argv])


def test_the_verb_is_listed_in_the_cli_help():
    for argv in (["--help"], ["gates", "--help"]):
        out = CliRunner().invoke(app, argv)
        assert out.exit_code == 0 and "predreg" in out.output, argv


def test_cli_json_matches_the_function(files):
    both = str(files["both"])
    res = _run("--returns", both, "--column", "ret", "--predictor", both, "--predictor-column", "dp", "--json")
    assert res.exit_code == 0, res.output
    got = json.loads(res.output)
    want = g.predreg_gate(files["r"], files["x"])
    for key, value in want.items():
        assert got[key] == (pytest.approx(value) if isinstance(value, (float, dict)) else value), key
    assert got["first_date"].startswith("1990-01-31")


def test_cli_exit_codes(files):
    both = str(files["both"])
    base = ["--returns", both, "--column", "ret", "--predictor", both, "--predictor-column", "dp"]
    assert _run(*base).exit_code == 0
    assert _run(*base, "--min-obs", "1000").exit_code == 1
    assert _run(*base, "--rho-bound", "1.5").exit_code == 2
    assert _run("--returns", both, "--predictor", both).exit_code == 2  # two numeric columns: refuses to guess


def test_cli_refuses_a_date_one_file_lacks_inside_the_shared_span(files):
    d = files["dir"]
    ret = d / "ret.csv"
    dp = d / "dp.csv"
    pd.DataFrame({"date": files["dates"], "ret": files["r"]}).to_csv(ret, index=False)
    pd.DataFrame({"date": files["dates"], "dp": files["x"]}).drop(index=150).to_csv(dp, index=False)
    res = _run("--returns", str(ret), "--predictor", str(dp), "--json")
    assert res.exit_code == 2
    assert "no partner" in res.output
