"""Predictive tests for a persistent predictor (TOM-3928): the Campbell-Yogo Q-test, Goyal-Welch OOS R², Clark-West.

Evidence, each named by what it reads:

- the lookup tables against the printed guide, and the lookup rule (grid, nearest delta, interpolation);
- a property test on simulated near-unit-root predictors with a KNOWN slope (seeded): the Q-test holds
  its 5% size where the OLS t-test does not, and still finds a real slope;
- the out-of-sample statistics against a naive per-period refit (a loop is fine in a test);
- the ``predreg`` gate and CLI: the new tests are opt-in, add keys only, and can only make the gate stricter;
- ``network``: the PUBLISHED numbers, Campbell-Yogo (2006) Tables 4-5 on the authors' own data and
  Goyal-Welch (2008) Table 1 on Goyal's data. Third-party data is never committed (the repo is public):
  these tests fetch it into the pytest cache, or read a path from an env var, and skip when neither works.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quantbox import _campbell_yogo_tables as cyt
from quantbox import gates as g
from quantbox import inference
from quantbox.cli import app


def _simulate(rng, T, beta, rho, corr, sigma_e=1.0, sigma_u=0.05):
    """``(r, x)`` aligned: r[t] = beta * x[t-1] + e[t], x[t] = rho * x[t-1] + u[t], corr(e, u) = corr."""
    e = rng.normal(size=T + 1)
    u = corr * e + np.sqrt(1 - corr**2) * rng.normal(size=T + 1)
    e, u = sigma_e * e, sigma_u * u
    x = np.empty(T + 1)
    x[0] = rng.normal(scale=sigma_u / np.sqrt(1 - rho**2))
    for t in range(1, T + 1):
        x[t] = rho * x[t - 1] + u[t]
    r = np.empty(T + 1)
    r[0] = 0.0
    r[1:] = beta * x[:-1] + e[1:]
    return r, x


# ── the tables and the lookup ──────────────────────────────────────────────


def test_the_tables_are_the_printed_grid_and_every_interval_is_ordered():
    assert cyt.DELTAS.shape == (40,) and cyt.DELTAS[0] == -0.999 and cyt.DELTAS[-1] == -0.025
    assert np.allclose(np.diff(cyt.DELTAS[1:]), 0.025)
    assert cyt.DFGLS.shape == (61,) and cyt.DFGLS[0] == 1.0 and cyt.DFGLS[-1] == -5.0
    assert cyt.C_LOWER.shape == cyt.C_UPPER.shape == (40, 61)
    assert (cyt.C_LOWER < cyt.C_UPPER).all()
    # a lower DF-GLS (stronger mean reversion) moves both ends of the c interval down
    assert (np.diff(cyt.C_LOWER, axis=1) < 0).all() and (np.diff(cyt.C_UPPER, axis=1) < 0).all()


@pytest.mark.parametrize(
    "delta, dfgls, lower, upper",
    [
        (-0.999, 1.0, -0.282, 4.148),  # Table 2, first row
        (-0.800, -2.2, -16.461, -4.662),  # Table 4
        (-0.025, -5.0, -54.286, -45.666),  # Table 11, last row
    ],
)
def test_printed_cells_read_back(delta, dfgls, lower, upper):
    i = int(np.flatnonzero(np.isclose(cyt.DELTAS, delta))[0])
    k = int(np.flatnonzero(np.isclose(cyt.DFGLS, dfgls))[0])
    assert (cyt.C_LOWER[i, k], cyt.C_UPPER[i, k]) == (lower, upper)


def test_the_lookup_reads_the_nearest_delta_and_interpolates_dfgls():
    look = inference._cy_c_interval
    i, k = 8, 32  # delta -0.800, DF-GLS -2.2
    assert look(-2.2, -0.8) == (cyt.C_LOWER[i, k], cyt.C_UPPER[i, k], -0.8)
    lo, hi, d = look(-2.25, -0.81)  # between rows -2.2 and -2.3, nearest delta -0.800
    assert d == -0.8
    assert lo == pytest.approx((cyt.C_LOWER[i, k] + cyt.C_LOWER[i, k + 1]) / 2)
    assert hi == pytest.approx((cyt.C_UPPER[i, k] + cyt.C_UPPER[i, k + 1]) / 2)
    assert (look(-2.2, -0.9874)[2], look(-2.2, -0.9876)[2]) == (-0.975, -0.999)  # the nearer grid delta
    assert look(-2.2, -1.0)[2] == -0.999  # the GAUSS grid's -1.0 is the paper's -0.999
    assert look(-2.2, 0.8) == look(-2.2, -0.8)  # delta > 0 mirrors
    assert look(-5.01, -0.8) is None and look(1.01, -0.8) is None


# ── the Q-test on simulated predictors ─────────────────────────────────────


def _mc(beta, rho, reps, seed):
    rng = np.random.default_rng(seed)
    return [inference.campbell_yogo(*_simulate(rng, 240, beta, rho, -0.9), lags=1) for _ in range(reps)]


def test_the_q_test_holds_its_size_where_the_t_test_over_rejects():
    runs = _mc(beta=0.0, rho=0.98, reps=400, seed=20261114)
    assert all(run["applicable"] for run in runs)
    q_size = np.mean([run["reject"] for run in runs])
    t_size = np.mean([run["beta_ols_tstat"] > 1.645 for run in runs])
    coverage = np.mean([run["beta_interval"][0] <= 0 <= run["beta_interval"][1] for run in runs])
    assert q_size <= 0.06  # 5% one-sided; 0.030 at this seed
    assert t_size >= 0.10  # the OLS t-test with delta -0.9 and rho 0.98; 0.148 at this seed
    assert coverage >= 0.88  # the two-sided 90% interval; 0.918 at this seed


def test_the_q_test_finds_a_real_slope():
    runs = _mc(beta=3.0, rho=0.98, reps=50, seed=1)
    assert np.mean([run["reject"] for run in runs]) >= 0.95


def test_the_interval_keys_are_consistent():
    r, x = _simulate(np.random.default_rng(2), 300, 1.0, 0.97, -0.8)
    out = inference.campbell_yogo(r, x, lags=1)
    t = out["n_obs"]
    assert t == 300 and out["ar_lags"] == 1 and out["lags_by"] == "given"
    assert out["rho_interval"] == pytest.approx([1 + c / t for c in out["c_interval"]])
    lo, hi = out["beta_interval"]
    assert lo < hi
    scale = out["sigma_e"] / out["sigma_u"]
    assert out["beta_interval_normalized"] == pytest.approx([scale * lo, scale * hi])
    assert out["reject"] == (lo > 0) and out["level_onesided"] == inference.CAMPBELL_YOGO_LEVEL == 0.05
    ols = inference.predictive_regression(r, x)
    assert out["beta_ols"] == pytest.approx(ols["beta_ols"]) and out["delta"] == pytest.approx(ols["corr_e_u"])
    assert inference.campbell_yogo(r, x)["lags_by"] == "bic, 1..4"


def test_a_stationary_predictor_is_off_the_tables_and_does_not_block_the_gate():
    r, x = _simulate(np.random.default_rng(3), 400, 1.0, 0.3, -0.5)
    out = inference.campbell_yogo(r, x)
    assert out["dfgls"] < -5 and not out["applicable"]
    assert out["reject"] is None and out["beta_interval"] is None and "outside" in out["note"]
    gate = g.predreg_gate(r, x, q_test=True)
    assert gate["q_test_pass"] is None and gate["gate_pass"] == (gate["predict_pass"] and gate["obs_pass"])


@pytest.mark.parametrize(
    "call",
    [
        lambda r, x: inference.campbell_yogo(r[:-1], x),
        lambda r, x: inference.campbell_yogo(np.where(np.arange(r.size) == 50, np.nan, r), x),
        lambda r, x: inference.campbell_yogo(r, x, lags=0),
        lambda r, x: inference.campbell_yogo(r, x, lags=True),
        lambda r, x: inference.campbell_yogo(r, x, max_lags=0),
        lambda r, x: inference.campbell_yogo(r[:12], x[:12], lags=4),  # 8 pairs < 10
        lambda r, x: inference.campbell_yogo(r, np.full_like(x, 0.03)),
        lambda r, x: inference.oos_forecast(r[:-1], x, start=50),
        lambda r, x: inference.oos_forecast(r, np.where(np.arange(x.size) == 9, np.inf, x), start=50),
        lambda r, x: inference.oos_forecast(r, x, start=2),
        lambda r, x: inference.oos_forecast(r, x, start=r.size - 2),
        lambda r, x: inference.oos_forecast(r, x, start=True),
        lambda r, x: inference.oos_forecast(r, np.r_[np.full(60, 0.03), x[60:]], start=50),
    ],
)
def test_inputs_that_cannot_compute_are_refused(call):
    r, x = _simulate(np.random.default_rng(4), 120, 0.1, 0.9, -0.5)
    with pytest.raises(inference.InferenceInputError):
        call(r, x)


# ── out-of-sample R² and Clark-West ────────────────────────────────────────


def _oos_reference(r, x, start):
    """Refit OLS and the mean at every forecast date from the pairs before it: the slow, obvious form."""
    f_a, f_n = [], []
    for t in range(start, r.size):
        y, z = r[1:t], x[: t - 1]  # pairs (r[s], x[s-1]) for s = 1..t-1
        a, b = np.linalg.lstsq(np.column_stack([np.ones(z.size), z]), y, rcond=None)[0]
        f_a.append(a + b * x[t - 1])
        f_n.append(y.mean())
    f_a, f_n, actual = np.array(f_a), np.array(f_n), r[start:]
    e_a, e_n = actual - f_a, actual - f_n
    cw = e_n**2 - (e_a**2 - (f_a - f_n) ** 2)
    return {
        "mse_model": np.mean(e_a**2),
        "mse_mean": np.mean(e_n**2),
        "r2_oos": 1 - np.mean(e_a**2) / np.mean(e_n**2),
        "clark_west_mean": cw.mean(),
        "clark_west_tstat": cw.mean() / (cw.std(ddof=1) / np.sqrt(cw.size)),
    }


@pytest.mark.parametrize("start", [3, 40, 117])
def test_the_expanding_fit_matches_a_refit_at_every_date(start):
    r, x = _simulate(np.random.default_rng(5), 119, 0.5, 0.95, -0.7)
    x = x + 3.0  # a level far from zero, as log D/P is: the cumulative sums must stay exact
    out = inference.oos_forecast(r, x, start=start)
    want = _oos_reference(r, x, start)
    for key, value in want.items():
        assert out[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key
    p = r.size - start
    assert (out["oos_start"], out["n_forecasts"], out["n_train_first"]) == (start, p, start - 1)
    assert out["r2_oos_adjusted"] == pytest.approx(1 - (1 - want["r2_oos"]) * (p - 1) / (p - 2))
    assert out["delta_rmse"] == pytest.approx(np.sqrt(want["mse_mean"]) - np.sqrt(want["mse_model"]))
    assert out["mse_f"] == pytest.approx(p * (want["mse_mean"] - want["mse_model"]) / want["mse_model"])


def test_no_forecast_reads_the_future():
    r, x = _simulate(np.random.default_rng(6), 200, 0.5, 0.95, -0.7)
    base = inference.oos_forecast(r, x, start=100)
    x2 = x.copy()
    x2[-1] = 1e6  # the last predictor value forecasts nothing inside the sample
    assert inference.oos_forecast(r, x2, start=100) == base
    # the prefix ending at row 150 gives the statistics of the first 50 forecasts, whatever follows
    r3, x3 = r.copy(), x.copy()
    r3[150:], x3[150:] = 99.0, -99.0
    assert inference.oos_forecast(r3[:150], x3[:150], start=100) == inference.oos_forecast(r[:150], x[:150], start=100)


def test_a_slope_that_dies_out_of_sample_fails_clark_west():
    rng = np.random.default_rng(7)
    r, x = _simulate(rng, 400, 0.0, 0.9, 0.0)
    r[1:201] += 6.0 * (x[:200] - x[:200].mean())  # predictable in the first half only
    out = g.predreg_gate(r, x, oos_start=201)
    assert out["predict_pass"] and out["obs_pass"]
    assert out["oos"]["r2_oos"] < 0 and out["oos_pass"] is False and out["gate_pass"] is False
    assert "Clark-West" in out["note"]


# ── the gate: opt-in, keys added, stricter only ────────────────────────────

NEW_KEYS = ("campbell_yogo", "q_test_pass", "oos", "oos_pass")


def test_not_asked_means_none_and_nothing_else_moves():
    r, x = _simulate(np.random.default_rng(8), 300, 1.0, 0.95, -0.7)
    plain = g.predreg_gate(r, x)
    assert all(plain[k] is None for k in NEW_KEYS)
    both = g.predreg_gate(r, x, q_test=True, oos_start=100)
    for key, value in plain.items():
        if key not in NEW_KEYS and key not in ("gate_pass", "note"):
            assert both[key] == value, key
    assert both["note"].startswith(plain["note"])
    assert both["campbell_yogo"] == inference.campbell_yogo(r, x)
    assert both["oos"] == inference.oos_forecast(r, x, start=100)
    assert both["gate_pass"] == bool(plain["gate_pass"] and both["q_test_pass"] and both["oos_pass"])


def test_a_q_test_that_cannot_reject_fails_a_gate_the_bias_adjusted_test_passes():
    r, x = _simulate(np.random.default_rng(103), 240, 0.0, 0.995, -0.95)  # no slope; the seed fools Stambaugh
    out = g.predreg_gate(r, x, q_test=True, q_lags=1)
    assert out["predict_pass"] and out["obs_pass"]
    assert out["q_test_pass"] is False and out["gate_pass"] is False
    assert "keeps beta <= 0" in out["note"]


# ── CLI ────────────────────────────────────────────────────────────────────


@pytest.fixture()
def files(tmp_path):
    r, x = _simulate(np.random.default_rng(9), 300, 1.0, 0.95, -0.7)
    dates = pd.date_range("1990-01-31", periods=r.size, freq="ME")
    both = tmp_path / "both.csv"
    pd.DataFrame({"date": dates, "ret": r, "dp": x}).to_csv(both, index=False)
    return {"both": str(both), "r": r, "x": x}


def _run(files, *argv):
    base = ["gates", "predreg", "--returns", files["both"], "--column", "ret"]
    return CliRunner().invoke(app, [*base, "--predictor", files["both"], "--predictor-column", "dp", *argv])


def test_the_options_are_in_the_help():
    out = CliRunner().invoke(app, ["gates", "predreg", "--help"])
    for option in ("--q-test", "--q-lags", "--q-max-lags", "--oos-start"):
        assert option in out.output, option


def test_cli_json_adds_the_keys_as_null_when_not_asked(files):
    got = json.loads(_run(files, "--json").output)
    assert all(got[k] is None for k in NEW_KEYS)


def test_cli_json_matches_the_functions(files):
    res = _run(files, "--q-test", "--q-max-lags", "6", "--oos-start", "1995-01-15", "--json")
    assert res.exit_code in (0, 1), res.output
    got = json.loads(res.output)
    start = 60  # 1995-01-31, the first row on or after the date
    want = g.predreg_gate(files["r"], files["x"], q_test=True, q_max_lags=6, oos_start=start)
    for key in ("campbell_yogo", "oos"):
        for k, value in want[key].items():
            assert got[key][k] == (pytest.approx(value) if isinstance(value, (float, list)) else value), (key, k)
    assert got["oos"]["oos_start_date"].startswith("1995-01-31")
    assert got["gate_pass"] == want["gate_pass"] and res.exit_code == (0 if want["gate_pass"] else 1)


def test_the_text_output_names_the_opt_in_tests_only_when_asked(files):
    assert _run(files).output.count("\n") == 1  # the one headline line, as before TOM-3928
    out = _run(files, "--q-test", "--oos-start", "1995-01-31").output
    assert "Campbell-Yogo" in out and "Clark-West" in out


def test_cli_refuses_a_bad_oos_start(files):
    assert _run(files, "--oos-start", "2030-01-01").exit_code == 2  # after the last row
    assert _run(files, "--oos-start", "not-a-date").exit_code == 2
    assert _run(files, "--oos-start", "1990-02-01").exit_code == 2  # row 1: no training pairs
    assert _run(files, "--q-test", "--q-lags", "0").exit_code == 2


# ── network: the published numbers ─────────────────────────────────────────


def _published(request, env: str, url: str, name: str, looks_right) -> bytes:
    """The bytes at ``$env`` if set, else at ``url`` (cached in the pytest cache); skip when neither works.

    ``looks_right(bytes)`` tells the file from an error page; only a file that passes is cached.
    """
    if os.environ.get(env):
        path = Path(os.environ[env])
        if not path.is_file():
            pytest.skip(f"${env}={path} is not a file: the published-sample check DID NOT RUN")
        blob = path.read_bytes()
    else:
        cached = Path(request.config.cache.mkdir("quantbox-published")) / name
        if cached.is_file():
            return cached.read_bytes()
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 — a fixed https URL
                blob = resp.read()
        except (OSError, http.client.HTTPException) as exc:
            pytest.skip(f"{name}: fetch failed ({exc}); set ${env} to a local copy. The check DID NOT RUN")
        if looks_right(blob):
            cached.write_bytes(blob)
    if not looks_right(blob):
        pytest.skip(f"{name}: what came back is not the published file: the check DID NOT RUN")
    return blob


# Campbell and Yogo's replication files: the "Programs" link on Motohiro Yogo's site (Predict_Programs.zip).
CY_URL = (
    "https://drive.usercontent.google.com/download?id=0BzR-ojpYuaFMcnZteHFyWUVIUFU"
    "&resourcekey=0-ZnbwJSS63wOF197JBFLYSA&export=download&confirm=t"
)
CY_SHA256 = {  # the data members (not the zip, which a re-pack would change)
    "CRSP_A": "90af5026b1370c3b319df171364eb48940c18ec80c87838942efc0468b2b2316",
    "CRSP_Q": "e7562024c474dccacb70634c3d5cbc5071531611c80737784bfde94e90b1814c",
    "CRSP_M": "74dcce401bcf4e02a806401cbb08dfaaa394808ce6f271ac09a43c442c888911",
    "SP_A": "3bda6efbeef8fc2df882cc3c1e7c06feb99c9f0df852218c14356283b264264d",
}


@pytest.fixture(scope="module")
def cy_data(request):
    blob = _published(
        request, "QUANTBOX_CY_PROGRAMS_ZIP", CY_URL, "Predict_Programs.zip", lambda b: zipfile.is_zipfile(io.BytesIO(b))
    )
    out = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name, sha in CY_SHA256.items():
            raw = z.read(f"Pred/data/txt/{name}.txt")
            assert hashlib.sha256(raw).hexdigest() == sha, f"{name}.txt is not the published file"
            out[name] = pd.read_csv(io.BytesIO(raw), sep="\t")  # time, ret, ldp, lep, rf, yield
    return out


# series -> (max BIC order, rows of the 1926/1880-1994 sample, first row of 1952-2002), as predtest.prg cuts them
CY_SAMPLES = {"SP_A": (4, 115, None), "CRSP_A": (4, 69, 26), "CRSP_Q": (6, 273, 101), "CRSP_M": (8, 817, 301)}
# Table 4 Panel A: p, delta, DF-GLS
CY_TABLE_4A = {
    ("SP_A", "ldp"): (3, -0.845, -0.855),
    ("SP_A", "lep"): (1, -0.962, -2.888),
    ("CRSP_A", "ldp"): (1, -0.721, -1.033),
    ("CRSP_A", "lep"): (1, -0.957, -2.229),
    ("CRSP_Q", "ldp"): (1, -0.942, -1.696),
    ("CRSP_Q", "lep"): (1, -0.986, -2.191),
    ("CRSP_M", "ldp"): (2, -0.950, -1.657),
    ("CRSP_M", "lep"): (1, -0.987, -1.859),
}
# Table 5: t-stat, beta_hat, 90% Q-test interval, Q lower bound at rho = 1 (beta in unit-variance units)
CY_TABLE_5 = {
    ("A", "SP_A", "ldp"): (1.967, 0.093, -0.033, 0.114, -0.017),
    ("A", "SP_A", "lep"): (2.762, 0.131, 0.042, 0.224, -0.023),
    ("A", "CRSP_A", "ldp"): (2.534, 0.125, 0.014, 0.188, 0.020),
    ("A", "CRSP_A", "lep"): (2.770, 0.169, 0.042, 0.277, 0.002),
    ("A", "CRSP_Q", "ldp"): (2.060, 0.034, -0.009, 0.044, -0.010),
    ("A", "CRSP_Q", "lep"): (2.908, 0.049, 0.010, 0.066, 0.002),
    ("A", "CRSP_M", "ldp"): (1.706, 0.009, -0.005, 0.010, -0.005),
    ("A", "CRSP_M", "lep"): (2.662, 0.014, 0.002, 0.018, 0.001),
    ("B", "SP_A", "ldp"): (2.233, 0.141, -0.048, 0.183, -0.081),
    ("B", "SP_A", "lep"): (3.321, 0.196, 0.093, 0.325, -0.030),
    ("B", "CRSP_A", "ldp"): (2.993, 0.212, 0.056, 0.332, 0.011),
    ("B", "CRSP_A", "lep"): (3.409, 0.279, 0.126, 0.448, 0.012),
    ("B", "CRSP_Q", "ldp"): (2.304, 0.053, -0.006, 0.076, -0.027),
    ("B", "CRSP_Q", "lep"): (3.506, 0.079, 0.027, 0.109, 0.005),
    ("B", "CRSP_M", "ldp"): (1.790, 0.013, -0.007, 0.017, -0.013),
    ("B", "CRSP_M", "lep"): (3.185, 0.022, 0.005, 0.028, 0.000),
    ("C", "CRSP_A", "ldp"): (2.289, 0.124, -0.007, 0.183, 0.020),
    ("C", "CRSP_A", "lep"): (1.733, 0.114, -0.031, 0.229, -0.025),
    ("C", "CRSP_A", "rf"): (-1.143, -0.095, -0.231, 0.042, None),
    ("C", "CRSP_A", "yield"): (1.124, 0.136, -0.075, 0.359, -0.156),
    ("C", "CRSP_Q", "ldp"): (2.236, 0.036, -0.010, 0.030, 0.005),
    ("C", "CRSP_Q", "lep"): (1.777, 0.029, -0.012, 0.042, -0.003),
    ("C", "CRSP_Q", "rf"): (-1.766, -0.042, -0.084, -0.004, -0.086),
    ("C", "CRSP_Q", "yield"): (1.991, 0.090, 0.006, 0.158, -0.002),
    ("C", "CRSP_M", "ldp"): (2.259, 0.012, -0.004, 0.010, 0.001),
    ("C", "CRSP_M", "lep"): (1.754, 0.009, -0.004, 0.012, -0.001),
    ("C", "CRSP_M", "rf"): (-2.431, -0.017, -0.030, -0.006, -0.030),
    ("C", "CRSP_M", "yield"): (2.963, 0.047, 0.020, 0.072, 0.016),
}


def _cy_run(cy_data, panel, series, column):
    max_lags, n_94, first_52 = CY_SAMPLES[series]
    df = cy_data[series]
    sub = {"A": df, "B": df.iloc[:n_94], "C": df.iloc[first_52 or 0 :]}[panel]
    return inference.campbell_yogo(sub["ret"].to_numpy(), sub[column].to_numpy(), max_lags=max_lags)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("key", list(CY_TABLE_4A), ids=lambda k: "-".join(k))
def test_campbell_yogo_2006_table_4_panel_a(cy_data, key):
    out = _cy_run(cy_data, "A", *key)
    p, delta, dfgls = CY_TABLE_4A[key]
    assert out["ar_lags"] == p
    assert out["delta"] == pytest.approx(delta, abs=5e-4) and out["dfgls"] == pytest.approx(dfgls, abs=5e-4)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("key", list(CY_TABLE_5), ids=lambda k: "-".join(k))
def test_campbell_yogo_2006_table_5(cy_data, key):
    out = _cy_run(cy_data, *key)
    tstat, beta, q_lo, q_hi, low_rho1 = CY_TABLE_5[key]
    scale = out["sigma_e"] / out["sigma_u"]
    assert out["applicable"]
    assert out["beta_ols_tstat"] == pytest.approx(tstat, abs=5e-4)
    assert out["beta_ols"] * scale == pytest.approx(beta, abs=5e-4)
    # 0.001: the guide's tables are read by interpolation, the authors' GAUSS code inverts a finer belt;
    # two of the 56 ends (annual e-p 1926-2002 low, S&P e-p 1880-1994 high) round the other way
    assert out["beta_interval_normalized"] == pytest.approx([q_lo, q_hi], abs=1.1e-3)
    if low_rho1 is None:  # delta > 0: the paper prints no bound at rho = 1
        assert out["beta_lower_rho1"] is None
    else:
        assert out["beta_lower_rho1"] * scale == pytest.approx(low_rho1, abs=1.1e-3)


@pytest.mark.network
@pytest.mark.slow
def test_campbell_yogo_annual_crsp_dp_is_the_papers_headline(cy_data):
    out = _cy_run(cy_data, "A", "CRSP_A", "ldp")
    assert [round(v, 3) for v in out["beta_interval_normalized"]] == [0.014, 0.188]
    assert out["reject"]  # the paper: d-p predicts annual CRSP returns 1926-2002 at 5%, one-sided


# Goyal's data (Goyal-Welch 2008, updated yearly): the "Data" link of Amit Goyal's site, the Annual sheet as CSV.
GW_URL = "https://docs.google.com/spreadsheets/d/17mw_IpaiLFDrGnrPRQ2o1ugV5nJsZuD1/gviz/tq?tqx=out:csv&sheet=Annual"


@pytest.fixture(scope="module")
def gw_annual(request):
    blob = _published(
        request, "QUANTBOX_GW_ANNUAL_CSV", GW_URL, "goyal_welch_annual.csv", lambda b: b.startswith(b'"yyyy","price"')
    )
    try:
        a = pd.read_csv(io.BytesIO(blob)).set_index("yyyy")
        a = a.loc[1871:2005, ["price", "d12", "Rfree"]].astype(float)
    except (KeyError, ValueError, pd.errors.ParserError) as exc:
        pytest.skip(f"goyal_welch_annual.csv is not Goyal's Annual sheet ({exc}): the check DID NOT RUN")
    if len(a) != 135 or a.isna().any().any():
        pytest.skip("goyal_welch_annual.csv lacks 1871-2005 price/d12/Rfree: the check DID NOT RUN")
    ret = (a["price"] + a["d12"]) / a["price"].shift(1) - 1  # the S&P with dividends, as their 2007 file
    premium = (np.log1p(ret) - np.log1p(a["Rfree"])).to_numpy().copy()
    premium[0] = 0.0  # 1871 carries the first predictor only
    return {"r": premium, "x": (np.log(a["d12"]) - np.log(a["price"])).to_numpy(), "years": a.index.to_numpy()}


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    "first, r2bar, drmse",
    [
        (1892, -2.06, -0.11),  # Table 1, d/p, forecasts 1892-2005 (OOS from 20 years of data)
        (1965, -3.69, -0.09),  # Table 1, d/p, forecasts 1965-2005
    ],
)
def test_goyal_welch_2008_table_1_dp(gw_annual, first, r2bar, drmse):
    start = int(np.flatnonzero(gw_annual["years"] == first)[0])
    out = inference.oos_forecast(gw_annual["r"], gw_annual["x"], start=start)
    assert out["n_forecasts"] == 2005 - first + 1
    # Goyal's current file, not the 2007 vintage of the paper: d12 is rounded to cents and the
    # history revised. On the 2007 file this gives -2.061 (1892); on the current one -2.13 and -3.67.
    assert 100 * out["r2_oos_adjusted"] == pytest.approx(r2bar, abs=0.1)
    assert 100 * out["delta_rmse"] == pytest.approx(drmse, abs=0.01)
    assert out["clark_west_pvalue_onesided"] > 0.10  # d/p does not beat the mean out of sample
