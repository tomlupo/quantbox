"""``quantbox gates`` — the one implementation of the acceptance gates (TOM-1351).

Pinned three ways, because every failure here is SILENT (a drifted gate still
prints a plausible probability, and a gate that passes what it should refuse
looks exactly like a good strategy):

- **parity**: the values pinned in qute-research's ``tests/test_acceptance_gates.py``
  (qute-plugins, 2026-09-28) reproduce here at rel 1e-9 — the plugin's copy of the
  maths and this one must agree before the plugin can delegate to quantbox;
- **known answers** for the two gates that exist nowhere else yet (the paired
  stationary block bootstrap and the largest-drawdown-episode exclusion);
- **exit codes**: 0 pass, 1 fail, 2 could not compute — and a 2 never borrows a
  verdict's code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quantbox.analysis import gates as g
from quantbox.cli import app

REL = 1e-9
GAMMA = 0.5772156649015329


def series():
    """The plugin's deterministic, autocorrelated, non-normal series and two factors."""
    t = np.arange(600)
    r = 0.0006 + 0.01 * np.sin(0.7 * t) + 0.006 * np.cos(1.3 * t + 0.4) + 0.004 * np.sin(0.05 * t) ** 3
    f1 = 0.008 * np.sin(0.7 * t + 0.2) + 0.002 * np.cos(0.11 * t)
    f2 = 0.005 * np.cos(1.3 * t) - 0.003 * np.sin(0.31 * t)
    return r, np.column_stack([f1, f2])


# ── DSR: parity with the plugin ────────────────────────────────────────────


def test_dsr_equals_the_textbook_formula_computed_without_scipy():
    N = NormalDist()
    sr, T, n = 3.3 / math.sqrt(365), 365, 20
    sr_std = math.sqrt((1 + (3.0 - 1) / 4 * sr**2) / (T - 1))
    e_max = (1 - GAMMA) * N.inv_cdf(1 - 1 / n) + GAMMA * N.inv_cdf(1 - 1 / (n * math.e))
    textbook = N.cdf((sr - sr_std * e_max) / sr_std)

    out = g.dsr_gate(sr=sr, T=T, skew=0.0, kurtosis=3.0, n_trials=[n], periods=365, threshold=0.95)
    assert out["by_n_trials"][str(n)]["dsr"] == pytest.approx(textbook, rel=REL)
    assert out["dsr_conservative"] == pytest.approx(0.9147282296900824, rel=REL)
    assert out["gate_pass"] is False


@pytest.mark.parametrize(
    "n_trials, expected",
    [(1, 0.9767210580374479), (5, 0.7874794762796605), (20, 0.5356943493190106), (100, 0.29449412587586443)],
)
def test_dsr_from_returns_reproduces_the_plugin_pinned_values(n_trials, expected):
    r, _ = series()
    out = g.dsr_gate_from_returns(r, n_trials=[n_trials])
    assert out["dsr_conservative"] == pytest.approx(expected, rel=REL)


def test_the_dsr_verdict_is_taken_at_the_most_deflated_end_of_the_range():
    r, _ = series()
    out = g.dsr_gate_from_returns(r, n_trials=[1, 5, 100], periods=252)
    assert out["by_n_trials"]["1"]["dsr"] > 0.95
    assert out["n_trials_conservative"] == 100 and out["gate_pass"] is False


def test_cross_trial_dispersion_raises_the_bar():
    base = g.dsr_gate(sr=0.1, T=500, skew=0.0, kurtosis=3.0, n_trials=[50])["by_n_trials"]["50"]
    wider = g.dsr_gate(sr=0.1, T=500, skew=0.0, kurtosis=3.0, n_trials=[50], trials_sr_std=2 * base["sr_std"])
    w = wider["by_n_trials"]["50"]
    assert w["sr0"] == pytest.approx(2 * base["sr0"], rel=REL) and w["dsr"] < base["dsr"]


@pytest.mark.parametrize("periods, dsr, passes", [(252, 0.9658711604387862, True), (365, 0.8845753797271305, False)])
def test_periods_moves_the_summary_verdict(periods, dsr, passes):
    out = g.dsr_gate(
        sr=2.2 / math.sqrt(periods), T=730, skew=0.0, kurtosis=3.0, n_trials=[20], periods=periods, threshold=0.95
    )
    assert out["dsr_conservative"] == pytest.approx(dsr, rel=REL) and out["gate_pass"] is passes


@pytest.mark.parametrize(
    "call",
    [
        lambda: g.dsr_gate(sr=0.1, T=500, skew=0.0, kurtosis=0.0, n_trials=[10]),  # excess kurtosis
        lambda: g.dsr_gate(sr=0.1, T=500, skew=0.0, kurtosis=3.0, n_trials=[0]),
        lambda: g.parse_n_trials("5,0,50"),
        lambda: g.dsr_gate(sr=0.1, T=500, skew=0.0, kurtosis=3.0, n_trials=[5], threshold=0.0),
        lambda: g.dsr_gate_from_returns(np.full(200, 0.001), n_trials=[5]),  # constant series
        lambda: g.dsr_gate_from_returns(np.array([0.01, np.nan, 0.02, -0.01]), n_trials=[5]),
    ],
    ids=["excess-kurtosis", "zero-trials", "zero-in-range", "threshold-0", "constant", "nan"],
)
def test_dsr_inputs_that_would_pass_by_accident_are_refused(call):
    with pytest.raises(g.GateInputError):
        call()


# ── Newey-West: parity ─────────────────────────────────────────────────────


def _textbook_nw_tstat(r, lags):
    n, mu = len(r), sum(r) / len(r)
    e = [x - mu for x in r]
    gamma = [sum(e[t] * e[t - j] for t in range(j, n)) / n for j in range(lags + 1)]
    lrv = gamma[0] + 2 * sum((1 - j / (lags + 1)) * gamma[j] for j in range(1, lags + 1))
    return mu / math.sqrt(lrv / n)


@pytest.mark.parametrize("lags, expected", [(None, 1.8348503448780675), (10, 2.0131233084373665)])
def test_newey_west_reproduces_the_plugin_pinned_values(lags, expected):
    r, _ = series()
    out = g.nw_gate(r, lags=lags)
    assert out["nw_lags"] == (5 if lags is None else lags)
    assert out["nw_tstat"] == pytest.approx(_textbook_nw_tstat(r.tolist(), out["nw_lags"]), rel=REL)
    assert out["nw_tstat"] == pytest.approx(expected, rel=REL)


def test_oos_periods_measures_the_t_stat_on_the_last_rows_only():
    r = np.concatenate([np.tile([0.02, 0.01, 0.015, 0.012], 100), series()[0][:300]])
    oos = g.nw_gate(r, oos_periods=300)
    alone = g.nw_gate(r[-300:])
    whole = g.nw_gate(r)
    assert oos["n_obs"] == 300 and oos["n_obs_file"] == r.size
    assert oos["nw_tstat"] == pytest.approx(alone["nw_tstat"], rel=REL)
    assert whole["gate_pass"] is True and oos["gate_pass"] is False


def test_a_short_window_fails_even_with_a_strong_t_and_cannot_be_inflated():
    r = np.tile([0.02, 0.01, 0.015, 0.012], 25)
    out = g.nw_gate(r, min_oos_periods=252)
    assert out["nw_pass"] is True and out["oos_window_pass"] is False and out["gate_pass"] is False
    with pytest.raises(g.GateInputError):
        g.nw_gate(r, min_oos_periods=50, oos_periods=300)


@pytest.mark.parametrize("n", [200, 337, 1000])
def test_a_constant_series_cannot_compute_rather_than_passing_on_a_noise_se(n):
    """`[0.001] * 337` carries std ~2e-19, not 0.0 — the old `== 0` guard in hac.py
    waved it through to a t-stat of 3.5e16, a PASS. It must be refused (exit 2)."""
    with pytest.raises(g.GateInputError):
        g.nw_gate(np.full(n, 0.001))


# ── factor: parity ─────────────────────────────────────────────────────────


def test_factor_alpha_reproduces_the_plugin_pinned_values():
    r, F = series()
    out = g.factor_gate(r, F, ["a", "b"])
    assert out["alpha"] == pytest.approx(0.0007002090691427459, rel=REL)
    assert out["alpha_tstat"] == pytest.approx(2.178202502400045, rel=REL)
    assert out["betas"]["a"] == pytest.approx(1.1478196310858937, rel=REL)
    assert out["betas"]["b"] == pytest.approx(0.8154644988372695, rel=REL)
    assert out["r_squared"] == pytest.approx(0.7712910830912337, rel=REL)
    assert out["gate_pass"] is True


def test_a_constant_risk_free_rate_shifts_alpha_by_exactly_rf():
    r, F = series()
    raw = g.factor_gate(r, F, ["a", "b"])
    net = g.factor_gate(r, F, ["a", "b"], rf=0.0001)
    assert net["alpha"] == pytest.approx(raw["alpha"] - 0.0001, rel=1e-9) and net["rf"] == 0.0001


@pytest.mark.parametrize(
    "call",
    [
        lambda r, F: g.factor_gate(r, np.empty((r.size, 0)), []),  # intercept-only
        lambda r, F: g.factor_gate(r, np.column_stack([F[:, 0], 2 * F[:, 0]]), ["a", "a2"]),  # collinear
        lambda r, F: g.factor_gate(r[:3], F[:3], ["a", "b"]),  # too few rows for 3 params
        lambda r, F: g.factor_gate(np.full(r.size, 0.001), F, ["a", "b"]),  # constant y: t was 3e16
        lambda r, F: g.factor_gate(0.001 + F @ [1.0, 2.0], F, ["a", "b"]),  # exact fit: no residual
    ],
    ids=["intercept-only", "collinear", "too-few", "constant-y", "perfect-fit"],
)
def test_factor_inputs_that_cannot_compute_are_refused(call):
    r, F = series()
    with pytest.raises(g.GateInputError):
        call(r, F)


# ── paired stationary block bootstrap: known answers ──────────────────────


def test_stationary_bootstrap_indices_walk_forward_circularly_within_blocks():
    rng = np.random.default_rng(0)
    n = 50
    idx = g.stationary_bootstrap_indices(n, mean_block=10, rng=rng)
    assert idx.shape == (n,) and idx.min() >= 0 and idx.max() < n
    steps = (np.diff(idx) % n) == 1
    # a new block starts with prob 1/mean_block, so most steps continue the block
    assert 0.6 < steps.mean() < 1.0


def test_mean_block_one_is_the_iid_bootstrap():
    rng = np.random.default_rng(1)
    idx = np.concatenate([g.stationary_bootstrap_indices(1000, mean_block=1, rng=rng) for _ in range(5)])
    # with p = 1 every observation starts a new block: consecutive steps are chance (~1/n)
    assert ((np.diff(idx) % 1000) == 1).mean() < 0.01


def test_identical_series_never_beat_each_other():
    """Every draw's difference is exactly 0; 'above 0' is strict, so P = 0 and it FAILS."""
    r, _ = series()
    out = g.paired_block_bootstrap(r, r, metric="sharpe", draws=200, mean_block=20, seed=3)
    assert out["point_estimate"] == 0.0 and out["probability"] == 0.0 and out["gate_pass"] is False


def test_a_constant_mean_uplift_is_found_in_every_draw():
    r, _ = series()
    out = g.paired_block_bootstrap(r + 0.001, r, metric="mean", draws=300, mean_block=20, seed=4)
    assert out["point_estimate"] == pytest.approx(0.001, rel=1e-9)
    assert out["quantiles"]["0.025"] == pytest.approx(0.001, rel=1e-9)
    assert out["quantiles"]["0.975"] == pytest.approx(0.001, rel=1e-9)
    assert out["probability"] == 1.0 and out["gate_pass"] is True


def test_sharpe_is_scale_invariant_in_every_paired_draw():
    r, _ = series()
    out = g.paired_block_bootstrap(3.0 * r, r, metric="sharpe", draws=200, mean_block=10, seed=5)
    assert abs(out["quantiles"]["0.025"]) < 1e-12 and abs(out["quantiles"]["0.975"]) < 1e-12


def test_the_circular_stationary_bootstrap_is_unbiased_for_the_mean():
    """E*[mean of a circular stationary resample] = the sample mean, exactly in expectation."""
    r, _ = series()
    base = np.zeros_like(r)
    out = g.paired_block_bootstrap(r, base, metric="mean", draws=4000, mean_block=20, seed=6)
    se = out["bootstrap_std"] / math.sqrt(4000)
    assert abs(out["bootstrap_mean"] - r.mean()) < 4 * se


def test_the_bootstrap_is_reproducible_from_its_seed():
    r, _ = series()
    b = np.roll(r, 7)
    a1 = g.paired_block_bootstrap(r, b, draws=200, seed=11)
    a2 = g.paired_block_bootstrap(r, b, draws=200, seed=11)
    assert a1 == a2


def test_max_drawdown_ratio_leg_below_threshold():
    """A leg 'candidate MDD <= 0.70 x baseline MDD': halving the exposure halves the
    log-free drawdown roughly, and the ratio leg passes with 'below'."""
    r, _ = series()
    out = g.paired_block_bootstrap(
        0.3 * r, r, metric="max_drawdown", compare="ratio", pass_if="below", threshold=0.70, draws=300, seed=2
    )
    assert out["point_estimate"] < 0.70 and out["probability"] > 0.95 and out["gate_pass"] is True


@pytest.mark.parametrize(
    "kw",
    [
        {"mean_block": 0},
        {"mean_block": 10_000},
        {"draws": 0},
        {"min_probability": 0.0},
        {"metric": "calmar"},
        {"compare": "pct"},
        {"pass_if": "sideways"},
        {"threshold": float("nan")},
    ],
)
def test_bootstrap_arguments_that_cannot_compute_are_refused(kw):
    r, _ = series()
    with pytest.raises(g.GateInputError):
        g.paired_block_bootstrap(r, r, **kw)


def test_bootstrap_refuses_unpaired_lengths_and_nan():
    r, _ = series()
    with pytest.raises(g.GateInputError):
        g.paired_block_bootstrap(r, r[:-1])
    bad = r.copy()
    bad[3] = np.nan
    with pytest.raises(g.GateInputError):
        g.paired_block_bootstrap(bad, r)


# ── largest drawdown episode: known answers ───────────────────────────────


def test_largest_episode_by_hand():
    # equity 1.1, .99, .891, .93555, 1.12266, 1.1338866, 1.0771923, 1.0987361
    # deepest: peak 1.1 (row 0) -> trough row 2 (-19%) -> recovered at row 4
    r = np.array([0.10, -0.10, -0.10, 0.05, 0.20, 0.01, -0.05, 0.02])
    ep = g.largest_drawdown_episode(r)
    assert (ep["start"], ep["trough"], ep["end"]) == (1, 2, 4)
    assert ep["depth"] == pytest.approx(0.19, rel=1e-12) and ep["recovered"] is True


def test_an_unrecovered_episode_runs_to_the_end_and_a_leading_loss_starts_at_row_zero():
    ep = g.largest_drawdown_episode(np.array([0.05, -0.20, 0.01, 0.02]))
    assert (ep["start"], ep["end"], ep["recovered"]) == (1, 3, False)
    lead = g.largest_drawdown_episode(np.array([-0.5, 0.1, 1.0, 0.01, 0.02]))
    assert (lead["start"], lead["trough"], lead["end"]) == (0, 0, 2)


def test_a_series_that_never_draws_down_has_no_episode():
    assert g.largest_drawdown_episode(np.array([0.01, 0.02, 0.0, 0.03])) is None


def test_episode_gate_re_evaluates_the_leg_without_the_episode():
    r = np.array([0.10, -0.10, -0.10, 0.05, 0.20, 0.01, -0.05, 0.02])
    out = g.episode_gate(r, metric="mean", pass_if="above", threshold=0.017)  # full mean 0.01625
    assert out["full"]["value"] == pytest.approx(r.mean(), rel=1e-12)
    assert out["ex_episode"]["value"] == pytest.approx(0.02, rel=1e-12)  # rows 0,5,6,7
    assert out["ex_episode"]["n_obs"] == 4 and out["episode"]["n_removed"] == 4
    assert out["full"]["pass"] is False and out["gate_pass"] is True


def test_the_episode_is_taken_from_the_baseline_when_one_is_given():
    base = np.array([0.10, -0.10, -0.10, 0.05, 0.20, 0.01, -0.05, 0.02])
    cand = base + 0.01
    out = g.episode_gate(cand, base, metric="mean", threshold=0.0)
    assert out["episode_source"] == "baseline" and out["episode"]["start"] == 1
    assert out["ex_episode"]["value"] == pytest.approx(0.01, rel=1e-9) and out["gate_pass"] is True


def test_passing_only_with_the_episode_is_flagged():
    # the whole edge sits in the rebound after the crash: drop the episode and it is gone
    r = np.array([0.0, -0.3, 0.6, 0.0, 0.0, 0.0])
    out = g.episode_gate(r, metric="mean", threshold=0.0)
    assert out["full"]["pass"] is True and out["gate_pass"] is False and out["passes_only_with_episode"] is True


def test_an_episode_that_swallows_the_series_cannot_compute():
    with pytest.raises(g.GateInputError):
        g.episode_gate(np.array([-0.5, 0.1, 1.0]), metric="mean")


# ── CLI: exit codes 0 / 1 / 2 and --json ───────────────────────────────────

runner = CliRunner()


def _csv(path: Path, frame: pd.DataFrame) -> str:
    frame.to_csv(path, index=False, lineterminator="\r\n")  # CRLF on purpose
    return str(path)


@pytest.fixture
def files(tmp_path):
    r, F = series()
    dates = pd.date_range("2020-01-01", periods=r.size, freq="D").strftime("%Y-%m-%d")
    nan = r.copy()
    nan[7] = np.nan
    return {
        "series": _csv(tmp_path / "r.csv", pd.DataFrame({"date": dates, "returns": r})),
        "uplift": _csv(tmp_path / "u.csv", pd.DataFrame({"date": dates, "returns": r + 0.001})),
        "strong": _csv(tmp_path / "s.csv", pd.DataFrame({"returns": np.tile([0.02, 0.01, 0.015, 0.012], 100)})),
        "nan": _csv(tmp_path / "n.csv", pd.DataFrame({"returns": nan})),
        "factors": _csv(tmp_path / "f.csv", pd.DataFrame({"date": dates, "a": F[:, 0], "b": F[:, 1]})),
        "factors_rf": _csv(
            tmp_path / "frf.csv", pd.DataFrame({"date": dates, "a": F[:, 0], "b": F[:, 1], "rf": 0.0001})
        ),
        "garbage": _csv(tmp_path / "g.csv", pd.DataFrame({"returns": ["x", "y"]})),
        "parquet": str(_parquet(tmp_path / "r.parquet", pd.DataFrame({"returns": r}))),
    }


def _parquet(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_parquet(path)
    return path


@pytest.mark.parametrize(
    "argv, code",
    [
        (["nw", "--returns", "{strong}", "--min-oos-periods", "252"], 0),
        (["dsr", "--returns", "{series}", "--n-trials", "1,5,100"], 1),
        (["dsr", "--returns", "{parquet}", "--n-trials", "1"], 0),
        (["factor", "--returns", "{series}", "--factors", "{factors}"], 0),
        (["factor", "--returns", "{series}", "--factors", "{factors_rf}", "--rf", "rf"], 1),  # rf eats the alpha
        (["bootstrap", "--returns", "{uplift}", "--baseline", "{series}", "--metric", "mean", "--draws", "200"], 0),
        (["bootstrap", "--returns", "{series}", "--baseline", "{series}", "--draws", "200"], 1),
        (["episode", "--returns", "{series}", "--metric", "mean", "--threshold", "1.0"], 1),
        (["nw", "--returns", "{nan}"], 2),  # NaN refused, not dropped
        (["nw", "--returns", "{nan}", "--allow-nonfinite-drop", "--min-oos-periods", "10"], 1),
        (["dsr", "--returns", "{garbage}"], 2),  # unreadable is not a FAIL
        (["dsr", "--returns", "/no/such/file.csv"], 2),
        (["dsr", "--sharpe", "1.5", "--n-trials", "10"], 2),  # no default moments
        (["dsr", "--sharpe", "2.2", "--skew", "0", "--kurtosis", "3", "--n-obs", "730"], 2),  # no periods
        (["factor", "--returns", "{series}", "--factors", "{factors}", "--rf", "nope"], 2),
        (["bootstrap", "--returns", "{series}", "--baseline", "{strong}"], 2),  # no shared dates
        (["episode", "--returns", "{series}", "--metric", "calmar"], 2),
    ],
    ids=[
        "nw-pass",
        "dsr-fail",
        "dsr-parquet-pass",
        "factor-pass",
        "factor-rf-column-fails",
        "bootstrap-pass",
        "bootstrap-fail",
        "episode-fail",
        "nan",
        "nan-dropped-on-request",
        "garbage",
        "missing-file",
        "sharpe-alone",
        "sharpe-without-periods",
        "rf-unknown",
        "bootstrap-unpaired",
        "episode-unknown-metric",
    ],
)
def test_exit_codes_keep_could_not_compute_apart_from_failed(files, argv, code):
    argv = ["gates", *[a.format(**files) for a in argv], "--json"]
    res = runner.invoke(app, argv)
    assert res.exit_code == code, res.output
    if code == 2:
        assert "error" in json.loads(res.stderr)
    else:
        payload = json.loads(res.stdout)
        assert payload["gate"] == argv[1] and payload["gate_pass"] is (code == 0)


def test_cli_json_matches_the_function(files):
    res = runner.invoke(app, ["gates", "nw", "--returns", files["strong"], "--json"])
    direct = g.nw_gate(np.tile([0.02, 0.01, 0.015, 0.012], 100))
    assert json.loads(res.stdout)["nw_tstat"] == pytest.approx(direct["nw_tstat"], rel=1e-12)


def test_cli_without_json_prints_a_verdict_line(files):
    res = runner.invoke(app, ["gates", "nw", "--returns", files["strong"]])
    assert res.exit_code == 0 and res.stdout.startswith("nw: PASS")
