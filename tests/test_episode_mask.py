"""Named episodes and the masked HAC calendar (TOM-3700).

The claim of ``episode_gate`` is re-tested without the episode, and its Newey-West
t-stat is taken on the kept rows WITH their calendar: a lag product u_t * u_{t-l}
never pairs the row before a removed window with a row after it. The hand
computations below sum the Bartlett-weighted products per run with plain numpy, so
they check statsmodels' ``hac-panel`` rather than restate it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quantbox import gates as g
from quantbox import inference
from quantbox.cli import app

runner = CliRunner()


def ar1(n: int = 300, phi: float = 0.6, mu: float = 0.003, seed: int = 11) -> np.ndarray:
    """An AR(1) return series: the serial correlation a HAC SE exists for."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 0.02, n)
    r = np.empty(n)
    r[0] = mu + e[0]
    for t in range(1, n):  # test fixture, not a hot path
        r[t] = mu + phi * (r[t - 1] - mu) + e[t]
    return r


def hand_hac(y: np.ndarray, X: np.ndarray, lags: int, runs: list[slice]) -> np.ndarray:
    """OLS SEs with a Bartlett HAC whose lag products stay inside each run (no small-sample correction)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    xu = X * (y - X @ beta)[:, None]
    S = xu.T @ xu
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        for run in runs:
            seg = xu[run]
            if len(seg) > lag:
                G = seg[lag:].T @ seg[:-lag]
                S += w * (G + G.T)
    inv = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(inv @ S @ inv))


# ── the card's proof: AR(1), one masked window ─────────────────────────────


def test_the_masked_hac_tstat_equals_the_hand_computation_whose_lags_never_cross_the_gap():
    r = ar1()
    out = g.episode_gate(r, episodes=[(100, 149)], metric="mean")
    kept = np.r_[r[:100], r[150:]]
    lags = inference.newey_west_auto_lags(kept.size)
    X = np.ones((kept.size, 1))
    se_gap = hand_hac(kept, X, lags, [slice(0, 100), slice(100, 250)])[0]
    se_glued = hand_hac(kept, X, lags, [slice(0, 250)])[0]

    nw = out["ex_episode"]["nw"]
    assert (nw["n_obs"], nw["n_segments"], nw["nw_lags"]) == (250, 2, lags)
    assert nw["mean"] == pytest.approx(kept.mean(), rel=1e-12)
    assert nw["nw_se"] == pytest.approx(se_gap, rel=1e-12)
    assert nw["nw_tstat"] == pytest.approx(kept.mean() / se_gap, rel=1e-12)
    # the compressed-time number is a different number: the test can tell them apart
    assert abs(se_gap - se_glued) > 1e-6 * se_gap


def test_the_unmasked_tstat_is_reported_unchanged_beside_it():
    r = ar1()
    out = g.episode_gate(r, episodes=[(100, 149)], metric="mean")
    plain = inference.newey_west_tstat(r)
    full = out["full"]["nw"]
    assert full["n_segments"] == 1 and full["n_obs"] == r.size
    assert full["nw_tstat"] == plain["nw_tstat"] and full["nw_se"] == plain["nw_se"]
    assert full["nw_tstat"] == g.nw_gate(r, min_oos_periods=1)["nw_tstat"]


def test_with_a_baseline_the_hac_tests_the_mean_difference():
    base = ar1(seed=3)
    cand = base + ar1(mu=0.001, seed=4) * 0.5
    out = g.episode_gate(cand, base, episodes=[(40, 79)], metric="sharpe")
    d = cand - base
    kept = np.r_[d[:40], d[80:]]
    lags = inference.newey_west_auto_lags(kept.size)
    se = hand_hac(kept, np.ones((kept.size, 1)), lags, [slice(0, 40), slice(40, kept.size)])[0]
    assert out["nw_series"] == "candidate - baseline"
    assert out["ex_episode"]["nw"]["nw_se"] == pytest.approx(se, rel=1e-12)
    assert out["full"]["nw"]["nw_tstat"] == inference.newey_west_tstat(d)["nw_tstat"]


# ── named episodes replace the largest drawdown; the default path is kept ──


R8 = np.array([0.10, -0.10, -0.10, 0.05, 0.20, 0.01, -0.05, 0.02])


def test_named_episodes_replace_the_largest_drawdown():
    out = g.episode_gate(R8, episodes=[(5, 6)], metric="mean")
    assert out["episode_source"] == "named" and out["episode"] is None
    assert out["episodes"] == [{"start": 5, "end": 6, "n_removed": 2}]
    assert out["ex_episode"]["value"] == pytest.approx(np.r_[R8[:5], R8[7:]].mean(), rel=1e-12)
    assert out["ex_episode"]["n_obs"] == 6


def test_several_windows_and_a_window_at_the_edge():
    out = g.episode_gate(ar1(), episodes=[(0, 9), (100, 149), (290, 299)], metric="mean")
    assert out["ex_episode"]["n_obs"] == 230
    assert out["ex_episode"]["nw"]["n_segments"] == 2  # 10..99 and 150..289: the edge windows make no gap


def test_the_default_path_keeps_its_leg_and_verdict_and_lists_its_one_episode():
    out = g.episode_gate(R8, metric="mean", pass_if="above", threshold=0.017)
    assert out["episode_source"] == "returns" and out["episode"]["start"] == 1
    assert out["episodes"] == [{"start": 1, "end": 4, "n_removed": 4}]
    assert out["full"]["value"] == pytest.approx(R8.mean(), rel=1e-12)
    assert out["ex_episode"]["value"] == pytest.approx(0.02, rel=1e-12) and out["gate_pass"] is True
    assert out["ex_episode"]["nw"]["n_segments"] == 2  # row 0, then rows 5..7
    assert g.episode_gate(np.array([0.01, 0.02, 0.0, 0.03]), metric="mean")["episodes"] == []


@pytest.mark.parametrize(
    "episodes",
    [[], [(5, 4)], [(-1, 2)], [(3, 8)], [(1.0, 2)], [(True, 2)], [(1, 2, 3)], [7]],
    ids=["empty", "reversed", "negative", "past-the-end", "float", "bool", "triple", "scalar"],
)
def test_a_malformed_episode_is_refused(episodes):
    with pytest.raises(g.GateInputError):
        g.episode_gate(R8, episodes=episodes, metric="mean")


def test_named_episodes_that_swallow_the_series_cannot_compute():
    with pytest.raises(g.GateInputError, match="named episodes leave 1"):
        g.episode_gate(R8, episodes=[(0, 6)], metric="mean")


# ── the calendar in inference ──────────────────────────────────────────────


def test_one_run_is_the_plain_hac_fit():
    r = ar1()
    one = inference.newey_west_tstat(r, segments=np.zeros(r.size))
    assert one["nw_se"] == pytest.approx(inference.newey_west_tstat(r)["nw_se"], rel=1e-12)


def test_a_dropped_row_breaks_its_run():
    r = ar1()
    holed = r.copy()
    holed[120] = np.nan
    out = inference.newey_west_tstat(holed, allow_nonfinite_drop=True, segments=np.zeros(r.size))
    kept = np.r_[r[:120], r[121:]]
    se = hand_hac(kept, np.ones((kept.size, 1)), out["nw_lags"], [slice(0, 120), slice(120, kept.size)])[0]
    assert out["n_nonfinite_dropped"] == 1 and out["nw_se"] == pytest.approx(se, rel=1e-12)


def test_segments_must_label_every_row():
    with pytest.raises(inference.InferenceInputError, match="one label per row"):
        inference.newey_west_tstat(ar1(), segments=np.zeros(10))
    with pytest.raises(inference.InferenceInputError, match="one label per row"):
        inference.factor_regression(ar1(), ar1(seed=2), ["f"], segments=np.zeros(10))


def test_factor_alpha_se_restarts_at_the_gap():
    y, f = ar1(seed=5), ar1(seed=6)
    keep = np.ones(y.size, dtype=bool)
    keep[100:150] = False
    yk, fk = y[keep], f[keep]
    out = inference.factor_regression(yk, fk, ["f"], lags=6, segments=inference.mask_runs(keep))
    X = np.column_stack([np.ones(yk.size), fk])
    se = hand_hac(yk, X, 6, [slice(0, 100), slice(100, yk.size)])
    assert out["alpha_se"] == pytest.approx(se[0], rel=1e-10)
    plain = inference.factor_regression(yk, fk, ["f"], lags=6)
    assert plain["alpha_se"] == pytest.approx(hand_hac(yk, X, 6, [slice(0, yk.size)])[0], rel=1e-10)
    assert plain["alpha"] == out["alpha"] and plain["alpha_se"] != out["alpha_se"]


# ── short runs: lag products a run is too short to hold are empty, not an error ──


#: the largest drawdown runs from row 1 to row 6, so the kept rows 0 and 7 are two one-row runs
SINGLES = np.random.default_rng(13).normal(0, 0.02, 8)


@pytest.mark.parametrize("lags", [None, 0, 3])
def test_kept_runs_of_one_row_each_give_the_white_se_in_default_mode(lags):
    out = g.episode_gate(SINGLES, metric="mean", lags=lags)
    kept = SINGLES[[0, 7]]
    assert out["episodes"] == [{"start": 1, "end": 6, "n_removed": 6}]
    ex = out["ex_episode"]
    # the leg and the verdict are what they were before the HAC block existed
    assert (ex["value"], ex["pass"], ex["n_obs"]) == (pytest.approx(kept.mean(), rel=1e-12), True, 2)
    assert out["gate_pass"] is True
    # no lag product exists inside a one-row run: the SE is the White (HC0) one
    assert ex["nw"]["n_segments"] == 2
    assert ex["nw"]["nw_se"] == pytest.approx(hand_hac(kept, np.ones((2, 1)), 0, [])[0], rel=1e-12)


def test_named_windows_that_leave_one_row_runs_compute():
    r10 = np.random.default_rng(13).normal(0, 0.02, 10)
    out = g.episode_gate(r10, episodes=[(1, 8)], metric="mean")
    kept = r10[[0, 9]]
    assert out["ex_episode"]["nw"]["nw_se"] == pytest.approx(hand_hac(kept, np.ones((2, 1)), 0, [])[0], rel=1e-12)


def test_runs_shorter_than_the_lags_keep_the_bartlett_weights_of_the_lags_asked():
    r = ar1(n=12)
    seg = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4])  # longest run 3 rows, so lags 2+ are empty past lag 2
    runs = [slice(0, 3), slice(3, 5), slice(5, 8), slice(8, 9), slice(9, 12)]
    for lags in (2, 3, 6):
        out = inference.newey_west_tstat(r, lags=lags, segments=seg)
        assert out["nw_lags"] == lags
        assert out["nw_se"] == pytest.approx(hand_hac(r, np.ones((12, 1)), lags, runs)[0], rel=1e-12)
    X = np.column_stack([np.ones(12), ar1(n=12, seed=2)])
    fr = inference.factor_regression(r, X[:, 1], ["f"], lags=6, segments=seg)
    assert fr["alpha_se"] == pytest.approx(hand_hac(r, X, 6, runs)[0], rel=1e-10)


def test_overlapping_named_windows_are_refused():
    # a shared row would be counted twice in n_removed; name each removed row once
    with pytest.raises(g.GateInputError, match="overlap"):
        g.episode_gate(ar1(), episodes=[(10, 20), (20, 30)], metric="mean")
    out = g.episode_gate(ar1(), episodes=[(10, 20), (21, 30)], metric="mean")  # adjacent is fine
    assert sum(e["n_removed"] for e in out["episodes"]) == 300 - out["ex_episode"]["n_obs"]


# ── CLI: --episodes ────────────────────────────────────────────────────────


@pytest.fixture
def monthly(tmp_path):
    r = ar1(n=240)
    dates = pd.date_range("1920-01-31", periods=r.size, freq="ME").strftime("%Y-%m-%d")
    path = tmp_path / "m.csv"
    pd.DataFrame({"date": dates, "returns": r}).to_csv(path, index=False)
    base = tmp_path / "b.csv"
    pd.DataFrame({"date": dates, "returns": ar1(n=240, seed=9)}).to_csv(base, index=False)
    undated = tmp_path / "u.csv"
    pd.DataFrame({"returns": r}).to_csv(undated, index=False)
    return {"r": r, "path": str(path), "base": str(base), "undated": str(undated)}


def _run(*argv):
    res = runner.invoke(app, ["gates", "episode", *argv, "--json"])
    return res.exit_code, (json.loads(res.stdout) if res.exit_code in (0, 1) else res.output)


def test_cli_episodes_remove_whole_months_and_carry_both_tests(monthly):
    code, out = _run("--returns", monthly["path"], "--metric", "mean", "--episodes", "1929-09:1932-06")
    assert code in (0, 1)
    (ep,) = out["episodes"]
    # 1929-09 is row 116 (Jan 1920 = row 0); June 1932 is row 149: the whole end month is removed
    assert (ep["start"], ep["end"], ep["n_removed"]) == (116, 149, 34)
    assert (ep["start_date"][:10], ep["end_date"][:10]) == ("1929-09-30", "1932-06-30")
    direct = g.episode_gate(monthly["r"], episodes=[(116, 149)], metric="mean")
    assert out["ex_episode"]["nw"]["nw_tstat"] == pytest.approx(direct["ex_episode"]["nw"]["nw_tstat"], rel=1e-12)
    assert out["ex_episode"]["nw"]["n_segments"] == 2
    assert out["full"]["nw"]["nw_tstat"] == pytest.approx(
        inference.newey_west_tstat(monthly["r"])["nw_tstat"], rel=1e-12
    )


def test_cli_episodes_with_a_baseline_and_two_windows(monthly):
    code, out = _run(
        "--returns", monthly["path"], "--baseline", monthly["base"], "--metric", "mean",
        "--episodes", "1921:1921, 1929-09:1932-06",
    )  # fmt: skip
    assert code in (0, 1) and out["nw_series"] == "candidate - baseline"
    assert [(e["start"], e["end"]) for e in out["episodes"]] == [(12, 23), (116, 149)]
    assert out["ex_episode"]["n_obs"] == 240 - 12 - 34 and out["ex_episode"]["nw"]["n_segments"] == 3


@pytest.mark.parametrize(
    "episodes, path",
    [
        ("1950-01:1950-12", "path"),  # no row inside
        ("1929-09", "path"),  # not START:END
        ("1929-09:nope", "path"),
        ("1932-06:1929-09", "path"),  # reversed
        ("1929-09:1932-06", "undated"),  # rows without dates cannot be named
    ],
    ids=["empty-window", "no-colon", "bad-bound", "reversed", "undated"],
)
def test_cli_refuses_an_episode_it_cannot_place(monthly, episodes, path):
    code, _ = _run("--returns", monthly[path], "--metric", "mean", "--episodes", episodes)
    assert code == 2


def test_cli_without_episodes_is_unchanged(monthly):
    code, out = _run("--returns", monthly["undated"], "--metric", "mean")
    direct = g.episode_gate(monthly["r"], metric="mean")
    assert code in (0, 1) and out["episode_source"] == "returns"
    assert out["episode"]["start"] == direct["episode"]["start"]
    assert out["ex_episode"]["value"] == pytest.approx(direct["ex_episode"]["value"], rel=1e-12)


def test_cli_end_bound_covers_the_whole_last_day_of_an_intraday_series(tmp_path):
    stamps = pd.date_range("1932-06-29 10:00", periods=12, freq="6h")  # 29 Jun 10:00 .. 2 Jul 04:00
    path = tmp_path / "h.csv"
    pd.DataFrame({"date": stamps, "returns": ar1(n=12)}).to_csv(path, index=False)
    code, out = _run("--returns", str(path), "--metric", "mean", "--episodes", "1932-06-30:1932-06-30")
    assert code in (0, 1)
    (ep,) = out["episodes"]
    # 30 Jun 04:00, 10:00, 16:00 and 22:00 are all removed, 22:00 included
    assert (ep["start"], ep["end"], ep["n_removed"]) == (3, 6, 4)
    assert ep["end_date"].startswith("1932-06-30T22:00") or ep["end_date"].startswith("1932-06-30 22:00")
