"""One metrics module computes the run statistics (TOM-1448).

`quantbox.metrics` holds Sharpe (and excess Sharpe vs a benchmark), IR,
tracking error, CAGR, max drawdown, top-N drawdowns and turnover. The old path
`quantbox.plugins.backtesting.metrics` re-exports the same objects (robo-lab
imports it), and `quantbox.performance.compute_performance` keeps working for
quantbox-live.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantbox
from quantbox import metrics


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-01-01", periods=400, freq="D")
    return pd.Series(rng.normal(0.0005, 0.015, 400), index=idx)


@pytest.fixture
def bench() -> pd.Series:
    rng = np.random.default_rng(8)
    idx = pd.date_range("2022-01-01", periods=400, freq="D")
    return pd.Series(rng.normal(0.0003, 0.012, 400), index=idx)


def test_old_import_path_is_the_same_module_objects():
    from quantbox.plugins.backtesting import metrics as old

    for name in (
        "compute_backtest_metrics",
        "compute_drawdown_series",
        "compute_rolling_sharpe",
        "compute_var",
        "compute_cvar",
        "compute_portfolio_var",
        "compute_portfolio_cvar",
    ):
        assert getattr(old, name) is getattr(metrics, name), name


def test_sharpe_ratio_is_the_one_in_compute_backtest_metrics(returns):
    m = metrics.compute_backtest_metrics(returns, trading_days=252, risk_free_rate=0.02)
    assert m["sharpe"] == metrics.sharpe_ratio(returns, 252, risk_free=0.02)


def test_sharpe_ratio_degenerate_inputs_are_zero():
    assert metrics.sharpe_ratio(np.array([0.01]), 365) == 0.0
    assert metrics.sharpe_ratio(np.array([0.01, 0.01, 0.01]), 365) == 0.0


def test_excess_sharpe_and_ir_vs_a_benchmark(returns, bench):
    excess = returns - bench
    expected = excess.mean() / excess.std() * np.sqrt(252)
    assert metrics.sharpe_ratio(returns, 252, risk_free=bench) == pytest.approx(expected, rel=1e-12)
    assert metrics.information_ratio(returns, bench, 252) == pytest.approx(expected, rel=1e-12)
    assert metrics.tracking_error(returns, bench, 252) == pytest.approx(excess.std() * np.sqrt(252), rel=1e-12)


def test_compute_backtest_metrics_adds_benchmark_and_turnover_keys_only_when_asked(returns, bench):
    base = metrics.compute_backtest_metrics(returns, trading_days=365)
    assert "information_ratio" not in base and "annual_turnover" not in base
    w = pd.DataFrame({"A": [0.5, 0.5, 1.0, 0.0]}, index=returns.index[:4])
    full = metrics.compute_backtest_metrics(returns, trading_days=365, benchmark=bench, weights=w)
    for k, v in base.items():  # the historical keys are untouched
        assert full[k] == v
    assert full["information_ratio"] == metrics.information_ratio(returns, bench, 365)
    assert full["excess_sharpe"] == full["information_ratio"]
    assert full["tracking_error"] == metrics.tracking_error(returns, bench, 365)
    # bars 2..4 move 0, 0.5, 1.0 → mean 0.5 per bar, x365
    assert full["annual_turnover"] == pytest.approx(0.5 * 365)


def test_max_drawdown_conventions():
    r = np.array([-0.1, 0.05, -0.2, 0.5])
    # From the first return's equity: the first-row loss is not a drawdown.
    eq = np.cumprod(1 + r)
    assert metrics.max_drawdown(r) == pytest.approx((eq[2] - eq[1]) / eq[1])
    # Starting equity 1.0 is a peak: the first-row loss opens the episode.
    assert metrics.max_drawdown(r, start_is_peak=True) == pytest.approx(eq[2] - 1.0)


def test_top_drawdowns_are_non_overlapping_and_deepest_first():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    r = pd.Series([0.0, -0.1, 0.2, 0.0, -0.3, 0.1, 0.5, -0.05], index=idx)
    top = metrics.top_drawdowns(r, n=5)
    assert list(top.columns) == ["peak", "trough", "recovery", "depth"]
    assert len(top) == 3
    assert list(top["depth"]) == sorted(top["depth"])  # deepest (most negative) first
    first = top.iloc[0]
    assert first["peak"] == idx[3] and first["trough"] == idx[4] and first["recovery"] == idx[6]
    assert first["depth"] == pytest.approx(-0.3)
    assert pd.isna(top.iloc[-1]["recovery"]) or top.iloc[-1]["trough"] == idx[1]
    assert len(metrics.top_drawdowns(r, n=1)) == 1


def test_turnover_series_conventions():
    w = pd.DataFrame({"A": [0.5, 0.5, 1.0], "B": [0.5, 0.0, 0.0]})
    assert metrics.turnover_series(w, from_flat=True).tolist() == [1.0, 0.5, 0.5]
    assert metrics.turnover_series(w).tolist() == [0.0, 0.5, 0.5]


def test_drawdown_series_on_ndarray_paths():
    paths = np.array([[1.0, 2.0, 1.0, 3.0], [1.0, 0.5, 0.25, 1.0]])
    dd = metrics.compute_drawdown_series(paths)
    assert dd.tolist() == [[0.0, 0.0, -0.5, 0.0], [0.0, -0.5, -0.75, 0.0]]


def test_total_return_compounds_every_return_the_first_included():
    """TOM-262: ``cum[-1] / cum[0] - 1`` dropped the first bar (an entry fee, a first day's loss)."""
    r = pd.Series([0.10, 0.10, -0.50], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    m = metrics.compute_backtest_metrics(r)
    assert m["total_return"] == pytest.approx(1.1 * 1.1 * 0.5 - 1, abs=1e-15)


def test_compute_performance_still_works_for_quantbox_live(returns):
    """quantbox-live's scripts/compute_performance.py imports this name; numbers pinned pre-TOM-1448.

    calmar moved -0.8428 -> -0.8368 with TOM-262: total_return (and so cagr) dropped the first
    return of the series. Every other number is unchanged.
    """
    from quantbox.performance import compute_performance

    idx = returns.index
    equity = 100 * (1 + returns).cumprod()
    flows = pd.DataFrame({"date": [idx[50], idx[200]], "amount_usdc": [10.0, -5.0]})
    out = compute_performance(equity, flows, 100.0, "2022-01-01", trading_days=365)
    assert out["risk_metrics"] == {
        "sharpe": -1.5496,
        "sortino": -2.0778,
        "max_drawdown": -0.4635,
        "max_drawdown_duration_days": 397,
        "annual_volatility": 0.2884,
        "calmar": -0.8368,
        "win_rate": 0.4687,
        "profit_factor": 0.803,
        "var_95": -0.024824,
        "cvar_95": -0.034251,
    }
    assert out["periods"]["ITD"] == {"return_pct": -41.4215, "pnl_usdc": -42.7447, "days": 399}


# ---------------------------------------------------------------------------
# Guard: no run metric is computed outside quantbox.metrics
# ---------------------------------------------------------------------------

_SRC = Path(quantbox.__file__).parent
_HOME = _SRC / "metrics.py"
# An annualised Sharpe / vol computed by hand, a hand-rolled drawdown, or a private Sharpe helper.
_DUPLICATE_RE = re.compile(
    "|".join(
        (
            r"def _annualized_sharpe",  # the four private Sharpe copies in validation/
            r"\.cummax\(\)",  # a hand-rolled drawdown
            r"np\.maximum\.accumulate\((eq|prices|wealth|equity|value)\b",  # the same, in numpy
            r"\.mean\(\)\s*/\s*[\w.()]*\.std\(\)",  # a hand-rolled (rolling) Sharpe
            r"\.mean\(\)\)?\s*/\s*std\b",  # a per-period Sharpe after a std guard
            r"np\.std\(\w+,\s*ddof=1\)\s*\*\s*np\.sqrt",  # a hand-rolled annualised tracking error
            r"\.diff\(\)(\.fillna\(\w+\))?\.abs\(\)\.sum\(axis=1\)",  # a hand-rolled turnover
            # TOM-1596: the research statistics. The HAC fit and its Bartlett weight are
            # inference: `_INFERENCE_PATTERNS["hac"]` below guards them (TOM-1618).
            r"spearmanr\(|method=[\"']spearman[\"']",  # a hand-rolled rank IC
            r"cov\([^)]*\)+(\[[^\]]*\])?\s*/\s*[\w.\[\]\"'()]*var\b",  # beta, or a risk contribution, by hand
            r"\w+\s*\*\s*\(\s*\w+\s*@\s*\w+\s*\)\s*/",  # a risk contribution w * (S @ w) / ...
            r"\(\s*[\w.]+(\[[^\]]+\])?\s*>\s*[\w.]+(\[[^\]]+\])?\s*\)\.mean\(\)",  # a hand-rolled hit rate
            r"0\.5\s*\*\s*\(.*\)\.abs\(\)\.sum\(axis=1\)",  # a hand-rolled active share
        )
    )
)

# robo-lab's own copies (robo-lab #11, TOM-1453): each must trip the guard, or a new
# pattern above is vacuous.
_LAB_COPIES = (
    'fit = sm.OLS(r, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})',
    "        w = 1 - L / (lags + 1)",
    "        corr, _ = spearmanr(chunk['signal'], chunk['fwd_ret'])",
    'ics = pd.Series({d: score.loc[d].corr(fwd.loc[d], method="spearman") for d in dates})',
    '        "beta": float(df["p"].cov(df["b"]) / df["b"].var()),',
    "            beta = float(np.cov(strategy, bench, ddof=0)[0, 1] / bench_var)",
    "        rc = w * r.apply(lambda col: col.cov(acct)) / acct.var()",
    "    rc = w * (cov @ w) / (w @ cov @ w)",
    '        "hit": float((m["p"] > m["b"]).mean()),',
    "    hit_rate = (ic_series > 0).mean()",
    '            "hit_rate": float((e > 0).mean()),',
    "    return float((0.5 * (a - b).abs().sum(axis=1))[ok].mean())",
)


@pytest.mark.parametrize("line", _LAB_COPIES)
def test_guard_catches_the_lab_copies(line):
    assert _DUPLICATE_RE.search(line) or any(rx.search(line) for rx in _INFERENCE_RE.values()), line


# Non-metric uses of the same idioms: a boolean "observed so far" mask, and a
# signal-block latch inside a strategy's entry logic.
_ALLOWED = {"instrument_calendar.py"}


def test_no_run_metric_is_hand_rolled_outside_the_metrics_module():
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        if path == _HOME or path.name in _ALLOWED or "strategies" in path.parts or "features" in path.parts:
            continue  # strategies/features compute SIZING inputs (realised vol), not run metrics
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if _DUPLICATE_RE.search(line):
                offenders.append(f"{path.relative_to(_SRC)}:{i}: {line.strip()}")
    assert not offenders, "run metric computed outside quantbox.metrics:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# Guard: no inference statistic is implemented outside quantbox.inference (TOM-1618)
# ---------------------------------------------------------------------------
#
# The metric guard above matched metric NAMES, so two NaN refusals and a second
# skew/kurtosis passed it (TOM-1618). These patterns match the MECHANICS of each
# inference class: counting non-finite rows to refuse them, the "cancelled to
# noise" test, a HAC fit, sample moments, a resample, a Monte-Carlo null.

_INFERENCE_HOMES = {_SRC / "inference.py", _SRC / "_numerics.py"}
_INFERENCE_PATTERNS = {
    "nan_refusal": r"\(~\s*(np\.isfinite\(|\w*finite\w*\)|mask\))",  # count the non-finite rows to refuse them
    "degenerate": r"<=\s*DEGENERATE_RTOL\s*\*",  # a second "constant to floating-point noise" test
    "hac": r"cov_type=[\"']HAC|1\s*-\s*\w+\s*/\s*\(\s*\w+\s*\+\s*1\s*\)",  # a HAC fit, or a Bartlett weight
    "moments": r"(stats\.|\.)(skew|kurtosis|kurt)\(|\*\*\s*[34]\)",  # sample skew / kurtosis
    "bootstrap": (  # an iid resample, a stationary block start, a moving block start
        r"\.choice\([^)]*replace=True|\.random\(\w+\)\s*<|\.integers\(0,\s*len\(\w+\)\s*-"
    ),
    "mc_null": r"\.normal\(0,\s*np\.std\(",  # a Monte-Carlo null at the series' own volatility
    # TOM-1646: a family-wise level split over the tests (Bonferroni), a Holm
    # step-down over the reversed ranks, or a second call into the library
    # multiple_testing wraps.
    "multiple_testing": (
        r"(?<![\w.+])(0?\.(05|01|1)|alpha\w*)\s*/\s*(float\()?\s*(n_\w+|len\(|m\b|ntests\b|k\b)"
        r"|np\.arange\(\s*\w+\s*,\s*0\s*,\s*-1\s*\)|multipletests\("
    ),
}
_INFERENCE_RE = {label: re.compile(p) for label, p in _INFERENCE_PATTERNS.items()}

# The copies that stood in the tree when this guard was written (origin/dev bd5865d):
# each must still trip its pattern, or the pattern is vacuous.
_INFERENCE_COPIES = (
    ("nan_refusal", "    dropped = int((~mask).sum())"),  # analysis.gates._finite
    ("nan_refusal", "    n_dropped = int((~finite_mask).sum())"),  # metrics.require_finite, analysis.dsr
    ("nan_refusal", "        n_nonfinite = int((~np.isfinite(rets)).sum())"),  # validation BLP
    ("nan_refusal", "    y_bad = int((~np.isfinite(y)).sum())"),  # analysis.hac.factor_regression
    ("nan_refusal", "        dropped = int((~finite).sum())"),  # gates_cli factor
    ("degenerate", "    if std <= DEGENERATE_RTOL * float(np.mean(np.abs(r))):"),  # analysis.gates
    ("degenerate", "        if std_period <= DEGENERATE_RTOL * scale:"),  # validation BLP
    ("hac", '    return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})'),
    ("moments", "    skew = float(np.mean(z**3))"),  # validation BLP
    ("moments", "    kurt = float(np.mean(z**4))"),
    ("moments", "        skew=float(stats.skew(r)),"),  # analysis.gates
    ("moments", "        return np.sum(((x - m) / s) ** 3) / n if s > 0 else 0"),  # simulation.engine
    ("moments", "        kurt = float(r_pct.kurt())"),  # pipeline blocks
    ("bootstrap", "            sample = rng.choice(rets, size=n, replace=True)"),  # validation statistical
    ("bootstrap", "    new = rng.random(n) < p"),  # analysis.gates stationary bootstrap
    ("bootstrap", "                start = rng.integers(0, len(returns) - block_size)"),  # simulation forecasting
    (
        "mc_null",
        "            [sharpe_ratio(rng.normal(0, np.std(rets, ddof=1), size=n), trading_days) for _ in range(n)]",
    ),
    # TOM-1646: the copies multiple_testing / bonferroni_alpha replace or must not grow.
    (
        "multiple_testing",
        "    q = 100 * 0.05 / n_tests / 2  # Bonferroni: family-wise 5% over the criteria",
    ),  # robo-lab
    ("multiple_testing", "    level = alpha / len(pvalues)"),
    ("multiple_testing", "    alphacBonf = alpha / float(ntests)"),  # statsmodels' own Bonferroni level
    ("multiple_testing", "    thresholds = alpha / np.arange(m, 0, -1)"),  # Holm step-down
    ("multiple_testing", "    adj = np.maximum.accumulate(p_sorted * np.arange(m, 0, -1))"),
    ("multiple_testing", "    reject, adj, _, _ = multipletests(p, alpha=alpha, method='holm')"),
)


@pytest.mark.parametrize(("label", "line"), _INFERENCE_COPIES)
def test_inference_guard_catches_the_copies_it_was_written_for(label, line):
    assert _INFERENCE_RE[label].search(line), (label, line)


def _inference_offenders() -> set[tuple[str, str]]:
    found = set()
    for path in sorted(_SRC.rglob("*.py")):
        if path in _INFERENCE_HOMES:
            continue
        text = path.read_text()
        for label, rx in _INFERENCE_RE.items():
            if any(rx.search(line) for line in text.splitlines()):
                found.add((str(path.relative_to(_SRC)), label))
    return found


def test_no_inference_statistic_is_implemented_outside_quantbox_inference():
    """On origin/dev bd5865d this found 22 (file, class) pairs; TOM-1618 moved every one."""
    found = sorted(_inference_offenders())
    assert not found, "inference statistic implemented outside quantbox.inference:\n" + "\n".join(map(str, found))
