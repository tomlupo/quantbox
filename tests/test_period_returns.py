"""``metrics.period_returns`` and ``quantbox gates --period/--start/--end`` (TOM-3698).

A period counts only when the data shows both its ends: an observation before it
(the opening anchor) and its last business day (the close). The report keeps its
edge months through ``keep_partial=True``, which is exactly the resample it had.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.metrics import Window, period_returns

runner = CliRunner()


@pytest.fixture(scope="module")
def daily() -> pd.Series:
    idx = pd.bdate_range("1926-07-01", "2015-12-31")
    return pd.Series(np.random.default_rng(1926).normal(0.0004, 0.01, len(idx)), index=idx, name="returns")


def _resampled(r: pd.Series, freq: str) -> pd.Series:
    return (1 + r).resample(freq).prod() - 1


# --- the card's proof -------------------------------------------------------


def test_the_1926_2015_fixture_gives_1073_months_from_august_1926(daily):
    out = period_returns(daily, "ME")
    assert len(out) == 1073
    assert out.index[0] == pd.Timestamp("1926-08-31") and out.index[-1] == pd.Timestamp("2015-12-31")
    pd.testing.assert_series_equal(out, _resampled(daily, "ME")["1926-08":"2015-12"], check_exact=True)


@pytest.mark.parametrize("freq, n, first", [("QE", 357, "1926-12-31"), ("YE", 89, "1927-12-31")])
def test_quarters_and_years_drop_the_unanchored_first_period_too(daily, freq, n, first):
    out = period_returns(daily, freq)
    assert len(out) == n and out.index[0] == pd.Timestamp(first) and out.index[-1] == pd.Timestamp("2015-12-31")
    pd.testing.assert_series_equal(out, _resampled(daily, freq)[first:], check_exact=True)


# --- the trailing period ----------------------------------------------------


def test_a_month_cut_before_its_last_business_day_is_dropped(daily):
    assert period_returns(daily[:"2015-12-30"], "ME").index[-1] == pd.Timestamp("2015-11-30")
    assert period_returns(daily, "ME", end="2015-12-15").index[-1] == pd.Timestamp("2015-11-30")


def test_a_month_that_ends_on_a_weekend_is_complete_on_its_friday():
    idx = pd.bdate_range("2022-10-03", "2022-12-30")  # 2022-12-31 is a Saturday
    r = pd.Series(0.001, index=idx)
    assert period_returns(r, "ME").index.tolist() == [pd.Timestamp("2022-11-30"), pd.Timestamp("2022-12-31")]


# --- the window -------------------------------------------------------------


def test_a_start_inside_the_data_keeps_the_month_it_opens_when_an_earlier_row_anchors_it(daily):
    # 1926-08-01 is a Sunday: a start on it or on Monday 08-02 cuts no business day of August
    for start in ("1926-08-01", "1926-08-02"):
        assert period_returns(daily, "ME", start=start).index[0] == pd.Timestamp("1926-08-31")
    assert period_returns(daily, "ME", start="1926-08-03").index[0] == pd.Timestamp("1926-09-30")


def test_the_window_is_inclusive_by_calendar_day(daily):
    w = Window("1950-03-01", "1950-03-31")
    clipped = w.clip(daily)
    assert clipped.index[0] == pd.Timestamp("1950-03-01") and clipped.index[-1] == pd.Timestamp("1950-03-31")
    intraday = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-31 16:00", "2020-02-01 09:00"]))
    assert Window(end="2020-01-31").clip(intraday).tolist() == [1.0]


def test_a_naive_window_on_a_tz_aware_series():
    r = pd.Series(0.001, index=pd.date_range("2021-01-01", "2021-04-30", freq="D", tz="UTC"))
    out = period_returns(r, "ME", start="2021-02-01", end="2021-03-31")
    assert [d.strftime("%Y-%m") for d in out.index] == ["2021-02", "2021-03"]


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(freq="W"), "freq"),
        (dict(freq="ME", start="2020-02-01", end="2020-01-01"), "after"),
    ],
)
def test_arguments_that_cannot_be_honoured_are_refused(daily, kw, match):
    with pytest.raises(ValueError, match=match):
        period_returns(daily, **kw)


def test_a_numeric_index_is_refused_not_read_as_epoch_dates():
    with pytest.raises(ValueError, match="dates"):
        period_returns(pd.Series([0.01, 0.02]), "ME")


# --- what is not a 0% month -------------------------------------------------


def test_a_period_with_a_nan_day_or_no_rows_is_nan_not_zero(daily):
    r = daily["2010-01-01":"2010-12-31"].copy()
    r["2010-03-10"] = np.nan
    r = r.drop(r["2010-06-01":"2010-06-30"].index)
    out = period_returns(r, "ME")
    assert np.isnan(out["2010-03-31"]) and np.isnan(out["2010-06-30"])
    assert out.drop([pd.Timestamp("2010-03-31"), pd.Timestamp("2010-06-30")]).notna().all()


def test_keep_partial_is_exactly_the_plain_resample_the_report_used(daily):
    cases = [
        daily["1990-03-14":"2001-06-12"],
        pd.concat([pd.Series([np.nan], index=[pd.Timestamp("1999-12-31")]), daily["2000-01-03":"2001-08-20"]]),
        pd.concat([daily["2010-01-05":"2010-04-30"], daily["2010-07-01":"2010-12-15"]]),
        pd.Series(0.002, index=pd.date_range("2021-03-17", periods=90, freq="D", tz="UTC")),
    ]
    for r in cases:
        pd.testing.assert_series_equal(
            period_returns(r, "ME", keep_partial=True), _resampled(r, "ME"), check_exact=True
        )


# --- quantbox gates ---------------------------------------------------------


@pytest.fixture
def files(tmp_path, daily):
    r = daily["1990-01-01":"2015-12-31"]
    base = r * 0.5
    f = pd.DataFrame({"mkt": r * 0.8 + 0.0001, "rf": 0.0001}, index=r.index)
    paths = {}
    for name, frame in {"r": r.to_frame("returns"), "b": base.to_frame("returns"), "f": f}.items():
        p = tmp_path / f"{name}.csv"
        frame.rename_axis("date").to_csv(p)
        paths[name] = str(p)
    return paths


def _json(argv):
    res = runner.invoke(app, ["gates", *argv, "--json"])
    return res, (json.loads(res.stdout) if res.exit_code in (0, 1) else None)


def test_nw_takes_period_start_end_and_gates_the_months(files):
    res, out = _json(["nw", "--returns", files["r"], "--period", "ME", "--min-oos-periods", "12"])
    assert res.exit_code in (0, 1), res.output
    assert out["n_obs"] == 311  # 1990-02 .. 2015-12, January 1990 has no anchor
    assert out["window"] == {"period": "ME", "start": None, "end": None}
    res, out = _json(
        [
            "nw",
            "--returns",
            files["r"],
            "--period",
            "ME",
            "--start",
            "2000-01-01",
            "--end",
            "2000-12-31",
            "--min-oos-periods",
            "12",
        ]
    )
    assert out["n_obs"] == 12


def test_start_end_alone_clip_the_daily_series(files):
    res, out = _json(
        ["nw", "--returns", files["r"], "--start", "2001-01-01", "--end", "2001-01-31", "--min-oos-periods", "5"]
    )
    assert res.exit_code in (0, 1), res.output
    assert out["n_obs"] == 23  # business days of January 2001


def test_paired_gates_compound_every_return_input(files):
    res, out = _json(
        ["bootstrap", "--returns", files["r"], "--baseline", files["b"], "--period", "QE", "--draws", "50"]
    )
    assert res.exit_code in (0, 1), res.output
    assert out["n_obs"] == 103  # 1990Q2 .. 2015Q4
    res, out = _json(
        [
            "factor",
            "--returns",
            files["r"],
            "--factors",
            files["f"],
            "--factor-columns",
            "mkt",
            "--rf",
            "rf",
            "--period",
            "YE",
            "--min-obs",
            "10",
        ]
    )
    assert res.exit_code in (0, 1), res.output
    assert out["n_obs"] == 25  # 1991 .. 2015


@pytest.mark.parametrize(
    "argv, match",
    [
        (["predreg", "--returns", "{r}", "--predictor", "{f}", "--predictor-column", "mkt", "--period", "ME"], "level"),
        (
            [
                "dsr",
                "--sharpe",
                "1.0",
                "--skew",
                "0",
                "--kurtosis",
                "3",
                "--n-obs",
                "100",
                "--periods",
                "12",
                "--start",
                "2000-01-01",
            ],
            "--returns",
        ),
        (["nw", "--returns", "{r}", "--period", "W"], "freq"),
    ],
    ids=["predreg-level", "dsr-summary", "bad-freq"],
)
def test_a_window_that_cannot_apply_cannot_compute(files, argv, match):
    res = runner.invoke(app, ["gates", *[a.format(**files) for a in argv]])
    assert res.exit_code == 2, res.output
    assert match in res.output
