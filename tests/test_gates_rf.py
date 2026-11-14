"""``quantbox gates dsr|nw --rf dataset:<name>:<column>`` (TOM-3764).

H58's loop built an excess-return file by hand to run DSR and NW on returns over
the T-bill (and crashed once on the returns file's ``date`` column). The gates now
read the risk-free column from a pinned returns dataset and subtract it on the
returns' own dates: period t from period t, never another period's rate.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from quantbox.cli import app

runner = CliRunner()
N = 300  # months


def _month_ends(n: int, start: str = "1990-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME", name="date")


@pytest.fixture
def line(tmp_path, monkeypatch):
    """A research line with its own frozen monthly returns dataset ``bills`` and a lock pinning it; cwd = the line."""
    monkeypatch.delenv("QUANTBOX_DATASETS_ROOT", raising=False)
    monkeypatch.setattr("quantbox.dataset_lock.datasets_root", _no_root)
    rng = np.random.default_rng(58)
    table = pd.DataFrame(
        {"EQ": rng.normal(0.008, 0.04, N), "TBILL": np.abs(rng.normal(0.003, 0.001, N))}, index=_month_ends(N)
    )
    for name, kind in (("bills", "returns"), ("levels", "prices")):
        ds = tmp_path / "datasets" / name
        ds.mkdir(parents=True)
        (table + (1.0 if kind == "prices" else 0.0)).to_parquet(ds / "monthly.parquet")
        sha = hashlib.sha256((ds / "monthly.parquet").read_bytes()).hexdigest()
        manifest = {
            "name": name,
            "refresh": "frozen",
            "source": "test",
            "pin_file": "monthly.parquet",
            "content_hashes": {"monthly.parquet": {"sha256": sha}},
            "serve": {"table": "monthly", "kind": kind, "frequency": "monthly", "columns": ["EQ", "TBILL"]},
        }
        (ds / "manifest.yaml").write_text(yaml.safe_dump(manifest))
        with (tmp_path / "datasets.lock").open("a") as lock:
            lock.write(f"{name}: {sha}\n")
    monkeypatch.chdir(tmp_path)
    return {"root": tmp_path, "table": table}


def _no_root():
    from quantbox.dataset_lock import DatasetResolveError

    raise DatasetResolveError("no datasets root")


def _returns_file(line, *, utc: bool = True, shuffle: bool = False, months: int = N - 10, start: int = 5) -> str:
    """A run's returns.parquet: a ``date`` column (UTC, as quantbox writes it) and ``returns``."""
    rng = np.random.default_rng(7)
    dates = _month_ends(N)[start : start + months]
    frame = pd.DataFrame(
        {"date": dates.tz_localize("UTC") if utc else dates, "returns": rng.normal(0.009, 0.04, months)}
    )
    if shuffle:
        frame = frame.sample(frac=1.0, random_state=3)
    path = line["root"] / f"returns-{utc}-{shuffle}-{months}-{start}.parquet"
    frame.to_parquet(path, index=False)
    return str(path)


def _hand_excess(line, returns: str) -> str:
    """What the loop did by hand: returns minus the bill, same month, written as a plain series."""
    r = pd.read_parquet(returns).sort_values("date")
    rf = line["table"]["TBILL"]
    excess = r["returns"].to_numpy() - rf.reindex(pd.DatetimeIndex(r["date"]).tz_localize(None)).to_numpy()
    path = line["root"] / "excess.csv"
    pd.DataFrame({"returns": excess}).to_csv(path, index=False)
    return str(path)


def _json(argv: list[str]) -> tuple[int, dict]:
    res = runner.invoke(app, ["gates", *argv, "--json"])
    return res.exit_code, json.loads(res.stdout if res.exit_code in (0, 1) else res.stderr)


@pytest.mark.parametrize(("gate", "key"), [("dsr", "dsr_conservative"), ("nw", "nw_tstat")])
@pytest.mark.parametrize("utc", [True, False])
def test_rf_gives_the_gate_the_excess_return_a_hand_subtraction_gives(line, gate, key, utc):
    returns = _returns_file(line, utc=utc)
    extra = ["--n-trials", "1,10"] if gate == "dsr" else ["--min-oos-periods", "10"]
    _, by_hand = _json([gate, "--returns", _hand_excess(line, returns), *extra])
    _, with_rf = _json([gate, "--returns", returns, "--rf", "dataset:bills:TBILL", *extra])
    assert with_rf[key] == by_hand[key]
    rf = with_rf["rf"]
    assert (rf["source"], rf["dataset"], rf["column"], rf["n_periods"]) == (
        "dataset:bills:TBILL",
        "bills",
        "TBILL",
        N - 10,
    )
    assert rf["sha256"] and (rf["first"], rf["last"]) == ("1990-06-30", "2014-07-31")


def test_rf_lowers_the_mean_by_the_rate_not_raises_it(line):
    returns = _returns_file(line)
    _, plain = _json(["nw", "--returns", returns, "--min-oos-periods", "10"])
    _, excess = _json(["nw", "--returns", returns, "--rf", "dataset:bills:TBILL", "--min-oos-periods", "10"])
    rate = line["table"]["TBILL"].iloc[5 : N - 5].mean()
    assert excess["mean_return"] == pytest.approx(plain["mean_return"] - rate, rel=1e-12)


def test_rf_pairs_by_date_in_date_order_whatever_the_file_order(line):
    _, ordered = _json(
        ["nw", "--returns", _returns_file(line), "--rf", "dataset:bills:TBILL", "--min-oos-periods", "10"]
    )
    _, shuffled = _json(
        ["nw", "--returns", _returns_file(line, shuffle=True), "--rf", "dataset:bills:TBILL", "--min-oos-periods", "10"]
    )
    assert shuffled["nw_tstat"] == ordered["nw_tstat"]


def test_rf_takes_the_same_window_as_the_returns(line):
    code, out = _json(
        [
            "dsr",
            "--returns",
            _returns_file(line),
            "--rf",
            "dataset:bills:TBILL",
            "--start",
            "2000-01-01",
            "--n-trials",
            "1",
        ]
    )
    assert code in (0, 1)
    assert (out["rf"]["first"], out["T"]) == ("2000-01-31", out["rf"]["n_periods"])


def test_a_return_date_with_no_rate_is_refused_not_dropped(line):
    # starts before the dataset: the first months have no bill return
    early = line["root"] / "early.parquet"
    dates = pd.date_range("1989-07-31", periods=40, freq="ME", tz="UTC")
    pd.DataFrame({"date": dates, "returns": 0.01}).to_parquet(early, index=False)
    code, out = _json(["nw", "--returns", str(early), "--rf", "dataset:bills:TBILL"])
    assert code == 2
    assert "no rate on 6 of 40 return dates" in out["error"] and "1989-07-31" in out["error"]


def test_daily_returns_need_period_to_meet_a_monthly_rate(line):
    daily = line["root"] / "daily.parquet"
    dates = pd.date_range("1995-01-01", "1999-12-31", freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    pd.DataFrame({"date": dates, "returns": rng.normal(0.0003, 0.01, len(dates))}).to_parquet(daily, index=False)
    code, out = _json(["nw", "--returns", str(daily), "--rf", "dataset:bills:TBILL"])
    assert code == 2 and "no rate on" in out["error"]
    code, out = _json(
        ["nw", "--returns", str(daily), "--rf", "dataset:bills:TBILL", "--period", "ME", "--min-oos-periods", "10"]
    )
    # 59, not 60: January 1995 starts on the file's first bar, so period_returns cannot show both its ends
    assert code in (0, 1) and out["rf"]["n_periods"] == out["n_obs"] == 59
    assert out["rf"]["first"] == "1995-02-28"


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["nw", "--rf", "bills:TBILL"], "must be dataset:<name>:<column>"),
        (["nw", "--rf", "dataset:bills"], "must be dataset:<name>:<column>"),
        (["nw", "--rf", "dataset:bills:NOPE"], "serves no column 'NOPE'"),
        (["nw", "--rf", "dataset:levels:TBILL"], "not returns"),
        (["nw", "--rf", "dataset:absent:TBILL"], "absent"),
        (["dsr", "--rf", "dataset:bills:TBILL"], ""),  # dsr with --returns: fine, checked below
    ],
)
def test_an_rf_that_cannot_be_read_cannot_compute(line, argv, match):
    code, out = _json([*argv, "--returns", _returns_file(line)])
    if not match:
        assert code in (0, 1)
        return
    assert code == 2 and match in out["error"]


def test_rf_needs_a_return_series_not_a_summary_sharpe(line):
    code, out = _json(
        [
            "dsr",
            "--sharpe",
            "1",
            "--skew",
            "0",
            "--kurtosis",
            "3",
            "--n-obs",
            "100",
            "--periods",
            "12",
            "--rf",
            "dataset:bills:TBILL",
        ]
    )
    assert code == 2 and "--rf is subtracted from a return series" in out["error"]


def test_the_text_line_says_what_was_subtracted(line):
    res = runner.invoke(app, ["gates", "nw", "--returns", _returns_file(line), "--rf", "dataset:bills:TBILL"])
    assert res.exit_code in (0, 1)
    assert "  rf: dataset:bills:TBILL subtracted on 290 dates, 1990-06-30..2014-07-31 (pin " in res.stdout


def test_without_rf_the_gates_read_as_before(line):
    returns = _returns_file(line)
    _, out = _json(["nw", "--returns", returns, "--min-oos-periods", "10"])
    assert "rf" not in out
    assert out["mean_return"] == pytest.approx(pd.read_parquet(returns)["returns"].mean(), rel=1e-12)
