"""Strategies whose histories end on different dates (TOM-1844).

`decided_weights` joins the strategy histories on an outer date join. With
OVERLAPPING tickers the per-ticker sum turned a missing date into 0. With
DISJOINT tickers there was no sum, so a strategy whose history ended earlier
left NaN on the last row, and `latest_weights` handed `nan` to live orders
(A = BTC to 10-12, B = ETH to 10-11: ETH was NaN on 10-12).

The decided rule: a strategy with no row on a date contributes 0 on that date,
for overlapping and disjoint tickers alike. These tests go through the runner's
public interface only: `decided_weights` and `latest_weights`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantbox.strategy_runner import decided_weights, latest_weights

TICKERS = ["BTC", "ETH", "SOL", "XRP", "ADA"]
CALENDAR = pd.date_range("2026-10-01", periods=12, freq="D")


def _result(frame: pd.DataFrame, weight: float) -> dict:
    return {"result": {"weights": frame}, "weight": weight}


def _expected(results: dict[str, dict]) -> pd.DataFrame:
    """The rule written out cell by cell: sum of weight x account over the
    strategies that HAVE a row on the date and the ticker as a column; any other
    strategy contributes 0."""
    dates = sorted(set().union(*(set(r["result"]["weights"].index) for r in results.values())))
    tickers = sorted(set().union(*(set(r["result"]["weights"].columns) for r in results.values())))
    out = pd.DataFrame(0.0, index=pd.DatetimeIndex(dates), columns=tickers)
    for info in results.values():
        frame = info["result"]["weights"]
        for date in frame.index:
            for ticker in frame.columns:
                out.loc[date, ticker] += float(frame.loc[date, ticker]) * info["weight"]
    return out


def test_the_card_case_disjoint_tickers_different_last_dates():
    """A = BTC ends 10-12, B = ETH ends 10-11: ETH is 0 on 10-12, never NaN."""
    a = pd.DataFrame({"BTC": [0.5, 0.5]}, index=pd.to_datetime(["2026-10-11", "2026-10-12"]))
    b = pd.DataFrame({"ETH": [0.3]}, index=pd.to_datetime(["2026-10-11"]))
    history = decided_weights({"a": _result(a, 1.0), "b": _result(b, 1.0)})

    assert not history.isna().any().any()
    latest = latest_weights(history)
    assert latest == {"BTC": 0.5, "ETH": 0.0}
    assert not any(math.isnan(v) for v in latest.values())


def test_overlapping_and_disjoint_follow_the_same_rule():
    """The same ragged ends, once with a shared ticker and once without: the
    missing date contributes 0 in both."""
    a_idx = pd.to_datetime(["2026-10-10", "2026-10-11", "2026-10-12"])
    b_idx = pd.to_datetime(["2026-10-10", "2026-10-11"])
    a = pd.DataFrame({"BTC": [0.2, 0.4, 0.6]}, index=a_idx)

    overlapping = decided_weights({"a": _result(a, 0.5), "b": _result(pd.DataFrame({"BTC": [0.8, 0.8]}, b_idx), 0.5)})
    disjoint = decided_weights({"a": _result(a, 0.5), "b": _result(pd.DataFrame({"ETH": [0.8, 0.8]}, b_idx), 0.5)})

    # On the last date B has no row: it adds 0 to BTC (overlapping) and puts 0 on ETH (disjoint).
    assert overlapping.loc[a_idx[-1], "BTC"] == pytest.approx(0.3)
    assert disjoint.loc[a_idx[-1], "BTC"] == pytest.approx(0.3)
    assert disjoint.loc[a_idx[-1], "ETH"] == 0.0
    pd.testing.assert_frame_equal(
        overlapping,
        _expected({"a": _result(a, 0.5), "b": _result(pd.DataFrame({"BTC": [0.8, 0.8]}, b_idx), 0.5)}),
        check_freq=False,
    )


@pytest.mark.parametrize("seed", range(200))
def test_property_no_nan_and_a_missing_row_contributes_zero(seed: int):
    """Random strategy sets: 2-4 strategies, each on a random ticker subset
    (disjoint, overlapping or mixed) and a random contiguous date window with
    its own first and last date. No input NaN -> no output NaN, on any row,
    and every cell is the rule's sum."""
    rng = np.random.default_rng(seed)
    results: dict[str, dict] = {}
    for i in range(int(rng.integers(2, 5))):
        n_tickers = int(rng.integers(1, 4))
        tickers = sorted(rng.choice(TICKERS, size=n_tickers, replace=False).tolist())
        start = int(rng.integers(0, len(CALENDAR) - 1))
        end = int(rng.integers(start + 1, len(CALENDAR) + 1))
        idx = CALENDAR[start:end]
        frame = pd.DataFrame(rng.uniform(-1, 1, size=(len(idx), n_tickers)), index=idx, columns=tickers)
        results[f"s{i}"] = _result(frame, float(rng.uniform(0.1, 1.0)))

    history = decided_weights(results)

    assert not history.isna().any().any(), f"NaN in decided weights:\n{history}"
    latest = latest_weights(history)
    assert not any(math.isnan(v) for v in latest.values())
    expected = _expected(results)
    got = history.reindex(index=expected.index, columns=expected.columns)
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-12)
    assert set(history.columns) == set(expected.columns)
    assert len(history.index) == len(expected.index)
