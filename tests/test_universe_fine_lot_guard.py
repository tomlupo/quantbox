"""Tests for the small-book fine-lot tradeability guard in ``select_universe``.

At a small book the minimum tradeable increment (1 lot = ``10**-szDecimals``
base units) can be a large fraction of a leg, so a coin whose 1-lot notional is
too coarse relative to the exchange min-notional floor cannot be sized to target.
The guard excludes such coins from the screened universe. These tests pin:

1. a coarse-lot coin is excluded while fine-lot coins are kept;
2. the guard is a no-op when disabled (default thresholds = 0);
3. unknown szDecimals fails closed (excluded), so a mis-wired guard never
   silently admits a coarse-lot coin;
4. the guard is point-in-time: a coin that becomes coarse-lot only after a price
   rally drops out of the universe on exactly the dates it is coarse.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.strategies._universe import select_universe


def _frame(cols_values: dict[str, list[float]], n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({c: v for c, v in cols_values.items()}, index=idx)


def test_coarse_lot_coin_excluded() -> None:
    # A: $1 @ szDec=2 -> lot $0.01 (fine). B: $2000 @ szDec=0 -> lot $2000
    # (coarse). C: $1 @ szDec=0 -> lot $1 (fine). threshold = 0.2 * 10 = $2.
    prices = _frame({"A": [1.0] * 10, "B": [2000.0] * 10, "C": [1.0] * 10})
    volume = _frame({"A": [1e6] * 10, "B": [1e6] * 10, "C": [1e6] * 10})

    mask = select_universe(
        prices,
        volume,
        market_cap=None,
        top_by_mcap=30,
        top_by_volume=3,
        exclude_tickers=[],
        volume_is_dollar=True,
        fine_lot_sz_decimals={"A": 2, "B": 0, "C": 0},
        fine_lot_min_notional=10.0,
        fine_lot_max_lot_fraction=0.2,
    )

    assert (mask["B"] == 0).all()  # coarse coin never selected
    assert (mask["A"] == 1).all()  # fine coins selected (all pass volume cut)
    assert (mask["C"] == 1).all()


def test_guard_disabled_keeps_coarse_coin() -> None:
    prices = _frame({"A": [1.0] * 10, "B": [2000.0] * 10})
    volume = _frame({"A": [1e6] * 10, "B": [1e6] * 10})

    # No fine-lot args -> guard off -> coarse coin B is selectable on volume.
    mask = select_universe(
        prices,
        volume,
        market_cap=None,
        top_by_mcap=30,
        top_by_volume=2,
        exclude_tickers=[],
        volume_is_dollar=True,
    )
    assert (mask["B"] == 1).all()


def test_unknown_szdecimals_fails_closed() -> None:
    # C has a fine lot ($1) but is absent from the szDecimals map -> excluded.
    prices = _frame({"A": [1.0] * 10, "C": [1.0] * 10})
    volume = _frame({"A": [1e6] * 10, "C": [1e6] * 10})

    mask = select_universe(
        prices,
        volume,
        market_cap=None,
        top_by_mcap=30,
        top_by_volume=2,
        exclude_tickers=[],
        volume_is_dollar=True,
        fine_lot_sz_decimals={"A": 2},  # C missing
        fine_lot_min_notional=10.0,
        fine_lot_max_lot_fraction=0.2,
    )
    assert (mask["A"] == 1).all()
    assert (mask["C"] == 0).all()


def test_guard_is_point_in_time() -> None:
    # D: $1 for first 5 days (lot $1 <= $2, fine), $5 thereafter (lot $5 > $2,
    # coarse). It must be in the universe only while it is fine-lot.
    prices = _frame({"A": [1.0] * 10, "D": [1.0] * 5 + [5.0] * 5})
    volume = _frame({"A": [1e6] * 10, "D": [1e6] * 10})

    mask = select_universe(
        prices,
        volume,
        market_cap=None,
        top_by_mcap=30,
        top_by_volume=2,
        exclude_tickers=[],
        volume_is_dollar=True,
        fine_lot_sz_decimals={"A": 2, "D": 0},
        fine_lot_min_notional=10.0,
        fine_lot_max_lot_fraction=0.2,
    )
    assert (mask["D"].iloc[:5] == 1).all()  # fine-lot early
    assert (mask["D"].iloc[5:] == 0).all()  # coarse-lot after the rally
    assert (mask["A"] == 1).all()
