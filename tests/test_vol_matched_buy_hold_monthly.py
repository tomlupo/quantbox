"""``strategy.vol_matched_buy_hold.v1`` with ``rv_window: ME``: Moreira-Muir c / RV2 (H59, TOM-3701).

A seeded excess-return index on business days from 1926-07 (UTC, as the engine serves it). The
expected numbers are computed here from the definition, not from the plugin's helpers. The exact
match against H59's own strategy.py on Ken French data is the card's proof (TOM-3701 run folder),
not a unit test: it needs the dataset and the lab's file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.strategies.vol_matched_buy_hold import VolMatchedBuyHoldStrategy

DAYS = pd.bdate_range("1926-07-01", "1940-12-31", tz="UTC")
WINDOW = ["1926-07", "1935-12"]


def _returns() -> np.ndarray:
    rng = np.random.default_rng(59)
    vol = np.repeat(rng.uniform(0.004, 0.02, len(DAYS) // 21 + 1), 21)[: len(DAYS)]
    return 0.0003 + vol * rng.standard_normal(len(DAYS))


def _prices(x: np.ndarray | None = None) -> pd.DataFrame:
    x = _returns() if x is None else x
    return pd.DataFrame({"MKT": np.cumprod(1 + x), "OTHER": 1.0}, index=DAYS)


def _run(prices: pd.DataFrame, **params) -> dict:
    base = {"ticker": "MKT", "vol_power": 2, "rv_window": "ME", "c_window": WINDOW}
    return VolMatchedBuyHoldStrategy().run({"prices": prices}, {**base, **params})


def _monthly(prices: pd.DataFrame, power: int = 2, centred: bool = False) -> pd.DataFrame:
    """The definition, by hand: per month its last bar, sum of squared daily returns, compounded return."""
    px = prices["MKT"]
    month = px.index.tz_localize(None).to_period("M")
    x = px.pct_change()
    if centred:
        x = x - x.groupby(month).transform("mean")
    rv2 = (x**2).groupby(month).sum(min_count=1)
    vol = rv2 if power == 2 else rv2**0.5
    f = px.groupby(month).last().pct_change()
    bar = pd.Series(px.index, index=px.index).groupby(month).last()
    return pd.DataFrame({"bar": bar, "vol": vol, "f": f, "pair": f / vol.shift(1)})


@pytest.mark.parametrize(("power", "centred"), [(2, False), (1, False), (2, True)])
def test_weight_is_full_window_c_over_this_months_vol(power: int, centred: bool) -> None:
    prices = _prices()
    form = "centred" if centred else "raw_excess"
    w = _run(prices, vol_power=power, rv_form=form)["weights"]["MKT"]
    m = _monthly(prices, power, centred)
    fit = m.loc["1926-07":"1935-12"].dropna(subset=["f", "pair"])
    c = fit["f"].std() / fit["pair"].std()
    for month in ("1926-07", "1931-06", "1940-11"):
        row = m.loc[pd.Period(month, "M")]
        assert w.loc[row["bar"]] == pytest.approx(c / row["vol"], rel=1e-12)


def test_weight_moves_only_on_decision_bars_and_starts_at_the_first() -> None:
    prices = _prices()
    out = _run(prices)
    w = out["weights"]["MKT"]
    bars = set(_monthly(prices)["bar"])
    changes = w.index[w.diff().fillna(1.0) != 0]
    assert set(changes) <= bars, "a weight moved off a decision bar"
    first = pd.Timestamp("1926-07-30", tz="UTC")  # July 1926's last business day
    # The book starts at the first decision, as H59's does; the engine lags it one bar (ADR-0005).
    assert w.index.equals(prices.index[prices.index >= first]), "no rows before the first decision, none shifted"
    assert (w > 0).all()
    assert out["details"]["first_decision_bar"] == str(first)
    assert (out["weights"]["OTHER"] == 0.0).all()


@pytest.mark.parametrize("rule", ["full", "expanding"])
def test_no_weight_uses_data_after_its_decision_bar(rule: str) -> None:
    """Shocking every return after 1937-03-31 leaves every weight through that bar unchanged."""
    cut = pd.Timestamp("1937-03-31", tz="UTC")
    x = _returns()
    shocked = x.copy()
    shocked[cut < DAYS] *= 5.0
    base = _run(_prices(x), c_rule=rule)["weights"]["MKT"]
    moved = _run(_prices(shocked), c_rule=rule)["weights"]["MKT"]
    pd.testing.assert_series_equal(base.loc[:cut], moved.loc[:cut])
    assert moved.loc[DAYS[cut < DAYS][0]] == base.loc[cut], "April 1937 starts on March's weight"


def test_expanding_c_waits_for_min_months_pairs() -> None:
    w = _run(_prices(), c_rule="expanding", min_months=36)["weights"]["MKT"]
    assert w.index[0] == pd.Timestamp("1929-07-31", tz="UTC"), "36 pairs 1926-08..1929-07"


def test_first_decision_drops_earlier_months() -> None:
    w = _run(_prices(), first_decision="1930-01", c_window=["1930-01", "1935-12"])["weights"]["MKT"]
    assert w.index[0] == pd.Timestamp("1930-01-31", tz="UTC")


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"vol_power": 3}, "vol_power"),
        ({"rv_window": "W"}, "rv_window"),
        ({"c_rule": "rolling"}, "unknown c_rule"),
        ({"rv_form": "log"}, "unknown rv_form"),
        ({"c_window": None}, "needs c_window"),
        ({"c_window": ["2050-01", "2060-12"]}, "holds 0 months"),
        ({"vol_lookback": 60}, "vol_lookback"),
    ],
)
def test_refuses_what_it_cannot_compute(params: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _run(_prices(), **params)


def test_daily_path_vol_power_2_squares_the_scale_and_defaults_are_untouched() -> None:
    prices = _prices()
    one = VolMatchedBuyHoldStrategy(ticker="MKT", target_annual_vol=0.2, trading_days=252).run({"prices": prices})
    two = VolMatchedBuyHoldStrategy(ticker="MKT", target_annual_vol=0.2, trading_days=252, vol_power=2).run(
        {"prices": prices}
    )
    sigma = prices["MKT"].pct_change().std()
    assert one["weights"]["MKT"].iloc[-1] == 0.2 / (sigma * np.sqrt(252))
    assert two["weights"]["MKT"].iloc[-1] == pytest.approx((0.2 / (sigma * np.sqrt(252))) ** 2, rel=1e-12)
