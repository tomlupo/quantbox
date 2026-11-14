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
        ({"c_window": ["2050-01", "2060-12"]}, "does not cover c_window"),
        ({"c_window": ["1930-01", "1930-01"]}, "holds 1 months"),
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


def _flat(month: str) -> np.ndarray:
    """The seeded returns with every return in ``month`` set to 0 (stale or forward-filled quotes)."""
    x = _returns()
    x[DAYS.tz_localize(None).to_period("M") == pd.Period(month, "M")] = 0.0
    return x


@pytest.mark.parametrize("rule", ["full", "expanding"])
def test_refuses_a_month_with_no_variance_and_names_it(rule: str) -> None:
    """RV2 = 0 would give w = c / 0 = inf, and an inf pair that poisons every later c (review, round 1)."""
    with pytest.raises(ValueError, match=r"RV2.*1931-03"):
        _run(_prices(_flat("1931-03")), c_rule=rule, min_months=6)


def test_a_flat_month_no_weight_reads_is_not_refused() -> None:
    """Only the months whose vol enters a weight or a pair are checked: first_decision drops 1927-03."""
    w = _run(_prices(_flat("1927-03")), first_decision="1930-01", c_window=["1930-01", "1935-12"])["weights"]["MKT"]
    assert np.isfinite(w).all() and w.index[0] == pd.Timestamp("1930-01-31", tz="UTC")


@pytest.mark.parametrize(
    "loaded",
    [slice("1928-01-01", None), slice(None, "1934-06-30")],  # loaded after c_window starts / ends before it
)
def test_full_refuses_a_c_window_the_loaded_data_does_not_cover(loaded: slice) -> None:
    """A partly loaded window would fit c on the months it has, a different level with no warning."""
    with pytest.raises(ValueError, match="does not cover c_window"):
        _run(_prices().loc[loaded])


def test_refuses_an_unknown_param_name() -> None:
    """A typo ('vol_pwr: 2') used to run inverse vol without a word."""
    with pytest.raises(ValueError, match="unknown param.*vol_pwr"):
        _run(_prices(), vol_pwr=2)


def test_pipeline_injected_params_are_still_accepted() -> None:
    out = VolMatchedBuyHoldStrategy(ticker="MKT").run({"prices": _prices()}, {"_pipeline_annualize": 252.0})
    assert (out["weights"]["MKT"] > 0).all()
