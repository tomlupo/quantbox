"""Rebalancing policies and group limits in the engine seam (TOM-1450 3d-2, docs/adr/0008).

The seam owns the schedule: a policy is an ORDERS mask (and, for tranche, the
targets) that :func:`quantbox.engine.simulate` builds once, the same for every
engine. Nothing here calls an adapter directly.

- A. a known answer per policy on a small fixture: periodic weekly/monthly on a
  market calendar, tranche, band, corridor;
- B. the same orders and the same trades on vectorbt and rsims, per policy;
- C. group limits: a known answer, an infeasible limit refuses loudly, and a
  seeded randomized loop (hypothesis is not a dev dependency) asserting the
  limits hold on every rebalance date;
- D. the pipeline: ``rebalancing_policy`` and ``group_limits`` are in
  ``params_schema`` (validate) and ``plan()`` resolves them (config explain). The
  run that writes them to ``data_validation.json`` is a pipeline smoke test in
  ``tests/pipeline/test_rebalancing_policies_e2e.py`` (CI's smoke job runs that
  directory only).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantbox.engine import Costs, TradedBook, get_engine, simulate
from quantbox.engine.groups import GroupLimits, apply_group_limits, resolve_group_limits
from quantbox.engine.policy import POLICIES, RebalancePolicy, resolve_policy
from quantbox.execution import resolve_execution

VECTORBT = get_engine("vectorbt", require_installed=False).installed()
NOT_CHECKED = "NOT CHECKED: engine parity needs vectorbt (the [vectorbt] extra) — this is not a pass"
ENGINES = ["vectorbt", "rsims"] if VECTORBT else ["rsims"]
RSIMS_COMMON = {"capitalise_profits": True, "trade_buffer": 0.0, "margin": 0.0, "initial_cash": 10_000.0}


def _simulate(
    engine: str,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    schedule: str = "calendar",
    fees: float = 0.0,
    **kw: Any,
) -> TradedBook:
    params = RSIMS_COMMON if engine == "rsims" else None
    return simulate(
        prices,
        weights,
        engine=engine,
        timing=resolve_execution({"schedule": schedule}),
        costs=Costs(fees=fees),
        engine_params=params,
        **kw,
    )


def _ordered_dates(book: TradedBook) -> list[pd.Timestamp]:
    return list(book.orders.index[book.orders.any(axis=1).to_numpy()])


def _trade_dates(book: TradedBook) -> pd.DatetimeIndex:
    trades = book.trades
    traded = trades[trades["value"].abs() > 1e-9]
    return pd.DatetimeIndex(sorted(set(pd.to_datetime(traded["date"]))))


def _daily_247(start: str, end: str, tickers: tuple[str, ...] = ("A", "B"), seed: int = 3) -> pd.DataFrame:
    """A 24/7 daily panel (crypto-style: weekends and US holidays print)."""
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.02, size=(len(idx), len(tickers)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + steps, axis=0), index=idx, columns=list(tickers))


# ----------------------------------------------------------------------
# A. Known answers, one per policy
# ----------------------------------------------------------------------


def test_periodic_monthly_on_a_market_calendar_is_the_last_session_of_the_month():
    """24/7 data, NYSE calendar: March 2024 ends on Good Friday eve (Thu 28 Mar), not Sun 31 Mar.

    Decisions: the last NYSE session of each month. Execution: the next NYSE
    session (the lag is counted in market-calendar execution bars), so the
    28 Mar decision fills on Mon 1 Apr, not on the Friday holiday or the weekend.
    """
    prices = _daily_247("2024-01-01", "2024-04-30")
    weights = pd.DataFrame({"A": 0.5, "B": 0.3}, index=prices.index)
    policy = {"policy": "periodic", "frequency": "monthly", "calendar": "NYSE"}
    book = _simulate("rsims", prices, weights, policy=policy)
    ts = pd.Timestamp
    assert list(book.schedule["decision_date"]) == [ts("2024-01-31"), ts("2024-02-29"), ts("2024-03-28")]
    assert list(book.schedule["execution_date"]) == [ts("2024-02-01"), ts("2024-03-01"), ts("2024-04-01")]
    assert _ordered_dates(book) == [ts("2024-02-01"), ts("2024-03-01"), ts("2024-04-01")]
    report = book.data_validation["rebalancing"]
    assert report["policy"]["policy"] == "periodic"
    assert report["policy"]["calendar"] == "NYSE"
    assert report["placed_rebalances"] == 3

    # Without the market calendar the month ends on the 24/7 bar: Sunday 31 March.
    plain = _simulate("rsims", prices, weights, policy={"policy": "periodic", "frequency": "monthly"})
    assert ts("2024-03-31") in set(plain.schedule["decision_date"])


def test_periodic_weekly_on_a_market_calendar_skips_the_holiday_friday():
    """Weekly = the last NYSE session of each calendar week: Thu 28 Mar 2024 (Good Friday is closed)."""
    prices = _daily_247("2024-03-01", "2024-04-14")
    weights = pd.DataFrame({"A": 0.5, "B": 0.3}, index=prices.index)
    book = _simulate("rsims", prices, weights, policy={"policy": "periodic", "frequency": "weekly", "calendar": "NYSE"})
    decisions = [d.strftime("%Y-%m-%d") for d in book.schedule["decision_date"]]
    assert decisions == ["2024-03-01", "2024-03-08", "2024-03-15", "2024-03-22", "2024-03-28", "2024-04-05"]
    executions = [d.strftime("%Y-%m-%d") for d in book.schedule["execution_date"]]
    assert executions == ["2024-03-04", "2024-03-11", "2024-03-18", "2024-03-25", "2024-04-01", "2024-04-08"]


def _two_asset_switch(n: int = 14, switch: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flat prices; the strategy decides 100% A on rows < *switch*, then 100% B."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    prices = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    decided = pd.DataFrame({"A": 1.0, "B": 0.0}, index=idx)
    decided.iloc[switch:] = [0.0, 1.0]
    return prices, decided


@pytest.mark.parametrize("engine", ENGINES)
def test_tranche_staggers_the_book_across_n_tranches_known_answer(engine):
    """Three tranches, a decision every bar: the book is the mean of the last three decided rows.

    Decided: A on rows 0-4, B from row 5. Targets (decided on row d, held from d+1):
      d=5: (B + A + A) / 3 -> A 2/3, B 1/3     d=6: A 1/3, B 2/3     d=7: B 1.
    """
    prices, decided = _two_asset_switch()
    book = _simulate(engine, prices, decided, schedule="bars", policy={"policy": "tranche", "tranches": 3})
    w = book.weights
    assert w.iloc[5]["A"] == pytest.approx(1.0)
    assert (w.iloc[6]["A"], w.iloc[6]["B"]) == (pytest.approx(2 / 3), pytest.approx(1 / 3))
    assert (w.iloc[7]["A"], w.iloc[7]["B"]) == (pytest.approx(1 / 3), pytest.approx(2 / 3))
    assert (w.iloc[8]["A"], w.iloc[8]["B"]) == (pytest.approx(0.0), pytest.approx(1.0))
    assert book.data_validation["rebalancing"]["tranches"] == 3


def test_tranche_counts_decisions_not_bars():
    """Every second bar decides; three tranches blend the last three DECISIONS (rows 6, 4, 2), not bars."""
    prices, decided = _two_asset_switch()
    book = _simulate(
        "rsims", prices, decided, schedule="bars", policy={"policy": "tranche", "tranches": 3, "frequency": 2}
    )
    w = book.weights
    assert w.iloc[7]["B"] == pytest.approx(1 / 3)  # decided on row 6: (B + A + A) / 3
    assert w.iloc[9]["B"] == pytest.approx(2 / 3)  # row 8: (B + B + A) / 3
    assert w.iloc[11]["B"] == pytest.approx(1.0)  # row 10: B
    assert _ordered_dates(book) == list(prices.index[1::2])


#: The drift band by hand (the same case as tests/test_engine_parity.py): A at 0.5, the rest cash, A +10% a bar
#: from bar 2. Drift 0.0238, 0.0476, 0.0709 -> a 5% band trades on bar 1 (entry), 4 and 7.
BAND_PRICES = pd.DataFrame(
    {"A": [100.0, 100.0] + [100.0 * 1.1**k for k in range(1, 9)]},
    index=pd.date_range("2024-01-01", periods=10, freq="D"),
)


@pytest.mark.parametrize("engine", ENGINES)
def test_band_trades_the_whole_book_when_a_weight_drifts_past_the_band_known_answer(engine):
    weights = pd.DataFrame({"A": 0.5}, index=BAND_PRICES.index)
    book = _simulate(engine, BAND_PRICES, weights, policy={"policy": "band", "band": 0.05})
    idx = BAND_PRICES.index
    assert _ordered_dates(book) == [idx[1], idx[4], idx[7]]
    assert _trade_dates(book).equals(pd.DatetimeIndex([idx[1], idx[4], idx[7]]))
    report = book.data_validation["rebalancing"]
    assert (report["scheduled_rebalances"], report["placed_rebalances"], report["skipped_rebalances"]) == (9, 3, 6)
    assert "threshold" not in book.data_validation  # the band is the declared policy, not the legacy threshold
    # The legacy `threshold` is the same band: one mask.
    legacy = _simulate(engine, BAND_PRICES, weights, threshold=0.05)
    assert legacy.orders.equals(book.orders)


#: Corridor by hand. A and B at 0.4 each, 0.2 cash; A +10% a bar from bar 2, B flat.
#:   after k bars: A = .4g / (.6 + .4g), B = .4 / (.6 + .4g), g = 1.1**k
#:   k=1: A .4231 B .3846   k=2: A .4465 B .3690   k=3: A .4701 (dev .0701) B .3532 (dev .0468)
#:   width 0.05: bar 4 is a HIT (A is outside), so EVERY asset trades to target: A and B (TOM-1513).
#:   bar 5 (A +10% from the reset): A .44 / 1.04 = .4231 (dev .0231), B .4 / 1.04 = .3846 (dev .0154): no hit.
CORRIDOR_PRICES = pd.DataFrame(
    {"A": [100.0, 100.0] + [100.0 * 1.1**k for k in range(1, 5)], "B": 50.0},
    index=pd.date_range("2024-01-01", periods=6, freq="D"),
)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_corridor_hit_rebalances_the_whole_book_known_answer(engine):
    """One asset outside its corridor -> every asset trades to target (Tom, 2026-10-05, TOM-1513)."""
    weights = pd.DataFrame({"A": 0.4, "B": 0.4}, index=CORRIDOR_PRICES.index)
    book = _simulate(engine, CORRIDOR_PRICES, weights, policy={"policy": "corridor", "width": 0.05})
    o = book.orders
    idx = CORRIDOR_PRICES.index
    assert _ordered_dates(book) == [idx[1], idx[4]]
    assert o.loc[idx[1]].tolist() == [True, True]  # the entry: both far outside
    assert o.loc[idx[4]].tolist() == [True, True]  # A drifted .0701 > .05: a hit; B (.0468, inside) trades too
    assert book.weights.loc[idx[4]].tolist() == pytest.approx([0.4, 0.4])
    report = book.data_validation["rebalancing"]
    assert (report["placed_rebalances"], report["partial_rebalances"]) == (2, 0)
    traded = book.trades[book.trades["value"].abs() > 1e-9]
    assert set(traded.loc[pd.to_datetime(traded["date"]) == idx[4], "symbol"]) == {"A", "B"}
    # The engine holds the target book after the hit: B's drifted .3532 was bought back up to .4.
    on_hit = traded[pd.to_datetime(traded["date"]) == idx[4]].set_index("symbol")["value"]
    assert on_hit["A"] < 0 < on_hit["B"]


def test_corridor_bounds_are_per_asset_and_asymmetric():
    """B's own corridor [-0.04, +0.04]: bar 4 trades both. An asymmetric corridor triggers on one side only."""
    weights = pd.DataFrame({"A": 0.4, "B": 0.4}, index=CORRIDOR_PRICES.index)
    idx = CORRIDOR_PRICES.index
    policy = {"policy": "corridor", "width": 0.05, "bounds": {"B": [0.04, 0.04]}}
    book = _simulate("rsims", CORRIDOR_PRICES, weights, policy=policy)
    assert book.orders.loc[idx[4]].tolist() == [True, True]
    # A may rise only 0.01 ABOVE its target: its +.0231 on bar 2 is a hit, and the whole book trades.
    up = {"policy": "corridor", "width": 0.05, "bounds": {"A": [0.3, 0.01], "B": [0.10, 0.5]}}
    assert _ordered_dates(_simulate("rsims", CORRIDOR_PRICES, weights, policy=up))[:2] == [idx[1], idx[2]]
    # Flipped: A may rise 0.3, B may fall 0.10. A peaks at +.094, B at -.0626: only the entry trades.
    down = {"policy": "corridor", "width": 0.05, "bounds": {"A": [0.01, 0.3], "B": [0.10, 0.5]}}
    assert _ordered_dates(_simulate("rsims", CORRIDOR_PRICES, weights, policy=down)) == [idx[1]]


def test_corridor_always_trades_an_exit_and_the_exit_rebalances_the_whole_book():
    """A target of 0 with a held weight is a hit even inside its corridor; like every hit, the whole book trades."""
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    weights = pd.DataFrame({"A": 0.3, "B": 0.5}, index=idx)
    weights.iloc[3:, 0] = 0.0  # enter A at 0.3 (0 < 0.3 - 0.25), then exit at 0 (0.3 < 0 + 0.35: inside)
    book = _simulate("rsims", prices, weights, policy={"policy": "corridor", "width": [0.25, 0.35]})
    assert _ordered_dates(book) == [idx[1], idx[4]]
    assert book.orders.loc[idx[4]].tolist() == [True, True]
    assert book.weights.loc[idx[4]].tolist() == [0.0, 0.5]


@pytest.mark.skipif(not VECTORBT, reason=NOT_CHECKED)
def test_a_corridor_on_a_fully_invested_book_never_runs_short_of_cash(caplog):
    """Net 1, no fees: after a whole-book hit the engine holds the target (#239 cut buys for lack of cash)."""
    prices = _daily_247("2024-01-01", "2024-06-30", tickers=("A", "B", "C"), seed=21)
    decided = pd.DataFrame({"A": 0.5, "B": 0.3, "C": 0.2}, index=prices.index)
    with caplog.at_level("WARNING", logger="quantbox.engine.vectorbt"):
        book = _simulate("vectorbt", prices, decided, policy={"policy": "corridor", "width": 0.02})
    assert book.data_validation["rebalancing"]["placed_rebalances"] > 2
    assert "buys were cut for lack of cash" not in caplog.text


# ----------------------------------------------------------------------
# A'. Resolution: every policy is declared, malformed ones refuse loudly
# ----------------------------------------------------------------------


def test_the_four_policies():
    assert POLICIES == ("periodic", "tranche", "band", "corridor")


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"policy": "monthly"}, "policy must be one of"),
        ({"policy": "periodic", "band": 0.05}, "does not take"),
        ({"policy": "tranche"}, "tranches"),
        ({"policy": "tranche", "tranches": 1}, "tranches"),
        ({"policy": "band"}, "band"),
        ({"policy": "band", "band": -0.1}, "band"),
        ({"policy": "corridor"}, "width"),
        ({"policy": "corridor", "width": [0.1]}, "width"),
        ({"policy": "corridor", "width": 0.1, "bounds": {"A": [0.1, -0.2]}}, "bounds"),
        ({"policy": "periodic", "calendar": "NOT_A_CALENDAR"}, "calendar"),
        ({"policy": "periodic", "frequency": "1m"}, "ambiguous"),
        ({"policy": "periodic", "frequency": 0}, "frequency"),
        ({"frequency": "monthly"}, "policy"),
    ],
)
def test_a_malformed_policy_is_refused(spec, match):
    with pytest.raises(ValueError, match=match):
        resolve_policy(spec)


def test_a_policy_and_the_legacy_schedule_keys_together_are_refused():
    prices, decided = _two_asset_switch()
    with pytest.raises(ValueError, match="rebalancing_policy"):
        _simulate("rsims", prices, decided, policy={"policy": "periodic"}, rebalancing_freq="ME")
    with pytest.raises(ValueError, match="rebalancing_policy"):
        _simulate("rsims", prices, decided, policy={"policy": "periodic"}, threshold=0.05)


def test_a_policy_records_itself():
    pol = resolve_policy({"policy": "corridor", "width": [0.02, 0.05], "bounds": {"A": 0.1}, "frequency": "weekly"})
    assert isinstance(pol, RebalancePolicy)
    assert pol.record() == {
        "policy": "corridor",
        "frequency": "weekly",
        "calendar": None,
        "width": [0.02, 0.05],
        "bounds": {"A": [0.1, 0.1]},
    }
    json.dumps(pol.record())


# ----------------------------------------------------------------------
# B. The same trades on both adapters, per policy
# ----------------------------------------------------------------------

PARITY_POLICIES = {
    "periodic-monthly-NYSE": {"policy": "periodic", "frequency": "monthly", "calendar": "NYSE"},
    "periodic-weekly-NYSE": {"policy": "periodic", "frequency": "weekly", "calendar": "NYSE"},
    "tranche-4-weekly": {"policy": "tranche", "tranches": 4, "frequency": "weekly"},
    "band-3pct": {"policy": "band", "band": 0.03},
    "corridor": {"policy": "corridor", "width": [0.02, 0.04], "bounds": {"C": [0.01, 0.01]}},
}


def _parity_book() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = _daily_247("2024-01-01", "2024-09-30", tickers=("A", "B", "C"), seed=11)
    rng = np.random.default_rng(5)
    raw = rng.dirichlet(np.ones(3), len(prices)) * 0.9
    # Strategic weights that move every 30 bars: drift policies have drift to act on.
    decided = pd.DataFrame(raw, index=prices.index, columns=prices.columns).iloc[::30].reindex(prices.index).ffill()
    return prices, decided


@pytest.mark.skipif(not VECTORBT, reason=NOT_CHECKED)
@pytest.mark.parametrize("schedule", ["calendar", "bars"])
@pytest.mark.parametrize("name", sorted(PARITY_POLICIES))
def test_both_adapters_trade_the_same_orders_for_every_policy(name, schedule):
    prices, decided = _parity_book()
    v = _simulate("vectorbt", prices, decided, schedule=schedule, policy=PARITY_POLICIES[name])
    r = _simulate("rsims", prices, decided, schedule=schedule, policy=PARITY_POLICIES[name])
    assert v.orders.equals(r.orders)
    assert v.weights.equals(r.weights)
    ordered = pd.DatetimeIndex(_ordered_dates(v))
    assert 1 <= len(ordered) < len(prices) // 2
    assert _trade_dates(v).equals(_trade_dates(r)), _trade_dates(v).symmetric_difference(_trade_dates(r))
    assert _trade_dates(v).isin(ordered).all()
    gap = (v.returns - r.returns).abs()
    assert gap.max() <= 1e-12, f"{name}: return gap {gap.max():.3e} on {gap.idxmax()}"


# ----------------------------------------------------------------------
# C. Group limits
# ----------------------------------------------------------------------

UNIVERSE = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "asset_class": ["equity", "equity", "bond", "gold"]})


def _limits(limits: dict[str, Any], excess: str = "redistribute") -> GroupLimits:
    return resolve_group_limits({"by": "asset_class", "limits": limits, "excess": excess}).bind(UNIVERSE)


def _row(**w: float) -> pd.DataFrame:
    return pd.DataFrame([w], index=pd.DatetimeIndex(["2024-01-01"]))


def test_a_group_max_redistributes_to_the_other_groups_known_answer():
    """Equity 0.8 capped at 0.6; the 0.2 goes to the uncapped groups pro rata (bond 0.2 -> 0.4)."""
    out, report = apply_group_limits(_row(A=0.5, B=0.3, C=0.2), _limits({"equity": {"max": 0.6}}))
    assert out.iloc[0].to_dict() == pytest.approx({"A": 0.375, "B": 0.225, "C": 0.4})
    assert report["rows_adjusted"] == 1


def test_a_group_max_with_excess_cash_leaves_the_rest_alone():
    out, _ = apply_group_limits(_row(A=0.5, B=0.3, C=0.2), _limits({"equity": {"max": 0.6}}, excess="cash"))
    assert out.iloc[0].to_dict() == pytest.approx({"A": 0.375, "B": 0.225, "C": 0.2})


def test_a_group_min_takes_from_the_other_groups_known_answer():
    """Bond 0.2 lifted to its 0.3 minimum; equity pays for it pro rata (0.8 -> 0.7). Gross stays 1."""
    out, _ = apply_group_limits(_row(A=0.5, B=0.3, C=0.2), _limits({"bond": {"min": 0.3}}))
    assert out.iloc[0].to_dict() == pytest.approx({"A": 0.4375, "B": 0.2625, "C": 0.3})


def test_a_compliant_row_is_untouched_and_a_flat_row_is_counted():
    w = pd.concat([_row(A=0.3, B=0.2, C=0.5), _row(A=0.0, B=0.0, C=0.0)])
    w.index = pd.date_range("2024-01-01", periods=2)
    out, report = apply_group_limits(w, _limits({"equity": {"max": 0.6}, "bond": {"min": 0.1}}))
    assert out.equals(w)
    assert report["rows_adjusted"] == 0
    assert report["flat_rows"] == 1


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"by": "asset_class", "limits": {"equity": {"min": 0.7, "max": 0.6}}}, "min .* above its max"),
        ({"by": "asset_class", "limits": {"equity": {"max": -0.1}}}, ">= 0"),
        ({"by": "asset_class", "limits": {"equity": {"cap": 0.6}}}, "min, max"),
        ({"by": "asset_class", "limits": {}}, "at least one"),
        ({"limits": {"equity": {"max": 0.6}}}, "by"),
        ({"by": "asset_class", "limits": {"equity": {"max": 0.6}}, "excess": "spill"}, "excess"),
    ],
)
def test_a_malformed_group_limit_is_refused(spec, match):
    with pytest.raises(ValueError, match=match):
        resolve_group_limits(spec)


def test_an_infeasible_group_limit_refuses_loudly():
    # The minimums need 0.9 of the book; this row only holds 0.5.
    with pytest.raises(ValueError, match="INFEASIBLE"):
        apply_group_limits(_row(A=0.2, B=0.1, C=0.2), _limits({"equity": {"min": 0.6}, "bond": {"min": 0.3}}))
    # A minimum on a group the row holds nothing in: no instrument to scale up.
    with pytest.raises(ValueError, match="INFEASIBLE"):
        apply_group_limits(_row(A=0.5, B=0.3, C=0.0), _limits({"bond": {"min": 0.1}}))


def test_group_limits_refuse_an_unknown_group_column_or_an_ungrouped_weight():
    spec = {"by": "sector", "limits": {"equity": {"max": 0.6}}}
    with pytest.raises(ValueError, match="sector"):
        resolve_group_limits(spec).bind(UNIVERSE)
    with pytest.raises(ValueError, match="not a group"):
        _limits({"equities": {"max": 0.6}})  # a typo of a group name
    with pytest.raises(ValueError, match="no asset_class"):
        apply_group_limits(_row(A=0.5, E=0.3), _limits({"equity": {"max": 0.6}}))


def test_group_limits_apply_before_execution_through_the_seam():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"A": 100.0, "B": 50.0, "C": 20.0}, index=idx)
    decided = pd.DataFrame({"A": 0.5, "B": 0.3, "C": 0.2}, index=idx)
    book = _simulate("rsims", prices, decided, groups=_limits({"equity": {"max": 0.6}}))
    assert book.weights.iloc[-1].to_dict() == pytest.approx({"A": 0.375, "B": 0.225, "C": 0.4})
    assert book.data_validation["groups"]["by"] == "asset_class"
    assert book.book_metrics["group_limit_rows_adjusted"] == 10.0


def test_the_backtest_helper_takes_a_policy_and_group_limits():
    """The L1 door, ``backtest()``, goes through the same seam: same policy keys, same group limits."""
    from quantbox.plugins.backtesting import backtest

    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"A": 100.0, "B": 50.0, "C": 20.0}, index=idx)
    decided = pd.DataFrame({"A": 0.5, "B": 0.3, "C": 0.2}, index=idx)
    out = backtest(
        prices,
        decided,
        engine="rsims",
        fees=0.0,
        policy={"policy": "tranche", "tranches": 2, "frequency": 3},
        group_limits={"by": "asset_class", "limits": {"equity": {"max": 0.6}}},
        universe=UNIVERSE,
        engine_params=RSIMS_COMMON,
    )
    book = out["book"]
    assert book.weights.iloc[-1].to_dict() == pytest.approx({"A": 0.375, "B": 0.225, "C": 0.4})
    assert book.data_validation["rebalancing"]["policy"]["tranches"] == 2
    with pytest.raises(ValueError, match="universe="):
        backtest(prices, decided, engine="rsims", group_limits={"by": "asset_class", "limits": {"e": {"max": 1}}})


def _random_case(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, GroupLimits, dict[str, Any]]:
    """Random prices, a random long-only book (gross in [0.3, 1]), random groups and FEASIBLE random limits."""
    n_inst = int(rng.integers(3, 7))
    n_groups = int(rng.integers(2, min(n_inst, 4) + 1))
    tickers = [f"T{i}" for i in range(n_inst)]
    groups = [f"g{k}" for k in range(n_groups)]
    member = [groups[i % n_groups] for i in range(n_inst)]  # every group has an instrument
    universe = pd.DataFrame({"symbol": tickers, "asset_class": member})
    idx = pd.date_range("2024-01-01", periods=int(rng.integers(40, 90)), freq="D")
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, size=(len(idx), n_inst)), axis=0), index=idx, columns=tickers
    )
    # Every instrument carries weight on every row, so a group minimum always has something to scale.
    raw = rng.dirichlet(np.ones(n_inst), len(idx)) * rng.uniform(0.3, 1.0, size=(len(idx), 1))
    decided = pd.DataFrame(raw, index=idx, columns=tickers).iloc[:: int(rng.integers(1, 6))].reindex(idx).ffill()
    limits: dict[str, dict[str, float]] = {}
    for g in rng.choice(groups, size=int(rng.integers(1, n_groups + 1)), replace=False):
        lim: dict[str, float] = {}
        if rng.random() < 0.7:
            lim["max"] = float(rng.uniform(0.05, 0.6))
        if rng.random() < 0.5:
            lim["min"] = float(rng.uniform(0.0, 0.3 / n_groups))  # sum of minimums <= 0.3 <= every row's gross
            lim.setdefault("max", 1.0)
            lim["max"] = max(lim["max"], lim["min"])
        limits[str(g)] = lim or {"max": 0.5}
    excess = str(rng.choice(["redistribute", "cash"]))
    bound = resolve_group_limits({"by": "asset_class", "limits": limits, "excess": excess}).bind(universe)
    policy = [
        {"policy": "periodic", "frequency": int(rng.integers(1, 8))},
        {"policy": "periodic", "frequency": "weekly"},
        {"policy": "tranche", "tranches": int(rng.integers(2, 5))},
        {"policy": "band", "band": float(rng.uniform(0.01, 0.08))},
    ][int(rng.integers(0, 4))]
    return prices, decided, bound, policy


def test_property_a_group_limit_holds_on_every_rebalance_date():
    """A seeded randomized loop (hypothesis is not a dev dependency): 200 random books, groups, limits, policies.

    On every bar the seam places orders, the TARGET book it trades to keeps every
    group inside [min, max]. ``schedule: bars`` so every cell of an order row is
    ordered (no deferral), long-only with gross <= 1 so ``venue.leverage`` never
    scales; every policy trades every ordered cell to its target on a placed bar
    (a corridor hit rebalances the whole book, TOM-1513). Every band case runs
    a corridor of the same width too, without drawing from the seeded stream.
    """
    rng = np.random.default_rng(20261005)
    checked_rows = 0
    corridor_rows = 0
    for case in range(200):
        prices, decided, limits, policy = _random_case(rng)
        policies = [policy]
        if policy["policy"] == "band":
            policies.append({"policy": "corridor", "width": policy["band"]})
        for pol in policies:
            book = _simulate("rsims", prices, decided, schedule="bars", policy=pol, groups=limits)
            rows = book.orders.any(axis=1).to_numpy()
            assert rows.any(), case
            held = book.weights.to_numpy()[rows]
            member = np.array([limits.membership[c] for c in book.weights.columns])
            for g, (lo, hi) in limits.limits.items():
                gross = np.abs(held[:, member == g]).sum(axis=1)
                assert (gross <= hi + 1e-9).all(), (case, g, gross.max(), hi, pol)
                assert (gross >= lo - 1e-9).all(), (case, g, gross.min(), lo, pol)
            checked_rows += int(rows.sum())
            corridor_rows += int(rows.sum()) if pol["policy"] == "corridor" else 0
    assert checked_rows > 2000  # the loop looked at real rebalances, not an empty schedule
    assert corridor_rows > 100


# ----------------------------------------------------------------------
# D. The pipeline: params_schema, plan (config explain) and the run's files
# ----------------------------------------------------------------------


def _pipeline_schema() -> dict[str, Any]:
    from quantbox.params_schema import resolve_params_schema
    from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline

    return resolve_params_schema(BacktestPipeline)


def test_every_policy_and_the_group_limits_are_declared_in_params_schema():
    props = _pipeline_schema()["properties"]
    assert set(props["rebalancing_policy"]["properties"]["policy"]["enum"]) == set(POLICIES)
    for key in ("frequency", "calendar", "tranches", "band", "width", "bounds"):
        assert props["rebalancing_policy"]["properties"][key]["description"]
    assert set(props["group_limits"]["properties"]) >= {"by", "limits", "excess"}


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ({"policy": "periodic", "frequency": "monthly", "calendar": "NYSE"}, True),
        ({"policy": "tranche", "tranches": 4, "frequency": "weekly"}, True),
        ({"policy": "band", "band": 0.05}, True),
        ({"policy": "corridor", "width": [0.02, 0.05], "bounds": {"SPY": [0.01, 0.03]}}, True),
        ({"policy": "monthly"}, False),
        ({"policy": "band"}, False),
        ({"policy": "periodic", "tranches": 3}, False),
        ({"policy": "corridor", "width": 0.05, "colour": "red"}, False),
    ],
)
def test_validate_checks_a_rebalancing_policy_against_the_schema(value, ok):
    from quantbox.params_schema import check_params

    unknown, violations = check_params(_pipeline_schema(), {"rebalancing_policy": value})
    assert not unknown
    assert (not violations) is ok, violations


def test_plan_resolves_the_policy_and_refuses_what_the_run_would_refuse():
    from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline

    plan = BacktestPipeline().plan(
        {
            "engine": "rsims",
            "rebalancing_policy": {"policy": "periodic", "frequency": "monthly", "calendar": "NYSE"},
            "group_limits": {"by": "asset_class", "limits": {"equity": {"max": 0.6}}},
        }
    )
    assert plan["rebalancing"]["policy"] == "periodic"
    assert plan["rebalancing"]["calendar"] == "NYSE"
    assert plan["group_limits"]["limits"] == {"equity": {"min": 0.0, "max": 0.6}}
    legacy = BacktestPipeline().plan({"engine": "rsims", "rebalancing_freq": "ME", "threshold": 0.05})
    assert legacy["rebalancing"] == {"policy": "band", "frequency": "ME", "calendar": None, "band": 0.05}
    with pytest.raises(ValueError, match="rebalancing_policy"):
        BacktestPipeline().plan(
            {"engine": "rsims", "rebalancing_policy": {"policy": "band", "band": 0.1}, "threshold": 0.1}
        )
    with pytest.raises(ValueError, match="tranches"):
        BacktestPipeline().plan({"engine": "rsims", "rebalancing_policy": {"policy": "tranche"}})
    with pytest.raises(ValueError, match="above its max"):
        BacktestPipeline().plan(
            {"engine": "rsims", "group_limits": {"by": "asset_class", "limits": {"e": {"min": 0.5, "max": 0.4}}}}
        )


# The end-to-end run of a policy and group limits through the pipeline is a
# pipeline smoke test: tests/pipeline/test_rebalancing_policies_e2e.py (TOM-1500).
