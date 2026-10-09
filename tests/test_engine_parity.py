"""Engine parity: the same books through the vectorbt and rsims adapters, via the seam (TOM-262, docs/adr/0008).

vectorbt is quantbox's default engine, so checking quantbox against vectorbt
checks it against itself. rsims is a separate implementation of the same
capability, so the two are each other's cross-check (part A). Part B is an
analytic closed-form case, worked out by hand and asserted to the cent: it
catches an assumption BOTH engines share, which part A cannot.

Every book goes in through :func:`quantbox.engine.simulate`, the one book
function every door calls. Nothing here calls an engine primitive directly.

The seam owns the schedule (TOM-1450, docs/adr/0008): both engines execute
the SAME orders mask, so for one schedule they trade on identical dates
(part C), whatever the schedule. The NaN policy and the default
``venue.leverage`` are the seam's too, one value for every engine.

THE COMMON SCOPE — where the two engines must agree on the numbers:

- spot: long-only, gross exposure at most 1;
- no funding (vectorbt charges none);
- both engines under their DEFAULTS (TOM-1500, Tom 2026-10-05: "same defaults for
  each engine"): rsims compounds (``capitalise_profits`` defaults to True), as
  vectorbt does, and both charge every field of :class:`~quantbox.engine.Costs`
  (``fees``, ``slippage``, ``fixed_fees``). A cost an engine cannot model is
  refused by the seam, never dropped.

THE TOLERANCE, stated:

- ``fees = 0``: per-bar returns agree to ``1e-12`` and the value to ``1e-9``
  relative. Only float round-off separates them.
- ``fees > 0``: the engines charge a fee differently on a FULLY invested book.
  rsims buys the full target notional and takes the fee from cash (its ledger
  may go to ``-fee``). vectorbt cannot spend cash it does not have, so it buys
  ``C / (p (1 + f))``. Both end the bar with the same equity, but rsims holds
  ``f * traded`` more exposure for one bar. The per-bar gap is therefore
  bounded by ``f * tau[t-1] * max_i |r_i[t]| + f**2 * tau[t]``, where ``tau`` is
  the traded notional over the equity before the bar. The assertion allows
  1.5 times that bound. A book with spare cash pays the fee from it on both
  engines, and agrees to round-off. The closed-form fee case below pins each
  engine's convention to the cent.

Without vectorbt the parity tests SKIP with "NOT CHECKED" in the reason: a
machine that cannot run both engines has not checked parity, so it must not
read as a pass. ``test_without_vectorbt_parity_reads_not_checked`` proves it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantbox.engine import Costs, EngineAdapter, TradedBook, get_engine, simulate
from quantbox.execution import resolve_execution

VECTORBT = get_engine("vectorbt", require_installed=False).installed()
NOT_CHECKED = "NOT CHECKED: engine parity needs vectorbt (the [vectorbt] extra) — this is not a pass"
needs_vectorbt = pytest.mark.skipif(not VECTORBT, reason=NOT_CHECKED)

TIMING = resolve_execution(None)  # next-bar, the mandatory default (docs/adr/0005)
CAPITAL = 10_000.0
#: rsims inside the common scope is rsims under its DEFAULTS (TOM-1500): nothing is passed.
RSIMS_COMMON = None
FEES = [0.0, 0.001]

# ----------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------

N_BARS = 250
TICKERS = ["A", "B", "C"]


def _prices() -> pd.DataFrame:
    """Three seeded random walks, daily, ~2% vol: enough movement for a wrong fill to show."""
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    rng = np.random.default_rng(42)
    steps = rng.normal(0.0005, 0.02, size=(N_BARS, len(TICKERS)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + steps, axis=0), index=idx, columns=TICKERS)


def _books() -> dict[str, pd.DataFrame]:
    """Decided weight books (NOT lagged; the seam lags them). Every book is spot: long-only, gross <= 1."""
    idx = _prices().index
    rng = np.random.default_rng(7)
    rotate = np.eye(len(TICKERS))[(np.arange(N_BARS) // 20) % len(TICKERS)]
    in_and_out = np.tile([0.5, 0.5, 0.0], (N_BARS, 1))
    in_and_out[60:120] = 0.0  # all to cash, then back in
    in_and_out[180:] = [0.0, 0.3, 0.7]
    return {
        # Fully invested, constant mix: daily drift trades only.
        "constant_mix": pd.DataFrame(np.tile([0.5, 0.3, 0.2], (N_BARS, 1)), index=idx, columns=TICKERS),
        # A cash buffer: the fee comes out of spare cash on both engines.
        "partial_cash": pd.DataFrame(np.tile([0.4, 0.2, 0.0], (N_BARS, 1)), index=idx, columns=TICKERS),
        # One asset at a time, fully invested: a full turnover every 20 bars.
        "rotation": pd.DataFrame(rotate, index=idx, columns=TICKERS),
        # New random weights every bar (sum 0.9): heavy turnover.
        "random_daily": pd.DataFrame(rng.dirichlet(np.ones(len(TICKERS)), N_BARS) * 0.9, index=idx, columns=TICKERS),
        # Entry, a full exit to cash, a re-entry into a different mix.
        "in_and_out": pd.DataFrame(in_and_out, index=idx, columns=TICKERS),
    }


BOOKS = sorted(_books())


def _simulate(
    engine: str, prices: pd.DataFrame, weights: pd.DataFrame, fees: float, schedule: str = "calendar", **kw
) -> TradedBook:
    params = RSIMS_COMMON if engine == "rsims" else None
    timing = resolve_execution({"schedule": schedule})
    return simulate(prices, weights, engine=engine, timing=timing, costs=Costs(fees=fees), engine_params=params, **kw)


def _capital(book: TradedBook) -> float:
    """The cash a book started with: vectorbt's ``init_cash``, rsims' ``initial_cash``."""
    if isinstance(book.native, pd.DataFrame):  # rsims' long results frame
        return CAPITAL
    return float(np.asarray(book.native.init_cash).ravel()[0])


def _traded_share(book: TradedBook, index: pd.Index) -> pd.Series:
    """``tau[t]``: traded notional on bar t over the equity before it."""
    traded = book.trades.assign(notional=lambda t: t["value"].abs()).groupby("date")["notional"].sum()
    before = book.value.shift(1).fillna(_capital(book))
    return (traded.reindex(index).fillna(0.0) / before.reindex(index)).fillna(0.0)


def _fee_gap_bound(prices: pd.DataFrame, rsims: TradedBook, fees: float) -> pd.Series:
    """The per-bar return gap the fee convention allows (module docstring)."""
    tau = _traded_share(rsims, prices.index)
    moves = prices.pct_change(fill_method=None).abs().max(axis=1).fillna(0.0)
    return fees * tau.shift(1).fillna(0.0) * moves + fees**2 * tau


@pytest.fixture(scope="module")
def runs() -> dict[tuple[str, float], tuple[TradedBook, TradedBook]]:
    """Every shared book, at every fee level, through both adapters (simulated once)."""
    if not VECTORBT:
        pytest.skip(NOT_CHECKED)
    prices = _prices()
    out = {}
    for name, weights in _books().items():
        for fees in FEES:
            out[(name, fees)] = (
                _simulate("vectorbt", prices, weights, fees),
                _simulate("rsims", prices, weights, fees),
            )
    return out


# ----------------------------------------------------------------------
# A. vectorbt <-> rsims parity on the common scope
# ----------------------------------------------------------------------


@needs_vectorbt
def test_the_fixtures_stay_inside_the_common_scope():
    """A book outside the scope would test a documented difference, not parity."""
    for name, w in _books().items():
        assert w.notna().all().all(), name
        assert (w >= 0).all().all(), name
        assert (w.sum(axis=1) <= 1.0 + 1e-12).all(), name
    assert len(BOOKS) >= 5


@needs_vectorbt
@pytest.mark.parametrize("fees", FEES)
@pytest.mark.parametrize("book", BOOKS)
def test_both_engines_produce_the_same_book(runs, book, fees):
    vbt_book, rsims_book = runs[(book, fees)]
    prices = _prices()
    assert (vbt_book.engine, rsims_book.engine) == ("vectorbt", "rsims")
    # One shape (docs/adr/0008): a value and a return on every bar.
    assert vbt_book.value.index.equals(rsims_book.value.index)
    assert vbt_book.returns.index.equals(rsims_book.returns.index)

    gap = (vbt_book.returns - rsims_book.returns).abs()
    if fees == 0.0:
        allowed = pd.Series(1e-12, index=gap.index)
    else:
        allowed = 1e-12 + 1.5 * _fee_gap_bound(prices, rsims_book, fees).reindex(gap.index)
    worst = (gap - allowed).idxmax()
    assert (gap <= allowed).all(), (
        f"{book} fees={fees}: return gap {gap[worst]:.3e} on {worst} exceeds the stated tolerance {allowed[worst]:.3e}"
    )

    v = vbt_book.value / _capital(vbt_book)
    r = rsims_book.value / _capital(rsims_book)
    value_tol = 1e-9 if fees == 0.0 else 1e-9 + 1.5 * _fee_gap_bound(prices, rsims_book, fees).sum()
    assert ((v - r).abs() / v).max() <= value_tol


@needs_vectorbt
@pytest.mark.parametrize("book", BOOKS)
def test_both_engines_trade_the_same_notional_without_fees(runs, book):
    """Same fills, not only the same curve: the traded notional per bar agrees."""
    vbt_book, rsims_book = runs[(book, 0.0)]
    index = _prices().index
    v = _traded_share(vbt_book, index)
    r = _traded_share(rsims_book, index)
    # rsims re-sizes every bar and leaves sub-1e-9 dust trades; vectorbt does not place them.
    assert (v - r).abs().max() <= 1e-9
    assert vbt_book.turnover.equals(rsims_book.turnover)


@needs_vectorbt
@pytest.mark.parametrize("fees", FEES)
@pytest.mark.parametrize("book", BOOKS)
def test_both_engines_report_the_same_metrics(runs, book, fees):
    """The headline numbers a researcher reads, from each adapter's own ``metrics``."""
    vbt_book, rsims_book = runs[(book, fees)]
    tol = 1e-9 if fees == 0.0 else 1.5 * _fee_gap_bound(_prices(), rsims_book, fees).sum() + 1e-9
    for name in ("total_return", "max_drawdown"):
        assert vbt_book.metrics[name] == pytest.approx(rsims_book.metrics[name], abs=tol), name
    assert vbt_book.metrics["sharpe"] == pytest.approx(rsims_book.metrics["sharpe"], rel=1e-3 if fees else 1e-9)


@needs_vectorbt
@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
@pytest.mark.parametrize("schedule", ["calendar", "bars"])
def test_total_return_is_the_value_the_book_ends_with(engine, schedule):
    """``metrics.total_return`` must equal the value curve's own total return, first bar included.

    A book that trades on its first bar pays its entry fee in the first
    return. A total return compounded from the SECOND return drops it.
    """
    book = _simulate(engine, _prices(), _books()["rotation"], 0.001, schedule=schedule)
    assert book.metrics["total_return"] == pytest.approx(book.value.iloc[-1] / _capital(book) - 1.0, abs=1e-12)


# ----------------------------------------------------------------------
# Same defaults, every cost charged (TOM-1500). These were strict xfails until
# rsims compounded by default and charged slippage and fixed fees.
# ----------------------------------------------------------------------

#: Every cost field at once, and each one alone: a book with spare cash (sum 0.9), so
#: vectorbt can always afford the full buy and both engines hold the same book.
COST_CASES = {
    "slippage": Costs(slippage=0.0005),
    "fixed_fees": Costs(fixed_fees=1.0),
    "all": Costs(fees=0.001, slippage=0.0005, fixed_fees=1.0),
}


@needs_vectorbt
def test_parity_under_the_defaults():
    """No engine params at all: rsims compounds by default, as vectorbt does."""
    prices, weights = _prices(), _books()["rotation"]
    v = simulate(prices, weights, engine="vectorbt", timing=TIMING, costs=Costs())
    r = simulate(prices, weights, engine="rsims", timing=TIMING, costs=Costs())
    assert (v.value / _capital(v)).iloc[-1] == pytest.approx((r.value / CAPITAL).iloc[-1], rel=1e-9)
    assert ((v.returns - r.returns).abs() <= 1e-12).all()


def test_rsims_compounds_by_default():
    """``capitalise_profits`` defaults to True on the adapter, the primitive and the pipeline schema."""
    import inspect

    from quantbox.engine.rsims_sim import fixed_commission_backtest_with_funding
    from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline

    adapter = get_engine("rsims")
    assert adapter.PARAMS["capitalise_profits"] is True
    assert adapter.check_params(None)["capitalise_profits"] is True
    default = inspect.signature(fixed_commission_backtest_with_funding).parameters["capitalise_profits"].default
    assert default is True
    schema = BacktestPipeline.meta.params_schema["properties"]["capitalise_profits"]
    assert schema["default"] is True


@needs_vectorbt
@pytest.mark.parametrize("case", sorted(COST_CASES))
@pytest.mark.parametrize("book", ["random_daily", "partial_cash"])
def test_parity_with_every_cost(book, case):
    """Slippage and fixed fees cost the same on both engines: the same value on every bar, the same fills."""
    prices, weights, costs = _prices(), _books()[book], COST_CASES[case]
    v = simulate(prices, weights, engine="vectorbt", timing=TIMING, costs=costs)
    r = simulate(prices, weights, engine="rsims", timing=TIMING, costs=costs, engine_params=RSIMS_COMMON)
    rel = ((v.value / _capital(v)) - (r.value / CAPITAL)).abs() / (v.value / _capital(v))
    assert rel.max() <= 1e-9, f"{book} {case}: value gap {rel.max():.3e} on {rel.idxmax()}"
    # A cost-free run would end higher: the costs were charged, not dropped.
    free = simulate(prices, weights, engine="rsims", timing=TIMING, costs=Costs())
    assert r.value.iloc[-1] < free.value.iloc[-1]
    # Same fills: the fill price carries the slippage, the fees carry the fixed fee.
    fv = v.trades.sort_values(["date", "symbol"]).reset_index(drop=True)
    fr = r.trades[r.trades["value"].abs() > 1e-9].sort_values(["date", "symbol"]).reset_index(drop=True)
    assert len(fv) == len(fr)
    np.testing.assert_allclose(fr["price"], fv["price"], rtol=1e-12)
    np.testing.assert_allclose(fr["fees"], fv["fees"], rtol=1e-6, atol=1e-9)


#: One asset at half the book, slippage 1%, a fixed fee of 1.00 per order; 100 -> 100 -> 125.
#:   bar 1: buy 5,000 / 100 = 50 units at 101.00: slippage 50.00, fee 1.00.  V = 9,949.00
#:   bar 2: 50 x 25 = 1,250 (V 11,199); target 0.5 x 11,199 / 125 = 44.796 units: sell 5.204
#:          at 123.75: slippage 5.204 x 1.25 = 6.505, fee 1.00.          V = 11,191.495 -> 11,191.50
SLIP_PRICES = pd.DataFrame({"A": [100.0, 100.0, 125.0]}, index=pd.date_range("2024-01-01", periods=3, freq="D"))
SLIP_WEIGHTS = pd.DataFrame({"A": 0.5}, index=SLIP_PRICES.index)
SLIP_COSTS = Costs(slippage=0.01, fixed_fees=1.0)


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
def test_closed_form_slippage_and_fixed_fees_to_the_cent(engine):
    if engine == "vectorbt" and not VECTORBT:
        pytest.skip(NOT_CHECKED)
    book = simulate(SLIP_PRICES, SLIP_WEIGHTS, engine=engine, timing=TIMING, costs=SLIP_COSTS)
    assert _to_the_cent(book).round(2).tolist() == [10_000.0, 9_949.0, 11_191.5]
    trades = book.trades[book.trades["value"].abs() > 1e-9]
    assert trades["price"].round(2).tolist() == [101.0, 123.75]
    assert trades["size"].round(3).tolist() == [50.0, -5.204]
    assert trades["fees"].round(2).tolist() == [1.0, 1.0]


def test_every_adapter_charges_every_cost_field():
    """Each adapter names the Costs fields it charges; a new Costs field is refused until an adapter charges it."""
    from dataclasses import fields

    from quantbox.engine import engine_names

    every = {f.name for f in fields(Costs)}
    assert every == {"fees", "fixed_fees", "slippage"}
    for name in engine_names():
        assert get_engine(name, require_installed=False).charged_costs() == every, name


class _FeesOnly(EngineAdapter):
    """A third engine that can model a proportional fee and nothing else (the refusal's control)."""

    name = "fees_only"
    distribution = "quantbox"
    charges_funding = False
    models_margin = False
    native_key = "native"

    def charged_costs(self) -> frozenset[str]:
        return frozenset({"fees"})

    def execute(self, prices, targets, orders, costs, funding=None, params=None, *, cash_legs=(), trading_days=365):
        return get_engine("rsims").execute(
            prices, targets, orders, Costs(fees=costs.fees), funding, None, cash_legs=cash_legs
        )

    def stats(self, book, names, *, trading_days=365):
        return {}


@pytest.mark.parametrize("costs", [Costs(slippage=0.0005), Costs(fixed_fees=1.0)], ids=["slippage", "fixed_fees"])
def test_a_cost_an_engine_cannot_model_is_refused(costs):
    """The seam refuses a non-zero cost the engine does not charge, naming both; it never drops it."""
    weights = _books()["random_daily"]
    with pytest.raises(ValueError, match=r"fees_only.*(slippage|fixed_fees)"):
        simulate(_prices(), weights, engine=_FeesOnly(), timing=TIMING, costs=costs)
    # The control: the costs it does model run.
    book = simulate(_prices(), weights, engine=_FeesOnly(), timing=TIMING, costs=Costs(fees=0.001))
    assert book.value.iloc[-1] > 0


# ----------------------------------------------------------------------
# C. One schedule, owned by the seam: both engines trade on identical dates (TOM-1450)
# ----------------------------------------------------------------------

#: (rebalancing_freq, threshold): every schedule shape the seam builds today.
SCHEDULES = [
    (1, None),
    (5, None),
    ("W-FRI", None),
    ("ME", None),
    (None, None),  # buy-and-hold: one decision
    (1, 0.05),  # a drift band
    (5, 0.02),
]
SCHEDULE_IDS = [f"freq={f}-threshold={t}" for f, t in SCHEDULES]


def _trade_dates(book: TradedBook) -> pd.DatetimeIndex:
    """The bars the engine actually traded on (a fill with a non-dust notional)."""
    trades = book.trades
    traded = trades[trades["value"].abs() > 1e-9]
    return pd.DatetimeIndex(sorted(set(pd.to_datetime(traded["date"]))))


def _ordered_dates(book: TradedBook) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(book.orders.index[book.orders.any(axis=1).to_numpy()])


@needs_vectorbt
@pytest.mark.parametrize(("freq", "threshold"), SCHEDULES, ids=SCHEDULE_IDS)
@pytest.mark.parametrize("schedule", ["calendar", "bars"])
@pytest.mark.parametrize("book", ["random_daily", "rotation"])
def test_both_engines_trade_on_identical_dates_for_the_same_schedule(book, schedule, freq, threshold):
    """The schedule is the seam's orders mask; each engine trades exactly on its bars, no other.

    Until TOM-1450 rsims decided on every bar whatever ``rebalancing_freq`` said
    and ignored ``threshold``; vectorbt followed both. Now both execute one mask.
    """
    prices, weights = _prices(), _books()[book]
    kw = {"rebalancing_freq": freq, "threshold": threshold}
    # A FULLY invested book with fees: rsims pays the fee from cash and trims the excess on the next
    # ordered bar, where vectorbt has nothing to trade (the fee convention, pinned to the cent in part B).
    fees = 0.0 if book == "rotation" else 0.001
    v = _simulate("vectorbt", prices, weights, fees, schedule=schedule, **kw)
    r = _simulate("rsims", prices, weights, fees, schedule=schedule, **kw)
    assert v.orders.equals(r.orders)
    ordered = _ordered_dates(v)
    assert len(ordered) >= 1
    traded_v, traded_r = _trade_dates(v), _trade_dates(r)
    assert traded_v.equals(traded_r), (
        f"the engines traded on different dates: {traded_v.symmetric_difference(traded_r)}"
    )
    assert traded_v.isin(ordered).all(), "an engine traded off the seam's schedule"
    if book == "random_daily":  # new weights every bar: every ordered bar has something to trade
        assert traded_v.equals(ordered)
    # (rotation: an ordered bar already 100% in the one asset has nothing to trade, on both engines.)
    if freq not in (1, None) and threshold is None:
        assert len(ordered) < len(prices) // 3  # a real, sparse schedule — not every bar


@needs_vectorbt
@pytest.mark.parametrize(("freq", "threshold"), [s for s in SCHEDULES if s[1] is None], ids=lambda x: str(x))
@pytest.mark.parametrize("book", BOOKS)
def test_both_engines_produce_the_same_book_on_any_schedule_without_fees(book, freq, threshold):
    """Same dates and the same numbers: without fees the two engines agree to round-off on every schedule."""
    prices, weights = _prices(), _books()[book]
    v = _simulate("vectorbt", prices, weights, 0.0, rebalancing_freq=freq)
    r = _simulate("rsims", prices, weights, 0.0, rebalancing_freq=freq)
    gap = (v.returns - r.returns).abs()
    assert gap.max() <= 1e-12, f"{book} freq={freq}: return gap {gap.max():.3e} on {gap.idxmax()}"


#: The drift band by hand. One asset at 0.5, the rest cash; A rises 10% a bar from bar 2.
#:   bar 0 decides 0.5 -> fills at close[1] (next-bar): held 0.5.
#:   drifted weight of A after k bars of +10%: 0.5 g / (0.5 + 0.5 g), g = 1.1**k
#:     k=1: 0.5238 (|dev| 0.0238)   k=2: 0.5476 (0.0476)   k=3: 0.5709 (0.0709 > 0.05: TRADE, back to 0.5)
#:   so with a 5% band: trades on bar 1 (entry) and bar 4, then bar 7, ...
BAND_PRICES = pd.DataFrame(
    {"A": [100.0, 100.0] + [100.0 * 1.1**k for k in range(1, 9)]},
    index=pd.date_range("2024-01-01", periods=10, freq="D"),
)


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
def test_threshold_is_a_seam_computed_drift_trigger_known_answer(engine):
    if engine == "vectorbt" and not VECTORBT:
        pytest.skip(NOT_CHECKED)
    weights = pd.DataFrame({"A": 0.5}, index=BAND_PRICES.index)
    book = _simulate(engine, BAND_PRICES, weights, 0.0, threshold=0.05)
    idx = BAND_PRICES.index
    assert list(_ordered_dates(book)) == [idx[1], idx[4], idx[7]]
    assert _trade_dates(book).equals(pd.DatetimeIndex([idx[1], idx[4], idx[7]]))
    assert book.data_validation["threshold"]["skipped_rebalances"] == 9 - 3
    assert book.book_metrics["threshold_skipped_rebalances"] == 6.0


def test_a_nan_cell_holds_on_every_engine():
    """The NaN policy is the seam's, one for every engine: a NaN cell holds the last target."""
    weights = _books()["in_and_out"].copy()
    weights.iloc[30:40] = np.nan
    engines = ["rsims", "vectorbt"] if VECTORBT else ["rsims"]
    for engine in engines:
        held = _simulate(engine, _prices(), weights, 0.0).weights
        assert (held.iloc[31:41] == held.iloc[30]).all().all(), engine
        assert held.iloc[31]["A"] == 0.5, engine


def test_the_default_leverage_is_one_value_for_every_engine():
    """The same config gives the same book when only ``engine`` changes: normalize, on every engine.

    Since TOM-1520 the normalisation is the decision's (:func:`quantbox.decision.final_targets`) and the
    seam executes final targets; a levered book handed to the seam directly is held to net 1 by its cash
    cap instead (it never borrows), the same on every engine.
    """
    from quantbox.decision import DecisionRules, final_targets
    from quantbox.financing import DEFAULT_LEVERAGE, resolve_leverage

    assert DEFAULT_LEVERAGE == "normalize"
    assert resolve_leverage(None) == "normalize"
    levered = pd.DataFrame({"A": 0.9, "B": 0.6, "C": 0.0}, index=_prices().index)
    final, _ = final_targets(levered, DecisionRules(leverage=DEFAULT_LEVERAGE))
    engines = ["rsims", "vectorbt"] if VECTORBT else ["rsims"]
    held = {e: _simulate(e, _prices(), final, 0.0).weights for e in engines}
    raw = {e: _simulate(e, _prices(), levered, 0.0).weights for e in engines}
    for e, w in held.items():
        assert w.sum(axis=1).max() == pytest.approx(1.0), e  # scaled to net 1
        assert w.iloc[-1]["A"] == pytest.approx(0.6), e
        assert raw[e].sum(axis=1).max() <= 1.0 + 1e-9, e  # the cash cap: never borrows
    if VECTORBT:
        assert held["rsims"].equals(held["vectorbt"])
        assert raw["rsims"].equals(raw["vectorbt"])


# ----------------------------------------------------------------------
# D. Capabilities: only charges_funding / models_margin are read outside the engine package
# ----------------------------------------------------------------------

SRC = Path(__file__).resolve().parents[1] / "src" / "quantbox"
ENGINE_PACKAGE = SRC / "engine"
#: The two capability differences an adapter may declare (docs/adr/0008).
ALLOWED_FLAGS = {"charges_funding", "models_margin"}
#: Adapter attributes that NAME the adapter, not a behaviour.
IDENTITY = {"name", "distribution", "extra", "native_key", "PARAMS"}
#: Flags the seam took over in TOM-1450: reading one again outside the engine is the regression.
RETIRED_FLAGS = {"nan_policy", "default_leverage", "decides_every_bar"}


def _adapter_flags() -> set[str]:
    """Every class-level attribute an adapter declares, minus its identity."""
    declared = set(EngineAdapter.__annotations__)
    return (declared - IDENTITY) | RETIRED_FLAGS


def _flag_reads(path: Path, flags: set[str]) -> list[str]:
    import ast

    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Attribute) and node.attr in flags:
            found.append(f"{path.name}:{node.lineno}:{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in flags
        ):
            found.append(f"{path.name}:{node.lineno}:{node.args[1].value}")
    return found


def test_an_adapter_declares_only_the_two_capability_flags():
    assert set(EngineAdapter.__annotations__) - IDENTITY == ALLOWED_FLAGS
    for name in ("vectorbt", "rsims"):
        adapter = get_engine(name, require_installed=False)
        for flag in RETIRED_FLAGS:
            assert not hasattr(adapter, flag), f"{name}.{flag} is back: the seam owns it (TOM-1450)"


def test_no_adapter_flag_but_the_capabilities_is_read_outside_the_engine_package():
    scanned = [p for p in SRC.rglob("*.py") if ENGINE_PACKAGE not in p.parents]
    assert len(scanned) > 100, "the scan saw almost no files — it is blind"
    forbidden = _adapter_flags() - ALLOWED_FLAGS
    assert forbidden >= RETIRED_FLAGS
    reads = [hit for p in scanned for hit in _flag_reads(p, forbidden)]
    assert reads == []


def test_the_flag_scan_sees_a_flag_read(tmp_path):
    """The control: the scan finds the shape it forbids, as an attribute and through getattr."""
    probe = tmp_path / "probe.py"
    probe.write_text("def f(a):\n    return a.decides_every_bar or getattr(a, 'nan_policy')\n", encoding="utf-8")
    assert _flag_reads(probe, RETIRED_FLAGS) == ["probe.py:2:decides_every_bar", "probe.py:2:nan_policy"]


# ----------------------------------------------------------------------
# B. Analytic closed form, to the cent
# ----------------------------------------------------------------------

#: Hand-picked prices with round returns. Bar 0's decision fills at close[1] (next-bar).
#:   A: 100 -> 100 -> 110 (+10%) -> 121 (+10%) -> 108.9 (-10%)
#:   B:  50 ->  50 ->  45 (-10%) ->  54 (+20%) ->  54   (0%)
CLOSED_FORM_PRICES = pd.DataFrame(
    {"A": [100.0, 100.0, 110.0, 121.0, 108.9], "B": [50.0, 50.0, 45.0, 54.0, 54.0]},
    index=pd.date_range("2024-01-01", periods=5, freq="D"),
)
#: 60/40, rebalanced every bar, no fees. Bars 0-1 hold cash; then the book earns
#: 0.6 rA + 0.4 rB:  +2%, +14%, -6%.
#: V = 10,000 x 1.02 x 1.14 x 0.94 = 10,930.32 exactly.
CLOSED_FORM_VALUE = 10_930.32


def _closed_form_weights() -> pd.DataFrame:
    return pd.DataFrame({"A": 0.6, "B": 0.4}, index=CLOSED_FORM_PRICES.index)


def _to_the_cent(book: TradedBook) -> pd.Series:
    """The value curve on a 10,000 start (vectorbt starts at its own ``init_cash``)."""
    return book.value * (CAPITAL / _capital(book))


def test_closed_form_rsims_to_the_cent():
    book = _simulate("rsims", CLOSED_FORM_PRICES, _closed_form_weights(), 0.0)
    curve = _to_the_cent(book)
    assert curve.round(2).tolist() == [10_000.0, 10_000.0, 10_200.0, 11_628.0, CLOSED_FORM_VALUE]
    assert book.metrics["total_return"] == pytest.approx(0.093032, abs=1e-12)


@needs_vectorbt
def test_closed_form_vectorbt_to_the_cent():
    book = _simulate("vectorbt", CLOSED_FORM_PRICES, _closed_form_weights(), 0.0)
    curve = _to_the_cent(book)
    assert curve.round(2).tolist() == [10_000.0, 10_000.0, 10_200.0, 11_628.0, CLOSED_FORM_VALUE]
    assert book.metrics["total_return"] == pytest.approx(0.093032, abs=1e-12)


#: One asset, weight 1, fee 0.1%: 100 -> 100 -> 125. Each engine's fee convention by hand:
#:
#: - rsims buys the full 10,000 / 100 = 100 units and pays 10.00 from cash (equity 9,990).
#:   Bar 2 earns 100 x 25 = 2,500 (equity 12,490), then re-sizes to 12,490 / 125 = 99.92
#:   units: it sells 0.08 units (10.00 notional) for a 0.01 fee.  V = 12,489.99.
#: - vectorbt cannot spend more than its cash: it buys 10,000 / (100 x 1.001) units, and on
#:   bar 2 the book is already 100% invested, so no trade. V = 12,500 / 1.001 = 12,487.51.
FEE_PRICES = pd.DataFrame({"A": [100.0, 100.0, 125.0]}, index=pd.date_range("2024-01-01", periods=3, freq="D"))
FEE_WEIGHTS = pd.DataFrame({"A": 1.0}, index=FEE_PRICES.index)


def test_closed_form_fee_convention_rsims_to_the_cent():
    book = _simulate("rsims", FEE_PRICES, FEE_WEIGHTS, 0.001)
    assert _to_the_cent(book).round(2).tolist() == [10_000.0, 9_990.0, 12_489.99]
    assert book.trades["fees"].round(2).tolist() == [10.0, 0.01]


@needs_vectorbt
def test_closed_form_fee_convention_vectorbt_to_the_cent():
    book = _simulate("vectorbt", FEE_PRICES, FEE_WEIGHTS, 0.001)
    assert _to_the_cent(book).round(2).tolist() == [10_000.0, 9_990.01, 12_487.51]
    assert len(book.trades) == 1


# ----------------------------------------------------------------------
# Without vectorbt: "not checked", never green
# ----------------------------------------------------------------------

_BLOCK_VECTORBT = textwrap.dedent(
    """
    import importlib.abc, sys

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {"vectorbt", "numba"}:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _Block())
    try:
        import vectorbt  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        raise SystemExit("HARNESS BROKEN: vectorbt still importable")
    import pytest
    sys.exit(pytest.main(sys.argv[1:]))
    """
)


@pytest.mark.slow
def test_without_vectorbt_parity_reads_not_checked(tmp_path):
    """This module, run with vectorbt blocked: every parity test SKIPS as NOT CHECKED; none passes.

    Only the cases that need no vectorbt (rsims closed forms, the seam's own rules, the source scans) may pass.
    """
    report = tmp_path / "junit.xml"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _BLOCK_VECTORBT,
            str(Path(__file__)),
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "not slow",
            f"--junitxml={report}",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert "HARNESS BROKEN" not in proc.stdout + proc.stderr, proc.stderr
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    cases = ET.parse(report).getroot().iter("testcase")
    outcome: dict[str, str] = {}
    for case in cases:
        skipped = case.find("skipped")
        if skipped is not None:
            outcome[case.get("name")] = "skipped" if NOT_CHECKED in (skipped.get("message") or "") else "other-skip"
        elif case.find("failure") is not None or case.find("error") is not None:
            outcome[case.get("name")] = "failed"
        else:
            outcome[case.get("name")] = "passed"
    passed = sorted(n for n, o in outcome.items() if o == "passed")
    assert passed == [
        "test_a_cost_an_engine_cannot_model_is_refused[fixed_fees]",
        "test_a_cost_an_engine_cannot_model_is_refused[slippage]",
        "test_a_nan_cell_holds_on_every_engine",
        "test_an_adapter_declares_only_the_two_capability_flags",
        "test_closed_form_fee_convention_rsims_to_the_cent",
        "test_closed_form_rsims_to_the_cent",
        "test_closed_form_slippage_and_fixed_fees_to_the_cent[rsims]",
        "test_every_adapter_charges_every_cost_field",
        "test_no_adapter_flag_but_the_capabilities_is_read_outside_the_engine_package",
        "test_rsims_compounds_by_default",
        "test_the_default_leverage_is_one_value_for_every_engine",
        "test_the_flag_scan_sees_a_flag_read",
        "test_threshold_is_a_seam_computed_drift_trigger_known_answer[rsims]",
    ], json.dumps(outcome, indent=1)
    assert set(outcome.values()) == {"passed", "skipped"}, json.dumps(outcome, indent=1)
    assert sum(o == "skipped" for o in outcome.values()) >= 20  # the parity tests were collected and refused
