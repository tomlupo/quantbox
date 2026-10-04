"""Engine parity: the same books through the vectorbt and rsims adapters, via the seam (TOM-262, docs/adr/0008).

vectorbt is quantbox's default engine, so checking quantbox against vectorbt
checks it against itself. rsims is a separate implementation of the same
capability, so the two are each other's cross-check (part A). Part B is an
analytic closed-form case, worked out by hand and asserted to the cent: it
catches an assumption BOTH engines share, which part A cannot.

Every book goes in through :func:`quantbox.engine.simulate_weights`, the seam
that ``backtest()``, ``optimize()`` and the sweep call. Nothing here calls an
engine primitive directly.

THE COMMON SCOPE — where the two engines must agree:

- spot: long-only, gross exposure at most 1;
- no funding, no slippage, no fixed fees (rsims charges ``Costs.fees`` only);
- a decision on every bar (``rebalancing_freq=1``): rsims decides every bar;
- no NaN weight cell (vectorbt holds the last target, rsims goes flat);
- rsims sized off current equity (``capitalise_profits: True``), as vectorbt is.

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

from quantbox.engine import Costs, TradedBook, get_engine, simulate_weights
from quantbox.execution import resolve_execution

VECTORBT = get_engine("vectorbt", require_installed=False).installed()
NOT_CHECKED = "NOT CHECKED: engine parity needs vectorbt (the [vectorbt] extra) — this is not a pass"
needs_vectorbt = pytest.mark.skipif(not VECTORBT, reason=NOT_CHECKED)

TIMING = resolve_execution(None)  # next-bar, the mandatory default (docs/adr/0005)
CAPITAL = 10_000.0
#: rsims inside the common scope: compounding like vectorbt, no buffer, no margin.
RSIMS_COMMON = {"capitalise_profits": True, "trade_buffer": 0.0, "margin": 0.0, "initial_cash": CAPITAL}
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


def _simulate(engine: str, prices: pd.DataFrame, weights: pd.DataFrame, fees: float, **kw) -> TradedBook:
    params = RSIMS_COMMON if engine == "rsims" else None
    book = simulate_weights(
        prices, weights, engine=engine, timing=TIMING, costs=Costs(fees=fees), engine_params=params, **kw
    )
    assert book is not None
    return book


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
@pytest.mark.parametrize("leading", ["flat", "drop"])
def test_total_return_is_the_value_the_book_ends_with(engine, leading):
    """``metrics.total_return`` must equal the value curve's own total return, first bar included.

    ``leading="drop"`` (the sweep) trades on the first bar, so its entry fee is
    the first return. A total return compounded from the SECOND return drops it.
    """
    book = _simulate(engine, _prices(), _books()["rotation"], 0.001, leading=leading)
    assert book.metrics["total_return"] == pytest.approx(book.value.iloc[-1] / _capital(book) - 1.0, abs=1e-12)


# ----------------------------------------------------------------------
# Known disagreements, outside the common scope. strict: a fix turns these red.
# ----------------------------------------------------------------------


@needs_vectorbt
@pytest.mark.xfail(
    strict=True,
    reason=(
        "rsims' default capitalise_profits=False sizes every bar off min(initial_cash, equity): "
        "a winning book stops compounding. vectorbt always sizes off equity. Under the DEFAULTS the "
        "engines disagree by ~1-3% on these fixtures; parity holds with capitalise_profits: True."
    ),
)
def test_parity_under_rsims_default_sizing():
    prices, weights = _prices(), _books()["rotation"]
    v = _simulate("vectorbt", prices, weights, 0.0)
    r = simulate_weights(prices, weights, engine="rsims", timing=TIMING, costs=Costs())
    assert (v.value / _capital(v)).iloc[-1] == pytest.approx((r.value / CAPITAL).iloc[-1], rel=1e-6)


@needs_vectorbt
@pytest.mark.xfail(
    strict=True,
    reason=(
        "rsims charges Costs.fees only: Costs.slippage (and fixed_fees) reach the adapter and are "
        "dropped without a word, so an rsims run with slippage reports a cheaper book than vectorbt."
    ),
)
def test_parity_with_slippage():
    prices, weights = _prices(), _books()["random_daily"]
    costs = Costs(slippage=0.0005)
    v = simulate_weights(prices, weights, engine="vectorbt", timing=TIMING, costs=costs)
    r = simulate_weights(prices, weights, engine="rsims", timing=TIMING, costs=costs, engine_params=RSIMS_COMMON)
    assert (v.value / _capital(v)).iloc[-1] == pytest.approx((r.value / CAPITAL).iloc[-1], rel=1e-4)


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

    Only the rsims closed-form cases (which need no vectorbt) may pass.
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
        "test_closed_form_fee_convention_rsims_to_the_cent",
        "test_closed_form_rsims_to_the_cent",
    ], json.dumps(outcome, indent=1)
    assert set(outcome.values()) == {"passed", "skipped"}, json.dumps(outcome, indent=1)
    assert sum(o == "skipped" for o in outcome.values()) >= 20  # the parity tests were collected and refused
