"""Independent contract tests for venue-derived portfolio valuation.

Written WITHOUT reading the branch's own test file, from the stated contract:

* ``src/quantbox/portfolio_value.py`` module docstring and per-function
  docstrings (the only written statement of the rule);
* ``BrokerPlugin.valuation_basis`` in ``src/quantbox/contracts.py``;
* ``KrakenBroker.get_equity`` docstring;
* ``FuturesRebalancer`` / ``StandardRebalancer`` module docstrings;
* ``TradingPipeline._rebalancer_params`` docstring.

Where the CODE is the only statement of intent, the test docstring says so
explicitly. That is recorded as a finding, not treated as a specification.

Every fixture constructs what it asserts on. Nothing here reads the repo tree,
the network, the environment, or another test's leftovers.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.pipeline.alloc2orders import _summed_holdings, _usd_marks
from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline, _valuation_metrics, _valuation_notes
from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer
from quantbox.plugins.rebalancing.standard_rebalancer import StandardRebalancer
from quantbox.portfolio_value import (
    BASIS_ATTR,
    BASIS_MARGIN,
    BASIS_MARK,
    DEFAULT_RECONCILIATION_TOLERANCE,
    REASON_BROKER_EQUITY_FAILED,
    REASON_MISMATCH,
    REASON_UNKNOWN_BASIS,
    REASON_UNPRICED,
    PortfolioValuationError,
    is_gated,
    resolve_portfolio_value,
    value_holdings,
    venue_valuation_basis,
)

# ======================================================================
# Fixtures -- broker doubles. Each CREATES the state it is asserted on.
# ======================================================================

_UNSET = object()


class _BrokerBase:
    """A broker double with NO ``get_equity``.

    ``resolve_portfolio_value`` branches on ``hasattr(broker, "get_equity")``,
    so the absence of the method is itself a fixture condition and must be
    modelled by a separate class rather than by an attribute.
    """

    def __init__(
        self,
        *,
        basis=_UNSET,
        cash: float = 0.0,
        positions: dict[str, float] | None = None,
        prices: dict[str, float] | None = None,
        quote: str = "USD",
        snapshot_error: BaseException | None = None,
    ) -> None:
        if basis is not _UNSET:
            # Instance attribute: `getattr` resolves it identically to the class
            # attribute real brokers declare. The CLASS-level requirement is
            # asserted against the real brokers in TestBrokerDeclarations.
            self.valuation_basis = basis
        self._cash = float(cash)
        self._positions = dict(positions or {})
        self._prices = dict(prices or {})
        self._quote = quote
        self._snapshot_error = snapshot_error
        self.equity_calls = 0
        self.snapshot_calls = 0

    def get_cash(self) -> dict[str, float]:
        return {self._quote: self._cash}

    def get_positions(self) -> pd.DataFrame:
        if not self._positions:
            return pd.DataFrame(columns=["symbol", "qty"])
        return pd.DataFrame([{"symbol": s, "qty": q} for s, q in self._positions.items()])

    def get_market_snapshot(self, symbols: list[str]) -> pd.DataFrame:
        self.snapshot_calls += 1
        if self._snapshot_error is not None:
            raise self._snapshot_error
        return pd.DataFrame(
            [
                {
                    "symbol": s,
                    "mid": self._prices.get(s),
                    "min_qty": 0.0,
                    "step_size": 0.0,
                    "min_notional": 0.0,
                }
                for s in symbols
            ]
        )


class _BrokerWithEquity(_BrokerBase):
    """A broker double that DOES answer ``get_equity``."""

    def __init__(self, *, equity=0.0, equity_error: BaseException | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._equity = equity
        self._equity_error = equity_error

    def get_equity(self) -> float:
        self.equity_calls += 1
        if self._equity_error is not None:
            raise self._equity_error
        return self._equity


def _price_fn(prices: dict[str, float]):
    return lambda sym: prices.get(sym)


def _must_not_refuse(what: str, fn, *args, **kwargs):
    """Run ``fn`` and turn a refusal into the TEST's own failure.

    "This must NOT refuse" is an assertion, so it has to be written as one. If
    the refusal were allowed to propagate, the test would go red on an
    incidental exception and a mutation run could not tell a real detection
    from a crash somewhere upstream.
    """
    try:
        return fn(*args, **kwargs)
    except PortfolioValuationError as exc:
        pytest.fail(f"{what}: refused with reason={exc.reason!r} ({exc})")


# ======================================================================
# 1. value_holdings -- marking a book, and what it COULD NOT mark
# ======================================================================


class TestValueHoldings:
    """Contract: ``value_holdings`` docstring + the module's "three outcomes"."""

    def test_complete_book_is_cash_plus_marks(self):
        v = value_holdings(cash=92.0, holdings={"BTC": 0.001}, get_price=_price_fn({"BTC": 32000.0}))
        assert v.value == pytest.approx(124.0)
        assert v.cash == pytest.approx(92.0)
        assert v.marked_value == pytest.approx(32.0)
        assert v.state == "complete"
        assert v.n_holdings == 1
        assert v.unpriced == ()

    def test_empty_book_is_cash_and_says_empty_book(self):
        """ "Nothing held" is a legitimate state whose value IS the cash balance."""
        v = value_holdings(cash=92.0, holdings={}, get_price=_price_fn({}))
        assert v.value == pytest.approx(92.0)
        assert v.state == "empty_book"
        assert v.is_empty_book is True
        assert v.n_holdings == 0

    def test_unpriceable_holding_is_collected_not_added_as_zero(self):
        """THE original defect: an unpriced book must not equal an empty one.

        The old loop was ``if p is not None and qty > 0: total += qty*p``, so a
        book with one unmarkable holding produced exactly the cash balance --
        the same number as a book holding nothing.
        """
        v = value_holdings(cash=92.0, holdings={"BTC": 0.001}, get_price=_price_fn({}))
        assert v.state == REASON_UNPRICED
        assert v.unpriced == ("BTC",)
        assert v.n_holdings == 1
        assert v.is_empty_book is False

        empty = value_holdings(cash=92.0, holdings={}, get_price=_price_fn({}))
        # The NUMBERS still coincide -- that is unavoidable, the mark is zero.
        # What must differ is the STATE, which is the whole point of the change.
        assert v.value == pytest.approx(empty.value)
        assert v.state != empty.state

    @pytest.mark.parametrize(
        "bad_price",
        [None, 0.0, -1.0, float("nan"), float("inf"), "not-a-number"],
        ids=["none", "zero", "negative", "nan", "inf", "unparseable"],
    )
    def test_every_unusable_price_is_unpriced(self, bad_price):
        """Contract: ``_is_finite_positive`` -- a price must be finite AND > 0."""
        v = value_holdings(cash=10.0, holdings={"X": 1.0}, get_price=lambda _s: bad_price)
        assert v.unpriced == ("X",), f"{bad_price!r} should not be a usable mark"
        assert v.marked_value == pytest.approx(0.0)

    def test_unreadable_quantity_is_unpriced_not_zero(self):
        """A quantity we cannot read is not a quantity of zero (``value_holdings`` docstring)."""
        v = value_holdings(cash=10.0, holdings={"X": float("nan")}, get_price=_price_fn({"X": 5.0}))
        assert v.unpriced == ("X",)
        assert v.n_holdings == 1
        assert math.isfinite(v.value)

    def test_unparseable_quantity_is_unpriced_not_zero(self):
        v = value_holdings(cash=10.0, holdings={"X": "abc"}, get_price=_price_fn({"X": 5.0}))
        assert v.unpriced == ("X",)
        assert v.n_holdings == 1

    def test_zero_quantity_is_not_a_holding(self):
        v = value_holdings(cash=10.0, holdings={"X": 0.0}, get_price=_price_fn({}))
        assert v.n_holdings == 0
        assert v.state == "empty_book"
        assert v.unpriced == ()

    def test_stable_coin_counted_at_par(self):
        """The quote currency sitting in the positions table is cash, not a mark."""
        v = value_holdings(
            cash=10.0,
            holdings={"USDC": 50.0},
            get_price=_price_fn({}),
            stable_coin="USDC",
        )
        assert v.value == pytest.approx(60.0)
        assert v.unpriced == ()
        assert v.priced == ("USDC",)
        assert v.n_holdings == 1

    def test_stable_coin_wins_over_exclusions(self):
        """Stated explicitly: every caller puts the stable coin in BOTH lists.

        The rebalancers do exactly this (``exclusions + [stable_coin]``), so if
        exclusion won, the quote balance would vanish from every valuation.
        """
        v = value_holdings(
            cash=10.0,
            holdings={"USDC": 50.0},
            get_price=_price_fn({}),
            stable_coin="USDC",
            exclusions=["USDC", "SHIB"],
        )
        assert v.value == pytest.approx(60.0)
        assert v.has_excluded_holdings is False

    def test_excluded_holding_is_neither_valued_nor_a_refusal(self):
        v = value_holdings(
            cash=10.0,
            holdings={"LOCKED": 3.0, "BTC": 0.001},
            get_price=_price_fn({"BTC": 32000.0}),
            exclusions=["LOCKED"],
        )
        assert v.value == pytest.approx(42.0)
        assert v.unpriced == ()
        assert v.n_holdings == 1  # LOCKED is not part of the tradable book
        assert v.has_excluded_holdings is True

    def test_excluded_holding_of_zero_does_not_flag(self):
        """A flat excluded balance does not make the two views disagree."""
        v = value_holdings(
            cash=10.0,
            holdings={"LOCKED": 0.0},
            get_price=_price_fn({}),
            exclusions=["LOCKED"],
        )
        assert v.has_excluded_holdings is False

    def test_excluded_holding_of_unreadable_quantity_does_flag(self):
        """Contract: ``if not math.isfinite(qty) or qty != 0``.

        An unreadable excluded balance might be anything, so it cannot be
        assumed flat -- reconciliation must still be skipped.
        """
        v = value_holdings(
            cash=10.0,
            holdings={"LOCKED": float("nan")},
            get_price=_price_fn({}),
            exclusions=["LOCKED"],
        )
        assert v.has_excluded_holdings is True

    def test_cash_is_signed(self):
        """Stated: the old ``max(0, cash)`` belonged to the MARGIN rule only.

        On a marked venue a debit balance is real and must reduce the book.
        """
        v = value_holdings(cash=-50.0, holdings={"BTC": 0.001}, get_price=_price_fn({"BTC": 32000.0}))
        assert v.value == pytest.approx(-18.0)
        assert v.cash == pytest.approx(-50.0)

    def test_value_holdings_leaves_basis_unset(self):
        """It marks a book without deciding what the mark MEANS for the venue."""
        v = value_holdings(cash=1.0, holdings={}, get_price=_price_fn({}))
        assert v.basis == ""
        assert v.source == "computed"


# ======================================================================
# 2. venue_valuation_basis -- the declaration
# ======================================================================


class TestVenueDeclaration:
    def test_declared_basis_is_returned(self):
        assert venue_valuation_basis(_BrokerBase(basis=BASIS_MARK)) == BASIS_MARK
        assert venue_valuation_basis(_BrokerBase(basis=BASIS_MARGIN)) == BASIS_MARGIN

    def test_no_broker_is_undeclared(self):
        assert venue_valuation_basis(None) is None

    def test_missing_attribute_is_undeclared(self):
        assert venue_valuation_basis(_BrokerBase()) is None

    @pytest.mark.parametrize("bad", ["spot", "MARK_TO_MARKET", "", 1, True, ["mark_to_market"]], ids=repr)
    def test_unrecognised_declaration_is_undeclared_and_logged(self, bad, caplog):
        """ "A declaration nobody can parse carries no more information than none."""
        with caplog.at_level(logging.ERROR, logger="quantbox.portfolio_value"):
            assert venue_valuation_basis(_BrokerBase(basis=bad)) is None
        assert any(BASIS_ATTR in r.getMessage() for r in caplog.records), (
            "an unparseable declaration must be LOUD, not silently undeclared"
        )

    def test_reading_the_declaration_costs_no_api_call(self):
        """Contract: "must be free of API calls and must not vary with account state".

        Modelled by a double whose every I/O surface raises: reading the basis
        must not touch any of them.
        """

        class _Exploding:
            valuation_basis = BASIS_MARK

            def get_cash(self):
                raise AssertionError("get_cash called while reading the declaration")

            def get_positions(self):
                raise AssertionError("get_positions called while reading the declaration")

            def get_equity(self):
                raise AssertionError("get_equity called while reading the declaration")

            def describe(self):
                raise AssertionError("describe() called while reading the declaration")

        assert venue_valuation_basis(_Exploding()) == BASIS_MARK


# ======================================================================
# 3. is_gated -- the allow-list, which must fail CLOSED
# ======================================================================


class TestGating:
    @pytest.mark.parametrize("mode", ["paper", "backtest", "dry_run", "dry-run", "PAPER", " Paper "])
    def test_declared_simulations_are_ungated(self, mode):
        assert is_gated(mode) is False

    @pytest.mark.parametrize(
        "mode",
        ["live", "LIVE", "Live", "", None, "prod", "papertrading", "paper_trade", "  "],
        ids=["live", "upper", "title", "empty", "none", "prod", "nearmiss1", "nearmiss2", "blank"],
    )
    def test_everything_not_positively_a_simulation_is_gated(self, mode):
        """An allow-list, so a forgotten or misspelled mode fails CLOSED."""
        assert is_gated(mode) is True


# ======================================================================
# 4. resolve_portfolio_value -- THE headline: the venue decides
# ======================================================================


class TestVenueDecidesTheRule:
    """The claim nobody could verify: same book, two venues, two numbers."""

    BOOK = {"cash": 92.0, "holdings": {"BTC": 0.001}, "prices": {"BTC": 32000.0}}

    def _resolve(self, basis, *, mode="live", **kw):
        return resolve_portfolio_value(
            broker=_BrokerBase(basis=basis),
            mode=mode,
            cash=self.BOOK["cash"],
            holdings=self.BOOK["holdings"],
            get_price=_price_fn(self.BOOK["prices"]),
            fallback_basis=BASIS_MARK,
            **kw,
        )

    def test_spot_venue_values_cash_plus_holdings(self):
        v = self._resolve(BASIS_MARK)
        assert v.value == pytest.approx(124.0)
        assert v.basis == BASIS_MARK
        assert v.state == "complete"

    def test_derivatives_venue_keeps_margin_balance_semantics(self):
        """A perps book's positions are leveraged: they do NOT add to equity."""
        v = self._resolve(BASIS_MARGIN)
        assert v.value == pytest.approx(92.0)
        assert v.basis == BASIS_MARGIN
        assert v.source == "margin_balance"

    def test_the_two_venues_disagree_by_the_whole_value_of_the_positions(self):
        """The A/B that the incident turned on, in one assertion."""
        spot = self._resolve(BASIS_MARK)
        perps = self._resolve(BASIS_MARGIN)
        assert spot.value - perps.value == pytest.approx(32.0)
        assert spot.marked_value == perps.marked_value  # the MARK is venue-agnostic

    def test_margined_venue_floors_a_debit_balance_at_zero(self):
        """``max(0, cash)`` is the margin-balance rule and lives only there."""
        v = resolve_portfolio_value(
            broker=_BrokerBase(basis=BASIS_MARGIN),
            mode="live",
            cash=-500.0,
            holdings={},
            get_price=_price_fn({}),
            fallback_basis=BASIS_MARGIN,
        )
        assert v.value == pytest.approx(0.0)

    def test_marked_venue_does_not_floor_a_debit_balance(self):
        v = resolve_portfolio_value(
            broker=_BrokerBase(basis=BASIS_MARK),
            mode="paper",
            cash=-500.0,
            holdings={"BTC": 0.001},
            get_price=_price_fn({"BTC": 32000.0}),
            fallback_basis=BASIS_MARK,
        )
        assert v.value == pytest.approx(-468.0)


class TestUndeclaredVenue:
    def test_live_refuses_rather_than_guessing(self):
        with pytest.raises(PortfolioValuationError) as exc:
            resolve_portfolio_value(
                broker=_BrokerBase(),
                mode="live",
                cash=92.0,
                holdings={"BTC": 0.001},
                get_price=_price_fn({"BTC": 32000.0}),
                fallback_basis=BASIS_MARGIN,
            )
        assert exc.value.reason == REASON_UNKNOWN_BASIS

    @pytest.mark.parametrize(("fallback", "expected"), [(BASIS_MARGIN, 92.0), (BASIS_MARK, 124.0)])
    def test_simulation_uses_the_call_sites_historic_basis_loudly(self, fallback, expected, caplog):
        """``fallback_basis`` = what this call site did before declarations existed.

        Both arms are asserted so the argument is OBSERVABLE: if the fallback
        were ignored, the two would return the same number.
        """
        with caplog.at_level(logging.WARNING, logger="quantbox.portfolio_value"):
            v = resolve_portfolio_value(
                broker=_BrokerBase(),
                mode="paper",
                cash=92.0,
                holdings={"BTC": 0.001},
                get_price=_price_fn({"BTC": 32000.0}),
                fallback_basis=fallback,
            )
        assert v.value == pytest.approx(expected)
        assert v.basis == fallback
        assert any(BASIS_ATTR in r.getMessage() for r in caplog.records), (
            "a fallback is a degraded answer and must be logged"
        )

    def test_an_undeclared_broker_is_never_asked_for_its_equity(self):
        """Stated: its equity is "a number whose meaning is exactly the thing
        that was not declared", and reading it would make ``fallback_basis``
        unobservable.
        """
        broker = _BrokerWithEquity(equity=999_999.0)  # declares nothing
        v = resolve_portfolio_value(
            broker=broker,
            mode="paper",
            cash=92.0,
            holdings={"BTC": 0.001},
            get_price=_price_fn({"BTC": 32000.0}),
            fallback_basis=BASIS_MARK,
        )
        assert broker.equity_calls == 0
        assert v.value == pytest.approx(124.0)
        assert v.broker_equity is None

    def test_invalid_fallback_basis_is_a_programming_error(self):
        with pytest.raises(ValueError, match="fallback_basis"):
            resolve_portfolio_value(
                broker=_BrokerBase(basis=BASIS_MARK),
                mode="paper",
                cash=1.0,
                holdings={},
                get_price=_price_fn({}),
                fallback_basis="cash",
            )


class TestMarkedVenueCompleteness:
    def test_live_refuses_on_a_single_unmarkable_holding(self):
        with pytest.raises(PortfolioValuationError) as exc:
            resolve_portfolio_value(
                broker=_BrokerBase(basis=BASIS_MARK),
                mode="live",
                cash=92.0,
                holdings={"BTC": 0.001, "PEPE": 1_000_000.0},
                get_price=_price_fn({"BTC": 32000.0}),
                fallback_basis=BASIS_MARK,
            )
        assert exc.value.reason == REASON_UNPRICED
        assert exc.value.unpriced == ("PEPE",)
        assert "PEPE" in str(exc.value), "the operator must see WHICH name failed"

    def test_simulation_warns_and_continues_understated(self, caplog):
        with caplog.at_level(logging.WARNING, logger="quantbox.portfolio_value"):
            v = resolve_portfolio_value(
                broker=_BrokerBase(basis=BASIS_MARK),
                mode="paper",
                cash=92.0,
                holdings={"BTC": 0.001, "PEPE": 1_000_000.0},
                get_price=_price_fn({"BTC": 32000.0}),
                fallback_basis=BASIS_MARK,
            )
        assert v.value == pytest.approx(124.0)
        assert v.state == REASON_UNPRICED
        assert any("INCOMPLETE" in r.getMessage() for r in caplog.records)

    def test_margined_venue_does_not_refuse_on_an_unmarkable_name_even_on_live(self, caplog):
        """Stated: on a margined venue the name only loses its TARGET.

        Conflating that with a valuation refusal is what made a spot book look
        like a perps book.
        """
        with caplog.at_level(logging.WARNING, logger="quantbox.portfolio_value"):
            v = _must_not_refuse(
                "a margined venue with an unmarkable name",
                resolve_portfolio_value,
                broker=_BrokerBase(basis=BASIS_MARGIN),
                mode="live",
                cash=92.0,
                holdings={"PEPE": 1_000_000.0},
                get_price=_price_fn({}),
                fallback_basis=BASIS_MARGIN,
            )
        assert v.value == pytest.approx(92.0)
        assert v.state == REASON_UNPRICED  # the MARK is still incomplete...
        assert v.basis == BASIS_MARGIN  # ...but equity is not understated
        assert any("lose their target" in r.getMessage() for r in caplog.records)


class TestBrokerEquityAndReconciliation:
    def _spot(self, **kw):
        kw.setdefault("basis", BASIS_MARK)
        return _BrokerWithEquity(**kw)

    def test_marked_venue_sizes_off_the_mark_not_the_brokers_equity(self):
        """Stated: broker equity is a CROSS-CHECK, not the sizing number.

        A broker marks every balance the account holds and knows nothing about
        exclusions, so sizing off it targets capital that cannot be raised.
        """
        broker = self._spot(equity=124.3)  # within tolerance, but NOT the mark
        v = resolve_portfolio_value(
            broker=broker,
            mode="live",
            cash=92.0,
            holdings={"BTC": 0.001},
            get_price=_price_fn({"BTC": 32000.0}),
            fallback_basis=BASIS_MARK,
        )
        assert v.value == pytest.approx(124.0), "the tradable mark must win"
        assert v.broker_equity == pytest.approx(124.3)
        assert v.reconciled is True
        assert v.source == "computed_reconciled"

    def test_margined_venue_prefers_broker_equity_because_it_carries_unrealised_pnl(self):
        broker = _BrokerWithEquity(basis=BASIS_MARGIN, equity=150.0)
        v = _must_not_refuse(
            "a margined venue whose equity exceeds the notional mark",
            resolve_portfolio_value,
            broker=broker,
            mode="live",
            cash=92.0,
            holdings={"BTC": 0.001},
            get_price=_price_fn({"BTC": 32000.0}),
            fallback_basis=BASIS_MARGIN,
        )
        assert v.value == pytest.approx(150.0)
        assert v.source == "broker_equity"
        assert v.reconciled is False, "reconciling margin against a notional mark is a false alarm"

    def test_reconciliation_mismatch_refuses_on_live(self):
        broker = self._spot(equity=92.0)  # the venue says the positions are gone
        with pytest.raises(PortfolioValuationError) as exc:
            resolve_portfolio_value(
                broker=broker,
                mode="live",
                cash=92.0,
                holdings={"BTC": 0.001},
                get_price=_price_fn({"BTC": 32000.0}),
                fallback_basis=BASIS_MARK,
            )
        assert exc.value.reason == REASON_MISMATCH
        assert exc.value.computed == pytest.approx(124.0)
        assert exc.value.broker_equity == pytest.approx(92.0)

    def test_reconciliation_mismatch_only_warns_off_live(self, caplog):
        with caplog.at_level(logging.WARNING, logger="quantbox.portfolio_value"):
            v = resolve_portfolio_value(
                broker=self._spot(equity=92.0),
                mode="paper",
                cash=92.0,
                holdings={"BTC": 0.001},
                get_price=_price_fn({"BTC": 32000.0}),
                fallback_basis=BASIS_MARK,
            )
        assert v.value == pytest.approx(124.0)
        assert any("reconciliation FAILED" in r.getMessage() for r in caplog.records)

    def test_default_tolerance_admits_bid_ask_drift_and_refuses_a_missing_position(self):
        """0.5% "absorbs the bid/ask and timing gap ... without absorbing a
        missing position". Asserted on BOTH sides of the default threshold."""
        computed = 100.0  # cash 100, nothing held
        near = computed / (1 - 0.004)  # 0.4% apart -- timing drift
        far = computed / (1 - 0.02)  # 2% apart -- a missing position

        v = resolve_portfolio_value(
            broker=self._spot(equity=near),
            mode="live",
            cash=computed,
            holdings={},
            get_price=_price_fn({}),
            fallback_basis=BASIS_MARK,
        )
        assert v.reconciled is True

        with pytest.raises(PortfolioValuationError):
            resolve_portfolio_value(
                broker=self._spot(equity=far),
                mode="live",
                cash=computed,
                holdings={},
                get_price=_price_fn({}),
                fallback_basis=BASIS_MARK,
            )
        assert DEFAULT_RECONCILIATION_TOLERANCE == 0.005

    def test_the_comparison_is_inclusive_at_the_exact_boundary(self):
        """``drift <= tolerance``, pinned EXACTLY rather than near it.

        The drift is measured first, then fed back as the tolerance: equal must
        pass, and one float below it must refuse. Constructing a book whose
        drift lands on 0.005 in binary floating point is not possible, so the
        boundary is probed from the tolerance side instead.
        """
        computed, equity = 100.0, 100.5
        drift = abs(equity - computed) / max(abs(equity), abs(computed))

        def _run(tol):
            return resolve_portfolio_value(
                broker=self._spot(equity=equity),
                mode="live",
                cash=computed,
                holdings={},
                get_price=_price_fn({}),
                fallback_basis=BASIS_MARK,
                tolerance=tol,
            )

        at_boundary = _must_not_refuse("drift exactly equal to the tolerance", _run, drift)
        assert at_boundary.reconciled is True
        with pytest.raises(PortfolioValuationError):
            _run(math.nextafter(drift, 0.0))

    def test_custom_tolerance_is_honoured(self):
        with pytest.raises(PortfolioValuationError):
            resolve_portfolio_value(
                broker=self._spot(equity=100.2),
                mode="live",
                cash=100.0,
                holdings={},
                get_price=_price_fn({}),
                fallback_basis=BASIS_MARK,
                tolerance=0.0001,
            )

    def test_reconciliation_is_skipped_when_an_excluded_asset_is_held(self, caplog):
        """The two views measure different books, so comparing them false-alarms."""
        with caplog.at_level(logging.WARNING, logger="quantbox.portfolio_value"):
            v = _must_not_refuse(
                "a fully marked tradable book holding an excluded asset",
                resolve_portfolio_value,
                broker=self._spot(equity=1_000.0),
                mode="live",
                cash=92.0,
                holdings={"BTC": 0.001, "LOCKED": 5.0},
                get_price=_price_fn({"BTC": 32000.0}),
                fallback_basis=BASIS_MARK,
                exclusions=["LOCKED"],
            )
        assert v.value == pytest.approx(124.0)
        assert v.reconciled is False
        assert v.source == "computed"
        assert v.broker_equity == pytest.approx(1_000.0)
        assert any("EXCLUDED" in r.getMessage() for r in caplog.records)

    def test_require_reconciliation_false_does_not_disable_the_completeness_gate(self):
        """Stated in the config schema: "Does NOT disable the completeness gate"."""
        # (a) a wild mismatch is tolerated
        v = resolve_portfolio_value(
            broker=self._spot(equity=10_000.0),
            mode="live",
            cash=92.0,
            holdings={"BTC": 0.001},
            get_price=_price_fn({"BTC": 32000.0}),
            fallback_basis=BASIS_MARK,
            require_reconciliation=False,
        )
        assert v.value == pytest.approx(124.0)
        assert v.reconciled is False
        # (b) an unmarkable holding STILL refuses
        with pytest.raises(PortfolioValuationError) as exc:
            resolve_portfolio_value(
                broker=self._spot(equity=10_000.0),
                mode="live",
                cash=92.0,
                holdings={"PEPE": 1.0},
                get_price=_price_fn({}),
                fallback_basis=BASIS_MARK,
                require_reconciliation=False,
            )
        assert exc.value.reason == REASON_UNPRICED

    def test_both_views_worth_nothing_agree(self):
        """denominator <= 0: the downstream zero-value guard is what acts."""
        v = resolve_portfolio_value(
            broker=self._spot(equity=0.0),
            mode="live",
            cash=0.0,
            holdings={},
            get_price=_price_fn({}),
            fallback_basis=BASIS_MARK,
        )
        assert v.value == pytest.approx(0.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
    def test_non_finite_broker_equity_refuses_on_live(self, bad):
        """``nan <= 0`` is False, so a NaN sails through every downstream guard."""
        with pytest.raises(PortfolioValuationError) as exc:
            resolve_portfolio_value(
                broker=self._spot(equity=bad),
                mode="live",
                cash=92.0,
                holdings={},
                get_price=_price_fn({}),
                fallback_basis=BASIS_MARK,
            )
        assert exc.value.reason == REASON_BROKER_EQUITY_FAILED

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
    def test_non_finite_broker_equity_is_never_published_even_off_live(self, bad):
        """The guard validates into a LOCAL before publishing -- see the comment.

        Assigning first and raising second would leave the NaN in
        ``broker_equity`` for the ungated arm, handing paper a NaN.
        """
        v = resolve_portfolio_value(
            broker=self._spot(equity=bad),
            mode="paper",
            cash=92.0,
            holdings={},
            get_price=_price_fn({}),
            fallback_basis=BASIS_MARK,
        )
        assert v.broker_equity is None
        assert math.isfinite(v.value)
        assert v.value == pytest.approx(92.0)

    def test_broker_equity_raising_refuses_on_live(self):
        with pytest.raises(PortfolioValuationError) as exc:
            resolve_portfolio_value(
                broker=self._spot(equity_error=RuntimeError("venue 503")),
                mode="live",
                cash=92.0,
                holdings={},
                get_price=_price_fn({}),
                fallback_basis=BASIS_MARK,
            )
        assert exc.value.reason == REASON_BROKER_EQUITY_FAILED
        assert exc.value.computed == pytest.approx(92.0)

    def test_broker_equity_raising_only_warns_off_live(self, caplog):
        with caplog.at_level(logging.WARNING, logger="quantbox.portfolio_value"):
            v = resolve_portfolio_value(
                broker=self._spot(equity_error=RuntimeError("venue 503")),
                mode="paper",
                cash=92.0,
                holdings={},
                get_price=_price_fn({}),
                fallback_basis=BASIS_MARK,
            )
        assert v.value == pytest.approx(92.0)
        assert v.broker_equity is None
        assert any("Broker equity unavailable" in r.getMessage() for r in caplog.records)

    def test_a_brokers_own_valuation_refusal_propagates_unchanged_on_live(self):
        """Stated: "the same refusal, raised closer to the data".

        On a GATED mode the broker already applied the rule, and its reason is
        more specific than ours, so it must survive intact rather than be
        relabelled `broker_equity_failed` by the generic arm.
        """
        inner = PortfolioValuationError("cannot mark XBT", reason=REASON_UNPRICED, unpriced=("XBT",))
        with pytest.raises(PortfolioValuationError) as exc:
            resolve_portfolio_value(
                broker=self._spot(equity_error=inner),
                mode="live",
                cash=92.0,
                holdings={},
                get_price=_price_fn({}),
                fallback_basis=BASIS_MARK,
            )
        assert exc.value is inner, "the broker's own reason must survive"
        assert exc.value.reason == REASON_UNPRICED

    def test_a_brokers_refusal_does_not_kill_an_ungated_run(self, caplog):
        """Both review axes, round 1: this used to propagate in EVERY mode.

        The module header promises paper and backtest keep working, LOUDLY, and
        `_resolve_marked` gates its own completeness check on `gated` — so
        re-raising here sat above that and killed the ungated arm outright. The
        broker's equity is only a CROSS-CHECK on a marked venue; losing it is a
        warning, not a stop, exactly as an unreadable `get_equity()` already was.
        """
        inner = PortfolioValuationError("cannot mark XBT", reason=REASON_UNPRICED, unpriced=("XBT",))
        with caplog.at_level(logging.WARNING, logger="quantbox.portfolio_value"):
            v = resolve_portfolio_value(
                broker=self._spot(equity_error=inner),
                mode="paper",
                cash=92.0,
                holdings={},
                get_price=_price_fn({}),
                fallback_basis=BASIS_MARK,
            )
        assert v.value == pytest.approx(92.0)
        assert v.broker_equity is None
        assert v.reconciled is False
        assert any("refused to value its own book" in r.getMessage() for r in caplog.records)

    def test_a_brokers_refusal_about_an_EXCLUDED_name_does_not_halt_live(self, caplog):
        """Both review axes, round 1 — the sharper half, and it is a LIVE path.

        A broker marks every balance the account holds and knows nothing about
        `exclusions`, so its refusal may be entirely about a name this run does
        not trade while the tradable mark is complete. `_resolve_marked` already
        skips the cross-check for that reason (`has_excluded_holdings`); the
        unconditional re-raise made that branch unreachable in the one case it
        was built for, halting the live Kraken book it was meant to let through.

        Positive control is the test above it: with NO excluded holding, live
        still refuses.
        """
        inner = PortfolioValuationError("cannot mark LOCKED", reason=REASON_UNPRICED, unpriced=("LOCKED",))
        with caplog.at_level(logging.WARNING, logger="quantbox.portfolio_value"):
            v = resolve_portfolio_value(
                broker=self._spot(equity_error=inner),
                mode="live",
                cash=92.0,
                holdings={"BTC": 0.001, "LOCKED": 5.0},
                get_price=_price_fn({"BTC": 32000.0}),
                fallback_basis=BASIS_MARK,
                exclusions=["LOCKED"],
            )
        assert v.value == pytest.approx(124.0)
        assert v.has_excluded_holdings is True
        assert v.broker_equity is None
        # The warning must NAME what the broker could not mark, so a reader can
        # check the claim rather than take it: round 1's text asserted only
        # that the book "holds an excluded asset", which was true of the venue
        # outages it was wrongly waving through.
        assert any("LOCKED" in r.getMessage() and "does not trade" in r.getMessage() for r in caplog.records)

    def test_a_venue_OUTAGE_still_halts_live_even_with_an_excluded_holding(self):
        """Round 2, on round 1's own fix: it asked the WRONG QUESTION.

        Round 1 gated the escape on "does this book hold an excluded name",
        which discards a total-venue-outage refusal on a LIVE run because the
        book happens to hold one excluded dust coin — trading with no
        cross-check at all. The question is what the broker REFUSED ABOUT.

        Here the refusal names nothing (an outage), so live must still halt.
        """
        inner = PortfolioValuationError("kraken timeout", reason=REASON_BROKER_EQUITY_FAILED)
        with pytest.raises(PortfolioValuationError) as exc:
            resolve_portfolio_value(
                broker=self._spot(equity_error=inner),
                mode="live",
                cash=92.0,
                holdings={"BTC": 0.001, "LOCKED": 5.0},
                get_price=_price_fn({"BTC": 32000.0}),
                fallback_basis=BASIS_MARK,
                exclusions=["LOCKED"],
            )
        assert exc.value is inner

    def test_a_refusal_naming_a_TRADED_name_still_halts_live_with_exclusions_present(self):
        """The other half of the same question, and the sharper one.

        The book holds an excluded name AND the broker's refusal names a name
        this run DOES trade. Round 1's predicate waved this through on live.
        """
        inner = PortfolioValuationError("cannot mark BTC", reason=REASON_UNPRICED, unpriced=("BTC",))
        with pytest.raises(PortfolioValuationError) as exc:
            resolve_portfolio_value(
                broker=self._spot(equity_error=inner),
                mode="live",
                cash=92.0,
                holdings={"BTC": 0.001, "LOCKED": 5.0},
                get_price=_price_fn({"BTC": 32000.0}),
                fallback_basis=BASIS_MARK,
                exclusions=["LOCKED"],
            )
        assert exc.value is inner
        assert exc.value.unpriced == ("BTC",)

    def test_an_ungated_reconciliation_mismatch_is_not_recorded_as_reconciled(self):
        """Round 1: `reconciled = True` was set from the CALL, not its result.

        Ungated, `_reconcile_or_raise` only warns on a real disagreement — so a
        wildly-apart run was stamped `source='computed_reconciled'` and
        `portfolio_value_reconciled=1.0`: a failed cross-check recorded as a
        passed one.
        """
        v = resolve_portfolio_value(
            broker=self._spot(equity=1_000.0),
            mode="paper",
            cash=92.0,
            holdings={"BTC": 0.001},
            get_price=_price_fn({"BTC": 32000.0}),
            fallback_basis=BASIS_MARK,
            tolerance=0.005,
        )
        assert v.value == pytest.approx(124.0)
        assert v.broker_equity == pytest.approx(1_000.0)
        assert v.reconciled is False, "the two views were 87% apart"
        assert v.source == "computed"
        assert v.as_metrics()["portfolio_value_reconciled"] == 0.0

    def test_an_ungated_reconciliation_AGREEMENT_is_still_recorded(self):
        """Positive control for the test above: `reconciled` must still go True.

        Without this, returning a hard `False` from `_reconcile_or_raise` would
        satisfy the mismatch test and silently stop recording every good run.
        """
        v = resolve_portfolio_value(
            broker=self._spot(equity=124.0),
            mode="paper",
            cash=92.0,
            holdings={"BTC": 0.001},
            get_price=_price_fn({"BTC": 32000.0}),
            fallback_basis=BASIS_MARK,
            tolerance=0.005,
        )
        assert v.reconciled is True
        assert v.source == "computed_reconciled"


# ======================================================================
# 5. What reaches the human: metrics and notes
# ======================================================================


class TestRunRecord:
    """Contract: "the three outcomes are distinguishable in the exit path,
    the metrics and the logs"."""

    def _resolve(self, holdings, prices, basis=BASIS_MARK):
        return resolve_portfolio_value(
            broker=_BrokerBase(basis=basis),
            mode="paper",
            cash=92.0,
            holdings=holdings,
            get_price=_price_fn(prices),
            fallback_basis=BASIS_MARK,
        )

    def test_the_three_states_are_distinguishable_in_metrics(self):
        empty = self._resolve({}, {}).as_metrics()
        complete = self._resolve({"BTC": 0.001}, {"BTC": 32000.0}).as_metrics()
        unpriced = self._resolve({"BTC": 0.001}, {}).as_metrics()

        assert empty["portfolio_value_n_holdings"] == 0.0
        assert complete["portfolio_value_n_holdings"] == 1.0
        assert complete["portfolio_value_n_unpriced"] == 0.0
        assert unpriced["portfolio_value_n_unpriced"] == 1.0
        assert unpriced["portfolio_valuation_complete"] == 0.0
        # The whole point: no two of the three collapse to the same record.
        assert len({tuple(sorted(m.items(), key=str)) for m in (empty, complete, unpriced)}) == 3

    def test_metrics_are_numeric_or_none_because_runresult_requires_it(self):
        m = self._resolve({"BTC": 0.001}, {"BTC": 32000.0}).as_metrics()
        for k, val in m.items():
            assert val is None or isinstance(val, float), f"{k}={val!r} is not RunResult-safe"

    def test_notes_carry_the_state_and_the_basis_separately(self):
        """On a margined book ``unpriced_holdings`` means "lost their target",
        not "equity is wrong" -- which is why ``basis`` sits next to it."""
        n = self._resolve({"PEPE": 1.0}, {}, basis=BASIS_MARGIN).as_notes()
        assert n["portfolio_valuation_state"] == REASON_UNPRICED
        assert n["portfolio_valuation_basis"] == BASIS_MARGIN
        assert n["portfolio_valuation_unpriced"] == ["PEPE"]

    def test_a_missing_valuation_reads_as_not_measured_not_as_clean(self):
        """Contract (``_valuation_metrics`` docstring, repo convention #92):
        a metric that DISAPPEARS reads as "fine" to every consumer downstream.
        """
        measured = _valuation_metrics(self._resolve({"BTC": 0.001}, {"BTC": 32000.0}))
        missing = _valuation_metrics(None)
        assert set(missing) == set(measured), "the key set must not shrink"
        assert all(v is None for v in missing.values())
        assert _valuation_notes(None)["portfolio_valuation_state"] == "unknown"
        assert missing["portfolio_valuation_complete"] is not 0.0  # noqa: F632 - identity is the point
        assert missing["portfolio_valuation_complete"] is None


# ======================================================================
# 6. Broker declarations -- every venue must say
# ======================================================================


class TestBrokerDeclarations:
    """Contract: ``BrokerPlugin.valuation_basis`` -- "a plain class attribute".

    Asserted against the REGISTERED brokers (production objects), because the
    scope of this change was 19 files precisely so that no venue is left
    undeclared. An undeclared one is a live refusal.
    """

    @staticmethod
    def _broker_classes():
        from quantbox.plugins.builtins import builtins

        return dict(builtins()["broker"])

    def test_the_sweep_actually_sees_brokers(self):
        """A scan that read 0 brokers did not find nothing -- it did not look."""
        classes = self._broker_classes()
        assert len(classes) >= 8, f"the sweep only saw {sorted(classes)}"
        assert "kraken.spot.v1" in classes, "the venue the incident was on must be in view"

    def test_every_registered_broker_declares_a_valid_basis_on_the_class(self):
        undeclared = {}
        for name, cls in self._broker_classes().items():
            declared = getattr(cls, BASIS_ATTR, None)
            if declared not in (BASIS_MARK, BASIS_MARGIN):
                undeclared[name] = declared
        assert not undeclared, f"undeclared or invalid valuation_basis: {undeclared}"

    @pytest.mark.parametrize(
        ("plugin_name", "expected"),
        [
            ("kraken.spot.v1", BASIS_MARK),
            ("binance.live.v1", BASIS_MARK),
            ("binance.paper.stub.v1", BASIS_MARK),
            ("ibkr.paper.stub.v1", BASIS_MARK),
            ("sim.paper.v1", BASIS_MARK),
            ("binance.futures.v1", BASIS_MARGIN),
            ("hyperliquid.perps.v1", BASIS_MARGIN),
            ("sim.futures_paper.v1", BASIS_MARGIN),
        ],
    )
    def test_spot_venues_mark_and_derivatives_venues_margin(self, plugin_name, expected):
        classes = self._broker_classes()
        if plugin_name not in classes:
            pytest.skip(f"{plugin_name} not registered in this build")
        assert getattr(classes[plugin_name], BASIS_ATTR, None) == expected

    def test_unregistered_broker_modules_declare_one_too(self):
        """``binance_live.py`` is in the diff but NOT in ``builtins()``.

        The registry sweep above cannot see it, so a broker reachable only by
        entry point would slip through into a live refusal.
        """
        from quantbox.plugins.broker.binance_live import BinanceLiveBroker

        assert getattr(BinanceLiveBroker, BASIS_ATTR, None) == BASIS_MARK

    def test_the_declaration_is_readable_without_instantiating(self):
        """ "must be free of API calls" -- so it cannot live on an instance."""
        for name, cls in self._broker_classes().items():
            assert BASIS_ATTR in dir(cls), f"{name} hides its basis behind construction"


# ======================================================================
# 7. The incident, end to end, through the rebalancer that caused it
# ======================================================================

# The live crypto-trend-kraken book at the moment the defect was found.
_CASH = 92.0
_BTC_QTY = 0.001
_BTC_PRICE = 32_000.0
_EQUITY = _CASH + _BTC_QTY * _BTC_PRICE  # 124.0
_WEIGHT = 0.4293


def _futures_orders(broker, *, mode, **extra):
    return FuturesRebalancer().generate_orders(
        weights={"BTC": _WEIGHT},
        broker=broker,
        params={
            "mode": mode,
            "stable_coin_symbol": "USD",
            "capital_at_risk": 1.0,
            "max_leverage": 1.0,
            **extra,
        },
    )


class TestTheIncident:
    """``configs/crypto_trend_kraken.yaml`` ran ``rebalancing.futures.v1``
    against ``kraken.spot.v1``. The futures rule (margin balance = cash) was
    applied faithfully to a book where it is wrong by the positions held.
    """

    def _book(self, basis):
        return _BrokerBase(
            basis=basis,
            cash=_CASH,
            positions={"BTC": _BTC_QTY},
            prices={"BTC": _BTC_PRICE},
            quote="USD",
        )

    def test_futures_rebalancer_on_a_spot_broker_now_sizes_off_equity(self):
        res = _futures_orders(self._book(BASIS_MARK), mode="live")
        assert res["total_value"] == pytest.approx(_EQUITY)
        assert res["valuation"].basis == BASIS_MARK
        row = res["rebalancing"].set_index("Asset").loc["BTC"]
        assert row["Target Value"] == pytest.approx(_EQUITY * _WEIGHT)  # 53.23

    def test_futures_rebalancer_on_a_perps_broker_is_unchanged(self):
        res = _futures_orders(self._book(BASIS_MARGIN), mode="live")
        assert res["total_value"] == pytest.approx(_CASH)
        assert res["valuation"].basis == BASIS_MARGIN
        row = res["rebalancing"].set_index("Asset").loc["BTC"]
        assert row["Target Value"] == pytest.approx(_CASH * _WEIGHT)  # 39.50

    def test_positive_control_the_legacy_rule_would_fail_the_spot_assertion(self):
        """POSITIVE CONTROL.

        Reproduces the deleted line ``total_value = max(0, cash_available)``
        and shows it produces a DIFFERENT number from the one the spot test
        asserts -- so that assertion genuinely discriminates rather than
        happening to hold for both rules.
        """
        legacy_total = max(0.0, _CASH)
        new_total = _futures_orders(self._book(BASIS_MARK), mode="live")["total_value"]
        assert legacy_total != pytest.approx(new_total)
        assert legacy_total * _WEIGHT == pytest.approx(39.4956, abs=1e-4)
        assert new_total * _WEIGHT == pytest.approx(53.2332, abs=1e-4)

    def test_the_understatement_grows_as_the_book_fills(self):
        """ "the targets shrank as positions accumulated -- it chased its own tail".

        Three snapshots of the SAME $124 book at different cash/position
        splits: the venue-derived value is invariant, the legacy one decays.
        """
        splits = [(124.0, 0.0), (92.0, 0.001), (60.0, 0.002)]
        venue_values, legacy_values = [], []
        for cash, qty in splits:
            broker = _BrokerBase(
                basis=BASIS_MARK,
                cash=cash,
                positions={"BTC": qty} if qty else {},
                prices={"BTC": _BTC_PRICE},
                quote="USD",
            )
            venue_values.append(_futures_orders(broker, mode="live")["total_value"])
            legacy_values.append(max(0.0, cash))
        assert venue_values == pytest.approx([124.0, 124.0, 124.0])
        assert legacy_values == pytest.approx([124.0, 92.0, 60.0])

    def test_a_live_run_on_an_undeclared_broker_refuses_through_the_rebalancer(self):
        broker = _BrokerBase(cash=_CASH, positions={"BTC": _BTC_QTY}, prices={"BTC": _BTC_PRICE}, quote="USD")
        with pytest.raises(PortfolioValuationError) as exc:
            _futures_orders(broker, mode="live")
        assert exc.value.reason == REASON_UNKNOWN_BASIS

    def test_a_rebalancer_called_without_a_mode_is_gated(self):
        """A forgotten ``mode`` must fail CLOSED, not default to paper."""
        broker = _BrokerBase(cash=_CASH, positions={"BTC": _BTC_QTY}, prices={"BTC": _BTC_PRICE}, quote="USD")
        with pytest.raises(PortfolioValuationError) as exc:
            FuturesRebalancer().generate_orders(
                weights={"BTC": _WEIGHT},
                broker=broker,
                params={"stable_coin_symbol": "USD", "capital_at_risk": 1.0, "max_leverage": 1.0},
            )
        assert exc.value.reason == REASON_UNKNOWN_BASIS

    def test_a_snapshot_failure_makes_a_spot_book_unmarkable_and_refuses_on_live(self, caplog):
        """Previously ``except Exception: pass`` -- the value collapsed to cash."""
        broker = _BrokerBase(
            basis=BASIS_MARK,
            cash=_CASH,
            positions={"BTC": _BTC_QTY},
            prices={"BTC": _BTC_PRICE},
            quote="USD",
            snapshot_error=RuntimeError("venue 503"),
        )
        with caplog.at_level(logging.ERROR), pytest.raises(PortfolioValuationError) as exc:
            _futures_orders(broker, mode="live")
        assert exc.value.reason == REASON_UNPRICED
        assert any("Market snapshot failed" in r.getMessage() for r in caplog.records), (
            "the swallowed cause must now be in the log"
        )

    def test_a_snapshot_failure_on_a_perps_book_is_not_a_valuation_refusal(self):
        broker = _BrokerBase(
            basis=BASIS_MARGIN,
            cash=_CASH,
            positions={"BTC": _BTC_QTY},
            prices={"BTC": _BTC_PRICE},
            quote="USD",
            snapshot_error=RuntimeError("venue 503"),
        )
        res = _must_not_refuse(
            "a live perps run whose snapshot failed",
            _futures_orders,
            broker,
            mode="live",
        )
        assert res["total_value"] == pytest.approx(_CASH)
        assert res["valuation"].state == REASON_UNPRICED

    def test_a_paper_run_on_an_undeclared_broker_keeps_this_call_sites_margin_rule(self):
        """``fallback_basis=BASIS_MARGIN`` is specific to THIS call site.

        Asserted through the rebalancer (not just the module) because the
        argument is the call site's claim about its own history, and the only
        way to see it is a number that differs from the other rebalancer's.
        """
        broker = _BrokerBase(cash=_CASH, positions={"BTC": _BTC_QTY}, prices={"BTC": _BTC_PRICE}, quote="USD")
        futures = _futures_orders(broker, mode="paper")
        assert futures["total_value"] == pytest.approx(_CASH)
        assert futures["valuation"].basis == BASIS_MARGIN

        standard = StandardRebalancer().generate_orders(
            weights={"BTC": _WEIGHT},
            broker=_BrokerBase(cash=_CASH, positions={"BTC": _BTC_QTY}, prices={"BTC": _BTC_PRICE}, quote="USD"),
            params={"mode": "paper", "stable_coin_symbol": "USD", "capital_at_risk": 1.0},
        )
        assert standard["total_value"] == pytest.approx(_EQUITY)
        assert futures["total_value"] != pytest.approx(standard["total_value"]), (
            "the two call sites must disagree, or fallback_basis is unobservable"
        )

    def test_the_valuation_is_carried_out_to_the_caller(self):
        """The run record must be able to say WHICH rule valued this book."""
        res = _futures_orders(self._book(BASIS_MARK), mode="live")
        assert "valuation" in res
        assert res["valuation"].as_notes()["portfolio_valuation_basis"] == BASIS_MARK


class TestStandardRebalancerGate:
    def _book(self, basis, **kw):
        kw.setdefault("cash", _CASH)
        kw.setdefault("positions", {"BTC": _BTC_QTY})
        kw.setdefault("prices", {"BTC": _BTC_PRICE})
        return _BrokerBase(basis=basis, quote="USD", **kw)

    def _orders(self, broker, *, mode, **extra):
        return StandardRebalancer().generate_orders(
            weights={"BTC": _WEIGHT},
            broker=broker,
            params={"mode": mode, "stable_coin_symbol": "USD", "capital_at_risk": 1.0, **extra},
        )

    def test_spot_book_is_marked(self):
        res = self._orders(self._book(BASIS_MARK), mode="live")
        assert res["total_value"] == pytest.approx(_EQUITY)

    def test_live_refuses_on_an_unmarkable_holding(self):
        broker = self._book(BASIS_MARK, positions={"BTC": _BTC_QTY, "PEPE": 1.0})
        with pytest.raises(PortfolioValuationError) as exc:
            self._orders(broker, mode="live")
        assert exc.value.reason == REASON_UNPRICED
        assert exc.value.unpriced == ("PEPE",)

    def test_paper_continues_understated(self):
        broker = self._book(BASIS_MARK, positions={"BTC": _BTC_QTY, "PEPE": 1.0})
        res = self._orders(broker, mode="paper")
        assert res["total_value"] == pytest.approx(_EQUITY)

    def test_the_valuation_survives_a_successful_standard_rebalance(self):
        """Was a strict xfail: the success return DROPPED `valuation`.

        `FuturesRebalancer` carried it on both exits; `StandardRebalancer`
        carried it only on the zero-value one. Fixed by adding the key to the
        final return dict in ``standard_rebalancer.py``; this is now a plain
        assertion, and the seam-level consequence is pinned below.
        """
        broker = self._book(BASIS_MARK, positions={"BTC": _BTC_QTY, "PEPE": 1.0})
        res = self._orders(broker, mode="paper")
        assert res["valuation"].unpriced == ("PEPE",)

    def test_a_successful_standard_rebalance_reports_as_MEASURED(self):
        """The CONSEQUENCE of the fix above, asserted at the seam that matters.

        Positive control on the run record itself: a healthy spot book must not
        reach ``portfolio_daily`` wearing this repo's "could not be MEASURED"
        signal. Delete the ``"valuation"`` key from the rebalancer's success
        return and this goes red.
        """
        broker = self._book(BASIS_MARK)
        res = self._orders(broker, mode="paper")
        assert res["total_value"] == pytest.approx(_EQUITY), "the book WAS valued..."
        metrics = _valuation_metrics(res.get("valuation"))
        notes = _valuation_notes(res.get("valuation"))
        assert not all(v is None for v in metrics.values()), "...and the run record must say so"
        assert notes["portfolio_valuation_state"] != "unknown"
        assert metrics["portfolio_value_n_unpriced"] == 0.0
        assert metrics["portfolio_valuation_complete"] == 1.0

    def test_a_zero_value_book_returns_the_valuation_rather_than_dropping_it(self):
        """The empty-orders exit must still carry WHY the value was zero."""
        broker = self._book(BASIS_MARK, cash=0.0, positions={})
        res = self._orders(broker, mode="paper")
        assert res["total_value"] == 0.0
        assert res["valuation"].state == "empty_book"

    def test_a_missing_mode_is_gated_here_too(self):
        broker = self._book(BASIS_MARK, positions={"BTC": _BTC_QTY, "PEPE": 1.0})
        with pytest.raises(PortfolioValuationError):
            StandardRebalancer().generate_orders(
                weights={"BTC": _WEIGHT},
                broker=broker,
                params={"stable_coin_symbol": "USD", "capital_at_risk": 1.0},
            )


# ======================================================================
# 8. The pipeline must not let a config ungate a live book
# ======================================================================


class TestPipelineModeThreading:
    """Contract: ``_rebalancer_params`` -- ``mode`` is ASSIGNED, never
    ``setdefault``-ed, because the rebalancer decides the gate from it."""

    def _params(self, cfg_params, mode):
        return TradingPipeline()._rebalancer_params(
            rebalancer_cfg={"params": dict(cfg_params)},
            params={},
            strategy_results={},
            mode=mode,
        )

    def test_a_config_cannot_declare_a_live_run_to_be_paper(self):
        assert self._params({"mode": "paper"}, "live")["mode"] == "live"

    def test_the_real_mode_is_threaded_when_the_config_is_silent(self):
        assert self._params({}, "live")["mode"] == "live"
        assert self._params({}, "paper")["mode"] == "paper"

    @pytest.mark.parametrize("key", ["capital_at_risk", "stable_coin_symbol", "exclusions", "strategy_weights"])
    def test_everything_else_remains_a_config_overridable_default(self, key):
        sentinel = {
            "capital_at_risk": 0.123,
            "stable_coin_symbol": "ZZZ",
            "exclusions": ["Q"],
            "strategy_weights": {"a": 1.0},
        }[key]
        assert self._params({key: sentinel}, "live")[key] == sentinel

    def test_a_forgotten_mode_argument_is_impossible_at_this_seam(self):
        """``_generate_orders`` requires ``mode`` rather than defaulting it."""
        import inspect

        sig = inspect.signature(TradingPipeline._generate_orders)
        assert sig.parameters["mode"].default is inspect.Parameter.empty


# ======================================================================
# 9. Kraken: the venue the incident was on
# ======================================================================


class _FakeCcxt:
    """A ccxt double. Constructs every market and balance it is asserted on."""

    def __init__(self, *, balances, tickers, markets=None, balance_error=None, ticker_error=None):
        self._balances = balances
        self._tickers = tickers
        self._markets = markets if markets is not None else {f"{a}/USD": {} for a in tickers}
        self._balance_error = balance_error
        self._ticker_error = ticker_error
        self.balance_calls = 0

    def load_markets(self):
        return self._markets

    def fetch_balance(self):
        self.balance_calls += 1
        if self._balance_error is not None:
            raise self._balance_error
        return {"total": dict(self._balances)}

    def fetch_ticker(self, market_symbol):
        if self._ticker_error is not None:
            raise self._ticker_error
        base = market_symbol.split("/")[0]
        if base not in self._tickers:
            raise KeyError(base)
        return {"last": self._tickers[base]}


def _kraken(**kw):
    from quantbox.plugins.broker.kraken import KrakenBroker

    return KrakenBroker(quote_asset="USD", _exchange=_FakeCcxt(**kw))


class TestKrakenEquity:
    """Contract: ``KrakenBroker.get_equity`` docstring."""

    def test_declares_a_spot_venue(self):
        from quantbox.plugins.broker.kraken import KrakenBroker

        assert getattr(KrakenBroker, BASIS_ATTR) == BASIS_MARK

    def test_equity_is_cash_plus_every_liquidatable_position(self):
        b = _kraken(balances={"USD": 92.0, "BTC": 0.001}, tickers={"BTC": 32_000.0})
        assert b.get_equity() == pytest.approx(124.0)

    def test_an_empty_book_is_the_cash_balance_and_is_a_complete_answer(self):
        b = _kraken(balances={"USD": 92.0}, tickers={})
        assert b.get_equity() == pytest.approx(92.0)

    def test_an_unmarkable_holding_raises_rather_than_understating(self):
        """ "a valuation missing a position is not a smaller portfolio,
        it is an unknown one"."""
        b = _kraken(
            balances={"USD": 92.0, "BTC": 0.001, "PEPE": 1_000_000.0},
            tickers={"BTC": 32_000.0},
            markets={"BTC/USD": {}, "PEPE/USD": {}},
        )
        with pytest.raises(PortfolioValuationError) as exc:
            b.get_equity()
        assert exc.value.reason == REASON_UNPRICED
        assert "PEPE" in exc.value.unpriced

    def test_a_holding_with_no_market_at_all_raises(self):
        """No ccxt market => no mid => the book is unknown, not smaller."""
        b = _kraken(balances={"USD": 92.0, "WEIRD": 5.0}, tickers={}, markets={})
        with pytest.raises(PortfolioValuationError) as exc:
            b.get_equity()
        assert exc.value.reason == REASON_UNPRICED

    def test_a_failed_balance_fetch_is_unknown_not_zero(self):
        b = _kraken(balances={}, tickers={}, balance_error=RuntimeError("kraken 520"))
        with pytest.raises(PortfolioValuationError) as exc:
            b.get_equity()
        assert exc.value.reason == REASON_BROKER_EQUITY_FAILED
        assert "UNKNOWN" in str(exc.value)

    def test_the_non_strict_readers_still_degrade_to_empty(self):
        """``strict=False`` stays fail-soft: a transient venue error must not
        take down the reporting callers. Asserted so the strict flag is shown
        to be a NARROWING, not a behaviour change for everyone."""
        b = _kraken(balances={}, tickers={}, balance_error=RuntimeError("kraken 520"))
        assert b.get_cash() == {"USD": 0.0}
        assert b.get_positions().empty

    def test_a_ticker_failure_leaves_the_position_unmarkable(self):
        b = _kraken(
            balances={"USD": 92.0, "BTC": 0.001},
            tickers={"BTC": 32_000.0},
            ticker_error=RuntimeError("ticker 503"),
        )
        with pytest.raises(PortfolioValuationError) as exc:
            b.get_equity()
        assert exc.value.reason == REASON_UNPRICED

    def test_equity_reads_the_balance_sheet_once(self):
        """Two reads of a moving book can disagree; the mark must be one snapshot.

        NOTE: the code is the only statement of this intent -- the ``_balances``
        threading in ``get_cash``/``get_positions`` implies it but no docstring
        states it. Recorded as a finding, asserted because a second fetch would
        be a real consistency defect.
        """
        b = _kraken(balances={"USD": 92.0, "BTC": 0.001}, tickers={"BTC": 32_000.0})
        b.get_equity()
        assert b._exchange.balance_calls == 1

    def test_staking_balances_are_skipped_and_the_docstring_saying_otherwise_is_wrong(self):
        """FINDING (pre-existing doc defect, NOT introduced by this branch).

        ``_fetch_balances``' first line says "earn/staking folded into spot"
        and its body says "fold ``.S/.F/...`` earn balances into their spot
        asset" -- but the code ``continue``s past them, with an inline comment
        giving the opposite rationale ("folding them in would overstate the
        sellable position"). ``get_equity``'s own docstring agrees with the
        CODE. One file, two contradictory statements of the same rule.

        The behaviour is defensible (staked coin is not sellable today) and it
        understates rather than overstates, so it is safe for sizing. It is
        pinned here so neither side can drift silently; the docstring is the
        half that should change.
        """
        b = _kraken(balances={"USD": 92.0, "BTC": 0.0005, "BTC.S": 0.0005}, tickers={"BTC": 32_000.0})
        assert b.get_equity() == pytest.approx(92.0 + 0.0005 * 32_000.0)
        assert b.get_positions().set_index("symbol")["qty"].to_dict() == {"BTC": 0.0005}

    def test_known_gap_quote_pegged_stablecoin_dust_is_omitted(self):
        """DOCUMENTED, DELIBERATE GAP (``get_equity`` docstring).

        A USD-pegged stable is excluded from ``get_positions`` as dust and is
        not in ``get_cash`` either, so it is in NEITHER term. That UNDERSTATES
        the book -- safe for sizing, and identical in the pipeline's own mark so
        the two still reconcile. Pinned here so the omission cannot change
        silently in either direction.
        """
        b = _kraken(balances={"USD": 92.0, "USDT": 500.0}, tickers={})
        assert b.get_equity() == pytest.approx(92.0), "dust counted: the documented gap has moved"

    def test_a_non_quote_pegged_stable_is_a_real_position(self):
        """A EUR-pegged stable on a USD book is an FX position, not dust --
        so it must be marked, and refuses if it cannot be."""
        b = _kraken(balances={"USD": 92.0, "EURT": 10.0}, tickers={"EURT": 1.08})
        assert b.get_equity() == pytest.approx(92.0 + 10.8)


# ======================================================================
# 10. alloc2orders NAV helpers -- the number written to portfolio_daily
# ======================================================================


class TestAllocationNavHelpers:
    def test_duplicate_position_rows_are_summed_not_overwritten(self):
        """Two lots / two accounts for one symbol. ``dict(zip(...))`` kept the
        LAST row, where the ``value_usd`` sum it replaced added them."""
        pos = pd.DataFrame({"symbol": ["AAPL", "AAPL", "MSFT"], "qty": [10.0, 5.0, 2.0]})
        assert _summed_holdings(pos) == {"AAPL": 15.0, "MSFT": 2.0}

    def test_an_all_nan_quantity_stays_nan_rather_than_becoming_zero(self):
        """``min_count=1`` is load bearing: a bare ``.sum()`` returns 0.0 for an
        all-NaN group, turning an unreadable quantity into a position worth
        nothing and hiding it from the non-finite branch."""
        pos = pd.DataFrame({"symbol": ["X", "X"], "qty": [np.nan, np.nan]})
        out = _summed_holdings(pos)
        assert math.isnan(out["X"])
        v = value_holdings(cash=0.0, holdings=out, get_price=_price_fn({"X": 1.0}))
        assert v.unpriced == ("X",), "the NaN must survive to the refusal branch"

    def test_a_partially_nan_quantity_still_sums(self):
        pos = pd.DataFrame({"symbol": ["X", "X"], "qty": [np.nan, 4.0]})
        assert _summed_holdings(pos) == {"X": 4.0}

    def test_duplicate_marks_keep_the_last_because_a_price_does_not_accumulate(self):
        pos = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "price": [100.0, 101.0],
                "multiplier": [1.0, 1.0],
                "fx_to_usd": [1.0, 1.0],
            }
        )
        assert _usd_marks(pos) == {"AAPL": 101.0}

    def test_a_mark_applies_multiplier_and_fx(self):
        pos = pd.DataFrame({"symbol": ["FESX"], "price": [5000.0], "multiplier": [10.0], "fx_to_usd": [1.1]})
        assert _usd_marks(pos)["FESX"] == pytest.approx(55_000.0)

    def test_an_unpriced_held_symbol_marks_as_nan_not_zero(self):
        """``fillna(0.0)`` here is what made an unpriceable book look empty."""
        pos = pd.DataFrame({"symbol": ["X"], "price": [np.nan], "multiplier": [1.0], "fx_to_usd": [1.0]})
        assert math.isnan(_usd_marks(pos)["X"])
