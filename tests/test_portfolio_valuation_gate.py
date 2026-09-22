"""The pre-trade portfolio valuation gate.

Context: the live ``crypto-trend-kraken`` book sized every target off CASH
instead of EQUITY since inception, and there were two reasons, one inside the
other.

The OUTER one is the design defect: the valuation rule came from which
rebalancer plugin the config named. ``configs/crypto_trend_kraken.yaml`` declared
``rebalancing.futures.v1`` against ``kraken.spot.v1`` -- a SPOT venue -- and
``FuturesRebalancer`` did exactly what its comment says, ``total_value = max(0,
cash_available)``, because for a perps book the margin balance genuinely is
equity. Correct code, wrong book, and nothing anywhere could tell the difference.
The rule is now derived from the BROKER (``valuation_basis``), and a live run
against a broker that does not declare one refuses.

The INNER one: four places computed ``cash + sum(qty * price)`` by hand and all
four SKIPPED any holding they could not price, so an unmarkable book and an empty
one produced the same number -- the cash balance -- and nothing in the metrics or
the logs said which had happened.

Every test here creates the condition it asserts on: the fixtures build their
own brokers, holdings and price maps rather than borrowing anything from the
machine or the network.

Mutation targets each test is meant to kill (restore the defect, watch it go
red) are named per test, along with WHICH BRANCH the assertion is reached by --
a red mutant proves nothing until you know it died on the intended assertion.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from quantbox.portfolio_value import (
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

# ---------------------------------------------------------------------------
# Fixtures -- each builds the whole condition it depends on
# ---------------------------------------------------------------------------

CASH = 214.33
HOLDINGS = {"BTC": 0.001, "ETH": 0.02, "SOL": 0.5}
PRICES = {"BTC": 60000.0, "ETH": 3000.0, "SOL": 140.0}
# 0.001*60000 + 0.02*3000 + 0.5*140 = 60 + 60 + 70 = 190.0
MARKED = 190.0


def _price_map(prices: dict[str, float]):
    return prices.get


class _SpotBroker:
    """A long-only spot broker whose equity is cash + marks, like Kraken."""

    valuation_basis = BASIS_MARK

    def __init__(self, equity: float | None = None, raises: Exception | None = None):
        self._equity = equity
        self._raises = raises

    def get_equity(self) -> float:
        if self._raises is not None:
            raise self._raises
        assert self._equity is not None
        return self._equity


class _PerpsBroker:
    """A derivatives broker: equity is margin + unrealised PnL, NOT cash + marks."""

    valuation_basis = BASIS_MARGIN

    def __init__(self, equity: float):
        self._equity = equity

    def get_equity(self) -> float:
        return self._equity


class _NoEquityBroker:
    """A spot broker with no equity surface at all -- the pre-fix Kraken shape."""

    valuation_basis = BASIS_MARK


class _UndeclaredBroker:
    """A broker that does not say what kind of venue it is.

    Deliberately carries a rich ``describe()`` AND a working ``get_equity()``: a
    valuation rule read off that free-form surface would have happily guessed
    "spot" here, which is the class of mistake that produced the incident, and
    the working equity surface is what makes the live refusal meaningful -- the
    run is stopped even though a number was available, because nobody knows what
    that number MEANS.
    """

    def describe(self) -> dict[str, object]:
        return {"name": "Undeclared", "type": "spot", "long_only": True}

    def get_equity(self) -> float:
        return 1_000_000.0


class _UndeclaredNoEquityBroker:
    """Undeclared AND no equity surface, so the two fallback bases really differ.

    With a ``get_equity()`` present both bases end at the broker's own number and
    the fallback is unobservable -- which is exactly how the first draft of
    ``test_a_simulation_falls_back_to_the_call_sites_historic_basis`` asserted
    something it could not see.
    """


class _MisdeclaredBroker:
    """A broker whose declaration cannot be parsed. Not a verdict, so: undeclared."""

    valuation_basis = "futures"  # not one of the two literals


class _MarginNoEquityBroker:
    """A perps venue with no get_equity -- the margin balance is all there is."""

    valuation_basis = BASIS_MARGIN


# ---------------------------------------------------------------------------
# value_holdings: the skip that started all of this
# ---------------------------------------------------------------------------


def test_unpriced_holding_is_collected_not_silently_skipped():
    """Mutation target: `unpriced.append(asset)` -> `continue`.

    Branch: SOL has no entry in the price map, so `get_price` returns None and
    the assertion is reached through the `price is None` arm of value_holdings.
    """
    partial = {"BTC": 60000.0, "ETH": 3000.0}  # SOL missing
    v = value_holdings(cash=CASH, holdings=HOLDINGS, get_price=_price_map(partial))

    assert v.unpriced == ("SOL",)
    assert not v.is_complete
    # The old code returned cash + 120.0 here and called it a portfolio value.
    assert v.n_holdings == 3


def test_empty_book_and_unpriceable_book_are_different_states():
    """THE defect, stated directly: these two used to be the same number.

    Mutation target: make `is_empty_book` return `not self.marked_value` (or
    make value_holdings skip unpriced names again) and this goes red.
    """
    empty = value_holdings(cash=CASH, holdings={}, get_price=_price_map({}))
    unpriceable = value_holdings(cash=CASH, holdings=HOLDINGS, get_price=_price_map({}))

    # Same value...
    assert empty.value == pytest.approx(unpriceable.value) == pytest.approx(CASH)
    # ...but they must NOT be the same state.
    assert empty.is_empty_book and empty.is_complete
    assert not unpriceable.is_empty_book and not unpriceable.is_complete
    assert empty.state == "empty_book"
    assert unpriceable.state == REASON_UNPRICED
    # The run record must carry the difference too, not just the object.
    assert empty.as_metrics()["portfolio_value_n_unpriced"] == 0.0
    assert unpriceable.as_metrics()["portfolio_value_n_unpriced"] == 3.0
    assert empty.as_metrics()["portfolio_valuation_complete"] == 1.0
    assert unpriceable.as_metrics()["portfolio_valuation_complete"] == 0.0


def test_fully_marked_book_values_cash_plus_positions():
    """Positive control: the gate must not be the only thing that works.

    Branch: every symbol resolves, so the assertion is reached through the
    successful `marked += qty * price` arm -- not through any refusal.
    """
    v = value_holdings(cash=CASH, holdings=HOLDINGS, get_price=_price_map(PRICES))

    assert v.is_complete
    assert v.marked_value == pytest.approx(MARKED)
    assert v.value == pytest.approx(CASH + MARKED)
    # And the whole point: equity exceeds cash by the value of what is held.
    assert v.value > CASH


def test_metrics_stay_numeric_per_the_runresult_contract():
    """`RunResult.metrics` is `dict[str, float | None]`.

    A string state in there would violate the declared contract and could trip
    artifact-schema validation at the far end of a live run -- so the label
    lives in notes and the metrics carry counts.
    """
    v = value_holdings(cash=CASH, holdings=HOLDINGS, get_price=_price_map({"BTC": 60000.0}))

    for key, value in v.as_metrics().items():
        assert value is None or isinstance(value, float), f"{key} is {type(value).__name__}"
    assert v.as_notes()["portfolio_valuation_state"] == REASON_UNPRICED


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), None])
def test_unusable_prices_are_unpriced_not_zero_valued(bad):
    """A NaN/zero/negative mid is an ABSENT price, not a position worth nothing.

    Mutation target: `_is_finite_positive` -> `lambda v: v is not None`.
    """
    v = value_holdings(cash=CASH, holdings={"BTC": 0.001}, get_price=lambda _s: bad)

    assert v.unpriced == ("BTC",)
    assert v.value == pytest.approx(CASH)


def test_stable_coin_counts_at_par_and_exclusions_are_ignored():
    """The quote currency sitting in positions is cash, not an unpriceable name."""
    v = value_holdings(
        cash=100.0,
        holdings={"USDC": 50.0, "BTC": 0.001, "JUNK": 5.0},
        get_price=_price_map({"BTC": 60000.0}),
        stable_coin="USDC",
        exclusions=("JUNK",),
    )

    assert v.is_complete, v.unpriced
    assert v.value == pytest.approx(100.0 + 50.0 + 60.0)
    assert "JUNK" not in v.unpriced and "JUNK" not in v.priced


# ---------------------------------------------------------------------------
# resolve_portfolio_value: live is a gate, paper is not
# ---------------------------------------------------------------------------


def test_live_refuses_when_even_one_position_cannot_be_marked():
    """Tom's acceptance bar: one unmarkable position fails the live run.

    Branch: reached through `gated and not valuation.is_complete`. The broker
    has no get_equity, so the broker-equity arm is not involved.
    """
    with pytest.raises(PortfolioValuationError) as exc:
        resolve_portfolio_value(
            broker=_NoEquityBroker(),
            mode="live",
            cash=CASH,
            holdings=HOLDINGS,
            get_price=_price_map({"BTC": 60000.0, "ETH": 3000.0}),
            fallback_basis=BASIS_MARK,
        )

    assert exc.value.reason == REASON_UNPRICED
    assert exc.value.unpriced == ("SOL",)


def test_live_allows_an_empty_book():
    """ "Nothing held" is a complete answer and must not be gated into uselessness.

    Mutation target: gate on `n_holdings == 0` as if it were a failure.
    """
    v = resolve_portfolio_value(
        broker=_NoEquityBroker(),
        mode="live",
        cash=CASH,
        holdings={},
        get_price=_price_map({}),
        fallback_basis=BASIS_MARK,
    )

    assert v.is_empty_book
    assert v.value == pytest.approx(CASH)


def test_live_allows_a_fully_marked_book():
    """Positive control on the live path: the gate passes real work through."""
    v = resolve_portfolio_value(
        broker=_NoEquityBroker(),
        mode="live",
        cash=CASH,
        holdings=HOLDINGS,
        get_price=_price_map(PRICES),
        fallback_basis=BASIS_MARK,
    )

    assert v.is_complete
    assert v.value == pytest.approx(CASH + MARKED)


@pytest.mark.parametrize("mode", ["paper", "backtest"])
def test_paper_and_backtest_are_not_gated(mode, caplog):
    """Stated explicitly because it is a deliberate asymmetry, not an oversight."""
    with caplog.at_level("WARNING"):
        v = resolve_portfolio_value(
            broker=_NoEquityBroker(),
            mode=mode,
            cash=CASH,
            holdings=HOLDINGS,
            get_price=_price_map({"BTC": 60000.0}),
            fallback_basis=BASIS_MARK,
        )

    assert not v.is_complete
    assert v.unpriced == ("ETH", "SOL")
    # Not silent, though: an understated value still has to reach the log.
    assert any("INCOMPLETE" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Reconciliation against the broker's own view
# ---------------------------------------------------------------------------


def test_live_refuses_when_broker_equity_disagrees_beyond_tolerance():
    """Reconciliation to the broker, per the acceptance bar.

    Branch: the book is fully marked, so the completeness arm passes and the
    assertion is reached through `_reconcile_or_raise` -- not through the
    unpriced refusal. That distinction is why this test asserts on `reason`.
    """
    computed = CASH + MARKED  # 404.33
    with pytest.raises(PortfolioValuationError) as exc:
        resolve_portfolio_value(
            broker=_SpotBroker(equity=computed * 1.10),
            mode="live",
            cash=CASH,
            holdings=HOLDINGS,
            get_price=_price_map(PRICES),
            fallback_basis=BASIS_MARK,
        )

    assert exc.value.reason == REASON_MISMATCH
    assert exc.value.computed == pytest.approx(computed)
    assert exc.value.broker_equity == pytest.approx(computed * 1.10)


def test_reconciliation_passes_within_tolerance_and_sizes_off_the_mark():
    """Positive control on the reconciliation path, and on WHICH number survives it.

    The broker's equity is a CROSS-CHECK, not the sizing number. It used to
    replace `value`, and that was wrong in the dangerous direction: a broker
    marks every balance the account holds and knows nothing about `exclusions`,
    so sizing off it targets capital that cannot be raised by trading -- see
    `test_an_excluded_holding_does_not_inflate_the_sizing_base`.

    Branch: fully marked and within tolerance, so this is the SUCCESS arm of
    `_reconcile_or_raise`; `reconciled` is what proves the check actually ran
    rather than being skipped.
    """
    computed = CASH + MARKED
    broker_equity = computed * (1 + DEFAULT_RECONCILIATION_TOLERANCE / 2)

    v = resolve_portfolio_value(
        broker=_SpotBroker(equity=broker_equity),
        mode="live",
        cash=CASH,
        holdings=HOLDINGS,
        get_price=_price_map(PRICES),
        fallback_basis=BASIS_MARK,
    )

    assert v.reconciled
    assert v.source == "computed_reconciled"
    assert v.broker_equity == pytest.approx(broker_equity)
    # The mark, NOT the broker's number, even though they agree here.
    assert v.value == pytest.approx(computed)


def test_tolerance_is_configurable_and_actually_applied():
    """Mutation target: hardcode the tolerance instead of reading the argument."""
    computed = CASH + MARKED
    broker_equity = computed * 1.02  # 2% apart: fails at 0.5%, passes at 5%

    with pytest.raises(PortfolioValuationError):
        resolve_portfolio_value(
            broker=_SpotBroker(equity=broker_equity),
            mode="live",
            cash=CASH,
            holdings=HOLDINGS,
            get_price=_price_map(PRICES),
            fallback_basis=BASIS_MARK,
            tolerance=0.005,
        )

    v = resolve_portfolio_value(
        broker=_SpotBroker(equity=broker_equity),
        mode="live",
        cash=CASH,
        holdings=HOLDINGS,
        get_price=_price_map(PRICES),
        fallback_basis=BASIS_MARK,
        tolerance=0.05,
    )
    # The looser tolerance lets the run PROCEED; it does not change the number
    # it proceeds with, which is always the mark.
    assert v.reconciled
    assert v.value == pytest.approx(computed)
    assert v.broker_equity == pytest.approx(broker_equity)


def test_derivatives_equity_is_used_without_a_false_reconciliation_alarm():
    """For perps, cash + marks is NOT supposed to equal equity.

    Reconciling them would fail on every live run, which is how a gate gets
    switched off. The broker's equity is authoritative and is used as-is.
    """
    v = resolve_portfolio_value(
        broker=_PerpsBroker(equity=1000.0),
        mode="live",
        cash=CASH,
        holdings=HOLDINGS,
        get_price=_price_map(PRICES),
        fallback_basis=BASIS_MARK,  # deliberately the WRONG fallback: the declaration must win
    )

    assert v.value == pytest.approx(1000.0)
    assert v.source == "broker_equity"
    assert not v.reconciled


def test_live_refuses_when_broker_equity_cannot_be_read():
    """A broker that cannot answer is not a broker reporting zero."""
    with pytest.raises(PortfolioValuationError) as exc:
        resolve_portfolio_value(
            broker=_SpotBroker(raises=RuntimeError("exchange unreachable")),
            mode="live",
            cash=CASH,
            holdings=HOLDINGS,
            get_price=_price_map(PRICES),
            fallback_basis=BASIS_MARK,
        )

    assert exc.value.reason == "broker_equity_failed"


def test_broker_refusal_propagates_unchanged():
    """A PortfolioValuationError from the broker keeps its own reason.

    Mutation target: catch it with the generic `except Exception` arm, which
    would relabel an `unpriced_holdings` refusal as `broker_equity_failed` and
    hide which name could not be marked.
    """
    inner = PortfolioValuationError("nope", reason=REASON_UNPRICED, unpriced=("SOL",))
    with pytest.raises(PortfolioValuationError) as exc:
        resolve_portfolio_value(
            broker=_SpotBroker(raises=inner),
            mode="live",
            cash=CASH,
            holdings=HOLDINGS,
            get_price=_price_map(PRICES),
            fallback_basis=BASIS_MARK,
        )

    assert exc.value is inner
    assert exc.value.unpriced == ("SOL",)


# ---------------------------------------------------------------------------
# KrakenBroker.get_equity -- the missing method that caused the incident
# ---------------------------------------------------------------------------


class _FakeKrakenExchange:
    """Minimal ccxt stand-in. Builds its own markets so nothing is borrowed."""

    def __init__(self, balances: dict[str, float], tickers: dict[str, float | None]):
        self._balances = balances
        self._tickers = tickers
        self.markets = {f"{a}/USD": {"spot": True, "base": a, "quote": "USD"} for a in tickers}

    def load_markets(self):
        return self.markets

    def fetch_balance(self):
        return {"total": dict(self._balances)}

    def fetch_ticker(self, market_symbol: str):
        base = market_symbol.split("/")[0]
        return {"last": self._tickers.get(base)}


def _kraken(balances: dict[str, float], tickers: dict[str, float | None]):
    from quantbox.plugins.broker.kraken import KrakenBroker

    return KrakenBroker(_exchange=_FakeKrakenExchange(balances, tickers))


def test_kraken_get_equity_is_cash_plus_marked_positions():
    """The method whose absence was the whole bug.

    Mutation target: `return cash_usd` before the marking loop -- i.e. restore
    the pre-fix behaviour -- and this goes red on the `> cash` assertion.
    """
    broker = _kraken(
        balances={"USD": 229.03, "BTC": 0.001, "ETH": 0.02},
        tickers={"BTC": 60000.0, "ETH": 3000.0},
    )

    equity = broker.get_equity()

    assert equity == pytest.approx(229.03 + 60.0 + 60.0)
    assert equity > 229.03  # the live report showed equity == cash, to the cent


def test_kraken_get_equity_raises_on_a_single_unmarkable_position():
    """One unpriceable name fails the valuation; it is not dropped.

    Branch: BTC prices fine, ETH's ticker returns None, so the assertion is
    reached through the `not valuation.is_complete` arm with a NON-empty
    priced list -- the partial case, which is the dangerous one.
    """
    broker = _kraken(
        balances={"USD": 229.03, "BTC": 0.001, "ETH": 0.02},
        tickers={"BTC": 60000.0, "ETH": None},
    )

    with pytest.raises(PortfolioValuationError) as exc:
        broker.get_equity()

    assert exc.value.reason == REASON_UNPRICED
    assert exc.value.unpriced == ("ETH",)


def test_kraken_get_equity_on_an_empty_book_returns_cash():
    """ "Nothing held" is a real answer and must not raise."""
    broker = _kraken(balances={"USD": 229.03}, tickers={})

    assert broker.get_equity() == pytest.approx(229.03)


def test_kraken_balance_fetch_failure_raises_instead_of_reporting_zero():
    """`_fetch_balances` is fail-soft by design; equity must not inherit that.

    Mutation target: drop `strict=True` in get_equity. The method then returns
    0.0 -- an account that the venue did not answer about, reported as empty.
    """

    class _Broken(_FakeKrakenExchange):
        def fetch_balance(self):
            raise RuntimeError("exchange unreachable")

    from quantbox.plugins.broker.kraken import KrakenBroker

    broker = KrakenBroker(_exchange=_Broken({"USD": 229.03}, {"BTC": 60000.0}))

    with pytest.raises(PortfolioValuationError) as exc:
        broker.get_equity()
    assert exc.value.reason == "broker_equity_failed"

    # Positive control: the fail-soft readers still degrade rather than raise,
    # so this change did not make a transient venue error fatal everywhere.
    assert broker.get_cash() == {"USD": 0.0}
    assert broker.get_positions().empty


def test_kraken_get_equity_is_discoverable_by_the_pipelines():
    """The pipelines reach get_equity via hasattr; a renamed method is invisible."""
    from quantbox.plugins.broker.kraken import KrakenBroker

    assert hasattr(KrakenBroker, "get_equity")


def test_kraken_declares_the_spot_valuation_basis():
    """The declaration that makes the live Kraken book value itself correctly.

    This is the whole design fix in one assertion. If it drifts, a live run does
    not silently revert to the margin rule -- it refuses -- but it does stop
    trading, so the declaration is worth pinning.

    Mutation target: delete the `valuation_basis` line on KrakenBroker; this test
    and `test_a_spot_venue_is_valued_off_its_holdings_whatever_the_rebalancer`
    both go red, the second one on the refusal.
    """
    from quantbox.plugins.broker.kraken import KrakenBroker

    assert KrakenBroker.valuation_basis == BASIS_MARK
    # Readable on the CLASS, with no instance and no API call -- the pipelines
    # must be able to ask before they have connected to anything.
    assert venue_valuation_basis(KrakenBroker) == BASIS_MARK

    broker = _kraken(balances={"USD": 1.0}, tickers={})
    assert venue_valuation_basis(broker) == BASIS_MARK
    # describe() still says spot, but it is no longer what the gate reads.
    assert broker.describe()["type"] == "spot"


# ---------------------------------------------------------------------------
# The sizing consequence, end to end through the rebalancer
# ---------------------------------------------------------------------------


class _SizingBroker:
    """A broker with real holdings, wired for the rebalancer's calls.

    Deliberately declares NO ``valuation_basis``: it stands for a third-party
    broker written before the declaration existed. The two subclasses below add
    one. Keeping the undeclared case as the BASE is what makes "this test needs a
    declared venue" a compile-time-visible choice rather than an inherited
    accident -- the first draft of this file had the base declare spot, which
    silently turned every "undeclared" test into a declared one.
    """

    def __init__(self, cash: float, holdings: dict[str, float], prices: dict[str, float]):
        self._cash = cash
        self._holdings = holdings
        self._prices = prices

    def get_cash(self):
        return {"USDC": self._cash}

    def get_positions(self):
        return pd.DataFrame([{"symbol": s, "qty": q} for s, q in self._holdings.items()])

    def get_market_snapshot(self, symbols):
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

    def get_equity(self) -> float:
        marked = sum(q * self._prices[s] for s, q in self._holdings.items())
        return self._cash + marked


class _SpotSizingBroker(_SizingBroker):
    """A spot venue -- what `kraken.spot.v1` is, and what prod actually had."""

    valuation_basis = BASIS_MARK


class _PerpsSizingBroker(_SizingBroker):
    """A perps venue, where the margin balance really is equity."""

    valuation_basis = BASIS_MARGIN


def test_rebalancer_sizes_off_equity_not_cash():
    """The live symptom, reproduced: a target sized off cash is ~35% too small.

    Branch: the book is fully marked and reconciles, so this is reached through
    the SUCCESS path of resolve_portfolio_value. The old code's `total_value`
    was `max(0, cash_available)` plus a loop that added zero.
    """
    from quantbox.plugins.rebalancing.standard_rebalancer import StandardRebalancer

    cash, prices = 214.33, {"BTC": 60000.0}
    holdings = {"BTC": 0.001}  # $60 held
    broker = _SpotSizingBroker(cash, holdings, prices)

    result = StandardRebalancer().generate_orders(
        weights={"BTC": 1.0},
        broker=broker,
        params={"mode": "live", "stable_coin_symbol": "USDC", "min_trade_size": 0.0},
    )

    equity = cash + 60.0
    assert result["total_value"] == pytest.approx(equity)
    assert result["total_value"] > cash
    # The understatement the book actually suffered, stated as a number.
    assert cash / equity == pytest.approx(0.7813, abs=1e-3)


def test_rebalancer_refuses_a_live_run_it_cannot_value():
    """The gate reaches the production sizing path, not just the helper.

    Branch: the snapshot returns a None mid for a HELD name, so this is reached
    through the unpriced arm inside the rebalancer's own call.
    """
    from quantbox.plugins.rebalancing.standard_rebalancer import StandardRebalancer

    broker = _SpotSizingBroker(214.33, {"BTC": 0.001}, {})  # no price for BTC
    broker.get_equity = lambda: 214.33  # broker itself reports cash only

    with pytest.raises(PortfolioValuationError) as exc:
        StandardRebalancer().generate_orders(
            weights={"BTC": 1.0},
            broker=broker,
            params={"mode": "live", "stable_coin_symbol": "USDC"},
        )

    assert exc.value.reason == REASON_UNPRICED
    assert "BTC" in exc.value.unpriced


def test_rebalancer_threads_mode_so_paper_still_runs():
    """Positive control on the degradation path: paper must not become fatal."""
    from quantbox.plugins.rebalancing.standard_rebalancer import StandardRebalancer

    broker = _SpotSizingBroker(214.33, {"BTC": 0.001}, {})
    broker.get_equity = lambda: 214.33

    result = StandardRebalancer().generate_orders(
        weights={"BTC": 1.0},
        broker=broker,
        params={"mode": "paper", "stable_coin_symbol": "USDC"},
    )

    assert math.isfinite(result["total_value"])


def test_an_omitted_mode_is_gated_not_assumed_to_be_paper():
    """The gate fails CLOSED on a caller that forgot to thread `mode`.

    An earlier draft defaulted to `"paper"`, which means any live caller that
    forgets one keyword silently loses the gate -- the same failure-open shape as
    the incident itself. Only a positively declared simulation is exempt.

    Branch: `params` has no `mode` key at all, so `is_gated("")` is True and the
    assertion is reached through the unpriced refusal, NOT through a mode string.
    """
    from quantbox.plugins.rebalancing.standard_rebalancer import StandardRebalancer

    broker = _SpotSizingBroker(214.33, {"BTC": 0.001}, {})
    broker.get_equity = lambda: 214.33

    with pytest.raises(PortfolioValuationError) as exc:
        StandardRebalancer().generate_orders(
            weights={"BTC": 1.0},
            broker=broker,
            params={"stable_coin_symbol": "USDC"},  # no mode
        )

    assert exc.value.reason == REASON_UNPRICED


@pytest.mark.parametrize("mode", ["live", "LIVE", "Live", "", None, "prod", "unknown"])
def test_only_a_declared_simulation_escapes_the_gate(mode):
    """Mutation target: `UNGATED_MODES` -> `mode == "live"` as a deny-list.

    A deny-list passes every one of these, which is how a typo or a missing
    keyword becomes an ungated live run.
    """
    assert is_gated(mode) is True


@pytest.mark.parametrize("mode", ["paper", "backtest", "PAPER", " paper "])
def test_declared_simulations_are_not_gated(mode):
    """Positive control on the other side of the same threshold."""
    assert is_gated(mode) is False


# ---------------------------------------------------------------------------
# The venue, not the config, decides HOW a book is valued
# ---------------------------------------------------------------------------


def test_the_basis_comes_from_the_broker_not_from_describe():
    """`describe()` is not the input, on purpose.

    `_UndeclaredBroker.describe()` says `type: "spot", long_only: True` -- the
    exact shape an earlier draft keyed the gate off. It is ignored: the venue
    either declares a basis or it does not.
    """
    assert venue_valuation_basis(_SpotBroker(equity=1.0)) == BASIS_MARK
    assert venue_valuation_basis(_PerpsBroker(equity=1.0)) == BASIS_MARGIN
    assert venue_valuation_basis(_UndeclaredBroker()) is None
    assert venue_valuation_basis(None) is None


def test_an_unparseable_declaration_is_undeclared_not_a_guess(caplog):
    """ "futures" is not one of the two literals, so it carries no information.

    Mutation target: accept any truthy string. The margin rule would then be
    applied to anything vaguely futures-sounding -- which is how the original
    defect reads in one sentence.
    """
    with caplog.at_level("ERROR"):
        assert venue_valuation_basis(_MisdeclaredBroker()) is None
    assert any("not one of" in r.getMessage() for r in caplog.records)


def test_live_refuses_a_venue_that_does_not_declare_its_basis():
    """The brief's rule: a plugin that cannot tell which it is FAILS on live.

    Branch: the refusal happens BEFORE any marking, so it fires even though this
    book is fully priceable and the broker answers get_equity() happily -- the
    point is that nobody knows what that answer MEANS.
    """
    with pytest.raises(PortfolioValuationError) as exc:
        resolve_portfolio_value(
            broker=_UndeclaredBroker(),
            mode="live",
            cash=CASH,
            holdings=HOLDINGS,
            get_price=_price_map(PRICES),
            fallback_basis=BASIS_MARK,
        )

    assert exc.value.reason == REASON_UNKNOWN_BASIS


def test_a_simulation_falls_back_to_the_call_sites_historic_basis(caplog):
    """Undeclared + not live: keep working, loudly. Third-party brokers exist.

    Branch: reached through the `basis is None and not gated` arm. The two
    fallbacks are asserted to give DIFFERENT numbers, so a mutant that ignores
    the argument cannot pass both halves.
    """
    with caplog.at_level("WARNING"):
        marked = resolve_portfolio_value(
            broker=_UndeclaredNoEquityBroker(),
            mode="paper",
            cash=CASH,
            holdings=HOLDINGS,
            get_price=_price_map(PRICES),
            fallback_basis=BASIS_MARK,
        )
    assert marked.basis == BASIS_MARK
    assert any("does not declare" in r.getMessage() for r in caplog.records)

    margined = resolve_portfolio_value(
        broker=_UndeclaredNoEquityBroker(),
        mode="paper",
        cash=CASH,
        holdings=HOLDINGS,
        get_price=_price_map(PRICES),
        fallback_basis=BASIS_MARGIN,
    )
    assert margined.basis == BASIS_MARGIN
    # Same inputs, same broker, different rule -> different portfolio value.
    # (This is the assertion the fixture has to EARN: with a get_equity present
    # both bases return the broker's number and the argument is unobservable.)
    assert marked.value == pytest.approx(CASH + MARKED)
    assert margined.value == pytest.approx(CASH)


def test_a_nonsense_fallback_basis_is_a_programming_error():
    """The fallback is a call-site declaration, so a typo must not resolve to anything."""
    with pytest.raises(ValueError, match="fallback_basis"):
        resolve_portfolio_value(
            broker=_SpotBroker(equity=1.0),
            mode="paper",
            cash=CASH,
            holdings={},
            get_price=_price_map({}),
            fallback_basis="spot",
        )


def test_a_margined_venue_keeps_the_margin_balance_when_it_has_no_equity_surface():
    """Today's behaviour for a perps book, preserved: value is the margin balance.

    Branch: BASIS_MARGIN with `broker_equity is None`, so the assertion is reached
    through the `margin_balance` source -- not through the broker-equity arm.
    """
    v = resolve_portfolio_value(
        broker=_MarginNoEquityBroker(),
        mode="live",
        cash=CASH,
        holdings=HOLDINGS,
        get_price=_price_map(PRICES),
        fallback_basis=BASIS_MARK,  # ignored: the venue declared
    )

    assert v.value == pytest.approx(CASH)
    assert v.source == "margin_balance"
    assert v.basis == BASIS_MARGIN
    # Positions were still marked for the record -- they just do not add to equity.
    assert v.marked_value == pytest.approx(MARKED)


def test_a_margined_venue_does_not_refuse_on_an_unmarkable_position(caplog):
    """The asymmetry that makes this venue-dependent rather than one global rule.

    On a margined book an unmarkable name loses its own target but does NOT
    understate equity, so a live run continues. Gating it would have been a guard
    that is too WIDE -- the mirror of the too-narrow one being fixed.

    Branch: BASIS_MARGIN with a non-empty `unpriced`, reached in live mode. The
    same inputs on BASIS_MARK raise, which the next assertion pins.
    """
    with caplog.at_level("WARNING"):
        v = resolve_portfolio_value(
            broker=_PerpsBroker(equity=1000.0),
            mode="live",
            cash=CASH,
            holdings=HOLDINGS,
            get_price=_price_map({"BTC": 60000.0}),
            fallback_basis=BASIS_MARGIN,
        )

    assert v.value == pytest.approx(1000.0)
    assert v.unpriced == ("ETH", "SOL")
    assert any("lose" in r.getMessage() for r in caplog.records)

    # Positive control that the two venues really do differ here: identical
    # inputs against a MARK venue refuse.
    with pytest.raises(PortfolioValuationError):
        resolve_portfolio_value(
            broker=_NoEquityBroker(),
            mode="live",
            cash=CASH,
            holdings=HOLDINGS,
            get_price=_price_map({"BTC": 60000.0}),
            fallback_basis=BASIS_MARGIN,
        )


# ---------------------------------------------------------------------------
# The live incident itself: a spot venue running the FUTURES rebalancer
# ---------------------------------------------------------------------------


def _futures_orders(broker, **params):
    from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer

    return FuturesRebalancer().generate_orders(
        weights={"BTC": 1.0},
        broker=broker,
        params={"stable_coin_symbol": "USDC", "min_trade_size": 0.0, **params},
    )


def test_futures_rebalancer_on_a_SPOT_venue_values_cash_plus_holdings():
    """THE live defect, reproduced at the site that was deliberately left alone.

    `configs/crypto_trend_kraken.yaml` named `rebalancing.futures.v1` while the
    broker was `kraken.spot.v1`. This plugin's `total_value = max(0, cash)` was
    right for perps and wrong for that book, and the run manifest recorded both
    plugin names happily.

    Mutation target: restore `total_value = max(0, cash_available)` in
    `futures_rebalancer._generate_orders`. Branch: reached through BASIS_MARK
    inside the FUTURES rebalancer, which is the combination that had no test.
    """
    cash, prices = 214.33, {"BTC": 60000.0}
    broker = _SpotSizingBroker(cash, {"BTC": 0.001}, prices)  # $60 held

    result = _futures_orders(broker, mode="live")

    equity = cash + 60.0
    assert result["total_value"] == pytest.approx(equity)
    assert result["valuation"].basis == BASIS_MARK
    # Sized off cash, every target was this fraction of what it should be.
    assert cash / equity == pytest.approx(0.7813, abs=1e-3)


def test_futures_rebalancer_on_a_PERPS_venue_is_unchanged():
    """The other half, and the one that must NOT move: perps sizing is untouched.

    Tom approved a ~28-35% growth in the KRAKEN targets. Nothing was approved for
    a perps book, so this asserts the number the plugin produced before the
    change: the margin balance, with the held position adding nothing.

    Branch: BASIS_MARGIN inside the futures rebalancer with a get_equity present,
    reached through the broker-equity arm. Hyperliquid's `get_cash()` and
    `get_equity()` return the same `balance["total"]`, so the two are equal there
    by construction -- which is why this is a no-op on live perps.
    """
    cash = 214.33
    broker = _PerpsSizingBroker(cash, {"BTC": 0.001}, {"BTC": 60000.0})
    broker.get_equity = lambda: cash  # as Hyperliquid does: total == get_cash

    result = _futures_orders(broker, mode="live")

    assert result["total_value"] == pytest.approx(cash)
    assert result["valuation"].basis == BASIS_MARGIN


def test_futures_rebalancer_refuses_an_undeclared_venue_on_live():
    """A third-party broker that says nothing does not get a guess on live."""
    broker = _SizingBroker(214.33, {"BTC": 0.001}, {"BTC": 60000.0})

    with pytest.raises(PortfolioValuationError) as exc:
        _futures_orders(broker, mode="live")

    assert exc.value.reason == REASON_UNKNOWN_BASIS


def test_futures_rebalancer_on_an_undeclared_venue_still_runs_in_paper():
    """Positive control on the degradation path at this call site."""
    cash = 214.33
    broker = _SizingBroker(cash, {"BTC": 0.001}, {"BTC": 60000.0})

    result = _futures_orders(broker, mode="paper")

    # The historic behaviour of THIS call site: the margin balance.
    assert result["total_value"] == pytest.approx(cash)
    assert result["valuation"].basis == BASIS_MARGIN


def test_every_broker_in_this_repo_declares_a_valuation_basis():
    """Closure check: a new broker cannot ship silently undeclared.

    A broker with no declaration is not a crash, it is a live run that refuses --
    which would be discovered in production. This is the check that finds it here
    instead.

    It reads TWO authorities, because neither alone is closed: the plugin
    registry (what a config can name) and the broker package's `__all__` (what a
    downstream repo can import). `BinanceLiveBroker` is in the second and not the
    first, so a registry-only check would have missed it.

    Mutation target: remove `valuation_basis` from any one broker; this names it.
    """
    import quantbox.plugins.broker as broker_pkg
    from quantbox.plugins.builtins import builtins

    registered = builtins()["broker"]
    assert len(registered) >= 9, f"read {len(registered)} brokers from the registry -- expected the full set"

    exported = {
        name: getattr(broker_pkg, name)
        for name in broker_pkg.__all__
        if getattr(broker_pkg, name, None) is not None and hasattr(getattr(broker_pkg, name), "meta")
    }
    assert "BinanceLiveBroker" in exported, "the exported-but-unregistered broker left this check's field of view"

    candidates = {f"registry:{k}": v for k, v in registered.items()} | {f"export:{k}": v for k, v in exported.items()}
    undeclared = sorted(name for name, cls in candidates.items() if venue_valuation_basis(cls) is None)
    assert not undeclared, f"brokers with no usable valuation_basis declaration: {undeclared}"


# ---------------------------------------------------------------------------
# The pipeline's own threading of `mode` to an injected rebalancer
# ---------------------------------------------------------------------------


def _rebal_params(mode, cfg_params=None):
    from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline

    return TradingPipeline()._rebalancer_params(
        rebalancer_cfg={"params": dict(cfg_params or {})},
        params={},
        strategy_results={},
        mode=mode,
    )


def test_the_pipeline_hands_the_rebalancer_the_real_run_mode():
    """Without this the rebalancer cannot tell a live book from a paper one.

    Branch: reached with NO `mode` in the rebalancer config, so the assertion is
    made on the assignment itself rather than on an override.

    Mutation target: drop the `rebal_params["mode"] = mode` line. The rebalancer
    would then see no mode, and `is_gated("")` would gate every PAPER run -- the
    safe direction, but still a break.
    """
    assert _rebal_params("live")["mode"] == "live"
    assert _rebal_params("paper")["mode"] == "paper"


def test_a_config_supplied_mode_cannot_outrank_the_real_run_mode():
    """A config saying `mode: paper` must not ungate a LIVE book.

    This is the incident's own shape -- a declaration honoured from the wrong
    source -- pointed at the gate built to stop it, so this is the one key here
    that must not be `setdefault`.

    Branch: the config DOES carry a conflicting `mode`, so this reaches the
    assignment through the collision path, which the test above cannot.

    Mutation target: `rebal_params["mode"] = mode` -> `.setdefault(...)`.
    """
    resolved = _rebal_params("live", {"mode": "paper"})

    assert resolved["mode"] == "live"
    assert is_gated(resolved["mode"]) is True


def test_an_excluded_holding_does_not_inflate_the_sizing_base():
    """The review BLOCKER: a broker's equity includes what the caller excluded.

    `trading_pipeline` builds `exclusions = params.exclusions + [stable_coin]`,
    so it is never empty, and `KrakenBroker.get_equity()` marks every balance
    `get_positions` returns -- it knows nothing about exclusions. Replacing the
    value with the broker's number therefore sized targets off capital that
    cannot be raised by trading the book.

    Branch: reached through `_resolve_marked` with an excluded holding present,
    so reconciliation is SKIPPED and the assertion lands on the value itself
    rather than on a reconciliation outcome.

    The price map deliberately carries NO price for the excluded name: every
    caller that passes `exclusions` also strips those symbols from the market
    snapshot it requests (`trading_pipeline.py:1492` and the two rebalancers),
    so in production no price for them exists. A fixture that supplied one would
    be lending the test a fact the real process cannot.

    Mutation target: `value=broker_equity` in `_resolve_marked`.
    """
    # The broker marks BOTH: 50 cash + 100 BTC + 1000 FROZEN.
    broker_equity = 50.0 + 100.0 + 1000.0

    v = resolve_portfolio_value(
        broker=_SpotBroker(equity=broker_equity),
        mode="live",
        cash=50.0,
        holdings={"BTC": 1.0, "FROZEN": 100.0},
        get_price=_price_map({"BTC": 100.0}),
        fallback_basis=BASIS_MARK,
        exclusions=["FROZEN"],
    )

    # Sizing base is the TRADABLE book only: 50 cash + 100 BTC. Sizing off the
    # broker's 1150 would target capital no trade in this book can raise.
    assert v.value == pytest.approx(150.0)
    assert v.broker_equity == pytest.approx(broker_equity)
    assert "FROZEN" not in v.unpriced  # excluded names are never a refusal


def test_an_excluded_holding_skips_reconciliation_instead_of_refusing():
    """The two views stop measuring the same book, so comparing them is a false alarm.

    Branch: `has_excluded_holdings` is True, so `_reconcile_or_raise` is never
    reached -- which is why this asserts `reconciled is False` rather than just
    "no exception". Without the flag the two sides differ by the whole excluded
    position (here 1000 vs a 0.5% tolerance) and every live run would refuse.

    Mutation target: `if valuation.has_excluded_holdings:` -> `if False:`.
    """
    v = resolve_portfolio_value(
        broker=_SpotBroker(equity=1150.0),
        mode="live",
        cash=50.0,
        holdings={"BTC": 1.0, "FROZEN": 100.0},
        get_price=_price_map({"BTC": 100.0}),
        fallback_basis=BASIS_MARK,
        exclusions=["FROZEN"],
    )

    assert v.has_excluded_holdings is True
    assert v.reconciled is False
    assert v.value == pytest.approx(150.0)


def test_a_book_with_no_exclusions_still_actually_reconciles():
    """Positive control: the skip above must not quietly disable the gate for everyone.

    Branch: no excluded holding, so this reaches `_reconcile_or_raise` and
    `reconciled` proves it ran. Without this, deleting the whole reconciliation
    call would still leave the test above green.
    """
    v = resolve_portfolio_value(
        broker=_SpotBroker(equity=150.0),
        mode="live",
        cash=50.0,
        holdings={"BTC": 1.0},
        get_price=_price_map({"BTC": 100.0}),
        fallback_basis=BASIS_MARK,
        exclusions=["FROZEN"],  # declared, but NOT held
    )

    assert v.has_excluded_holdings is False
    assert v.reconciled is True


def test_a_nan_broker_equity_is_a_refusal_not_a_nan_target():
    """`nan <= 0` is False, so every downstream zero-guard would wave it through.

    Branch: reached through the `except Exception` arm after the finiteness
    check, so the reason is BROKER_EQUITY_FAILED rather than a mismatch.
    """
    with pytest.raises(PortfolioValuationError) as exc:
        resolve_portfolio_value(
            broker=_PerpsBroker(equity=float("nan")),
            mode="live",
            cash=CASH,
            holdings=HOLDINGS,
            get_price=_price_map(PRICES),
            fallback_basis=BASIS_MARGIN,
        )

    assert exc.value.reason == REASON_BROKER_EQUITY_FAILED


def test_a_debit_cash_balance_reduces_a_marked_book_but_not_a_margin_balance():
    """`max(0, cash)` is the MARGIN rule and must not silently pad a spot book.

    A margin debit is real money owed: on a mark-to-market venue it reduces the
    book, and clamping it to zero overstates equity by the whole debt -- the
    too-large direction. On the margin path the clamp is the historic behaviour
    and is kept.

    Mutation target: `cash_value = max(0.0, float(cash))` in `value_holdings`.
    """
    marked = value_holdings(
        cash=-40.0,
        holdings={"BTC": 1.0},
        get_price=_price_map({"BTC": 100.0}),
    )
    assert marked.value == pytest.approx(60.0)

    margined = resolve_portfolio_value(
        broker=_MarginNoEquityBroker(),
        mode="live",
        cash=-40.0,
        holdings={"BTC": 1.0},
        get_price=_price_map({"BTC": 100.0}),
        fallback_basis=BASIS_MARGIN,
    )
    assert margined.value == pytest.approx(0.0)


def test_a_symbol_held_across_two_rows_is_summed_not_overwritten():
    """`dict(zip(...))` keeps the LAST row; the `value_usd.sum()` it replaced added them.

    A position reported across two rows (two accounts, two lots) would otherwise
    lose all but one -- an understatement, which is the same defect class this
    module exists to close.

    Mutation target: `.sum()` -> `.last()` in `_summed_holdings`.
    """
    from quantbox.plugins.pipeline.alloc2orders import _summed_holdings, _usd_marks

    pos = pd.DataFrame(
        {
            "symbol": ["BTC", "BTC", "ETH"],
            "qty": [1.5, 2.5, 10.0],
            "price": [100.0, 100.0, 5.0],
            "multiplier": [1.0, 1.0, 1.0],
            "fx_to_usd": [1.0, 1.0, 1.0],
        }
    )

    assert _summed_holdings(pos) == {"BTC": 4.0, "ETH": 10.0}
    # A per-unit PRICE, unlike a quantity, does not accumulate.
    assert _usd_marks(pos)["BTC"] == pytest.approx(100.0)


def test_an_unreadable_quantity_survives_summing():
    """A bare `.sum()` returns 0.0 for an all-NaN group — a position worth nothing.

    That would hide the holding from `value_holdings`' non-finite branch, so a
    quantity nobody could read would be silently valued at zero instead of
    refusing a live run. This is the understatement the whole change exists to
    stop, re-entering through the aggregation added to fix a different one.

    Mutation target: drop `min_count=1` in `_summed_holdings`.
    """
    from quantbox.plugins.pipeline.alloc2orders import _summed_holdings

    pos = pd.DataFrame({"symbol": ["GHOST", "GHOST"], "qty": [float("nan"), float("nan")]})

    assert math.isnan(_summed_holdings(pos)["GHOST"])

    # ...and that NaN must still read as unmarkable, not as an empty book.
    v = value_holdings(cash=100.0, holdings=_summed_holdings(pos), get_price=_price_map({}))
    assert v.unpriced == ("GHOST",)
    assert v.state == REASON_UNPRICED


def test_a_nan_equity_does_not_leak_into_an_ungated_run():
    """The finiteness guard must not protect live while handing paper a NaN.

    Assigning `broker_equity` before raising leaves the NaN in place for the
    ungated arm, which only warns — so the value would carry NaN into
    `_resolve_margined` and out, invisible to every `total_value <= 0` guard.

    Branch: `mode="paper"`, so the refusal is NOT taken and the assertion lands
    on the value that a degraded run actually proceeds with.

    Mutation target: assign `broker_equity` first, validate second.
    """
    v = resolve_portfolio_value(
        broker=_PerpsBroker(equity=float("nan")),
        mode="paper",
        cash=CASH,
        holdings={},
        get_price=_price_map({}),
        fallback_basis=BASIS_MARGIN,
    )

    assert v.broker_equity is None, "a NaN equity must never be published"
    assert math.isfinite(v.value)
    assert v.value == pytest.approx(CASH)


def test_the_other_rebalancer_params_stay_config_overridable():
    """Positive control: `mode` is the exception, not a new blanket rule.

    If every key became an assignment this goes red, which is what stops the
    test above from being read as "the pipeline overrides everything".
    """
    resolved = _rebal_params("paper", {"capital_at_risk": 0.25, "stable_coin_symbol": "USDT"})

    assert resolved["capital_at_risk"] == 0.25
    assert resolved["stable_coin_symbol"] == "USDT"
