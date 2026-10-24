"""Brokers DECLARE their capabilities; the trade path reads the declaration (TOM-1459).

Contract: ``BrokerCapability`` and ``BrokerPlugin.broker_capabilities`` in
``src/quantbox/contracts.py``, read by ``quantbox.trade.capabilities``.

Three things are pinned here:

1. Every broker's declaration matches its class, in both directions (closure:
   the registry AND the package's ``__all__``, as for ``valuation_basis``).
2. The paper brokers' declared capabilities do what they declare when
   exercised: injection lands, funding is booked, fees accrue.
3. A broker that declares nothing supports nothing, and the pipeline's
   readers take the "not supported" branch for it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.contracts import BROKER_CAPABILITY_MEMBERS, BrokerCapability
from quantbox.trade.capabilities import declared, mismatches, supports

C = BrokerCapability


def _all_brokers() -> dict[str, type]:
    import quantbox.plugins.broker as broker_pkg
    from quantbox.plugins.builtins import builtins

    registered = builtins()["broker"]
    assert len(registered) >= 9, f"read {len(registered)} brokers from the registry -- expected the full set"
    exported = {
        name: getattr(broker_pkg, name)
        for name in broker_pkg.__all__
        if getattr(broker_pkg, name, None) is not None and hasattr(getattr(broker_pkg, name), "meta")
    }
    assert set(exported) == set(broker_pkg.__all__), "an exported broker left this check's field of view"
    return {f"registry:{k}": v for k, v in registered.items()} | {f"export:{k}": v for k, v in exported.items()}


#: What each broker declares — the table a reviewer reads. A change to a
#: broker's capabilities is a change to this table, on purpose.
EXPECTED: dict[str, frozenset[BrokerCapability]] = {
    "sim.paper.v1": frozenset({C.PRICE_INJECTION, C.FEES_ACCRUED, C.COST_MODEL}),
    "sim.futures_paper.v1": frozenset(
        {
            C.PRICE_INJECTION,
            C.FUNDING_RATE_INJECTION,
            C.POSITION_LIMIT_INJECTION,
            C.FUNDING_APPLIED,
            C.FEES_ACCRUED,
            C.COST_MODEL,
        }
    ),
    "kraken.spot.v1": frozenset({C.EQUITY, C.ORDER_RESULT}),
    "hyperliquid.perps.v1": frozenset({C.EQUITY, C.FUNDING_HISTORY, C.ORDER_RESULT}),
    "binance.futures.v1": frozenset({C.ORDER_RESULT}),
    "binance.live.v1": frozenset(),
    "ibkr.live.v1": frozenset(),
    "ibkr.paper.stub.v1": frozenset(),
    "binance.paper.stub.v1": frozenset(),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_broker_declares_what_its_class_implements(name):
    from quantbox.plugins.builtins import builtins

    cls = builtins()["broker"][name]
    assert declared(cls) == EXPECTED[name]
    assert mismatches(cls) == []


def test_every_broker_in_this_repo_has_a_consistent_declaration():
    """Closure: a new broker, or a new method on an old one, cannot drift from its declaration."""
    found = {name: mismatches(cls) for name, cls in _all_brokers().items()}
    assert not {k: v for k, v in found.items() if v}


def test_every_registered_broker_is_in_the_expected_table():
    from quantbox.plugins.builtins import builtins

    assert set(builtins()["broker"]) == set(EXPECTED)


# --- the check can fail: it is not vacuous --------------------------------


class _Undeclared:
    def get_equity(self) -> float:
        return 1.0

    def fetch_order_result(self, order_id, symbol):
        return None


class _Overdeclared:
    broker_capabilities = frozenset({C.EQUITY, C.FUNDING_HISTORY})

    def get_equity(self) -> float:
        return 1.0


def test_mismatches_names_an_implemented_but_undeclared_capability():
    found = mismatches(_Undeclared)
    assert any("does not declare equity" in m for m in found)
    assert any("does not declare order_result" in m for m in found)


def test_mismatches_names_a_declared_but_missing_member():
    assert mismatches(_Overdeclared) == ["declares funding_history but has no fetch_funding_payments"]


def test_an_undeclared_broker_supports_nothing():
    """The default is 'not supported', whatever methods the class has."""
    assert declared(_Undeclared()) == frozenset()
    assert declared(None) == frozenset()
    assert not any(supports(_Undeclared(), cap) for cap in BrokerCapability)


def test_an_unparseable_declaration_is_ignored_loudly(caplog):
    class _Typo:
        broker_capabilities = frozenset({"equity", "equitty"})

    class _NotASet:
        broker_capabilities = "equity"

    with caplog.at_level("ERROR"):
        assert declared(_Typo()) == frozenset({C.EQUITY})
        assert declared(_NotASet()) == frozenset()
    assert "equitty" in caplog.text
    assert "not a set of BrokerCapability" in caplog.text


def test_a_forwarding_wrapper_carries_its_brokers_declaration():
    """quantbox-live's shadow run wraps the broker in a proxy that forwards attributes."""
    from quantbox.plugins.broker.kraken import KrakenBroker

    class _Proxy:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def __getattr__(self, name):
            return getattr(self._real, name)

    real = KrakenBroker.__new__(KrakenBroker)
    assert declared(_Proxy(real)) == EXPECTED["kraken.spot.v1"]


def test_every_capability_names_its_members():
    assert set(BROKER_CAPABILITY_MEMBERS) == set(BrokerCapability)


# --- the paper brokers do what they declare --------------------------------


def test_sim_paper_declared_capabilities_behave():
    from quantbox.plugins.broker.sim import SimPaperBroker

    b = SimPaperBroker(cash=10_000.0)
    b.set_prices({"BTC": 100.0})
    assert b.get_market_snapshot(["BTC"])["mid"].tolist() == [100.0]
    assert b.cumulative_fees == 0.0
    b.place_orders(pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0}]))
    assert b.cumulative_fees > 0.0
    assert b.cumulative_fees == b._cumulative_fees
    assert {k: float(getattr(b, k)) for k in BROKER_CAPABILITY_MEMBERS[C.COST_MODEL]} == {
        "spread_bps": 2.0,
        "slippage_bps": 5.0,
        "maker_fee_bps": 10.0,
        "taker_fee_bps": 10.0,
    }


def test_futures_paper_declared_capabilities_behave():
    from quantbox.plugins.broker.futures_paper import FuturesPaperBroker

    b = FuturesPaperBroker(margin_balance=10_000.0, default_funding_rate=0.0)
    b.set_prices({"BTC": 100.0})
    b.set_funding_rates({"BTC": 0.001})
    b.set_position_limits({"BTC": 5_000.0})
    assert b.funding_rates["BTC"] == 0.001
    assert b.position_limits["BTC"] == 5_000.0
    b.place_orders(pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0}]))
    assert b.cumulative_fees > 0.0
    charge = b.apply_funding()
    assert charge is not None and charge < 0.0  # a long pays positive funding


# --- the pipeline's readers ------------------------------------------------


def test_funding_follows_the_declaration_not_the_method():
    from quantbox.plugins.pipeline.trading_pipeline import _resolve_funding_charge

    class _HasMethodUndeclared:
        def fetch_funding_payments(self, since, until):  # pragma: no cover - must not be called
            raise AssertionError("an undeclared capability was used")

    class _Declared:
        broker_capabilities = frozenset({C.FUNDING_HISTORY})

        def fetch_funding_payments(self, since, until):
            return -1.25

    assert _resolve_funding_charge(_HasMethodUndeclared(), "2026-10-13") is None
    assert _resolve_funding_charge(_Declared(), "2026-10-13") == -1.25
