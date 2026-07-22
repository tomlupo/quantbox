"""NAV must be the SUM of the spot and perps pots, not the larger one.

quantbox-live#93. A Hyperliquid account holds USDC in two separate places — the
perps side (`accountValue` = collateral + unrealized) and the spot side — and
moving between them takes an explicit transfer. `get_balance` previously returned
`max(perps, spot)`, discarding the smaller pot.

Measured live 2026-07-22: spot $85.3955 + perps $5.5140 = $90.9095 true NAV,
against $85.3955 reported — NAV understated by 6.25%.

This matters twice over: `get_equity()` feeds position sizing, and performance is
`ΔNAV - flows`, so an error that MOVES injects phantom PnL. The missing amount is
posted margin, which tracks gross exposure ($0.10–$8.20 over carver-HL's history)
— so every leverage change wrote a fake profit/loss into the equity curve.
"""

from __future__ import annotations

import pytest

from quantbox.exceptions import BrokerExecutionError
from quantbox.plugins.broker.hyperliquid import HyperliquidBroker

# The exact live reading that exposed the bug.
LIVE_SPOT = 85.3955
LIVE_PERPS = 5.5140
LIVE_TRUE_NAV = 90.9095


def _broker(perps: dict, spot: dict | None, spot_raises: bool = False):
    b = HyperliquidBroker.__new__(HyperliquidBroker)
    b.telegram_token = ""
    b.telegram_chat_id = ""

    class _Ex:
        def fetch_balance(self, params=None):
            if params and params.get("type") == "spot":
                if spot_raises:
                    raise RuntimeError("spot endpoint 503")
                return {"USDC": spot}
            return {"USDC": perps}

    b._exchange = _Ex()
    return b


def test_nav_is_spot_plus_perps_live_numbers():
    b = _broker(
        perps={"total": LIVE_PERPS, "free": 0.0, "used": LIVE_PERPS},
        spot={"total": LIVE_SPOT, "free": LIVE_SPOT, "used": 0.0},
    )
    assert b.get_balance()["total"] == pytest.approx(LIVE_TRUE_NAV, abs=1e-4)
    assert b.get_equity() == pytest.approx(LIVE_TRUE_NAV, abs=1e-4)


def test_perps_pot_is_not_discarded_when_spot_is_larger():
    """The regression: spot > perps used to mean 'return spot, forget perps'."""
    b = _broker(
        perps={"total": LIVE_PERPS, "free": 0.0, "used": LIVE_PERPS},
        spot={"total": LIVE_SPOT, "free": LIVE_SPOT, "used": 0.0},
    )
    total = b.get_balance()["total"]
    assert total != pytest.approx(LIVE_SPOT, abs=1e-4), "must not collapse to max(spot, perps)"
    assert total > LIVE_SPOT


def test_spot_pot_is_not_discarded_when_perps_is_larger():
    """Symmetric case — the other branch of the old max()."""
    b = _broker(
        perps={"total": 80.0, "free": 70.0, "used": 10.0},
        spot={"total": 12.0, "free": 12.0, "used": 0.0},
    )
    assert b.get_balance()["total"] == pytest.approx(92.0)


def test_free_and_used_are_summed_too():
    b = _broker(
        perps={"total": 10.0, "free": 4.0, "used": 6.0},
        spot={"total": 20.0, "free": 20.0, "used": 0.0},
    )
    bal = b.get_balance()
    assert (bal["total"], bal["free"], bal["used"]) == (30.0, 24.0, 6.0)


def test_unreadable_spot_pot_fails_closed():
    """Half a NAV is not a NAV. Sizing on the perps pot alone would massively
    understate capital, so an unread pot must abort, not degrade."""
    b = _broker(perps={"total": LIVE_PERPS, "free": 0.0, "used": 0.0}, spot=None, spot_raises=True)
    with pytest.raises(BrokerExecutionError, match="unknown NAV"):
        b.get_balance()
