"""Real-broker coverage for HyperliquidBroker.get_positions() (issue #93).

The FLATTEN reduce-only gate (`TradingPipeline._filter_reduce_only`) reads
`broker.get_positions()` and relies on the returned `DataFrame[symbol, qty]`
with **signed** qty (perp shorts negative). Before enforce is armed on a live
book we lock down that schema + sign convention here.

ccxt is never touched: we build the broker via ``__new__`` and inject a fake
exchange exposing only ``fetch_positions``. No credentials, no network, no
capital.
"""

from __future__ import annotations

from quantbox.plugins.broker.hyperliquid import HyperliquidBroker

_EXPECTED_COLS = ["symbol", "qty", "notional", "entry_price", "unrealized_pnl"]


class _FakeExchange:
    """Minimal ccxt.hyperliquid stand-in — only ``fetch_positions``."""

    def __init__(self, positions=None, raises=False):
        self._positions = positions or []
        self._raises = raises

    def fetch_positions(self):
        if self._raises:
            raise RuntimeError("simulated hyperliquid API error")
        return self._positions


def _pos(symbol, side, contracts, notional=1000.0, entry=100.0, upnl=0.0):
    """A ccxt unified position row."""
    return {
        "symbol": symbol,
        "side": side,
        "contracts": contracts,
        "notional": notional,
        "entryPrice": entry,
        "unrealizedPnl": upnl,
    }


def _broker(positions=None, raises=False, base_to_coin=None):
    """Build a HyperliquidBroker without touching ccxt / __post_init__."""
    b = object.__new__(HyperliquidBroker)
    b._exchange = _FakeExchange(positions, raises=raises)
    b._base_to_coin = base_to_coin or {}
    return b


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_get_positions_schema_columns():
    b = _broker([_pos("BTC/USDC:USDC", "long", 2.0)])
    pos = b.get_positions()
    assert list(pos.columns) == _EXPECTED_COLS
    assert "symbol" in pos.columns and "qty" in pos.columns


def test_get_positions_empty_has_expected_schema():
    b = _broker([])
    pos = b.get_positions()
    assert pos.empty
    assert list(pos.columns) == _EXPECTED_COLS


# ---------------------------------------------------------------------------
# Signed-qty semantics (the load-bearing property for the FLATTEN clamp)
# ---------------------------------------------------------------------------


def test_long_position_is_positive_qty():
    b = _broker([_pos("BTC/USDC:USDC", "long", 2.5)])
    pos = b.get_positions()
    row = pos.set_index("symbol").loc["BTC"]
    assert row["qty"] == 2.5


def test_short_position_is_negative_qty():
    # A perp SHORT must surface as a NEGATIVE qty so the reduce-only clamp can
    # tell that a BUY (positive delta) reduces it. This is the exact futures-short
    # semantics the #92 review flagged as untested.
    b = _broker([_pos("ETH/USDC:USDC", "short", 3.0)])
    pos = b.get_positions()
    row = pos.set_index("symbol").loc["ETH"]
    assert row["qty"] == -3.0


def test_mixed_long_and_short_book():
    b = _broker(
        [
            _pos("BTC/USDC:USDC", "long", 1.0),
            _pos("ETH/USDC:USDC", "short", 4.0),
        ]
    )
    pos = b.get_positions().set_index("symbol")
    assert pos.loc["BTC", "qty"] == 1.0
    assert pos.loc["ETH", "qty"] == -4.0


# ---------------------------------------------------------------------------
# Dust / canonical-coin / fail-safe behaviour
# ---------------------------------------------------------------------------


def test_dust_position_below_epsilon_excluded():
    b = _broker(
        [
            _pos("BTC/USDC:USDC", "long", 1e-9),  # dust, dropped
            _pos("ETH/USDC:USDC", "short", 2.0),
        ]
    )
    pos = b.get_positions()
    assert pos["symbol"].tolist() == ["ETH"]


def test_canonical_coin_name_remapped():
    # ccxt uppercases the base ("KPEPE"); get_positions must map it back to the
    # canonical Hyperliquid coin ("kPEPE") so it matches strategy targets.
    b = _broker(
        [_pos("KPEPE/USDC:USDC", "long", 100.0)],
        base_to_coin={"KPEPE": "kPEPE"},
    )
    pos = b.get_positions()
    assert pos["symbol"].tolist() == ["kPEPE"]


def test_fetch_error_returns_empty_frame_fail_safe():
    # _get_positions_dict swallows API errors and returns {} → get_positions must
    # yield an empty (not malformed) frame. The FLATTEN gate then fails SAFE
    # (drops all orders) rather than acting on phantom positions.
    b = _broker(raises=True)
    pos = b.get_positions()
    assert pos.empty
    assert list(pos.columns) == _EXPECTED_COLS
