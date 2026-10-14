"""The marked book (TOM-1455): one module assembles positions, cash and marks, and values them.

Two halves:

1. The book's own interface (``read_book``, ``MarkedBook``): lots summed, the
   snapshot request, the cash rule, and a missing mark is UNKNOWN, never zero.
2. The valuation-gate cases extended past pre-trade, as the card's AC asks: the
   post-trade NAV, the reconciliation drift and the rebalancing frame read
   their values from the book, so a held name with no mark is reported as
   unknown at each of them instead of being valued at 0.

Every fixture builds the condition it asserts on; nothing is read from the
machine or the network.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantbox.contracts import BrokerCapability
from quantbox.plugins.broker.sim import SimPaperBroker
from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer
from quantbox.plugins.rebalancing.standard_rebalancer import StandardRebalancer
from quantbox.store import FileArtifactStore
from quantbox.trade.book import BASIS_MARGIN, BASIS_MARK, MarkedBook, PortfolioValuationError, read_book


class _Broker:
    """Positions, balances and a snapshot, each scripted; every call is counted."""

    def __init__(
        self,
        positions: list[dict[str, Any]],
        marks: dict[str, Any],
        cash: dict[str, float] | None = None,
        snapshot_raises: Exception | None = None,
    ):
        self._positions = positions
        self._marks = marks
        self._cash = cash if cash is not None else {"USDC": 1000.0}
        self._snapshot_raises = snapshot_raises
        self.requests: list[list[str]] = []
        self.calls: dict[str, int] = {}

    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def get_positions(self) -> pd.DataFrame:
        self._count("get_positions")
        return pd.DataFrame(self._positions, columns=["symbol", "qty"])

    def get_cash(self) -> dict[str, float]:
        self._count("get_cash")
        return dict(self._cash)

    def get_market_snapshot(self, symbols: list[str]) -> pd.DataFrame:
        self._count("get_market_snapshot")
        self.requests.append(list(symbols))
        if self._snapshot_raises is not None:
            raise self._snapshot_raises
        rows = [
            {"symbol": s, "mid": self._marks[s], "min_qty": 0.0, "min_notional": 0.0}
            for s in symbols
            if s in self._marks
        ]
        return pd.DataFrame(rows, columns=["symbol", "mid", "min_qty", "min_notional"])


# ---------------------------------------------------------------------------
# 1. The book's interface
# ---------------------------------------------------------------------------


def test_a_position_is_qty_times_mark_and_a_missing_mark_is_unknown_not_zero():
    book = MarkedBook.of(positions={"BTC": 0.5, "GHOST": 3.0, "FLAT": 0.0}, marks={"BTC": 100.0})
    assert book.value_of("BTC") == pytest.approx(50.0)
    assert book.value_of("GHOST") is None  # held, no mark: UNKNOWN
    assert book.value_of("FLAT") == 0.0  # flat: known zero whatever its mark
    assert book.value_of("NEVER_HELD") == 0.0
    assert book.value_of("BTC", 2.0) == pytest.approx(200.0)  # a given quantity at the book's mark


def test_an_unreadable_quantity_is_unknown_not_zero():
    book = MarkedBook.of(positions={"X": float("nan")}, cash=0.0, marks={"X": 1.0})
    assert book.value_of("X") is None
    assert book.marked().unpriced == ("X",)


def test_a_book_read_without_cash_cannot_be_valued():
    """Unread cash is not zero cash: `marked()` and `valuation()` refuse rather than value it at 0."""
    book = MarkedBook.of(positions={"X": 1.0}, marks={"X": 1.0})
    with pytest.raises(ValueError, match="without cash"):
        book.marked()
    with pytest.raises(ValueError, match="without cash"):
        book.valuation(broker=None, mode="paper", fallback_basis=BASIS_MARK)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), None, "n/a"])
def test_a_mark_that_is_not_finite_and_positive_is_no_mark(bad):
    book = MarkedBook.of(positions={"X": 1.0}, marks={"X": bad})
    assert book.mark("X") is None
    assert book.value_of("X") is None


def test_read_book_sums_lots_and_requests_targets_and_held_names_minus_exclusions():
    broker = _Broker(
        positions=[
            {"symbol": "BTC", "qty": 0.2},
            {"symbol": "BTC", "qty": 0.3},
            {"symbol": "FLAT", "qty": 0.0},
            {"symbol": "USDC", "qty": 5.0},
        ],
        marks={"BTC": 100.0, "ETH": 10.0},
    )
    book = read_book(broker, wanted=["ETH", "BTC"], exclude=["USDC"], cash=("USDC", "USD"))
    assert book.positions["BTC"] == pytest.approx(0.5)
    assert broker.requests == [["BTC", "ETH"]]  # sorted; the flat and the excluded are not requested
    assert book.cash == pytest.approx(1000.0)
    assert book.value_of("BTC") == pytest.approx(50.0)
    assert broker.calls == {"get_positions": 1, "get_cash": 1, "get_market_snapshot": 1}


def test_read_book_cash_rules():
    broker = _Broker(positions=[], marks={}, cash={"USDT": 3.0, "USD": 7.0})
    assert read_book(broker, cash=("USDC", "USD", "USDT")).cash == pytest.approx(7.0)  # first key present
    assert read_book(broker, cash=("USDC",)).cash == 0.0  # none present
    calls_before = broker.calls.get("get_cash", 0)
    assert read_book(broker).cash is None  # not read at all
    assert broker.calls.get("get_cash", 0) == calls_before


def test_read_book_requests_nothing_for_an_empty_book():
    broker = _Broker(positions=[], marks={})
    read_book(broker)
    assert "get_market_snapshot" not in broker.calls


def test_a_failed_snapshot_leaves_every_held_name_unmarked_and_says_so(caplog):
    broker = _Broker(positions=[{"symbol": "BTC", "qty": 1.0}], marks={}, snapshot_raises=RuntimeError("venue down"))
    with caplog.at_level("ERROR"):
        book = read_book(broker, cash=("USDC",))
    assert book.marks == {}
    assert book.marked().unpriced == ("BTC",)
    assert "Market snapshot failed" in caplog.text


def test_the_first_usable_mark_wins_on_a_duplicated_snapshot_row():
    class _Dup(_Broker):
        def get_market_snapshot(self, symbols):
            return pd.DataFrame({"symbol": ["BTC", "BTC", "BTC"], "mid": [float("nan"), 101.0, 102.0]})

    assert read_book(_Dup(positions=[{"symbol": "BTC", "qty": 1.0}], marks={})).mark("BTC") == 101.0


@pytest.mark.parametrize(
    "rows",
    [[{"asset": "BTC", "qty": 10.0}], [{"symbol": "BTC", "size": 10.0}]],
    ids=["no-symbol-column", "no-qty-column"],
)
def test_an_unreadable_positions_frame_is_unknown_holdings_not_an_empty_book(rows):
    """Replay of independent-review round 1, PR #284 (FINDING 101): a frame with rows but
    no `symbol` column was valued as an EMPTY book, so a live spot run sized off cash
    alone and ignored 10 BTC. The base refused. Unread is not empty."""

    class _OddBroker(_SpotBroker):
        def get_positions(self):
            return pd.DataFrame(rows)

    broker = _OddBroker(positions=[], marks={"BTC": 100.0}, cash={"USDC": 1000.0})
    with pytest.raises(PortfolioValuationError) as exc:
        StandardRebalancer().generate_orders(
            weights={"BTC": 0.5}, broker=broker, params={"mode": "live", "stable_coin_symbol": "USDC"}
        )
    assert exc.value.reason == "unpriced_holdings"


def test_a_position_with_one_unreadable_lot_refuses_a_live_run():
    """Replay of the independent review of promotion PR #285 (v0.16.0): BTC on two lots,
    [1.0, NaN], was summed to a known 1.0, so a live spot run sized where 0.15.0 refused."""

    class _Lots(_SpotBroker):
        def get_positions(self):
            return pd.DataFrame({"symbol": ["BTC", "BTC"], "qty": [1.0, float("nan")]})

    broker = _Lots(positions=[], marks={"BTC": 100.0}, cash={"USDC": 1000.0})
    with pytest.raises(PortfolioValuationError) as exc:
        StandardRebalancer().generate_orders(
            weights={"BTC": 0.5}, broker=broker, params={"mode": "live", "stable_coin_symbol": "USDC"}
        )
    assert exc.value.unpriced == ("BTC",)


def test_positions_keep_the_venue_row_order():
    """Float addition is order-sensitive: the book is marked in the order the sites marked it."""
    book = MarkedBook.of(positions=pd.DataFrame({"symbol": ["SOL", "BTC", "SOL"], "qty": [1.0, 2.0, 3.0]}))
    assert list(book.positions) == ["SOL", "BTC"]


# ---------------------------------------------------------------------------
# 2a. The rebalancing frame: an unpriced held name's value is NaN, never 0
# ---------------------------------------------------------------------------


class _MarginBroker(_Broker):
    """A perps venue: an unmarkable name does not understate equity, so pre-trade does not refuse."""

    valuation_basis = BASIS_MARGIN
    broker_capabilities = frozenset({BrokerCapability.EQUITY})  # TOM-1459: declared, not probed

    def get_equity(self) -> float:
        return 1000.0


class _SpotBroker(_Broker):
    valuation_basis = BASIS_MARK


def test_futures_rebalancer_frame_reports_an_unpriced_held_name_as_unknown():
    broker = _MarginBroker(
        positions=[{"symbol": "BTC", "qty": 1.0}, {"symbol": "GHOST", "qty": 4.0}],
        marks={"BTC": 100.0},
    )
    out = FuturesRebalancer().generate_orders(
        weights={"BTC": 0.5}, broker=broker, params={"mode": "live", "stable_coin_symbol": "USDC"}
    )
    frame = out["rebalancing"].set_index("Asset")
    assert math.isnan(frame.loc["GHOST", "Current Value"])  # was 0: "missing mark = 0"
    assert math.isnan(frame.loc["GHOST", "Current Weight"])
    assert frame.loc["BTC", "Current Value"] == pytest.approx(100.0)
    orders = out["orders"].set_index("Asset")
    # The frame's Price column holds NaN for it (pandas turns None into NaN beside
    # a float), so the order is "Invalid (NaN)" -- unchanged by this card, never sent.
    assert orders.loc["GHOST", "Order Status"] == "Invalid (NaN)"
    assert not bool(orders.loc["GHOST", "Executable"])


def test_standard_rebalancer_frame_reports_an_unpriced_held_name_as_unknown():
    broker = _SpotBroker(
        positions=[{"symbol": "BTC", "qty": 1.0}, {"symbol": "GHOST", "qty": 4.0}],
        marks={"BTC": 100.0},
    )
    out = StandardRebalancer().generate_orders(
        weights={"BTC": 0.5}, broker=broker, params={"mode": "paper", "stable_coin_symbol": "USDC"}
    )
    frame = out["rebalancing"].set_index("Asset")
    assert math.isnan(frame.loc["GHOST", "Current Value"])
    assert frame.loc["BTC", "Current Value"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 2b. Post-trade NAV and reconciliation drift, through a pipeline run
# ---------------------------------------------------------------------------

ASOF = "2026-09-22"


class _Store:
    def __init__(self):
        self.run_id = "testrun"
        self.written: dict[str, object] = {}

    def put_parquet(self, name, df):
        self.written[name] = df
        return f"artifacts/{name}.parquet"

    def put_json(self, name, obj):
        self.written[name] = obj
        return f"artifacts/{name}.json"


class _Data:
    def load_universe(self, params):
        return pd.DataFrame({"symbol": ["BTC"]})

    def load_market_data(self, universe, asof, params):
        idx = pd.to_datetime([ASOF])
        return {
            "prices": pd.DataFrame({"BTC": [100.0]}, index=idx),
            "volume": pd.DataFrame({"BTC": [5.0]}, index=idx),
            "market_cap": pd.DataFrame({"BTC": [1e12]}, index=idx),
        }


class _Strategy:
    meta = type("M", (), {"name": "stub.strategy.v1"})()

    def run(self, *, data, params):
        return {"weights": pd.DataFrame({"BTC": [1.0]}, index=pd.to_datetime([ASOF]))}


class _NoOrders:
    """An injected order generator that sizes nothing, so the run reaches post-trade untouched."""

    def generate_orders(self, weights, broker, params):
        return {
            "weights": weights,
            "rebalancing": pd.DataFrame(),
            "orders": pd.DataFrame(),
            "total_value": 2000.0,
        }  # a pre-trade value, as a real generator reports


class _NavBroker(_Broker):
    """No get_equity: the post-trade NAV is the pipeline's own mark of the book."""


def _run(broker, tmp_path, store=None):
    store = store or _Store()
    result = TradingPipeline().run(
        mode="live",
        asof=ASOF,
        params={
            "book_key": "b",
            "stable_coin_symbol": "USDC",
            "trading_enabled": False,
            "reconciliation": {"book_key": "b", "data_dir": str(tmp_path), "mode": "observe"},
        },
        data=_Data(),
        store=store,
        broker=broker,
        risk=[],
        strategies=[_Strategy()],
        rebalancer=_NoOrders(),
    )
    return result, store


def test_post_trade_nav_reports_an_unmarked_holding_instead_of_valuing_it_at_zero(tmp_path):
    broker = _NavBroker(
        positions=[{"symbol": "BTC", "qty": 2.0}, {"symbol": "GHOST", "qty": 9.0}],
        marks={"BTC": 100.0},
        cash={"USDC": 1000.0},
    )
    result, store = _run(broker, tmp_path)
    nav = store.written["portfolio_daily"]["portfolio_value_usd"].iloc[0]
    assert nav == pytest.approx(1200.0)  # the marked part, stated as such
    errors = result.notes["exceptions"]["api_errors"]
    assert {"stage": "portfolio_snapshot", "error": "unmarkable holdings: GHOST"} in errors


def test_post_trade_nav_with_a_snapshot_lacking_mid_reports_every_holding(tmp_path):
    """A snapshot with no `mid` column used to make the NAV plain cash with no trace (TOM-1455)."""

    class _NoMid(_NavBroker):
        def get_market_snapshot(self, symbols):
            return pd.DataFrame({"symbol": list(symbols), "bid": [1.0] * len(symbols)})

    broker = _NoMid(positions=[{"symbol": "BTC", "qty": 2.0}], marks={}, cash={"USDC": 1000.0})
    result, _ = _run(broker, tmp_path)
    errors = result.notes["exceptions"]["api_errors"]
    assert {"stage": "portfolio_snapshot", "error": "unmarkable holdings: BTC"} in errors


def test_post_trade_nav_after_a_failed_snapshot_keeps_the_pre_trade_value(tmp_path):
    """A failed read is not a book of unmarked names: the NAV is not re-marked to cash alone."""
    broker = _NavBroker(
        positions=[{"symbol": "BTC", "qty": 10.0}],
        marks={},
        cash={"USDC": 1000.0},
        snapshot_raises=RuntimeError("venue down"),
    )
    result, store = _run(broker, tmp_path)
    nav = store.written["portfolio_daily"]["portfolio_value_usd"].iloc[0]
    assert nav == result.metrics["portfolio_value_usd_pre"]  # not 1000.0, the cash alone
    stages = [e["stage"] for e in result.notes["exceptions"]["api_errors"]]
    assert "portfolio_snapshot" in stages


def test_reconciliation_drift_of_an_unmarked_holding_is_unknown_not_a_break(tmp_path):
    broker = _NavBroker(
        positions=[{"symbol": "BTC", "qty": 2.0}, {"symbol": "GHOST", "qty": 9.0}],
        marks={"BTC": 100.0},
    )
    result, _ = _run(broker, tmp_path)
    recon = result.notes["reconciliation"]
    assert recon["unpriced"] == ["GHOST"]
    assert all(b["symbol"] != "GHOST" for b in recon["breaks"])


def test_reconciliation_after_a_failed_snapshot_reads_unknown_drift_not_an_empty_book(tmp_path):
    """The snapshot raising used to leave the book EMPTY to the drift check: every target a -100% hard drift."""
    broker = _NavBroker(
        positions=[{"symbol": "BTC", "qty": 10.0}],
        marks={},
        snapshot_raises=RuntimeError("venue down"),
    )
    result, _ = _run(broker, tmp_path)
    recon = result.notes["reconciliation"]
    assert recon["unpriced"] == ["BTC"]
    assert [b for b in recon["breaks"] if b["class"] == "drift"] == []


def test_the_screen_inputs_are_stored_next_to_prices(tmp_path):
    """TOM-1420: volume, market_cap and screen_volume beside prices.parquet; an absent key is stored empty."""
    broker = _NavBroker(positions=[], marks={"BTC": 100.0})
    _, store = _run(broker, tmp_path)
    for key in ("prices", "volume", "market_cap"):
        frame = store.written[key]
        assert list(frame.columns) == ["index", "BTC"] or "BTC" in frame.columns
        assert len(frame) == 1
    assert store.written["volume"]["BTC"].iloc[0] == 5.0
    assert store.written["market_cap"]["BTC"].iloc[0] == 1e12
    assert store.written["screen_volume"].empty  # the data plugin returned none


# ---------------------------------------------------------------------------
# 2c. The rebalancing policy cannot read an unknown held weight as flat
# ---------------------------------------------------------------------------


def _policy_prices() -> pd.DataFrame:
    idx = pd.date_range(end="2024-03-30", periods=30, freq="D")
    return pd.DataFrame({"A": np.linspace(90, 110, 30)}, index=idx)


class _PolicyData:
    def load_universe(self, params):
        return pd.DataFrame({"symbol": ["A"]})

    def load_market_data(self, universe, asof, params):
        return {"prices": _policy_prices()}


class _PolicyStrategy:
    meta = type("M", (), {"name": "strategy.flat.v1"})()

    def run(self, data, params=None, context=None):
        return {"weights": pd.DataFrame({"A": 0.5}, index=data["prices"].index)}


def test_a_policy_reading_the_held_book_refuses_an_unknown_held_weight(tmp_path):
    """On a book that does not refuse pre-trade (paper here), a held name with no mark
    used to reach the band trigger as weight 0. It is now a valuation refusal: no orders."""
    broker = SimPaperBroker(cash=10_000.0, quote_currency="USD", positions={"GHOST": 5.0})
    result = TradingPipeline().run(
        mode="paper",
        asof="2024-03-30",
        params={
            "strategies": [{"name": "strategy.flat.v1", "weight": 1.0, "params": {}}],
            "stable_coin_symbol": "USD",
            "trading_enabled": False,
            "rebalancing_policy": {"policy": "band", "band": 0.02},
            "_rebalancer_cfg": {"params": {}},
        },
        data=_PolicyData(),
        store=FileArtifactStore(str(tmp_path / "run"), "run"),
        broker=broker,
        risk=[],
        strategies=[_PolicyStrategy()],
        rebalancer=StandardRebalancer(),
    )
    assert result.notes["exceptions"]["pipeline_ok"] is False
    errors = [e for e in result.notes["exceptions"]["api_errors"] if e["stage"] == "portfolio_valuation"]
    assert errors and "GHOST" in errors[0]["error"]
    assert result.metrics["n_orders"] == 0.0
