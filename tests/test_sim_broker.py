"""Tests for SimPaperBroker plugin (sim.paper.v1)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quantbox.plugins.broker.sim import SimPaperBroker


class TestSimPaperBroker:
    """Unit tests for the SimPaperBroker paper-trading broker."""

    # ------------------------------------------------------------------
    # 1. Initialization
    # ------------------------------------------------------------------

    def test_default_initialization(self):
        """Default broker starts with 100k cash, no positions, no fills."""
        broker = SimPaperBroker()
        assert broker.cash == 100_000.0
        assert broker.quote_currency == "USDT"
        assert broker.positions == {}
        assert broker._fill_log == []
        assert broker._cumulative_fees == 0.0
        assert broker.state_file is None

    def test_custom_cash_initialization(self):
        """Broker accepts custom starting cash via dataclass field."""
        broker = SimPaperBroker(cash=50_000.0)
        assert broker.cash == 50_000.0

    # ------------------------------------------------------------------
    # 2. get_cash
    # ------------------------------------------------------------------

    def test_get_cash_returns_initial(self):
        """get_cash returns a dict with the quote currency and initial amount."""
        broker = SimPaperBroker(cash=25_000.0, quote_currency="USD")
        result = broker.get_cash()
        assert result == {"USD": 25_000.0}

    def test_get_cash_type(self):
        """get_cash value is always a Python float, not numpy or int."""
        broker = SimPaperBroker(cash=10_000)
        result = broker.get_cash()
        assert isinstance(result["USDT"], float)

    # ------------------------------------------------------------------
    # 3. get_positions (initially empty)
    # ------------------------------------------------------------------

    def test_get_positions_initially_empty(self):
        """get_positions returns an empty DataFrame when no trades have occurred."""
        broker = SimPaperBroker()
        pos = broker.get_positions()
        assert isinstance(pos, pd.DataFrame)
        assert len(pos) == 0

    # ------------------------------------------------------------------
    # 4. place_orders: buy order
    # ------------------------------------------------------------------

    def test_buy_order_updates_positions_and_cash(self):
        """A single buy order increases position qty and decreases cash."""
        broker = SimPaperBroker(
            cash=100_000.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            impact_factor=0.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )
        orders = pd.DataFrame(
            [
                {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 50_000.0},
            ]
        )
        fills = broker.place_orders(orders)

        assert len(fills) == 1
        assert broker.positions["BTC"] == 1.0
        assert broker.cash == pytest.approx(50_000.0)

    # ------------------------------------------------------------------
    # 5. place_orders: sell order
    # ------------------------------------------------------------------

    def test_sell_order_updates_positions_and_cash(self):
        """A sell order decreases position qty and increases cash."""
        broker = SimPaperBroker(
            cash=100_000.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            impact_factor=0.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )
        # Seed an existing position.
        broker.positions["ETH"] = 10.0
        orders = pd.DataFrame(
            [
                {"symbol": "ETH", "side": "sell", "qty": 5.0, "price": 3_000.0},
            ]
        )
        fills = broker.place_orders(orders)

        assert len(fills) == 1
        assert broker.positions["ETH"] == pytest.approx(5.0)
        # Cash increases by qty * price (no fees/slippage).
        assert broker.cash == pytest.approx(100_000.0 + 5.0 * 3_000.0)

    # ------------------------------------------------------------------
    # 6. place_orders: multiple orders in one batch
    # ------------------------------------------------------------------

    def test_batch_orders(self):
        """Multiple orders in a single DataFrame are all executed."""
        broker = SimPaperBroker(
            cash=500_000.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            impact_factor=0.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )
        orders = pd.DataFrame(
            [
                {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 50_000.0},
                {"symbol": "ETH", "side": "buy", "qty": 10.0, "price": 3_000.0},
                {"symbol": "SOL", "side": "buy", "qty": 100.0, "price": 150.0},
            ]
        )
        fills = broker.place_orders(orders)

        assert len(fills) == 3
        assert broker.positions["BTC"] == pytest.approx(1.0)
        assert broker.positions["ETH"] == pytest.approx(10.0)
        assert broker.positions["SOL"] == pytest.approx(100.0)
        expected_cash = 500_000.0 - (50_000.0 + 30_000.0 + 15_000.0)
        assert broker.cash == pytest.approx(expected_cash)

    # ------------------------------------------------------------------
    # 7. Fee deduction
    # ------------------------------------------------------------------

    def test_fee_deduction(self):
        """Trading fees are subtracted from cash and tracked cumulatively."""
        broker = SimPaperBroker(
            cash=100_000.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            impact_factor=0.0,
            taker_fee_bps=10.0,  # 0.10%
            assume_taker=True,
        )
        orders = pd.DataFrame(
            [
                {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 50_000.0},
            ]
        )
        fills = broker.place_orders(orders)

        # Fee = notional * 10 bps = 50_000 * 0.001 = 50
        expected_fee = 50_000.0 * 10.0 / 10_000
        assert fills.iloc[0]["fee"] == pytest.approx(expected_fee)
        assert broker._cumulative_fees == pytest.approx(expected_fee)
        # Cash = initial - notional - fee
        assert broker.cash == pytest.approx(100_000.0 - 50_000.0 - expected_fee)

    def test_maker_fee_when_not_taker(self):
        """When assume_taker=False, maker_fee_bps is used instead."""
        broker = SimPaperBroker(
            cash=100_000.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            impact_factor=0.0,
            maker_fee_bps=5.0,  # 0.05%
            taker_fee_bps=10.0,
            assume_taker=False,
        )
        orders = pd.DataFrame(
            [
                {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 40_000.0},
            ]
        )
        fills = broker.place_orders(orders)

        expected_fee = 40_000.0 * 5.0 / 10_000  # maker fee
        assert fills.iloc[0]["fee"] == pytest.approx(expected_fee)

    # ------------------------------------------------------------------
    # 8. Position tracking across sequential buys and sells
    # ------------------------------------------------------------------

    def test_sequential_buys_and_sells(self):
        """Sequential trades accumulate and reduce positions correctly."""
        broker = SimPaperBroker(
            cash=200_000.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            impact_factor=0.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )

        # Buy 2 BTC
        broker.place_orders(
            pd.DataFrame(
                [
                    {"symbol": "BTC", "side": "buy", "qty": 2.0, "price": 50_000.0},
                ]
            )
        )
        assert broker.positions["BTC"] == pytest.approx(2.0)

        # Buy 3 more BTC
        broker.place_orders(
            pd.DataFrame(
                [
                    {"symbol": "BTC", "side": "buy", "qty": 3.0, "price": 51_000.0},
                ]
            )
        )
        assert broker.positions["BTC"] == pytest.approx(5.0)

        # Sell 4 BTC
        broker.place_orders(
            pd.DataFrame(
                [
                    {"symbol": "BTC", "side": "sell", "qty": 4.0, "price": 52_000.0},
                ]
            )
        )
        assert broker.positions["BTC"] == pytest.approx(1.0)

        # Fill log should have 3 entries total
        assert len(broker._fill_log) == 3

    # ------------------------------------------------------------------
    # 9. Selling more than held (goes negative = short)
    # ------------------------------------------------------------------

    def test_sell_more_than_held_goes_negative(self):
        """Selling more than the held quantity results in a negative position (short)."""
        broker = SimPaperBroker(
            cash=200_000.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            impact_factor=0.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )
        broker.positions["ETH"] = 2.0

        orders = pd.DataFrame(
            [
                {"symbol": "ETH", "side": "sell", "qty": 5.0, "price": 3_000.0},
            ]
        )
        broker.place_orders(orders)

        # Position goes to 2 - 5 = -3 (short)
        assert broker.positions["ETH"] == pytest.approx(-3.0)
        # Cash increases by 5 * 3000 = 15_000
        assert broker.cash == pytest.approx(200_000.0 + 15_000.0)

    # ------------------------------------------------------------------
    # 10. describe() / meta attributes
    # ------------------------------------------------------------------

    def test_meta_attributes(self):
        """PluginMeta is set correctly as a class attribute."""
        assert SimPaperBroker.meta.name == "sim.paper.v1"
        assert SimPaperBroker.meta.kind == "broker"
        assert SimPaperBroker.meta.version == "0.1.0"
        assert "paper" in SimPaperBroker.meta.tags
        assert "paper" in SimPaperBroker.meta.capabilities

    def test_get_market_snapshot(self):
        """get_market_snapshot returns a DataFrame with mid prices."""
        broker = SimPaperBroker()
        broker.set_prices({"BTC": 60_000.0, "ETH": 3_500.0})
        snap = broker.get_market_snapshot(["BTC", "ETH", "UNKNOWN"])

        assert len(snap) == 3
        assert snap.loc[snap["symbol"] == "BTC", "mid"].iloc[0] == 60_000.0
        assert snap.loc[snap["symbol"] == "ETH", "mid"].iloc[0] == 3_500.0
        assert pd.isna(snap.loc[snap["symbol"] == "UNKNOWN", "mid"].iloc[0])

    # ------------------------------------------------------------------
    # 11. State persistence (save and reload via state_file)
    # ------------------------------------------------------------------

    def test_state_persistence_save_and_reload(self, tmp_path):
        """Broker state is persisted to JSON and reloaded on new instance creation."""
        state_path = str(tmp_path / "broker_state.json")

        # Create broker, execute a trade, which triggers _save_state
        broker1 = SimPaperBroker(
            cash=100_000.0,
            state_file=state_path,
            spread_bps=0.0,
            slippage_bps=0.0,
            impact_factor=0.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )
        orders = pd.DataFrame(
            [
                {"symbol": "BTC", "side": "buy", "qty": 0.5, "price": 60_000.0},
            ]
        )
        broker1.place_orders(orders)

        # Verify the state file was created
        assert (tmp_path / "broker_state.json").exists()

        # Create a new broker pointing to the same state file -- __post_init__ loads state
        broker2 = SimPaperBroker(state_file=state_path)

        assert broker2.cash == pytest.approx(broker1.cash)
        assert broker2.positions == broker1.positions
        assert broker2._cumulative_fees == pytest.approx(broker1._cumulative_fees)
        assert len(broker2._fill_log) == len(broker1._fill_log)

    def test_state_file_contents(self, tmp_path):
        """The persisted JSON contains the expected keys."""
        state_path = str(tmp_path / "state.json")
        broker = SimPaperBroker(
            cash=75_000.0,
            state_file=state_path,
            spread_bps=0.0,
            slippage_bps=0.0,
            impact_factor=0.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )
        broker.place_orders(
            pd.DataFrame(
                [
                    {"symbol": "SOL", "side": "buy", "qty": 100.0, "price": 150.0},
                ]
            )
        )

        state = json.loads((tmp_path / "state.json").read_text())
        assert "cash" in state
        assert "positions" in state
        assert "cumulative_fees" in state
        assert "fill_log" in state
        assert "last_updated" in state
        assert state["positions"]["SOL"] == pytest.approx(100.0)

    def test_no_state_file_does_not_crash(self):
        """When state_file is None, save/load are no-ops and don't raise."""
        broker = SimPaperBroker(state_file=None)
        # Manually calling save/load should be safe
        broker._save_state()
        broker._load_state()
        assert broker.cash == 100_000.0

    # ------------------------------------------------------------------
    # 12. Price impact model (slippage)
    # ------------------------------------------------------------------

    def test_price_impact_buy_increases_fill_price(self):
        """Buy orders fill above mid price when slippage is enabled."""
        broker = SimPaperBroker(
            cash=200_000.0,
            spread_bps=2.0,
            slippage_bps=5.0,
            impact_factor=0.01,
            max_impact_bps=20.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )
        mid_price = 50_000.0
        orders = pd.DataFrame(
            [
                {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": mid_price},
            ]
        )
        fills = broker.place_orders(orders)

        fill_price = fills.iloc[0]["price"]
        # Buy fill price must be above mid
        assert fill_price > mid_price

    def test_price_impact_sell_decreases_fill_price(self):
        """Sell orders fill below mid price when slippage is enabled."""
        broker = SimPaperBroker(
            cash=200_000.0,
            spread_bps=2.0,
            slippage_bps=5.0,
            impact_factor=0.01,
            max_impact_bps=20.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )
        broker.positions["BTC"] = 5.0
        mid_price = 50_000.0
        orders = pd.DataFrame(
            [
                {"symbol": "BTC", "side": "sell", "qty": 1.0, "price": mid_price},
            ]
        )
        fills = broker.place_orders(orders)

        fill_price = fills.iloc[0]["price"]
        # Sell fill price must be below mid
        assert fill_price < mid_price

    def test_price_impact_computation(self):
        """_compute_impact_bps scales linearly and respects the cap."""
        broker = SimPaperBroker(impact_factor=0.01, max_impact_bps=20.0)

        # Small order: 10k notional -> 0.01 * 10_000 / 10_000 = 0.01 bps
        assert broker._compute_impact_bps(10_000.0) == pytest.approx(0.01)

        # Large order: should be capped at max_impact_bps
        huge_notional = 1e12
        assert broker._compute_impact_bps(huge_notional) == pytest.approx(20.0)

    def test_slippage_exact_fill_price(self):
        """Verify the exact fill price calculation for a buy order."""
        broker = SimPaperBroker(
            cash=200_000.0,
            spread_bps=2.0,
            slippage_bps=5.0,
            impact_factor=0.0,  # disable volume impact for clean calculation
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )
        mid = 10_000.0
        orders = pd.DataFrame(
            [
                {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": mid},
            ]
        )
        fills = broker.place_orders(orders)

        # cost_bps = (2.0 + 5.0 + 0.0) / 10_000 = 0.0007
        # fill_price = 10_000 * (1 + 1 * 0.0007) = 10_007.0
        expected = mid * (1 + (2.0 + 5.0) / 10_000)
        assert fills.iloc[0]["price"] == pytest.approx(expected)

    # ------------------------------------------------------------------
    # Extra: edge cases
    # ------------------------------------------------------------------

    def test_order_with_zero_price_and_no_set_prices_is_skipped(self):
        """Orders with price=0 and no set_prices entry are skipped."""
        broker = SimPaperBroker(cash=100_000.0)
        orders = pd.DataFrame(
            [
                {"symbol": "UNKNOWN", "side": "buy", "qty": 10.0, "price": 0.0},
            ]
        )
        fills = broker.place_orders(orders)

        assert len(fills) == 0
        assert broker.cash == 100_000.0

    def test_order_uses_set_prices_when_price_column_missing(self):
        """Orders fall back to set_prices when no price is provided in the order."""
        broker = SimPaperBroker(
            cash=100_000.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            impact_factor=0.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )
        broker.set_prices({"BTC": 45_000.0})
        orders = pd.DataFrame(
            [
                {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 0.0},
            ]
        )
        fills = broker.place_orders(orders)

        assert len(fills) == 1
        assert fills.iloc[0]["price"] == pytest.approx(45_000.0)
        assert broker.cash == pytest.approx(100_000.0 - 45_000.0)

    def test_fetch_fills_filters_by_timestamp(self):
        """fetch_fills returns only fills at or after the given timestamp."""
        broker = SimPaperBroker(
            cash=500_000.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            impact_factor=0.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
        )
        broker.place_orders(
            pd.DataFrame(
                [
                    {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 50_000.0},
                ]
            )
        )
        # Use a future timestamp so nothing matches
        future = "2099-01-01T00:00:00+00:00"
        result = broker.fetch_fills(future)
        assert len(result) == 0

        # Use a past timestamp so everything matches
        past = "2000-01-01T00:00:00+00:00"
        result = broker.fetch_fills(past)
        assert len(result) == 1
