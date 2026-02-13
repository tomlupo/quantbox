"""Tests for StandardRebalancer plugin.

Covers order generation, risk transforms, threshold filtering, edge cases,
and plugin meta attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from quantbox.contracts import PluginMeta
from quantbox.plugins.rebalancing.standard_rebalancer import StandardRebalancer

# ---------------------------------------------------------------------------
# Fake broker
# ---------------------------------------------------------------------------


@dataclass
class FakeBroker:
    """Minimal BrokerPlugin stand-in for unit tests."""

    meta = PluginMeta(
        name="test.fake_broker.v1",
        kind="broker",
        version="0.0.1",
        core_compat=">=0.1,<0.2",
        description="Fake broker for tests",
    )

    positions: pd.DataFrame | None = None
    cash: dict[str, float] | None = None
    snapshot: pd.DataFrame | None = None

    def get_positions(self) -> pd.DataFrame:
        if self.positions is not None:
            return self.positions
        return pd.DataFrame(columns=["symbol", "qty"])

    def get_cash(self) -> dict[str, float]:
        return self.cash or {}

    def get_market_snapshot(self, symbols: list[str]) -> pd.DataFrame:
        if self.snapshot is not None:
            # Return rows matching requested symbols
            mask = self.snapshot["symbol"].isin(symbols)
            return self.snapshot[mask].reset_index(drop=True)
        return pd.DataFrame(columns=["symbol", "mid"])

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        return orders

    def fetch_fills(self, since: str) -> pd.DataFrame:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(prices: dict[str, float]) -> pd.DataFrame:
    """Build a market snapshot DataFrame from a {symbol: price} dict."""
    rows = []
    for sym, mid in prices.items():
        rows.append(
            {
                "symbol": sym,
                "mid": mid,
                "min_qty": 0.0,
                "step_size": 0.0,
                "min_notional": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _make_positions(holdings: dict[str, float]) -> pd.DataFrame:
    """Build a positions DataFrame from a {symbol: qty} dict."""
    rows = [{"symbol": sym, "qty": qty} for sym, qty in holdings.items()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["symbol", "qty"])


def _default_params(**overrides: Any) -> dict[str, Any]:
    """Default params dict for generate_orders; override any key."""
    p: dict[str, Any] = {
        "min_trade_size": 0.01,
        "min_notional": 1.0,
        "capital_at_risk": 1.0,
        "stable_coin_symbol": "USDC",
        "scaling_factor_min": 0.0,  # allow any scaling so buys always pass
        "max_leverage": 1.0,
        "allow_short": False,
        "exclusions": [],
    }
    p.update(overrides)
    return p


# ===================================================================
# Test: plugin meta
# ===================================================================


class TestStandardRebalancer:
    """Plugin-level checks (meta, instantiation)."""

    def test_meta_name(self):
        assert StandardRebalancer.meta.name == "rebalancing.standard.v1"

    def test_meta_kind(self):
        assert StandardRebalancer.meta.kind == "rebalancing"

    def test_meta_version(self):
        assert StandardRebalancer.meta.version == "0.1.0"

    def test_meta_is_class_attribute(self):
        """meta must be a class-level attribute, not per-instance."""
        r1 = StandardRebalancer()
        r2 = StandardRebalancer()
        assert r1.meta is r2.meta

    def test_meta_tags(self):
        assert "rebalancing" in StandardRebalancer.meta.tags
        assert "trading" in StandardRebalancer.meta.tags


# ===================================================================
# Test: generate_orders
# ===================================================================


class TestGenerateOrders:
    """Order-generation logic tests."""

    # --- 1. Basic buy/sell -------------------------------------------

    def test_basic_buy_and_sell(self):
        """Target weights vs current positions produce correct buy/sell."""
        prices = {"BTC": 50_000.0, "ETH": 3_000.0}
        # Currently hold 0.1 BTC (5000 USD), 1 ETH (3000 USD)
        # Cash = 2000 USDC => total = 10000 USD
        broker = FakeBroker(
            positions=_make_positions({"BTC": 0.1, "ETH": 1.0}),
            cash={"USDC": 2_000.0},
            snapshot=_make_snapshot(prices),
        )

        # Want 60% BTC, 10% ETH
        weights = {"BTC": 0.6, "ETH": 0.1}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(),
        )

        reb = result["rebalancing"]
        assert not reb.empty

        btc_row = reb.loc[reb["Asset"] == "BTC"].iloc[0]
        eth_row = reb.loc[reb["Asset"] == "ETH"].iloc[0]

        # BTC target = 10000*0.6 / 50000 = 0.12  => delta = +0.02 => Buy
        assert btc_row["Trade Action"] == "Buy"
        assert btc_row["Delta Quantity"] == pytest.approx(0.02, abs=1e-6)

        # ETH target = 10000*0.1 / 3000 ~= 0.3333 => delta = -0.6667 => Sell
        assert eth_row["Trade Action"] == "Sell"
        assert eth_row["Delta Quantity"] < 0

    # --- 2. No rebalancing needed -----------------------------------

    def test_no_rebalancing_when_weights_match(self):
        """When current weights equal target weights, deltas are ~0."""
        # Total = 10000 (5000 cash + 5000 in BTC)
        prices = {"BTC": 50_000.0}
        broker = FakeBroker(
            positions=_make_positions({"BTC": 0.1}),
            cash={"USDC": 5_000.0},
            snapshot=_make_snapshot(prices),
        )
        # BTC is 5000/10000 = 50%
        weights = {"BTC": 0.5}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(),
        )

        reb = result["rebalancing"]
        btc_row = reb.loc[reb["Asset"] == "BTC"].iloc[0]
        assert btc_row["Trade Action"] == "Hold"
        assert btc_row["Delta Quantity"] == pytest.approx(0.0, abs=1e-8)

    # --- 3. New positions (all buys) --------------------------------

    def test_new_positions_all_buys(self):
        """No existing positions: all targets become buy orders."""
        prices = {"BTC": 50_000.0, "ETH": 3_000.0, "SOL": 100.0}
        broker = FakeBroker(
            positions=_make_positions({}),
            cash={"USDC": 10_000.0},
            snapshot=_make_snapshot(prices),
        )

        weights = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(),
        )

        reb = result["rebalancing"]
        assert len(reb) == 3
        assert (reb["Trade Action"] == "Buy").all()
        assert result["total_value"] == pytest.approx(10_000.0, rel=1e-6)

    # --- 4. Full exit (target weight = 0) ---------------------------

    def test_full_exit_generates_sell(self):
        """Setting target weight to 0 for a held asset produces a sell."""
        prices = {"BTC": 50_000.0}
        broker = FakeBroker(
            positions=_make_positions({"BTC": 0.2}),
            cash={"USDC": 0.0},
            snapshot=_make_snapshot(prices),
        )

        # Total value = 0.2 * 50000 = 10000.  Target BTC = 0% => sell all 0.2
        weights = {"BTC": 0.0}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(min_trade_size=0.0),
        )

        reb = result["rebalancing"]
        assert len(reb) == 1
        btc_row = reb.iloc[0]
        assert btc_row["Trade Action"] == "Sell"
        assert btc_row["Target Quantity"] == pytest.approx(0.0, abs=1e-10)
        assert btc_row["Delta Quantity"] == pytest.approx(-0.2, abs=1e-8)

    # --- 5. Threshold filtering (skip small trades) -----------------

    def test_threshold_filters_small_trades(self):
        """Trades with weight delta below min_trade_size are skipped."""
        # BTC = 50% of 10000; target = 50.5% => delta ~0.5% < 1% threshold
        prices = {"BTC": 50_000.0}
        broker = FakeBroker(
            positions=_make_positions({"BTC": 0.1}),
            cash={"USDC": 5_000.0},
            snapshot=_make_snapshot(prices),
        )
        weights = {"BTC": 0.505}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(min_trade_size=0.01),  # 1% threshold
        )

        orders = result["orders"]
        # The tiny buy should be below threshold
        if not orders.empty:
            btc_orders = orders.loc[orders["Asset"] == "BTC"]
            if not btc_orders.empty:
                assert btc_orders.iloc[0]["Order Status"] == "Below threshold"

    # --- 6. Order sizing correctness --------------------------------

    def test_order_sizing(self):
        """Verify buy quantities are correct based on prices and portfolio value."""
        prices = {"ETH": 2_000.0}
        broker = FakeBroker(
            positions=_make_positions({}),
            cash={"USDC": 10_000.0},
            snapshot=_make_snapshot(prices),
        )

        weights = {"ETH": 0.4}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(),
        )

        reb = result["rebalancing"]
        eth_row = reb.loc[reb["Asset"] == "ETH"].iloc[0]

        # Target qty = 10000 * 0.4 / 2000 = 2.0
        expected_qty = 10_000.0 * 0.4 / 2_000.0
        assert eth_row["Target Quantity"] == pytest.approx(expected_qty, rel=1e-6)
        assert eth_row["Delta Quantity"] == pytest.approx(expected_qty, rel=1e-6)

        # total_value should reflect cash only (no existing positions)
        assert result["total_value"] == pytest.approx(10_000.0, rel=1e-6)

    # --- 7. Multiple assets simultaneously --------------------------

    def test_multiple_assets(self):
        """Simultaneous rebalancing across many assets."""
        prices = {
            "BTC": 60_000.0,
            "ETH": 3_000.0,
            "SOL": 150.0,
            "AVAX": 30.0,
            "DOGE": 0.10,
        }
        broker = FakeBroker(
            positions=_make_positions(
                {
                    "BTC": 0.05,  # 3000
                    "ETH": 2.0,  # 6000
                    "DOGE": 10000,  # 1000
                }
            ),
            cash={"USDC": 0.0},
            snapshot=_make_snapshot(prices),
        )

        # Total value = 3000 + 6000 + 1000 = 10000
        weights = {
            "BTC": 0.30,  # target 3000 => 0.05 BTC => hold
            "ETH": 0.30,  # target 3000 => 1.0 ETH => sell 1
            "SOL": 0.20,  # target 2000 => 13.33 SOL => buy
            "AVAX": 0.10,  # target 1000 => 33.33 AVAX => buy
            "DOGE": 0.10,  # target 1000 => 10000 DOGE => hold
        }
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(),
        )

        reb = result["rebalancing"]
        assert len(reb) == 5

        actions = dict(zip(reb["Asset"], reb["Trade Action"], strict=False))
        assert actions["ETH"] == "Sell"
        assert actions["SOL"] == "Buy"
        assert actions["AVAX"] == "Buy"

        # BTC should be Hold (exact match)
        assert actions["BTC"] == "Hold"
        # DOGE should also be Hold
        assert actions["DOGE"] == "Hold"

    # --- 8. Edge case: zero price -----------------------------------

    def test_zero_price_skips_order(self):
        """Assets with zero price should not generate executable orders."""
        prices = {"SHIB": 0.0}
        broker = FakeBroker(
            positions=_make_positions({}),
            cash={"USDC": 5_000.0},
            snapshot=_make_snapshot(prices),
        )

        weights = {"SHIB": 0.5}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(),
        )

        # With zero price, target_positions will be empty (p > 0 check fails),
        # so rebalancing df should be empty or the asset won't appear.
        reb = result["rebalancing"]
        if not reb.empty:
            # If it does appear (due to current holdings union), no buy
            shib_rows = reb.loc[reb["Asset"] == "SHIB"]
            if not shib_rows.empty:
                assert shib_rows.iloc[0]["Target Quantity"] == pytest.approx(0.0, abs=1e-10)

    # --- 9. Edge case: empty targets --------------------------------

    def test_empty_targets(self):
        """Empty weight dict with no positions produces empty results."""
        broker = FakeBroker(
            positions=_make_positions({}),
            cash={"USDC": 10_000.0},
            snapshot=_make_snapshot({}),
        )

        weights: dict[str, float] = {}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(),
        )

        assert result["orders"].empty
        assert result["rebalancing"].empty

    # --- 10. Empty targets with existing positions (sell all) -------

    def test_empty_targets_with_positions_sells_all(self):
        """Empty weights but existing positions should produce sells."""
        prices = {"BTC": 50_000.0}
        broker = FakeBroker(
            positions=_make_positions({"BTC": 0.1}),
            cash={"USDC": 5_000.0},
            snapshot=_make_snapshot(prices),
        )

        weights: dict[str, float] = {}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(min_trade_size=0.0),
        )

        reb = result["rebalancing"]
        assert len(reb) == 1
        assert reb.iloc[0]["Trade Action"] == "Sell"
        assert reb.iloc[0]["Delta Quantity"] == pytest.approx(-0.1, abs=1e-8)

    # --- Risk transforms: leverage capping --------------------------

    def test_leverage_cap_scales_down(self):
        """Weights exceeding max_leverage are scaled down proportionally."""
        prices = {"BTC": 50_000.0, "ETH": 3_000.0}
        broker = FakeBroker(
            positions=_make_positions({}),
            cash={"USDC": 10_000.0},
            snapshot=_make_snapshot(prices),
        )

        # Gross exposure = 0.8 + 0.6 = 1.4, but max_leverage = 1.0
        weights = {"BTC": 0.8, "ETH": 0.6}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(max_leverage=1.0),
        )

        reb = result["rebalancing"]
        # After scaling, gross should be <= 1.0
        actual_gross = reb["Target Weight"].abs().sum()
        assert actual_gross <= 1.0 + 1e-6

    # --- Risk transforms: allow_short clamp -------------------------

    def test_negative_weights_clamped_when_short_disabled(self):
        """Negative weights are clamped to 0 unless allow_short=True."""
        prices = {"BTC": 50_000.0, "ETH": 3_000.0}
        broker = FakeBroker(
            positions=_make_positions({}),
            cash={"USDC": 10_000.0},
            snapshot=_make_snapshot(prices),
        )

        weights = {"BTC": 0.8, "ETH": -0.3}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(allow_short=False),
        )

        reb = result["rebalancing"]
        # ETH should be removed entirely (clamped to 0 => dropped)
        assert "ETH" not in reb["Asset"].values

    # --- Zero portfolio value edge case -----------------------------

    def test_zero_portfolio_value(self):
        """Zero cash and no positions returns empty orders."""
        broker = FakeBroker(
            positions=_make_positions({}),
            cash={"USDC": 0.0},
            snapshot=_make_snapshot({"BTC": 50_000.0}),
        )

        weights = {"BTC": 1.0}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(),
        )

        assert result["total_value"] == 0.0
        assert result["orders"].empty

    # --- Capital at risk factor -------------------------------------

    def test_capital_at_risk_scales_targets(self):
        """capital_at_risk < 1 scales all target positions proportionally."""
        prices = {"BTC": 50_000.0}
        broker = FakeBroker(
            positions=_make_positions({}),
            cash={"USDC": 10_000.0},
            snapshot=_make_snapshot(prices),
        )

        weights = {"BTC": 1.0}
        result = StandardRebalancer().generate_orders(
            weights=weights,
            broker=broker,
            params=_default_params(capital_at_risk=0.5),
        )

        reb = result["rebalancing"]
        btc_row = reb.loc[reb["Asset"] == "BTC"].iloc[0]
        # target = 10000 * 1.0 * 0.5 / 50000 = 0.1
        assert btc_row["Target Quantity"] == pytest.approx(0.1, rel=1e-6)
