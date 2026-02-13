"""Tests for TradingRiskManager plugin (risk.trading_basic.v1).

Covers check_targets(), check_orders(), edge cases, and plugin metadata.
Self-contained — no conftest.py required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.plugins.risk.trading_risk import TradingRiskManager


class TestTradingRiskManager:
    """Test suite for the TradingRiskManager risk plugin."""

    # ---- helpers ----

    @staticmethod
    def _make_targets(rows: list[dict]) -> pd.DataFrame:
        """Build a targets DataFrame with expected columns."""
        return pd.DataFrame(rows, columns=["symbol", "weight", "price", "target_qty"])

    @staticmethod
    def _make_orders(rows: list[dict]) -> pd.DataFrame:
        """Build an orders DataFrame with expected columns."""
        return pd.DataFrame(rows, columns=["symbol", "side", "qty", "price"])

    @staticmethod
    def _default_params(**overrides) -> dict:
        """Return default params dict, merging in any overrides."""
        p = {
            "max_leverage": 1.0,
            "max_concentration": 0.30,
            "allow_negative_weights": False,
            "min_notional": 1.0,
            "max_order_notional": 0,  # 0 = unlimited
        }
        p.update(overrides)
        return p

    # ---- fixtures ----

    @pytest.fixture
    def rm(self) -> TradingRiskManager:
        return TradingRiskManager()

    # ================================================================
    # 1. check_targets passes with valid targets within limits
    # ================================================================
    def test_check_targets_valid(self, rm: TradingRiskManager) -> None:
        targets = self._make_targets(
            [
                {"symbol": "BTC", "weight": 0.25, "price": 50000, "target_qty": 0.01},
                {"symbol": "ETH", "weight": 0.25, "price": 3000, "target_qty": 1.0},
                {"symbol": "SOL", "weight": 0.20, "price": 100, "target_qty": 10.0},
            ]
        )
        params = self._default_params(max_leverage=1.0, max_concentration=0.30)
        findings = rm.check_targets(targets, params)
        assert findings == [], f"Expected no findings, got {findings}"

    # ================================================================
    # 2. check_targets flags excessive concentration
    # ================================================================
    def test_check_targets_concentration_exceeded(self, rm: TradingRiskManager) -> None:
        targets = self._make_targets(
            [
                {"symbol": "BTC", "weight": 0.50, "price": 50000, "target_qty": 0.01},
                {"symbol": "ETH", "weight": 0.10, "price": 3000, "target_qty": 1.0},
            ]
        )
        params = self._default_params(max_concentration=0.30)
        findings = rm.check_targets(targets, params)

        conc_findings = [f for f in findings if f["rule"] == "concentration_exceeded"]
        assert len(conc_findings) == 1
        assert "BTC" in conc_findings[0]["detail"]
        assert conc_findings[0]["level"] == "warn"

    # ================================================================
    # 3. check_targets flags leverage above limit
    # ================================================================
    def test_check_targets_leverage_exceeded(self, rm: TradingRiskManager) -> None:
        targets = self._make_targets(
            [
                {"symbol": "BTC", "weight": 0.25, "price": 50000, "target_qty": 0.01},
                {"symbol": "ETH", "weight": 0.25, "price": 3000, "target_qty": 1.0},
                {"symbol": "SOL", "weight": 0.25, "price": 100, "target_qty": 10.0},
                {"symbol": "AVAX", "weight": 0.30, "price": 40, "target_qty": 50.0},
            ]
        )
        # Gross = 1.05 > max_leverage 1.0
        params = self._default_params(max_leverage=1.0, max_concentration=0.50)
        findings = rm.check_targets(targets, params)

        lev_findings = [f for f in findings if f["rule"] == "max_leverage_exceeded"]
        assert len(lev_findings) == 1
        assert "1.05" in lev_findings[0]["detail"]
        assert lev_findings[0]["level"] == "warn"

    # ================================================================
    # 4. check_orders passes with valid orders
    # ================================================================
    def test_check_orders_valid(self, rm: TradingRiskManager) -> None:
        orders = self._make_orders(
            [
                {"symbol": "BTC", "side": "buy", "qty": 0.001, "price": 50000},
                {"symbol": "ETH", "side": "sell", "qty": 0.5, "price": 3000},
            ]
        )
        # notionals: 50, 1500 — both above min_notional=1, below unlimited max
        params = self._default_params(min_notional=1.0, max_order_notional=0)
        findings = rm.check_orders(orders, params)
        assert findings == [], f"Expected no findings, got {findings}"

    # ================================================================
    # 5. check_orders flags orders exceeding max notional
    # ================================================================
    def test_check_orders_exceeds_max_notional(self, rm: TradingRiskManager) -> None:
        orders = self._make_orders(
            [
                {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 50000},
            ]
        )
        # notional = 50000, max = 10000
        params = self._default_params(max_order_notional=10000)
        findings = rm.check_orders(orders, params)

        max_findings = [f for f in findings if f["rule"] == "exceeds_max_order_notional"]
        assert len(max_findings) == 1
        assert "BTC" in max_findings[0]["detail"]
        assert max_findings[0]["level"] == "error"

    # ================================================================
    # 6. check_orders flags orders below min notional
    # ================================================================
    def test_check_orders_below_min_notional(self, rm: TradingRiskManager) -> None:
        orders = self._make_orders(
            [
                {"symbol": "DOGE", "side": "buy", "qty": 0.1, "price": 0.08},
            ]
        )
        # notional = 0.008 < min_notional 1.0
        params = self._default_params(min_notional=1.0)
        findings = rm.check_orders(orders, params)

        min_findings = [f for f in findings if f["rule"] == "below_min_notional"]
        assert len(min_findings) == 1
        assert "DOGE" in min_findings[0]["detail"]
        assert min_findings[0]["level"] == "warn"

    # ================================================================
    # 7. Empty targets -> no findings
    # ================================================================
    def test_check_targets_empty(self, rm: TradingRiskManager) -> None:
        empty_df = pd.DataFrame(columns=["symbol", "weight", "price", "target_qty"])
        params = self._default_params()

        assert rm.check_targets(empty_df, params) == []
        assert rm.check_targets(None, params) == []

    # ================================================================
    # 8. Empty orders -> no findings
    # ================================================================
    def test_check_orders_empty(self, rm: TradingRiskManager) -> None:
        empty_df = pd.DataFrame(columns=["symbol", "side", "qty", "price"])
        params = self._default_params()

        assert rm.check_orders(empty_df, params) == []
        assert rm.check_orders(None, params) == []

    # ================================================================
    # 9. Multiple violations in a single check
    # ================================================================
    def test_multiple_violations(self, rm: TradingRiskManager) -> None:
        # Build targets that violate leverage, concentration, AND negative weights
        targets = self._make_targets(
            [
                {"symbol": "BTC", "weight": 0.50, "price": 50000, "target_qty": 0.01},
                {"symbol": "ETH", "weight": 0.40, "price": 3000, "target_qty": 1.0},
                {"symbol": "SOL", "weight": -0.20, "price": 100, "target_qty": -10.0},
            ]
        )
        # gross leverage = 1.10 > 1.0
        # BTC 0.50 > max_conc 0.30, ETH 0.40 > max_conc 0.30
        # SOL is negative, allow_negative_weights=False
        params = self._default_params(
            max_leverage=1.0,
            max_concentration=0.30,
            allow_negative_weights=False,
        )
        findings = rm.check_targets(targets, params)

        rules = [f["rule"] for f in findings]
        assert "max_leverage_exceeded" in rules
        assert "concentration_exceeded" in rules
        assert "negative_weight_disallowed" in rules
        # BTC and ETH both exceed concentration
        conc_count = rules.count("concentration_exceeded")
        assert conc_count == 2, f"Expected 2 concentration findings, got {conc_count}"
        # Total: 1 leverage + 2 concentration + 1 negative = 4
        assert len(findings) == 4

    # ================================================================
    # 10. Plugin meta attributes
    # ================================================================
    def test_plugin_meta_attributes(self) -> None:
        meta = TradingRiskManager.meta

        assert meta.name == "risk.trading_basic.v1"
        assert meta.kind == "risk"
        assert meta.version == "0.1.0"
        assert "trading" in meta.tags
        assert "risk" in meta.tags
        assert "paper" in meta.capabilities
        assert "live" in meta.capabilities
        assert meta.description is not None and len(meta.description) > 0
        assert meta.params_schema is not None
        assert "max_leverage" in meta.params_schema["properties"]
        assert "max_concentration" in meta.params_schema["properties"]
        assert "allow_negative_weights" in meta.params_schema["properties"]
        assert "min_notional" in meta.params_schema["properties"]
        assert "max_order_notional" in meta.params_schema["properties"]

    # ================================================================
    # Bonus: negative weights allowed when flag is True
    # ================================================================
    def test_check_targets_negative_weights_allowed(self, rm: TradingRiskManager) -> None:
        targets = self._make_targets(
            [
                {"symbol": "BTC", "weight": 0.25, "price": 50000, "target_qty": 0.01},
                {"symbol": "ETH", "weight": -0.20, "price": 3000, "target_qty": -1.0},
            ]
        )
        params = self._default_params(
            max_leverage=1.0,
            max_concentration=0.30,
            allow_negative_weights=True,
        )
        findings = rm.check_targets(targets, params)
        neg_findings = [f for f in findings if f["rule"] == "negative_weight_disallowed"]
        assert neg_findings == [], "Negative weights should be allowed when flag is True"
