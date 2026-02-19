"""Tests for FactorExposureRiskManager plugin (risk.factor_exposure.v1).

Covers single-position weight checks, sector concentration, clean portfolio,
check_orders delegation, and plugin metadata. Self-contained.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.plugins.risk.factor_exposure import FactorExposureRiskManager


class TestFactorExposureRiskManager:
    """Test suite for the FactorExposureRiskManager risk plugin."""

    @pytest.fixture()
    def rm(self) -> FactorExposureRiskManager:
        return FactorExposureRiskManager()

    @staticmethod
    def _make_targets(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------
    # 1. Single weight exceeds limit
    # ----------------------------------------------------------------
    def test_single_weight_exceeded(self, rm: FactorExposureRiskManager) -> None:
        targets = self._make_targets([
            {"symbol": "BTC", "weight": 0.7},
            {"symbol": "ETH", "weight": 0.3},
        ])
        findings = rm.check_targets(targets, {"max_single_weight": 0.5})

        weight_findings = [f for f in findings if f["rule"] == "single_weight_exceeded"]
        assert len(weight_findings) == 1
        assert "BTC" in weight_findings[0]["detail"]
        assert weight_findings[0]["level"] == "warn"

    # ----------------------------------------------------------------
    # 2. Multiple positions exceeding weight limit
    # ----------------------------------------------------------------
    def test_multiple_weights_exceeded(self, rm: FactorExposureRiskManager) -> None:
        targets = self._make_targets([
            {"symbol": "BTC", "weight": 0.6},
            {"symbol": "ETH", "weight": 0.55},
            {"symbol": "SOL", "weight": 0.1},
        ])
        findings = rm.check_targets(targets, {"max_single_weight": 0.5})

        weight_findings = [f for f in findings if f["rule"] == "single_weight_exceeded"]
        assert len(weight_findings) == 2

    # ----------------------------------------------------------------
    # 3. Sector concentration exceeded
    # ----------------------------------------------------------------
    def test_sector_concentration_exceeded(self, rm: FactorExposureRiskManager) -> None:
        targets = self._make_targets([
            {"symbol": "BTC", "weight": 0.5},
            {"symbol": "ETH", "weight": 0.4},
            {"symbol": "AAPL", "weight": 0.1},
        ])
        params = {
            "max_single_weight": 1.0,
            "sectors": {"BTC": "crypto", "ETH": "crypto", "AAPL": "tech"},
            "max_sector_weight": 0.5,
        }
        findings = rm.check_targets(targets, params)

        sector_findings = [f for f in findings if f["rule"] == "sector_concentration_exceeded"]
        assert len(sector_findings) == 1
        assert "crypto" in sector_findings[0]["detail"]
        assert sector_findings[0]["level"] == "warn"

    # ----------------------------------------------------------------
    # 4. Clean portfolio: no findings
    # ----------------------------------------------------------------
    def test_clean_portfolio(self, rm: FactorExposureRiskManager) -> None:
        targets = self._make_targets([
            {"symbol": "BTC", "weight": 0.3},
            {"symbol": "ETH", "weight": 0.3},
            {"symbol": "SOL", "weight": 0.2},
        ])
        findings = rm.check_targets(targets, {"max_single_weight": 0.5})
        assert findings == []

    # ----------------------------------------------------------------
    # 5. Clean portfolio with sectors
    # ----------------------------------------------------------------
    def test_clean_portfolio_with_sectors(self, rm: FactorExposureRiskManager) -> None:
        targets = self._make_targets([
            {"symbol": "BTC", "weight": 0.25},
            {"symbol": "AAPL", "weight": 0.25},
            {"symbol": "MSFT", "weight": 0.25},
            {"symbol": "GOOG", "weight": 0.25},
        ])
        params = {
            "max_single_weight": 0.5,
            "sectors": {"BTC": "crypto", "AAPL": "tech", "MSFT": "tech", "GOOG": "tech"},
            "max_sector_weight": 0.8,
        }
        findings = rm.check_targets(targets, params)
        assert findings == []

    # ----------------------------------------------------------------
    # 6. check_orders returns empty
    # ----------------------------------------------------------------
    def test_check_orders_returns_empty(self, rm: FactorExposureRiskManager) -> None:
        orders = pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 1, "price": 50000}])
        findings = rm.check_orders(orders, {})
        assert findings == []

    # ----------------------------------------------------------------
    # 7. Empty targets
    # ----------------------------------------------------------------
    def test_empty_targets(self, rm: FactorExposureRiskManager) -> None:
        targets = pd.DataFrame(columns=["symbol", "weight"])
        findings = rm.check_targets(targets, {})
        assert findings == []

    # ----------------------------------------------------------------
    # 8. Default max_single_weight
    # ----------------------------------------------------------------
    def test_default_max_single_weight(self, rm: FactorExposureRiskManager) -> None:
        targets = self._make_targets([
            {"symbol": "BTC", "weight": 0.6},
            {"symbol": "ETH", "weight": 0.4},
        ])
        # Default max_single_weight is 0.5
        findings = rm.check_targets(targets, {})
        weight_findings = [f for f in findings if f["rule"] == "single_weight_exceeded"]
        assert len(weight_findings) == 1
        assert "BTC" in weight_findings[0]["detail"]

    # ----------------------------------------------------------------
    # 9. Negative weights checked by absolute value
    # ----------------------------------------------------------------
    def test_negative_weight_absolute_check(self, rm: FactorExposureRiskManager) -> None:
        targets = self._make_targets([
            {"symbol": "BTC", "weight": -0.6},
            {"symbol": "ETH", "weight": 0.4},
        ])
        findings = rm.check_targets(targets, {"max_single_weight": 0.5})
        weight_findings = [f for f in findings if f["rule"] == "single_weight_exceeded"]
        assert len(weight_findings) == 1
        assert "BTC" in weight_findings[0]["detail"]

    # ----------------------------------------------------------------
    # 10. Meta correct
    # ----------------------------------------------------------------
    def test_meta_name(self) -> None:
        assert FactorExposureRiskManager.meta.name == "risk.factor_exposure.v1"

    def test_meta_kind(self) -> None:
        assert FactorExposureRiskManager.meta.kind == "risk"

    def test_meta_version(self) -> None:
        assert FactorExposureRiskManager.meta.version == "0.1.0"
