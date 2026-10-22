"""binance_live's own valuation never reads a missing mark as 0 (TOM-1860).

A held position with no usable mark is REFUSED by ``get_portfolio_value()`` and
``generate_rebalancing()`` (as ``KrakenBroker.get_equity`` refuses) and REPORTED
as unmarked by ``describe()``. Before TOM-1860 the first skipped it and the other
two valued it at 0, so the book was understated with nothing to say so.
"""

from __future__ import annotations

import json
import math

import pytest

from quantbox.plugins.broker.binance_live import BinanceLiveBroker
from quantbox.trade.book import REASON_UNPRICED, PortfolioValuationError


def _paper_broker(tmp_path, monkeypatch, holdings: dict[str, float], marks: dict[str, float]) -> BinanceLiveBroker:
    """A paper broker with no Binance client: an asset not in ``marks`` has no mark."""
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    broker = BinanceLiveBroker(paper_trading=True, account_name="t1860", trades_dir=str(tmp_path))
    account_dir = tmp_path / "t1860"
    account_dir.mkdir(parents=True, exist_ok=True)
    (account_dir / "paper_portfolio.json").write_text(json.dumps(holdings))
    for asset, price in marks.items():
        broker._price_cache.set(asset, price)
    return broker


def test_a_complete_book_is_cash_plus_marked_longs(tmp_path, monkeypatch):
    broker = _paper_broker(tmp_path, monkeypatch, {"USDC": 1000.0, "BTC": 0.5}, {"BTC": 20000.0})
    assert broker.get_portfolio_value() == pytest.approx(11000.0)
    desc = broker.describe()
    assert desc["portfolio_value"] == pytest.approx(11000.0)
    assert desc["unmarked"] == []


def test_portfolio_value_refuses_an_unmarkable_holding(tmp_path, monkeypatch):
    broker = _paper_broker(tmp_path, monkeypatch, {"USDC": 1000.0, "BTC": 0.5, "ETH": 4.0}, {"BTC": 20000.0})
    with pytest.raises(PortfolioValuationError, match="ETH") as excinfo:
        broker.get_portfolio_value()
    assert excinfo.value.reason == REASON_UNPRICED
    assert excinfo.value.unpriced == ("ETH",)


def test_portfolio_value_refuses_an_unmarkable_short(tmp_path, monkeypatch):
    broker = _paper_broker(tmp_path, monkeypatch, {"USDC": 1000.0, "SOL": -10.0}, {})
    with pytest.raises(PortfolioValuationError, match="SOL"):
        broker.get_portfolio_value()


def test_an_excluded_unmarkable_holding_is_not_refused(tmp_path, monkeypatch):
    broker = _paper_broker(tmp_path, monkeypatch, {"USDC": 1000.0, "ETHW": 3.0}, {})
    assert broker.get_portfolio_value() == pytest.approx(1000.0)


def test_rebalancing_refuses_an_unmarkable_holding(tmp_path, monkeypatch):
    broker = _paper_broker(tmp_path, monkeypatch, {"USDC": 1000.0, "BTC": 0.5, "ETH": 4.0}, {"BTC": 20000.0})
    with pytest.raises(PortfolioValuationError, match="ETH"):
        broker.generate_rebalancing({"BTC": 0.5})


def test_describe_reports_an_unmarkable_holding_never_at_zero(tmp_path, monkeypatch):
    broker = _paper_broker(tmp_path, monkeypatch, {"USDC": 1000.0, "BTC": 0.5, "ETH": 4.0}, {"BTC": 20000.0})
    desc = broker.describe()

    assert desc["unmarked"] == ["ETH"]
    # A total missing a position is unknown, not smaller.
    assert desc["portfolio_value"] is None
    by_symbol = {p["symbol"]: p for p in desc["positions"]}
    assert by_symbol["ETH"]["value"] is None
    assert by_symbol["ETH"]["weight"] is None
    assert by_symbol["ETH"]["marked"] is False
    assert by_symbol["BTC"]["value"] == pytest.approx(10000.0)
    assert by_symbol["BTC"]["marked"] is True


def test_an_unpriced_target_that_is_not_held_is_reported_never_silent(tmp_path, monkeypatch):
    """TOM-1979: a target with no price was sized to 0 and never bought, with nothing to say so.

    The book is still valued (the asset is not held), so the plan is not refused.
    The asset's row says ``Unpriced`` with an UNKNOWN target, and the orders frame
    carries it with ``Order_Status == "Unpriced"``, where a skipped order is reported.
    """
    broker = _paper_broker(tmp_path, monkeypatch, {"USDC": 1000.0, "BTC": 0.5}, {"BTC": 20000.0})
    broker.readonly = True

    plan = broker.generate_rebalancing({"BTC": 0.5, "SOL": 0.2}).set_index("Asset")
    assert plan.loc["SOL", "Action"] == "Unpriced"
    assert math.isnan(plan.loc["SOL", "Target_Qty"])
    assert math.isnan(plan.loc["SOL", "Delta_Qty"])
    assert plan.loc["BTC", "Action"] in {"Buy", "Sell", "Hold"}

    report = broker.execute_rebalancing({"BTC": 0.5, "SOL": 0.2})
    orders = report["orders"].set_index("Asset")
    assert orders.loc["SOL", "Order_Status"] == "Unpriced"
    assert not bool(orders.loc["SOL", "Executable"])
