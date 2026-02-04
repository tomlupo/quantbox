from quantbox.registry import PluginRegistry

def test_discovery():
    reg = PluginRegistry.discover()
    assert "fund_selection.simple.v1" in reg.pipelines
    assert "eod.duckdb_parquet.v1" in reg.data
    assert "trade.allocations_to_orders.v1" in reg.pipelines
    assert "ibkr.paper.stub.v1" in reg.brokers
    assert "binance.paper.stub.v1" in reg.brokers

    assert "ibkr.paper.v1" in reg.brokers
    assert "ibkr.live.v1" in reg.brokers
    assert "binance.paper.v1" in reg.brokers
    assert "binance.live.v1" in reg.brokers
