from quantbox.registry import PluginRegistry

def test_discovery():
    reg = PluginRegistry.discover()
    assert "fund_selection.simple.v1" in reg.pipelines
    assert "eod.duckdb_parquet.v1" in reg.data
