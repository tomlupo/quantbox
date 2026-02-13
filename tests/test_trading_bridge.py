from pathlib import Path

import yaml

from quantbox.registry import PluginRegistry
from quantbox.runner import run_from_config


def test_trading_bridge(tmp_path):
    reg = PluginRegistry.discover()

    # 1) run fund selection
    cfg1 = yaml.safe_load(Path("configs/run_fund_selection.yaml").read_text(encoding="utf-8"))
    cfg1["artifacts"]["root"] = str(tmp_path)
    res1 = run_from_config(cfg1, reg)
    alloc_path = Path(tmp_path) / res1.run_id / "allocations.parquet"
    assert alloc_path.exists()

    # 2) run trading bridge consuming allocations
    cfg2 = yaml.safe_load(Path("configs/run_trade_from_allocations.yaml").read_text(encoding="utf-8"))
    cfg2["artifacts"]["root"] = str(tmp_path)
    cfg2["plugins"]["pipeline"]["params"]["allocations_path"] = str(alloc_path)
    cfg2["plugins"]["pipeline"]["params"]["instrument_map"] = "configs/instruments.yaml"
    cfg2["plugins"]["broker"]["name"] = "ibkr.paper.stub.v1"
    cfg2["plugins"]["pipeline"]["params"]["approval_required"] = False
    res2 = run_from_config(cfg2, reg)
    # validate manifest and key artifacts exist
    run_dir = Path(tmp_path) / res2.run_id
    assert (run_dir / "targets.parquet").exists()
    assert (run_dir / "orders.parquet").exists()
    assert (run_dir / "fills.parquet").exists()
    assert (run_dir / "portfolio_daily.parquet").exists()
    assert (run_dir / "run_manifest.json").exists()
