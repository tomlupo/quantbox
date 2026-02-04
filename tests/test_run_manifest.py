import yaml
from pathlib import Path
from quantbox.registry import PluginRegistry
from quantbox.runner import run_from_config

def test_manifest_written(tmp_path):
    # Load sample config and redirect artifacts to tmp_path
    cfg = yaml.safe_load(Path("configs/run_fund_selection.yaml").read_text(encoding="utf-8"))
    cfg["artifacts"]["root"] = str(tmp_path)
    reg = PluginRegistry.discover()
    res = run_from_config(cfg, reg)
    manifest = Path(tmp_path) / res.run_id / "run_manifest.json"
    assert manifest.exists()
