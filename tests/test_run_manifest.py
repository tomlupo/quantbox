from pathlib import Path

import yaml

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


def test_manifest_includes_reproducibility_evidence(tmp_path):
    cfg_path = Path("configs/run_fund_selection.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["artifacts"]["root"] = str(tmp_path)
    reg = PluginRegistry.discover()

    res = run_from_config(cfg, reg, config_path=cfg_path)
    manifest_path = Path(tmp_path) / res.run_id / "run_manifest.json"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest["config"]["path"].endswith("configs/run_fund_selection.yaml")
    assert len(manifest["config"]["sha256"]) == 64
    assert manifest["config"]["git_blob_sha"]
    assert manifest["git"]["commit"]
    assert manifest["plugin_versions"]["pipeline"]["name"]
    assert "dataset" in manifest
    assert "tier" in manifest["dataset"]
    assert isinstance(manifest["warnings"], list)
