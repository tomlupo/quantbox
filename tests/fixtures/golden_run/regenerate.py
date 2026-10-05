"""Regenerate the golden ``quantbox/run@1`` run directory beside this file.

    uv run python tests/fixtures/golden_run/regenerate.py

A tiny deterministic rsims backtest (funding modelled). The run directory's
canonical files and its ``run_manifest.json`` are copied here with every
machine-specific absolute path replaced, so consumers can point reader tests at
a real manifest. Regenerate after a run@1 MINOR bump (a field added) and commit
the result; a MAJOR bump (run@2) gets its own golden directory.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from quantbox.registry import PluginRegistry
from quantbox.run_manifest import run_files, validate_run_manifest
from quantbox.runner import run_from_config

HERE = Path(__file__).resolve().parent


def _inputs(root: Path) -> tuple[Path, Path]:
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    a = 100.0 * np.cumprod(1 + np.random.default_rng(7).normal(0, 0.01, len(idx)))
    prices = pd.DataFrame({"A": a, "USD": 100.0}, index=idx)
    funding = pd.DataFrame({"A": 0.0001, "USD": 0.0}, index=idx)
    paths = []
    for name, frame, col in (("prices", prices, "close"), ("funding", funding, "rate")):
        path = root / f"{name}.parquet"
        frame.rename_axis("date").reset_index().melt("date", var_name="symbol", value_name=col).to_parquet(
            path, index=False
        )
        paths.append(path)
    return paths[0], paths[1]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prices_path, funding_path = _inputs(root)
        cfg = {
            "run": {"mode": "backtest", "asof": "2024-02-09", "pipeline": "backtest.pipeline.v1", "n_trials": 3},
            "artifacts": {"root": str(root / "artifacts")},
            "plugins": {
                "pipeline": {
                    "name": "backtest.pipeline.v1",
                    "params": {
                        "engine": "rsims",
                        "fees": 0.0,
                        "venue": {"allow_shorts": False},
                        "risk": {"max_leverage": 1.5},
                        "universe": {"symbols": ["A", "USD"]},
                    },
                },
                "strategies": [
                    {
                        "name": "strategy.static_weights.v1",
                        "weight": 1.0,
                        "params_init": {"weights": {"A": 1.0}},
                    }
                ],
                "data": {
                    "name": "local_file_data",
                    "params_init": {"prices_path": str(prices_path), "funding_rates_path": str(funding_path)},
                },
            },
        }
        result = run_from_config(cfg, PluginRegistry.discover())
        run_dir = root / "artifacts" / result.run_id
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

        for old in HERE.iterdir():
            if old.is_file() and old.name != Path(__file__).name:  # not __pycache__/
                old.unlink()
        for rel in run_files(manifest).values():
            shutil.copy2(run_dir / rel, HERE / rel)

        # Strip what only this machine knows; a golden must not carry it.
        manifest["artifacts"] = {k: Path(v).name for k, v in manifest["artifacts"].items()}
        manifest["funding"]["source_path"] = "funding.parquet"
        manifest["git"] = {"repo_root": None, "branch": None, "commit": "0" * 40, "dirty": False}
        manifest["installed_packages"] = {}
        problems = validate_run_manifest(manifest)
        if problems:
            raise SystemExit(f"golden manifest does not validate: {problems}")
        (HERE / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {HERE}")


if __name__ == "__main__":
    main()
