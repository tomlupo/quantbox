"""Canonical reproduction tests — diff each `cookbook/canonical/configs/*.yaml`
run against the committed golden in `cookbook/canonical/expected/*.json`.

Parametrized over every config in cookbook/canonical/configs/. Each test
runs the actual `quantbox run` pipeline end-to-end on the bundled
fixture parquet, then asserts headline metric equivalence within a
tight tolerance.

If a test fails, the change is either intentional (regen the golden via
`cookbook/canonical/regen_goldens.py`) or it's drift to investigate.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COOKBOOK = REPO_ROOT / "cookbook" / "canonical"
CONFIG_DIR = COOKBOOK / "configs"
EXPECTED_DIR = COOKBOOK / "expected"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"

ATOL = 1e-4
RTOL = 1e-3
HEADLINE_KEYS = (
    "total_return",
    "cagr",
    "sharpe",
    "max_drawdown",
    "annual_volatility",
)

CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.canonical_reproduction
@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_canonical_matches_golden(config: Path) -> None:
    expected_path = EXPECTED_DIR / f"{config.stem}.json"
    assert expected_path.exists(), f"no golden at {expected_path}. Run `cookbook/canonical/regen_goldens.py`."
    expected = json.loads(expected_path.read_text())

    if ARTIFACTS_ROOT.exists():
        shutil.rmtree(ARTIFACTS_ROOT)

    result = subprocess.run(
        ["uv", "run", "quantbox", "run", "-c", str(config)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"backtest failed:\n{result.stderr[-1000:]}"

    runs = sorted(ARTIFACTS_ROOT.glob("*"), key=lambda p: p.stat().st_mtime)
    assert runs, "no artifact directory produced"
    actual = json.loads((runs[-1] / "metrics.json").read_text())

    for key in HEADLINE_KEYS:
        if key not in expected:
            continue
        assert key in actual, f"{config.stem}: actual run missing metric {key}"
        exp, act = expected[key], actual[key]
        assert math.isclose(exp, act, rel_tol=RTOL, abs_tol=ATOL), (
            f"{config.stem}.{key}: expected {exp:.6f}, got {act:.6f} "
            f"(atol={ATOL}, rtol={RTOL}). If intentional, regen goldens."
        )
