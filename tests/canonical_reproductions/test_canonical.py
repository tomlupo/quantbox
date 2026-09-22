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
SAME_BAR_EXPECTED_DIR = COOKBOOK / "expected_same_bar"  # frozen pre-lag goldens; never regenerated
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


def _run_and_compare(config: Path, expected_path: Path, label: str) -> None:
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

    compared = 0
    for key in HEADLINE_KEYS:
        if key not in expected:
            continue
        assert key in actual, f"{label}: actual run missing metric {key}"
        exp, act = expected[key], actual[key]
        assert math.isclose(exp, act, rel_tol=RTOL, abs_tol=ATOL), (
            f"{label}.{key}: expected {exp:.6f}, got {act:.6f} "
            f"(atol={ATOL}, rtol={RTOL}). If intentional, regen goldens."
        )
        compared += 1
    assert compared, f"{label}: golden {expected_path} carries none of {HEADLINE_KEYS} — nothing was compared"


@pytest.mark.canonical_reproduction
@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_canonical_matches_golden(config: Path) -> None:
    """The config as written — i.e. the DEFAULT execution timing (next-bar)."""
    _run_and_compare(config, EXPECTED_DIR / f"{config.stem}.json", config.stem)


@pytest.mark.canonical_reproduction
@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_same_bar_reproduces_the_pre_lag_goldens(config: Path, tmp_path: Path) -> None:
    """`execution.lag_bars: 0` reproduces the goldens committed BEFORE the default changed.

    `expected_same_bar/` is a frozen copy of those goldens and is never
    regenerated: it is the proof that a historical `quantbox run -c` number can
    still be reproduced on purpose.
    """
    import yaml

    cfg = yaml.safe_load(config.read_text())
    cfg["plugins"]["pipeline"]["params"]["execution"] = {"lag_bars": 0}
    same_bar = tmp_path / config.name
    same_bar.write_text(yaml.safe_dump(cfg))
    _run_and_compare(same_bar, SAME_BAR_EXPECTED_DIR / f"{config.stem}.json", f"{config.stem}[same-bar]")
