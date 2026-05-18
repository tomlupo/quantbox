"""Run every canonical reproduction and diff against the committed goldens.

User-facing entry point — no pytest required. Exits 0 on all-match, 1 on
any divergence with a per-metric diff printed.

Usage:
    uv run python cookbook/canonical/run_all.py
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = sorted((HERE / "configs").glob("*.yaml"))
EXPECTED_DIR = HERE / "expected"
ATOL = 1e-4
RTOL = 1e-3

HEADLINE_KEYS = (
    "total_return",
    "cagr",
    "sharpe",
    "max_drawdown",
    "annual_volatility",
)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def _run_one(config: Path) -> tuple[bool, dict[str, str]]:
    name = config.stem
    expected_path = EXPECTED_DIR / f"{name}.json"
    if not expected_path.exists():
        return False, {"error": f"no expected golden at {expected_path}"}

    expected = json.loads(expected_path.read_text())
    artifacts_root = HERE.parent.parent / "artifacts"
    if artifacts_root.exists():
        shutil.rmtree(artifacts_root)

    result = subprocess.run(
        ["uv", "run", "quantbox", "run", "-c", str(config)],
        cwd=HERE.parent.parent,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        return False, {"error": f"backtest failed: {result.stderr[-500:]}"}

    runs = sorted(artifacts_root.glob("*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        return False, {"error": "no artifact produced"}
    actual = json.loads((runs[-1] / "metrics.json").read_text())

    diffs: dict[str, str] = {}
    for key in HEADLINE_KEYS:
        if key not in expected or key not in actual:
            continue
        exp, act = expected[key], actual[key]
        if not _close(exp, act):
            diffs[key] = f"expected {exp:.6f}, got {act:.6f}"

    return len(diffs) == 0, diffs


def main() -> int:
    print(f"Running {len(CONFIGS)} canonical reproductions (atol={ATOL}, rtol={RTOL})\n")
    overall_ok = True
    for config in CONFIGS:
        ok, diffs = _run_one(config)
        if ok:
            print(f"  [PASS] {config.name}")
        else:
            overall_ok = False
            print(f"  [FAIL] {config.name}")
            for k, msg in diffs.items():
                print(f"         {k}: {msg}")
    print()
    if overall_ok:
        print("All canonical reproductions match the committed goldens.")
        return 0
    print("Some reproductions diverged. If the change is intentional, regenerate")
    print("the goldens via: uv run python cookbook/canonical/regen_goldens.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
