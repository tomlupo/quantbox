"""Regenerate every canonical reproduction's expected/<name>.json from a fresh run.

Use this after an intentional change to strategy/pipeline semantics. The
test suite (and ``run_all.py``) then enforces the new baseline.

Usage:
    uv run python cookbook/canonical/regen_goldens.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = sorted((HERE / "configs").glob("*.yaml"))
EXPECTED_DIR = HERE / "expected"


def main() -> int:
    artifacts_root = HERE.parent.parent / "artifacts"
    for config in CONFIGS:
        name = config.stem
        out = EXPECTED_DIR / f"{name}.json"
        print(f"  regen {name} → {out}")
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
            print(f"  [FAIL] {name}: {result.stderr[-400:]}")
            return 1
        runs = sorted(artifacts_root.glob("*"), key=lambda p: p.stat().st_mtime)
        if not runs:
            print(f"  [FAIL] {name}: no artifact produced")
            return 1
        shutil.copyfile(runs[-1] / "metrics.json", out)
    print("\nAll goldens regenerated. Review the diff before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
