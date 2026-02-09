"""Inspect artifacts from past runs.

Shows how to list runs, open a specific run, and read its artifacts
using the FileArtifactStore API.

Usage:
    uv run python examples/03_inspect_artifacts.py [artifacts_root]
"""

import sys

from quantbox.store import FileArtifactStore

ARTIFACTS_ROOT = sys.argv[1] if len(sys.argv) > 1 else "./artifacts"

# 1. List recent runs
runs = FileArtifactStore.list_runs(ARTIFACTS_ROOT, limit=5)
if not runs:
    print(f"No runs found in {ARTIFACTS_ROOT}")
    print("Run 01_quickstart_backtest.py first to generate artifacts.")
    sys.exit(0)

print(f"Found {len(runs)} recent run(s):\n")
for run in runs:
    print(f"  {run.get('run_id', 'unknown'):60s}  mode={run.get('mode')}  asof={run.get('asof')}")

# 2. Open the latest run
latest = runs[0]
run_id = latest["run_id"]
store = FileArtifactStore.open_run(ARTIFACTS_ROOT, run_id)

print(f"\n--- Latest run: {run_id} ---")
print(f"Artifacts: {store.list_artifacts()}")

# 3. Read the manifest
manifest = store.get_manifest()
print(f"Pipeline:  {manifest.get('pipeline')}")
print(f"Metrics:   {manifest.get('metrics')}")
if manifest.get("warnings"):
    print(f"Warnings:  {manifest['warnings']}")

# 4. Read a parquet artifact (if available)
artifacts = store.list_artifacts()
for name in ["strategy_weights", "aggregated_weights", "prices", "universe"]:
    if name in artifacts:
        df = store.read_parquet(name)
        print(f"\n{name}: {df.shape[0]} rows x {df.shape[1]} cols")
        print(df.head(3).to_string())
        break
