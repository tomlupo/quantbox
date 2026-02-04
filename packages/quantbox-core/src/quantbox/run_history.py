from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import json

def find_latest_run(artifacts_root: str | Path, pipeline_name: str) -> Optional[Tuple[str, Path]]:
    root = Path(artifacts_root)
    if not root.exists():
        return None

    candidates = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        man = d / "run_manifest.json"
        meta = d / "run_meta.json"
        info = man if man.exists() else meta if meta.exists() else None
        if not info:
            continue
        try:
            obj = json.loads(info.read_text(encoding="utf-8"))
            pname = obj.get("pipeline") or obj.get("pipeline_name")
            if pname == pipeline_name:
                candidates.append((d.stat().st_mtime, d.name, d))
        except Exception:
            continue

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, run_id, run_dir = candidates[0]
    return run_id, run_dir

def resolve_latest_artifact(artifacts_root: str | Path, pipeline_name: str, artifact_file: str) -> Path:
    found = find_latest_run(artifacts_root, pipeline_name)
    if not found:
        raise FileNotFoundError(f"No runs found for pipeline '{pipeline_name}' under {artifacts_root}")
    run_id, run_dir = found
    p = run_dir / artifact_file
    if not p.exists():
        raise FileNotFoundError(f"Latest run {run_id} for pipeline '{pipeline_name}' does not have artifact {artifact_file}")
    return p
