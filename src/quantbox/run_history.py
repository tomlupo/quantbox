from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

#: The UTC start timestamp that ends every run id (``asof__pipeline__cfghash__ts``).
RUN_TS_FORMAT = "%Y%m%dT%H%M%SZ"


def run_started_at(run_id: str) -> datetime | None:
    """When the run started, parsed from its id's last ``__`` segment; None if it carries none.

    The way to order runs in time. A run id sorts by asof, pipeline and config
    hash before its timestamp, so a name sort does not put the newest run last;
    a directory mtime moves on every copy or sync.
    """
    _, sep, ts = run_id.rpartition("__")
    if not sep:
        return None
    try:
        return datetime.strptime(ts, RUN_TS_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def find_latest_run(artifacts_root: str | Path, pipeline_name: str) -> tuple[str, Path] | None:
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
                started = run_started_at(d.name)
                # the run id's own timestamp; mtime only for a dir that carries none
                when = started.timestamp() if started else d.stat().st_mtime
                candidates.append((when, d.name, d))
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
        raise FileNotFoundError(
            f"Latest run {run_id} for pipeline '{pipeline_name}' does not have artifact {artifact_file}"
        )
    return p
