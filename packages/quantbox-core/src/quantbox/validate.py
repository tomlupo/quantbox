from __future__ import annotations
from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class ValidationFinding:
    level: str  # "error" or "warning"
    message: str

def validate_config(cfg: Dict[str, Any]) -> List[ValidationFinding]:
    findings: List[ValidationFinding] = []
    for k in ("run","artifacts","plugins"):
        if k not in cfg:
            findings.append(ValidationFinding("error", f"missing_top_level_key:{k}"))
    if "run" in cfg:
        mode = cfg["run"].get("mode")
        if mode not in ("backtest","paper","live"):
            findings.append(ValidationFinding("error", "run.mode must be backtest|paper|live"))
        if not cfg["run"].get("asof"):
            findings.append(ValidationFinding("error", "run.asof is required (YYYY-MM-DD)"))
    if "plugins" in cfg:
        if "pipeline" not in cfg["plugins"] or "data" not in cfg["plugins"]:
            findings.append(ValidationFinding("error", "plugins.pipeline and plugins.data are required"))
    return findings
