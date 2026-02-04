from __future__ import annotations
from typing import Any, Dict, List
from dataclasses import dataclass
from .plugin_manifest import load_manifest, resolve_profile

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
        plugins = cfg["plugins"]
        has_pipeline = "pipeline" in plugins
        has_data = "data" in plugins
        profile = plugins.get("profile")

        if not has_pipeline or not has_data:
            if not profile:
                findings.append(ValidationFinding("error", "plugins.pipeline and plugins.data are required"))
            else:
                manifest = load_manifest()
                prof = resolve_profile(str(profile), manifest)
                if not prof:
                    findings.append(ValidationFinding("error", f"plugins.profile not found in manifest: {profile}"))
                else:
                    if "pipeline" not in prof or "data" not in prof:
                        findings.append(ValidationFinding("error", f"profile_missing_required_plugins:{profile}"))
    return findings
