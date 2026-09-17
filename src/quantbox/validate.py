from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from .plugin_manifest import load_manifest, resolve_profile


@dataclass
class ValidationFinding:
    level: str  # "error" or "warning"
    message: str


def _check_legacy_dataset_params(cfg: dict) -> None:
    data = (cfg.get("plugins") or {}).get("data") or {}
    params = data.get("params_init") or {}
    has_legacy = "dataset_root" in params or "dataset" in params
    has_new = "dataset_id" in params
    if has_legacy and not has_new:
        warnings.warn(
            "config uses legacy dataset_root/dataset params; switch to dataset_id "
            "(see quantbox-qute/docs/decisions/0004-quantbox-dataset-plugin-tiers.md)",
            DeprecationWarning,
            stacklevel=2,
        )


def validate_config(cfg: dict[str, Any]) -> list[ValidationFinding]:
    _check_legacy_dataset_params(cfg)
    findings: list[ValidationFinding] = []
    for k in ("run", "artifacts", "plugins"):
        if k not in cfg:
            findings.append(ValidationFinding("error", f"missing_top_level_key:{k}"))
    if "run" in cfg:
        mode = cfg["run"].get("mode")
        if mode not in ("backtest", "paper", "live"):
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
        findings.extend(_check_backtest_execution(plugins))
    return findings


def _check_backtest_execution(plugins: dict[str, Any]) -> list[ValidationFinding]:
    """Validate ``execution:`` / ``venue:`` for backtest pipelines with the pipeline's own resolver."""
    from .execution import resolve_allow_shorts, resolve_lag_bars

    pipeline = plugins.get("pipeline")
    if not isinstance(pipeline, dict) or not str(pipeline.get("name", "")).startswith("backtest.pipeline."):
        return []
    params = pipeline.get("params") or {}
    findings: list[ValidationFinding] = []
    try:
        if resolve_lag_bars(params.get("execution")) == 0:
            findings.append(
                ValidationFinding(
                    "warning",
                    "execution.lag_bars=0: SAME-BAR fills (look-ahead for close-based signals); "
                    "use only to reproduce a historical number",
                )
            )
        resolve_allow_shorts(params.get("venue"), params.get("risk"))
    except ValueError as exc:
        findings.append(ValidationFinding("error", str(exc)))
    return findings
