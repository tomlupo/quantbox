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
    # A bare ``dataset`` name is current (loaded by name, pinned by datasets.lock);
    # only the filesystem root is legacy.
    has_legacy = "dataset_root" in params
    has_new = "dataset_id" in params
    if has_legacy and not has_new:
        warnings.warn(
            "config uses legacy dataset_root param; switch to dataset_id or a pinned dataset name "
            "(see quantbox-qute/docs/decisions/0004-quantbox-dataset-plugin-tiers.md)",
            DeprecationWarning,
            stacklevel=2,
        )


def validate_config(cfg: dict[str, Any], registry: Any = None, *, check_params: bool = True) -> list[ValidationFinding]:
    """Findings for a run config.

    ``check_params`` checks every plugin block's ``params_init`` / ``params``
    against that plugin's params schema (see ``check_plugin_params``);
    ``registry`` defaults to ``PluginRegistry.discover()``.
    """
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
        if check_params:
            findings.extend(check_plugin_params(plugins, registry))
    return findings


# config slot -> (registry group, is a list of blocks)
_PARAM_SLOTS: dict[str, tuple[str, bool]] = {
    "pipeline": ("pipeline", False),
    "data": ("data", False),
    "broker": ("broker", False),
    "rebalancing": ("rebalancing", False),
    "aggregator": ("strategy", False),
    "strategies": ("strategy", True),
    "risk": ("risk", True),
    "publishers": ("publisher", True),
    "validation": ("validation", True),
    "monitors": ("monitor", True),
}


def _plugin_blocks(plugins: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(where, group, block) for every named plugin block, backtest variants included."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for slot, (group, is_list) in _PARAM_SLOTS.items():
        val = plugins.get(slot)
        items = (val or []) if is_list else [val]
        for i, block in enumerate(items):
            if isinstance(block, dict) and block.get("name"):
                out.append((f"plugins.{slot}[{i}]" if is_list else f"plugins.{slot}", group, block))
    pipeline = plugins.get("pipeline")
    variants = ((pipeline or {}).get("params") or {}).get("variants") if isinstance(pipeline, dict) else None
    for i, v in enumerate(variants or []):
        strat = v.get("strategy") if isinstance(v, dict) else None
        if isinstance(strat, dict) and strat.get("name"):
            out.append((f"plugins.pipeline.params.variants[{i}].strategy", "strategy", strat))
    return out


def check_plugin_params(plugins: dict[str, Any], registry: Any = None) -> list[ValidationFinding]:
    """Every key a config sets on a plugin must be a property of that plugin's params schema."""
    import difflib

    from .params_schema import PLUGIN_GROUPS, check_params, resolve_params_schema

    blocks = _plugin_blocks(plugins)
    if not blocks:
        return []
    if registry is None:
        from .registry import PluginRegistry

        try:
            registry = PluginRegistry.discover()
        except Exception as exc:
            return [ValidationFinding("warning", f"params_not_checked: plugin registry failed to load ({exc})")]

    findings: list[ValidationFinding] = []
    for where, group, block in blocks:
        name = block["name"]
        cls = (getattr(registry, PLUGIN_GROUPS[group], None) or {}).get(name)
        if cls is None:
            findings.append(ValidationFinding("warning", f"params_not_checked:{name}: not a registered {group} plugin"))
            continue
        schema = resolve_params_schema(cls)
        if schema is None:
            findings.append(
                ValidationFinding("warning", f"params_not_checked:{name}: plugin declares no params_schema")
            )
            continue
        props = schema.get("properties", {})
        for channel in ("params_init", "params"):
            params = block.get(channel) or {}
            if not isinstance(params, dict):
                findings.append(ValidationFinding("error", f"{where}.{channel} must be a mapping ({name})"))
                continue
            unknown, violations = check_params(schema, params)
            for key in unknown:
                close = difflib.get_close_matches(key, list(props), n=1)
                hint = f"; did you mean '{close[0]}'?" if close else ""
                findings.append(
                    ValidationFinding(
                        "error",
                        f"unknown_param: '{key}' is not a parameter of plugin '{name}' ({where}.{channel}){hint}",
                    )
                )
            for msg in violations:
                findings.append(ValidationFinding("error", f"invalid_param: plugin '{name}' ({where}.{channel}) {msg}"))
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
