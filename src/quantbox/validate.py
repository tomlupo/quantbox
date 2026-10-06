from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from typing import Any

from .plugin_manifest import load_manifest, resolve_profile

#: Finding code of a block naming a plugin this environment does not register. ``validate``
#: and the runner refuse it alike (TOM-1528, TOM-1529).
UNKNOWN_PLUGIN = "unknown_plugin"


@dataclass
class ValidationFinding:
    level: str  # "error" or "warning"
    message: str
    #: Machine-readable kind, set where a consumer acts on it (``UNKNOWN_PLUGIN``); None otherwise.
    code: str | None = None
    #: What the finding is about, for a consumer that acts on it: for ``UNKNOWN_PLUGIN``,
    #: ``{"plugin_name", "group", "where"}``.
    subject: dict[str, str] | None = None


def _check_legacy_dataset_params(cfg: dict) -> None:
    data = (cfg.get("plugins") or {}).get("data") or {}
    params = data.get("params_init") or {}
    # A bare ``dataset`` name is current: rooted by $QUANTBOX_DATASETS_ROOT, pinned by
    # datasets.lock (TOM-1349). An inline root or inline sha is the deprecated alias — it
    # still runs. FutureWarning, not DeprecationWarning: Python hides the latter from a
    # CLI user by default, and this one is meant for the person who wrote the config.
    legacy = [k for k in ("dataset_root", "expected_prices_sha256") if k in params]
    has_new = "dataset_id" in params
    if legacy and not has_new:
        warnings.warn(
            f"config pins its dataset inline ({', '.join(legacy)}) — deprecated; name the dataset "
            "and pin it in datasets.lock (`quantbox-datasets pin <name>`), with the root from "
            "$QUANTBOX_DATASETS_ROOT; `quantbox dataset resolve <name> -c <config> --json` shows the result "
            "(see quantbox-qute/docs/decisions/0004-quantbox-dataset-plugin-tiers.md)",
            FutureWarning,
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
            findings.extend(_check_overlay_host(plugins, registry))
    return findings


def _check_overlay_host(plugins: dict[str, Any], registry: Any = None) -> list[ValidationFinding]:
    """``plugins.overlays`` needs a pipeline that applies them; the runner refuses the rest."""
    pipeline = plugins.get("pipeline")
    if not plugins.get("overlays") or not isinstance(pipeline, dict) or not pipeline.get("name"):
        return []
    if registry is None:
        from .registry import PluginRegistry

        try:
            registry = PluginRegistry.discover()
        except Exception:
            return []  # check_plugin_params already reported the registry failure
    cls = (getattr(registry, "pipelines", None) or {}).get(pipeline["name"])
    if cls is None or getattr(cls, "accepts_overlays", False):
        return []
    return [
        ValidationFinding(
            "error",
            f"plugins.overlays is set but pipeline '{pipeline['name']}' does not apply overlays "
            "(overlays run in backtest.pipeline.*)",
        )
    ]


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
    "overlays": ("overlay", True),
}


# Blocks whose ``params`` NOTHING reads (review of #218): the runner builds the data and
# broker plugins from ``params_init`` alone and never forwards their ``params``; a data
# plugin's load-time params reach load_universe() / load_market_data() only through
# ``plugins.pipeline.params.universe`` / ``.prices``. Any key here is refused, valid or not.
_UNREAD_PARAMS: dict[str, str] = {
    "plugins.data": (
        "plugins.data.params is never read; constructor params go under params_init, "
        "load-time params under plugins.pipeline.params.universe (load_universe) "
        "or plugins.pipeline.params.prices (load_market_data)"
    ),
    "plugins.broker": "plugins.broker.params is never read; set constructor params under params_init",
}


# Pseudo-group for ``pipeline.params.strategies``: there ``name`` is a module under
# ``quantbox.plugins.strategies`` whose module-level ``run(data, params)`` the backtest
# and trading pipelines call when ``plugins.strategies`` is absent.
_STRATEGY_MODULE = "strategy_module"
_STRATEGY_PKG = "quantbox.plugins.strategies"


def _plugin_blocks(plugins: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(where, group, block) for every named plugin block, backtest variants and
    ``pipeline.params.strategies`` included."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for slot, (group, is_list) in _PARAM_SLOTS.items():
        val = plugins.get(slot)
        items = (val or []) if is_list else [val]
        for i, block in enumerate(items):
            if isinstance(block, dict) and block.get("name"):
                out.append((f"plugins.{slot}[{i}]" if is_list else f"plugins.{slot}", group, block))
    pipeline = plugins.get("pipeline")
    pparams = ((pipeline or {}).get("params") or {}) if isinstance(pipeline, dict) else {}
    for i, v in enumerate(pparams.get("variants") or []):
        strat = v.get("strategy") if isinstance(v, dict) else None
        if isinstance(strat, dict) and strat.get("name"):
            out.append((f"plugins.pipeline.params.variants[{i}].strategy", "strategy", strat))
    for i, strat in enumerate(pparams.get("strategies") or []):
        if isinstance(strat, dict) and strat.get("name"):
            out.append((f"plugins.pipeline.params.strategies[{i}]", _STRATEGY_MODULE, strat))
    return out


def _resolve_block_plugin(registry: Any, group: str, name: str) -> tuple[Any, str, str]:
    """(plugin class, plugin name for messages, why it is unresolved) for one block."""
    from .params_schema import PLUGIN_GROUPS

    if group != _STRATEGY_MODULE:
        cls = (getattr(registry, PLUGIN_GROUPS[group], None) or {}).get(name)
        return cls, name, f"not a registered {group} plugin"
    module = f"{_STRATEGY_PKG}.{name}"
    for cls in (getattr(registry, PLUGIN_GROUPS["strategy"], None) or {}).values():
        if getattr(cls, "__module__", None) == module:
            return cls, cls.meta.name, ""
    return None, name, f"no registered strategy plugin in module {module}"


def _strategy_module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(f"{_STRATEGY_PKG}.{name}") is not None
    except (ImportError, ValueError):
        return False


def _unknown_plugin(registry: Any, where: str, group: str, name: str) -> ValidationFinding:
    """The error for a block that names a plugin this environment does not register (TOM-1528).

    ``run_from_config`` raises this same finding before any work (TOM-1529); validate used
    to pass it with a ``params_not_checked`` warning and exit 0.
    """
    import difflib

    from .params_schema import PLUGIN_GROUPS
    from .registry import ENTRYPOINT_GROUPS

    subject = {"plugin_name": name, "group": group, "where": where}
    if group == _STRATEGY_MODULE:
        return ValidationFinding(
            "error",
            f"{UNKNOWN_PLUGIN}: '{name}' ({where}) is not a module under {_STRATEGY_PKG}; "
            "the pipeline imports it by that name",
            UNKNOWN_PLUGIN,
            subject,
        )
    registered = sorted(getattr(registry, PLUGIN_GROUPS[group], None) or {})
    close = difflib.get_close_matches(name, registered, n=3, cutoff=0.6)
    closest = (
        f"closest registered: {', '.join(close)}" if close else "`quantbox plugins list` shows the registered names"
    )
    return ValidationFinding(
        "error",
        f"{UNKNOWN_PLUGIN}: '{name}' ({where}) is not a registered {group} plugin; {closest}. "
        f"A plugin from another package registers under the '{ENTRYPOINT_GROUPS[group]}' entry point: "
        "install that package in this environment (`uv add <package>`).",
        UNKNOWN_PLUGIN,
        subject,
    )


def check_plugin_params(plugins: dict[str, Any], registry: Any = None) -> list[ValidationFinding]:
    """Every key a config sets on a plugin must be a property of that plugin's params schema."""
    import difflib

    from .params_schema import check_params, config_fields, resolve_params_schema

    blocks = _plugin_blocks(plugins)
    if not blocks:
        return []
    if registry is None:
        from .registry import PluginRegistry

        try:
            registry = PluginRegistry.discover()
        except Exception as exc:
            # A run's discover() fails the same way, and nothing below can be checked (TOM-1528).
            return [
                ValidationFinding(
                    "error",
                    f"plugin registry failed to load ({exc}); no plugin was checked. An installed plugin "
                    "package fails to import: install its dependencies or remove it",
                )
            ]

    findings: list[ValidationFinding] = []
    for where, group, block in blocks:
        cls, name, unresolved = _resolve_block_plugin(registry, group, block["name"])
        if cls is None:
            if block.get("source"):
                # Local source: the runner loads the class from this file, not the registry,
                # and validate does not execute it. Its params go unchecked; the name is free.
                findings.append(
                    ValidationFinding("warning", f"params_not_checked:{name}: local-source plugin ({block['source']})")
                )
            elif group == _STRATEGY_MODULE and _strategy_module_exists(name):
                # The module exists and the pipeline can call its run(); only its params go unchecked.
                findings.append(ValidationFinding("warning", f"params_not_checked:{name}: {unresolved}"))
            else:
                findings.append(_unknown_plugin(registry, where, group, name))
            continue
        schema = resolve_params_schema(cls)
        if schema is None:
            findings.append(
                ValidationFinding("warning", f"params_not_checked:{name}: plugin declares no params_schema")
            )
            continue
        props = schema.get("properties", {})
        init_names = {p.name for p in config_fields(cls)}
        for channel in ("params_init", "params"):
            params = block.get(channel) or {}
            if not isinstance(params, dict):
                findings.append(ValidationFinding("error", f"{where}.{channel} must be a mapping ({name})"))
                continue
            unread = _UNREAD_PARAMS.get(where) if channel == "params" else None
            if unread:
                findings.extend(
                    ValidationFinding(
                        "error", f"unknown_param: '{key}' is set on plugin '{name}' ({where}.params); {unread}"
                    )
                    for key in params
                )
                continue
            allowed = init_names if channel == "params_init" else None
            unknown, violations = check_params(schema, params, allowed)
            for key in unknown:
                if key in props:
                    hint = "; it is a run-time param, set it under params"
                else:
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
    from .execution import check_schedule_venue, resolve_allow_shorts, resolve_execution
    from .financing import resolve_financing, resolve_leverage

    pipeline = plugins.get("pipeline")
    if not isinstance(pipeline, dict) or not str(pipeline.get("name", "")).startswith("backtest.pipeline."):
        return []
    params = pipeline.get("params") or {}
    findings: list[ValidationFinding] = []
    try:
        timing = resolve_execution(params.get("execution"))
        resolve_allow_shorts(params.get("venue"), params.get("risk"))
        venue = params.get("venue") if isinstance(params.get("venue"), dict) else {}
        resolve_financing(venue.get("financing"))
        resolve_leverage(venue.get("leverage"))
        check_schedule_venue(timing, params.get("venue"))
    except ValueError as exc:
        findings.append(ValidationFinding("error", str(exc)))
    return findings
