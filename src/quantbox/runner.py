from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quantbox.parquet_io import read_parquet

from . import run_manifest as _run_manifest
from .contracts import (
    BrokerPlugin,
    DataPlugin,
    Mode,
    PipelinePlugin,
    PublisherPlugin,
    RebalancingPlugin,
    RiskPlugin,
    RunResult,
    StrategyPlugin,
)
from .exceptions import ConfigValidationError, PluginNotFoundError
from .execution import run_record
from .llm_utils import event_line, load_schema, validate_table
from .params_schema import PLUGIN_GROUPS
from .plugin_manifest import load_manifest, resolve_profile
from .run_history import RUN_TS_FORMAT
from .run_manifest import _sha256_file
from .store import FileArtifactStore
from .strict import get_capability
from .validate import UNKNOWN_PLUGIN, check_plugin_params, validate_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local-source plugin loading
# ---------------------------------------------------------------------------
#
# A plugin spec in YAML can take two forms:
#
#   - Registered:    {"name": "lab.strategy.regime_taa.v1", ...}
#                    Looked up in the entry-point registry.
#
#   - Local-source:  {"source": "research/regime-taa/strategy.py:RegimeTaa", ...}
#                    Imported from a local file at runtime; no package needed.
#
# Local-source is the scratch-plugin escape hatch — it lets LLM-authored or
# one-shot research code participate in a normal pipeline without going through
# package release. See:
#   - docs/architecture/plugin-authoring.md (registration paths)
#   - docs/architecture/skills.md (capability-gap branch)
#   - docs/adr/0003-autoresearch-as-driver-not-runtime.md (where this fits)
#
# Safety rails:
#   - Local-source is REFUSED for broker plugins (arbitrary order-submitting code).
#   - Local-source is REFUSED in paper/live mode regardless of plugin kind.
#   - Loaded classes must declare a ``meta`` attribute; ``meta.status`` defaults
#     to "research" — production runs (``--strict``) should reject these.
#
# Local-source is allowed for: strategy, data, feature, validation, monitor,
# rebalancing, risk, aggregator, overlay. Not allowed for: broker, pipeline.


_LOCAL_SOURCE_FORBIDDEN_KINDS: frozenset[str] = frozenset({"broker", "pipeline"})


def _load_local_source_class(source: str, expected_kind: str | None = None) -> type:
    """Load a plugin class from a local file path.

    Args:
        source: ``"path/to/file.py:ClassName"`` (path may be relative to cwd
            or absolute; resolved at load time).
        expected_kind: Expected ``meta.kind`` value (e.g. ``"strategy"``).
            If provided and the loaded class's ``meta.kind`` mismatches, raises.

    Returns:
        The plugin class (not an instance). Caller instantiates as usual.

    Raises:
        ValueError: malformed source string, or ``meta.kind`` mismatch.
        FileNotFoundError: source file does not exist.
        AttributeError: class not found in module, or class has no ``meta`` attr.
    """
    if ":" not in source:
        raise ValueError(
            f"Local source must be 'path:ClassName', got: {source!r}. "
            f"Example: 'research/regime-taa/strategy.py:RegimeTaa'"
        )
    file_part, class_name = source.rsplit(":", 1)
    path = Path(file_part).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Plugin source file not found: {path}")

    # Use a unique-ish module name to avoid sys.modules collisions across runs.
    module_name = f"_quantbox_localsource__{path.stem}__{abs(hash(str(path))) & 0xFFFFFFFF:x}"
    spec_obj = importlib.util.spec_from_file_location(module_name, path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Cannot create module spec for: {path}")
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"Class {class_name!r} not found in {path}")

    meta = getattr(cls, "meta", None)
    if meta is None:
        raise AttributeError(
            f"Plugin class {class_name!r} in {path} has no 'meta' attribute. "
            f"Local-source plugins must declare ``meta = PluginMeta(...)`` like any other plugin."
        )
    if expected_kind is not None:
        actual_kind = getattr(meta, "kind", None)
        if actual_kind != expected_kind:
            raise ValueError(
                f"Plugin class {class_name!r} has meta.kind={actual_kind!r}, "
                f"but config block requires meta.kind={expected_kind!r}"
            )
    return cls


def _resolve_plugin_cls(
    spec: dict[str, Any],
    registry_dict: dict[str, type],
    kind: str,
    *,
    mode: Mode,
):
    """Resolve a plugin class from a YAML spec — either registry name or local source.

    Args:
        spec: The plugin block dict from YAML (e.g. ``{"name": "..."}``
            or ``{"source": "path:Class"}``).
        registry_dict: The relevant registry slot (e.g. ``registry.strategies``).
        kind: Expected plugin kind (e.g. ``"strategy"``); used for safety check
            and for ``meta.kind`` validation when local-source is used.
        mode: Run mode. Local-source is refused in paper/live mode.

    Returns:
        Plugin class, ready to instantiate with ``cls(**params_init)``.
    """
    if "source" in spec:
        if kind in _LOCAL_SOURCE_FORBIDDEN_KINDS:
            raise ValueError(
                f"Local-source plugins are forbidden for kind={kind!r} (safety rail). "
                f"Use a registered entry-point instead."
            )
        if mode in ("paper", "live"):
            raise ValueError(
                f"Local-source plugins are forbidden in mode={mode!r} (safety rail). "
                f"Use a registered entry-point for production runs; local-source is research-only."
            )
        return _load_local_source_class(spec["source"], expected_kind=kind)

    name = spec.get("name")
    if name is None:
        raise ValueError(f"Plugin spec must have either 'name' (registered) or 'source' (local), got: {spec!r}")
    if name not in registry_dict:
        raise PluginNotFoundError(name, kind, list(registry_dict.keys()))
    return registry_dict[name]


def _hash_config(cfg: dict[str, Any]) -> str:
    b = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:12]


def _hash_config_full(cfg: dict[str, Any]) -> str:
    b = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _git_value(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return out or None


def _git_info(cwd: Path | None = None) -> dict[str, Any]:
    root = cwd or Path.cwd()
    return {
        "repo_root": _git_value(["rev-parse", "--show-toplevel"], root),
        "branch": _git_value(["branch", "--show-current"], root),
        "commit": _git_value(["rev-parse", "HEAD"], root),
        "dirty": _git_value(["status", "--porcelain"], root) not in (None, ""),
    }


_TRACKED_PACKAGES: tuple[str, ...] = (
    "quantbox",
    "quantbox-datasets",
    "quantbox-lab",
    "quantbox-live",
)


def _installed_packages(names: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, dict[str, Any]]:
    """Capture installed-package provenance for the named distributions.

    For each package, returns version + (where available) the PEP 610
    ``direct_url.json`` data, which gives the git commit SHA + URL for VCS
    installs and the editable flag for ``pip install -e`` installs. This
    closes the audit gap where ``run_manifest.json`` only recorded version
    strings (e.g. ``0.2.0``) but not which git commit produced that build.
    """
    import importlib.metadata as md

    out: dict[str, dict[str, Any]] = {}
    for name in names:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        info: dict[str, Any] = {"version": dist.version}
        try:
            durl_str = dist.read_text("direct_url.json")
        except FileNotFoundError:
            durl_str = None
        if durl_str:
            try:
                durl = json.loads(durl_str)
            except (ValueError, TypeError):
                durl = {}
            if "url" in durl:
                info["url"] = durl["url"]
            vcs = durl.get("vcs_info") or {}
            if vcs:
                info["vcs"] = vcs.get("vcs")
                if "commit_id" in vcs:
                    info["commit_id"] = vcs["commit_id"]
                if "requested_revision" in vcs:
                    info["requested_revision"] = vcs["requested_revision"]
            if durl.get("dir_info", {}).get("editable"):
                info["editable"] = True
        out[name] = info
    return out


def _plugin_meta(plugin: Any, fallback_name: str | None = None) -> dict[str, Any] | None:
    if plugin is None and fallback_name is None:
        return None
    meta = getattr(plugin, "meta", None)
    return {
        "name": getattr(meta, "name", fallback_name),
        "kind": getattr(meta, "kind", None),
        "version": getattr(meta, "version", None),
        "schema_version": getattr(meta, "schema_version", None),
        "core_compat": getattr(meta, "core_compat", None),
    }


def _bind_dataset_lock(data: Any, config_path: str | Path | None) -> None:
    """Pin a by-name dataset with the datasets.lock nearest the CONFIG, not the cwd.

    The same config run from the lab root or from a worktree then reads the same lock
    and so the same build (TOM-1349). With no lock above the config, the plugin falls
    back to the nearest one above the cwd, as before.
    """
    if config_path is None or not getattr(data, "dataset", None) or getattr(data, "dataset_lock", "") is not None:
        return
    from .dataset_lock import lock_for_config

    lock = lock_for_config(config_path)
    if lock is not None:
        data.dataset_lock = str(lock)


def _dataset_block(data: Any) -> dict[str, Any]:
    """Return the typed dataset evidence block for run_manifest.json.

    Accepts a DataPlugin. A dataset read by name records its resolution (tier
    ``lock``) — exactly what ``quantbox dataset resolve -c <config> --json`` prints. If the
    DataPlugin exposes ``.resolve()`` returning a DatasetPlugin (Tier 1+),
    evidence is read from it. Otherwise (Tier 0) a raw marker is emitted.
    """
    resolution = getattr(data, "dataset_resolution", None)
    if isinstance(resolution, dict):
        if resolution["sha256"]:
            return {"tier": "lock", "id": resolution["name"], "resolved": resolution}
        return {"tier": "raw", "warning": "dataset not pinned in datasets.lock", "resolved": resolution}
    plugin = None
    if hasattr(data, "resolve"):
        try:
            plugin = data.resolve()
        except Exception:
            plugin = None
    if plugin is None and hasattr(data, "dataset_id") and hasattr(data, "manifest_hash"):
        plugin = data  # caller passed a DatasetPlugin directly (used by tests)

    if plugin is None:
        return {"tier": "raw", "warning": "no dataset plugin used"}
    try:
        m = plugin.manifest()
    except Exception as exc:
        return {
            "tier": "plugin",
            "id": getattr(plugin, "dataset_id", None),
            "warning": f"manifest_error:{exc}",
        }
    return {
        "tier": "plugin",
        "id": plugin.dataset_id,
        "version": plugin.dataset_version,
        "plugin_name": getattr(getattr(plugin, "meta", None), "name", None),
        "manifest": {
            "format": "yaml",
            "sha256": plugin.manifest_hash(),
            "name": m.name,
            "version": m.version,
            "date_range": dict(m.date_range),
            "symbols_count": m.symbols_count,
            "data_fields": list(m.data_fields),
        },
        "coverage_report": {"present": plugin.coverage_report() is not None},
        "capabilities_declared": list(plugin.capabilities),
    }


def _run_capability_checks(data: Any, run_ctx: Any) -> dict[str, dict[str, Any]]:
    plugin = None
    if hasattr(data, "resolve"):
        try:
            plugin = data.resolve()
        except Exception:
            plugin = None
    if plugin is None and hasattr(data, "capabilities"):
        plugin = data
    if plugin is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for cap in getattr(plugin, "capabilities", ()) or ():
        chk = get_capability(cap)
        if chk is None:
            out[cap] = {"passed": False, "message": "unknown_capability"}
            continue
        r = chk.check(plugin, run_ctx)
        out[cap] = {"passed": r.passed, "details": dict(r.details), "message": r.message}
    return out


def _run_id(asof: str, pipeline_name: str, cfg_hash: str) -> str:
    ts = datetime.now(timezone.utc).strftime(RUN_TS_FORMAT)
    safe = pipeline_name.replace(".", "_")
    return f"{asof}__{safe}__{cfg_hash}__{ts}"


def _lookup(registry_dict: dict[str, type], name: str, group: str) -> type:
    if name not in registry_dict:
        raise PluginNotFoundError(name, group, list(registry_dict.keys()))
    return registry_dict[name]


def prepare_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill plugin blocks from ``plugins.profile`` (IN PLACE) and refuse an invalid config.

    The first step of :func:`run_from_config` and of ``quantbox config explain``:
    both hash and resolve the config as it stands after this.
    """
    cfg["run"]  # noqa: B018 — a config without `run` is a KeyError before anything else, as always
    if "plugins" in cfg and cfg["plugins"].get("profile"):
        profile_name = str(cfg["plugins"]["profile"])
        prof = resolve_profile(profile_name, load_manifest())
        if prof:
            # Fill missing plugin blocks from profile, without overwriting explicit config
            for key in ("pipeline", "data", "broker", "publishers", "risk"):
                if key in prof and key not in cfg["plugins"]:
                    cfg["plugins"][key] = prof[key]
    # Basic config validation (LLM-friendly)
    findings = validate_config(cfg, check_params=False)
    if any(f.level == "error" for f in findings):
        msgs = "; ".join(f.message for f in findings)
        raise ConfigValidationError(f"config_validation_failed: {msgs}", findings=findings)
    return cfg


@dataclass
class ResolvedRun:
    """The plugins a run of a config uses, instantiated, and the params its pipeline gets.

    Built by :func:`resolve_run`, the one place a config becomes plugins — shared by
    :func:`run_from_config` and ``quantbox config explain``.
    """

    mode: Mode
    asof: str
    pipeline_key: str
    pipe_name: str
    data_name: str
    pipeline: PipelinePlugin
    data: DataPlugin
    broker_block: dict[str, Any] | None
    broker_cls: type | None  # set only when the run instantiates a broker (trading, paper/live)
    risk_plugins: list[RiskPlugin]
    strategy_plugins: list[StrategyPlugin] | None
    aggregator: StrategyPlugin | None
    rebalancer: RebalancingPlugin | None
    pipeline_params: dict[str, Any]
    variant_plugins: dict[str, Any]
    #: ``(overlay plugin, params)`` per ``plugins.overlays`` entry, in config order (ADR-0004).
    overlay_chain: list[tuple[Any, dict[str, Any]]]
    #: ``(block, plugin class)`` per ``plugins.validation`` / ``plugins.monitors`` entry. Resolved
    #: here, before any work, though they run after the pipeline (TOM-1529): a block the run
    #: cannot load fails the run up front, never after the book has traded.
    validations: list[tuple[dict[str, Any], type]] = field(default_factory=list)
    monitors: list[tuple[dict[str, Any], type]] = field(default_factory=list)


def plugin_refs(cfg: dict[str, Any]) -> list[tuple[str, str, str, dict[str, Any]]]:
    """``(role, group, registry attribute, spec)`` for every plugin a prepared config names.

    Includes the publishers, which a run resolves only AFTER the pipeline, and the
    validation and monitor plugins, which run after it, so a pre-flight can check
    every id up front.
    """
    plugins = cfg.get("plugins") or {}
    refs: list[tuple[str, str, str, dict[str, Any]]] = []

    def add(role: str, group: str, attr: str, spec: Any) -> None:
        if isinstance(spec, dict):
            refs.append((role, group, attr, spec))

    add("pipeline", "pipeline", "pipelines", plugins.get("pipeline"))
    add("data", "data", "data", plugins.get("data"))
    add("broker", "broker", "brokers", plugins.get("broker"))
    for i, r in enumerate(plugins.get("risk") or []):
        add(f"risk[{i}]", "risk", "risk", r)
    for i, s in enumerate(plugins.get("strategies") or []):
        add(f"strategies[{i}]", "strategy", "strategies", s)
    add("aggregator", "strategy", "strategies", plugins.get("aggregator"))
    add("rebalancing", "rebalancing", "rebalancing", plugins.get("rebalancing"))
    for i, o in enumerate(plugins.get("overlays") or []):
        add(f"overlays[{i}]", "overlay", "overlays", o)
    for v in ((plugins.get("pipeline") or {}).get("params") or {}).get("variants") or []:
        strat = v.get("strategy") or {}
        if not isinstance(strat, dict):
            strat = {"name": strat}
        # As resolve_run reads it: a `source:` (TOM-1363) wins over a name.
        spec = {"source": strat["source"]} if strat.get("source") else {"name": strat.get("name")}
        add(f"variants[{v.get('name')}]", "strategy", "strategies", spec)
    for i, p in enumerate(plugins.get("publishers") or []):
        add(f"publishers[{i}]", "publisher", "publishers", p)
    for i, v in enumerate(plugins.get("validation") or []):
        add(f"validation[{i}]", "validation", "validations", v)
    for i, m in enumerate(plugins.get("monitors") or []):
        add(f"monitors[{i}]", "monitor", "monitors", m)
    return refs


def resolve_run(
    cfg: dict[str, Any],
    registry,
    *,
    config_path: str | Path | None = None,
) -> ResolvedRun:
    """Resolve and instantiate the plugins of a PREPARED config (:func:`prepare_config`).

    Touches no artifact store, loads no data and contacts no broker: the broker
    CLASS is resolved here and :func:`run_from_config` instantiates it.
    """
    run_cfg = cfg["run"]
    mode: Mode = run_cfg["mode"]
    plugins = cfg["plugins"]

    pipe_name = plugins["pipeline"]["name"]
    pipeline_cls = _lookup(registry.pipelines, pipe_name, "pipeline")
    pipeline: PipelinePlugin = pipeline_cls(**plugins["pipeline"].get("params_init", {}))

    data_name = plugins["data"]["name"]
    data_cls = _lookup(registry.data, data_name, "data")
    data: DataPlugin = data_cls(**plugins["data"].get("params_init", {}))
    _bind_dataset_lock(data, config_path)

    broker_cls: type | None = None
    broker_block = plugins.get("broker")
    if broker_block and getattr(pipeline, "kind", None) == "trading" and mode in ("paper", "live"):
        broker_cls = _lookup(registry.brokers, broker_block["name"], "broker")

    risk_plugins: list[RiskPlugin] = []
    for r in plugins.get("risk", []) or []:
        risk_cls = _lookup(registry.risk, r["name"], "risk")
        risk_plugins.append(risk_cls(**r.get("params_init", {})))

    # --- Strategy plugins (registered or local-source) ---
    strategy_plugins: list[StrategyPlugin] | None = None
    strategies_cfg = plugins.get("strategies", [])
    if strategies_cfg:
        strategy_plugins = []
        named_cfg = []
        for s in strategies_cfg:
            cls = _resolve_plugin_cls(s, registry.strategies, "strategy", mode=mode)
            strategy_plugins.append(cls(**s.get("params_init", {})))
            # The pipeline keys strategies by name; a `source:` block may omit it,
            # so it carries the loaded class's meta.name (the config is not mutated).
            named_cfg.append(s if s.get("name") else {**s, "name": cls.meta.name})
        strategies_cfg = named_cfg

    # --- Aggregator (it's a strategy plugin) ---
    aggregator: StrategyPlugin | None = None
    agg_cfg = plugins.get("aggregator")
    if agg_cfg:
        agg_cls = _lookup(registry.strategies, agg_cfg["name"], "strategy")
        aggregator = agg_cls(**agg_cfg.get("params_init", {}))

    # --- Rebalancer ---
    rebalancer: RebalancingPlugin | None = None
    rebal_cfg = plugins.get("rebalancing")
    if rebal_cfg:
        rebal_cls = _lookup(registry.rebalancing, rebal_cfg["name"], "rebalancing")
        rebalancer = rebal_cls(**rebal_cfg.get("params_init", {}))

    # --- Overlay chain (registered or local-source), applied in config order (ADR-0004) ---
    overlay_chain: list[tuple[Any, dict[str, Any]]] = []
    overlays_cfg = plugins.get("overlays") or []
    if overlays_cfg and not getattr(pipeline, "accepts_overlays", False):
        raise ValueError(
            f"plugins.overlays is set but pipeline {pipe_name!r} does not apply overlays; "
            "they would be silently dropped (overlays run in backtest.pipeline.*)"
        )
    for o in overlays_cfg:
        cls = _resolve_plugin_cls(o, registry.overlays, "overlay", mode=mode)
        overlay_chain.append((cls(**o.get("params_init", {})), dict(o.get("params") or {})))

    # Build pipeline params, merging in strategy/aggregator/rebalancer config
    pipeline_params = dict(plugins["pipeline"].get("params", {}))
    if strategies_cfg:
        pipeline_params["_strategies_cfg"] = strategies_cfg
    if agg_cfg:
        pipeline_params["_aggregator_cfg"] = agg_cfg
    if rebal_cfg:
        pipeline_params["_rebalancer_cfg"] = rebal_cfg
    risk_cfg_list = plugins.get("risk", []) or []
    if risk_cfg_list:
        merged_risk_params: dict[str, Any] = {}
        for r in risk_cfg_list:
            merged_risk_params.update(r.get("params", {}))
        pipeline_params["_risk_cfg"] = merged_risk_params

    # --- Variant strategies (multi-variant pipelines) ---
    variant_plugins: dict[str, Any] = {}
    variants_cfg = pipeline_params.get("variants") or []
    for v in variants_cfg:
        vname = str(v["name"])
        strat_cfg = v.get("strategy") or {}
        spec = strat_cfg if isinstance(strat_cfg, dict) else {"name": str(strat_cfg)}
        if not (spec.get("name") or spec.get("source")):
            raise ValueError(f"Variant {vname!r}: missing strategy.name or strategy.source")
        spec = {"source": spec["source"]} if spec.get("source") else {"name": spec["name"]}
        cls = _resolve_plugin_cls(spec, registry.strategies, "strategy", mode=mode)
        params_init = strat_cfg.get("params_init", {}) if isinstance(strat_cfg, dict) else {}
        variant_plugins[vname] = cls(**params_init)

    # Registered or local-source; local source is refused in paper/live (the safety rail).
    validations = [
        (v, _resolve_plugin_cls(v, registry.validations, "validation", mode=mode))
        for v in plugins.get("validation") or []
    ]
    monitors = [
        (m, _resolve_plugin_cls(m, registry.monitors, "monitor", mode=mode)) for m in plugins.get("monitors") or []
    ]

    return ResolvedRun(
        mode=mode,
        asof=run_cfg["asof"],
        # Optional: validate never required it and `run --dry-run` names plugins.pipeline.name (TOM-1526).
        pipeline_key=run_cfg.get("pipeline") or pipe_name,
        pipe_name=pipe_name,
        data_name=data_name,
        pipeline=pipeline,
        data=data,
        broker_block=broker_block,
        broker_cls=broker_cls,
        risk_plugins=risk_plugins,
        strategy_plugins=strategy_plugins,
        aggregator=aggregator,
        rebalancer=rebalancer,
        pipeline_params=pipeline_params,
        variant_plugins=variant_plugins,
        overlay_chain=overlay_chain,
        validations=validations,
        monitors=monitors,
    )


def strict_refusal(cfg: dict[str, Any], mode: str, dataset_tier: str | None) -> str | None:
    """Why ``run.strict`` (or a promotion run) refuses this config, or None.

    Refused: a same-bar RESEARCH run (docs/adr/0006), and a raw dataset tier.

    One check for :func:`run_from_config` and ``quantbox config explain``: the run
    raises it after writing its manifest, explain reports it before any work (TOM-1362).
    """
    if not (bool(cfg.get("run", {}).get("strict")) or mode == "promotion"):
        return None
    pipeline_params = ((cfg.get("plugins") or {}).get("pipeline") or {}).get("params") or {}
    same_bar = (pipeline_params.get("execution") or {}).get("same_bar") if isinstance(pipeline_params, dict) else None
    if isinstance(same_bar, dict) and same_bar.get("allow") is True:
        return (
            "strict mode refuses a same-bar run: execution.same_bar makes it a RESEARCH run, not a "
            "backtest (docs/adr/0006-same-bar-explicit-override.md)"
        )
    # "lock" (a by-name dataset verified against datasets.lock, TOM-1349) was "raw" before
    # it had a tier of its own; accepting it in strict mode is a separate decision.
    if dataset_tier in ("raw", "lock"):
        return (
            f"strict mode rejects Tier-0 raw ingest (dataset tier {dataset_tier!r}) — see "
            "quantbox-qute/docs/decisions/0004-quantbox-dataset-plugin-tiers.md"
        )
    return None


def _config_block(cfg: dict[str, Any], config_path: str | Path | None) -> dict[str, Any]:
    """The run@1 ``config`` block of a PREPARED config."""
    path = Path(config_path).resolve() if config_path is not None else None
    return {
        "path": str(path) if path else None,
        "sha256": _hash_config_full(cfg),
        "file_sha256": _sha256_file(path) if path else None,
        "git_blob_sha": _git_value(["hash-object", str(path)], Path.cwd()) if path else None,
    }


def _plugins_block(resolved: ResolvedRun, broker: Any = None) -> dict[str, Any]:
    """The run@1 ``plugins`` block; *broker* is the instance, when the run made one."""
    broker_name = None
    if resolved.broker_cls is not None:
        broker_name = getattr(getattr(broker or resolved.broker_cls, "meta", None), "name", None) or (
            resolved.broker_block or {}
        ).get("name")
    return {
        "pipeline": getattr(getattr(resolved.pipeline, "meta", None), "name", resolved.pipe_name),
        "data": getattr(getattr(resolved.data, "meta", None), "name", resolved.data_name),
        "broker": broker_name,
    }


def run_from_config(
    cfg: dict[str, Any],
    registry,
    *,
    config_path: str | Path | None = None,
) -> RunResult:
    prepare_config(cfg)
    # Plugin params (TOM-1350): `quantbox validate` REFUSES an unknown or invalid
    # param; a run only warns. Configs in use today carry keys their plugins have
    # always ignored silently, and turning that into a refusal here would halt a
    # live book on its next pin bump rather than at a deliberate migration.
    try:
        param_findings = check_plugin_params(cfg["plugins"], registry)
    except Exception as exc:  # a params check must never be what breaks a run
        logger.warning("config params: not checked (%s)", exc)
        param_findings = []
    # An unknown PLUGIN is refused, as validate refuses it (TOM-1529): validate's finding,
    # verbatim, before any work. A validation or monitor block used to be skipped with a
    # warning, so a typo silently dropped a check. Still a PluginNotFoundError, as before.
    unknown = [f for f in param_findings if f.code == UNKNOWN_PLUGIN]
    if unknown:
        first = unknown[0].subject or {}
        group = first.get("group", "")
        attr = PLUGIN_GROUPS.get(group)  # none for a pipeline.params.strategies module
        raise PluginNotFoundError(
            first.get("plugin_name", ""),
            group,
            sorted(getattr(registry, attr, None) or {}) if attr else [],
            message="; ".join(f.message for f in unknown),
        )
    for f in param_findings:
        logger.warning("config params: %s", f.message)
    n_trials = _run_manifest.n_trials(cfg)  # refuses a malformed value before any work
    resolved = resolve_run(cfg, registry, config_path=config_path)
    mode, asof, pipeline_key = resolved.mode, resolved.asof, resolved.pipeline_key
    pipe_name, data_name = resolved.pipe_name, resolved.data_name
    pipeline, data = resolved.pipeline, resolved.data
    broker_block = resolved.broker_block
    risk_plugins, strategy_plugins = resolved.risk_plugins, resolved.strategy_plugins
    aggregator, rebalancer = resolved.aggregator, resolved.rebalancer
    pipeline_params, variant_plugins = resolved.pipeline_params, resolved.variant_plugins
    overlay_chain = resolved.overlay_chain

    cfg_hash = _hash_config(cfg)
    cfg_hash_full = _hash_config_full(cfg)
    run_id = _run_id(asof, pipeline_key, cfg_hash)

    store = FileArtifactStore(cfg["artifacts"]["root"], run_id)
    store.append_event(event_line("RUN_START", run_id=run_id, asof=asof, mode=mode, pipeline=pipeline_key))

    broker: BrokerPlugin | None = None
    if resolved.broker_cls is not None:
        broker = resolved.broker_cls(**broker_block.get("params_init", {}))

    store.append_event(
        event_line(
            "PLUGINS_RESOLVED",
            pipeline=pipe_name,
            data=data_name,
            broker=(broker_block["name"] if broker_block else None),
        )
    )

    result = pipeline.run(
        mode=mode,
        asof=asof,
        params=pipeline_params,
        data=data,
        store=store,
        broker=broker,
        risk=risk_plugins,
        strategies=strategy_plugins,
        rebalancer=rebalancer,
        aggregator=aggregator,
        variant_plugins=variant_plugins or None,
        overlays=overlay_chain or None,
    )

    # --- Validation plugins (post-backtest) ---
    if resolved.validations and mode == "backtest":
        validation_results = []
        for v_cfg, v_cls in resolved.validations:
            v_name = v_cfg.get("name") or v_cls.meta.name
            v_plugin = v_cls(**v_cfg.get("params_init", {}))
            # Load returns and weights from artifacts
            returns_path = result.artifacts.get("returns", "")
            weights_path = result.artifacts.get("weights_history", "")
            returns_df = read_parquet(returns_path) if returns_path else pd.DataFrame()
            weights_df = read_parquet(weights_path) if weights_path else pd.DataFrame()
            benchmark_df = None
            v_result = v_plugin.validate(returns_df, weights_df, benchmark_df, v_cfg.get("params", {}))
            validation_results.append({"plugin": v_name, **v_result})
        if validation_results:
            validation_path = store.put_json("validation", validation_results)
            result.artifacts["validation"] = validation_path
            result.notes["validation"] = validation_results

    # --- Monitor plugins (paper/live) ---
    if resolved.monitors and mode in ("paper", "live"):
        all_alerts = []
        for m_cfg, m_cls in resolved.monitors:
            m_plugin = m_cls(**m_cfg.get("params_init", {}))
            alerts = m_plugin.check(result, None, m_cfg.get("params", {}))
            all_alerts.extend(alerts)
        if all_alerts:
            result.notes["monitor_alerts"] = all_alerts
            # Kill-switch: if any alert has action="halt", write halt file
            if any(a.get("action") == "halt" for a in all_alerts):
                halt_path = store.put_json("halt", {"reason": "monitor_halt", "alerts": all_alerts})
                result.artifacts["halt"] = halt_path
                logger.critical("HALT triggered by monitor alerts")

    store.put_json(
        "run_meta",
        {
            "run_id": result.run_id,
            "pipeline": result.pipeline_name,
            "mode": result.mode,
            "asof": result.asof,
            "config_hash": cfg_hash,
            "config_sha256": cfg_hash_full,
            "artifacts": result.artifacts,
            "metrics": result.metrics,
            "notes": result.notes,
        },
    )

    for p in cfg["plugins"].get("publishers", []) or []:
        pub_cls = _lookup(registry.publishers, p["name"], "publisher")
        pub: PublisherPlugin = pub_cls(**p.get("params_init", {}))
        pub.publish(result, p.get("params", {}))

    # LLM-friendly manifest (single file to understand the run)
    plugin_versions = {
        "pipeline": _plugin_meta(pipeline, pipe_name),
        "data": _plugin_meta(data, data_name),
        "broker": _plugin_meta(broker, broker_block["name"]) if broker_block and broker else None,
        "risk": [_plugin_meta(plugin) for plugin in risk_plugins],
        "strategies": [_plugin_meta(plugin) for plugin in strategy_plugins or []],
        "aggregator": _plugin_meta(aggregator) if aggregator else None,
        "rebalancer": _plugin_meta(rebalancer) if rebalancer else None,
        "overlays": [_plugin_meta(plugin) for plugin, _ in overlay_chain],
    }
    notes = result.notes or {}
    dataset = _dataset_block(data)
    dataset.update(_run_manifest.dataset_fields(data, dataset))
    manifest = {
        "schema": _run_manifest.SCHEMA_ID,
        "run_id": result.run_id,
        "asof": result.asof,
        "mode": result.mode,
        "pipeline": result.pipeline_name,
        "config_hash": cfg_hash,
        "config": _config_block(cfg, config_path),
        "git": _git_info(Path.cwd()),
        "installed_packages": _installed_packages(),
        "plugins": _plugins_block(resolved, broker),
        "plugin_versions": plugin_versions,
        "engine": _run_manifest.engine_block(notes),
        "dataset": dataset,
        "funding": _run_manifest.funding_block(data, notes),
        "n_trials": n_trials,
        "capability_results": _run_capability_checks(data, run_ctx=None),
        "artifacts": result.artifacts,
        "files": _run_manifest.files_block(result.artifacts or {}, store.root),
        "metrics": result.metrics,
        "warnings": [],
    }
    # Backtest pipelines state their execution timing and venue; a reader of
    # the manifest must never have to infer either (quantbox.execution).
    # ``overlays`` is the chain the pipeline APPLIED (name, version, params, in order),
    # reported by the pipeline itself — the traded_weights file is its output.
    # ``data_validation`` is the instrument-calendar summary; the full report is data_validation.json.
    for block in ("execution", "venue", "overlays", "data_validation"):
        if block in notes:
            manifest[block] = notes[block]
    if "execution" in manifest:
        # research (same-bar, docs/adr/0006) or backtest — what every reader of a result checks.
        manifest["run"] = run_record(manifest["execution"])

    # Validate artifacts against JSON schemas when available (best-effort)
    from importlib.resources import files as _res_files

    schema_dir = Path(str(_res_files("quantbox").joinpath("artifact_schemas")))
    for logical, path in (result.artifacts or {}).items():
        schema_path = schema_dir / f"{logical}.schema.json"
        if schema_path.exists() and path.endswith(".parquet"):
            try:
                df = read_parquet(path)
                schema = load_schema(schema_path)
                manifest["warnings"].extend([f"{logical}:{w}" for w in validate_table(df, schema)])
            except Exception as e:
                manifest["warnings"].append(f"{logical}:schema_check_error:{e}")

    strict_mode = bool(cfg.get("run", {}).get("strict")) or result.mode == "promotion"
    if strict_mode:
        refusal = strict_refusal(cfg, result.mode, manifest["dataset"]["tier"])
        if refusal:
            store.put_json("run_manifest", _run_manifest.json_safe(manifest))
            raise RuntimeError(refusal)
        failures = [c for c, r in manifest["capability_results"].items() if not r["passed"]]
        if failures:
            store.put_json("run_manifest", _run_manifest.json_safe(manifest))
            raise RuntimeError(f"strict mode capability failures: {failures}")

    store.put_json("run_manifest", _run_manifest.json_safe(manifest))
    if manifest.get("engine"):
        # The slim default report (TOM-1365): the run's qute-research/finding-report@1
        # data, read back through the manifest just written. A failed export does not fail
        # the run (its results are written; `quantbox report export` re-derives the report),
        # but it is never only a log line (TOM-1529): the manifest records that the report
        # was not produced and why, and `quantbox run` prints it in its summary.
        from .finding_export import FILENAME, write_finding_report

        try:
            write_finding_report(store.root)
            record: dict[str, Any] = {"produced": True, "file": FILENAME}
        except Exception as exc:
            logger.warning("finding_report.json export failed: %s", exc)
            record = {"produced": False, "error": f"{type(exc).__name__}: {exc}"}
            manifest["warnings"].append(f"finding_report:not_produced:{record['error']}")
        manifest["reports"] = {"finding_report": record}
        result.notes["reports"] = manifest["reports"]
        store.put_json("run_manifest", _run_manifest.json_safe(manifest))
    store.append_event(event_line("RUN_END", run_id=run_id, metrics=result.metrics, warnings=len(manifest["warnings"])))

    # Optional: ingest artifacts into warehouse
    wh_cfg = cfg.get("warehouse")
    if wh_cfg and wh_cfg.get("auto_ingest"):
        try:
            from .warehouse import Warehouse
            from .warehouse.ingestion import ingest_run

            with Warehouse(wh_cfg["root"], wh_cfg.get("database")) as wh:
                ingest_run(wh, store, tables=wh_cfg.get("ingest_tables"))
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("Warehouse auto-ingest failed: %s", exc)

    return result
