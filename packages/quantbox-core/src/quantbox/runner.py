from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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
from .llm_utils import event_line, load_schema, validate_table
from .plugin_manifest import load_manifest, repo_root, resolve_profile
from .store import FileArtifactStore
from .strict import get_capability
from .validate import validate_config

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
# rebalancing, risk, aggregator. Not allowed for: broker, pipeline.


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


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


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


def _dataset_block(data: Any) -> dict[str, Any]:
    """Return the typed dataset evidence block for run_manifest.json.

    Accepts a DataPlugin. If the DataPlugin exposes ``.resolve()`` returning a
    DatasetPlugin (Tier 1+), evidence is read from it. Otherwise (Tier 0)
    a raw marker is emitted.
    """
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
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = pipeline_name.replace(".", "_")
    return f"{asof}__{safe}__{cfg_hash}__{ts}"


def run_from_config(
    cfg: dict[str, Any],
    registry,
    *,
    config_path: str | Path | None = None,
) -> RunResult:
    run_cfg = cfg["run"]
    if "plugins" in cfg and cfg["plugins"].get("profile"):
        profile_name = str(cfg["plugins"]["profile"])
        prof = resolve_profile(profile_name, load_manifest())
        if prof:
            # Fill missing plugin blocks from profile, without overwriting explicit config
            for key in ("pipeline", "data", "broker", "publishers", "risk"):
                if key in prof and key not in cfg["plugins"]:
                    cfg["plugins"][key] = prof[key]
    # Basic config validation (LLM-friendly)
    findings = validate_config(cfg)
    if any(f.level == "error" for f in findings):
        msgs = "; ".join(f.message for f in findings)
        raise ConfigValidationError(f"config_validation_failed: {msgs}", findings=findings)

    mode: Mode = run_cfg["mode"]
    asof: str = run_cfg["asof"]
    pipeline_key: str = run_cfg["pipeline"]

    cfg_hash = _hash_config(cfg)
    cfg_hash_full = _hash_config_full(cfg)
    run_id = _run_id(asof, pipeline_key, cfg_hash)

    store = FileArtifactStore(cfg["artifacts"]["root"], run_id)
    store.append_event(event_line("RUN_START", run_id=run_id, asof=asof, mode=mode, pipeline=pipeline_key))

    pipe_name = cfg["plugins"]["pipeline"]["name"]
    if pipe_name not in registry.pipelines:
        raise PluginNotFoundError(pipe_name, "pipeline", list(registry.pipelines.keys()))
    pipeline_cls = registry.pipelines[pipe_name]
    pipeline: PipelinePlugin = pipeline_cls(**cfg["plugins"]["pipeline"].get("params_init", {}))

    data_name = cfg["plugins"]["data"]["name"]
    if data_name not in registry.data:
        raise PluginNotFoundError(data_name, "data", list(registry.data.keys()))
    data_cls = registry.data[data_name]
    data: DataPlugin = data_cls(**cfg["plugins"]["data"].get("params_init", {}))

    broker: BrokerPlugin | None = None
    broker_block = cfg["plugins"].get("broker")
    if broker_block and getattr(pipeline, "kind", None) == "trading" and mode in ("paper", "live"):
        broker_cls = registry.brokers[broker_block["name"]]
        broker = broker_cls(**broker_block.get("params_init", {}))

    store.append_event(
        event_line(
            "PLUGINS_RESOLVED",
            pipeline=pipe_name,
            data=data_name,
            broker=(broker_block["name"] if broker_block else None),
        )
    )

    risk_plugins: list[RiskPlugin] = []
    for r in cfg["plugins"].get("risk", []) or []:
        risk_cls = registry.risk[r["name"]]
        risk_plugins.append(risk_cls(**r.get("params_init", {})))

    # --- Strategy plugins (registered or local-source) ---
    strategy_plugins: list[StrategyPlugin] | None = None
    strategies_cfg = cfg["plugins"].get("strategies", [])
    if strategies_cfg:
        strategy_plugins = []
        for s in strategies_cfg:
            cls = _resolve_plugin_cls(s, registry.strategies, "strategy", mode=mode)
            strategy_plugins.append(cls(**s.get("params_init", {})))

    # --- Aggregator (it's a strategy plugin) ---
    aggregator: StrategyPlugin | None = None
    agg_cfg = cfg["plugins"].get("aggregator")
    if agg_cfg:
        agg_cls = registry.strategies[agg_cfg["name"]]
        aggregator = agg_cls(**agg_cfg.get("params_init", {}))

    # --- Rebalancer ---
    rebalancer: RebalancingPlugin | None = None
    rebal_cfg = cfg["plugins"].get("rebalancing")
    if rebal_cfg:
        rebal_cls = registry.rebalancing[rebal_cfg["name"]]
        rebalancer = rebal_cls(**rebal_cfg.get("params_init", {}))

    # Build pipeline params, merging in strategy/aggregator/rebalancer config
    pipeline_params = dict(cfg["plugins"]["pipeline"].get("params", {}))
    if strategies_cfg:
        pipeline_params["_strategies_cfg"] = strategies_cfg
    if agg_cfg:
        pipeline_params["_aggregator_cfg"] = agg_cfg
    if rebal_cfg:
        pipeline_params["_rebalancer_cfg"] = rebal_cfg
    risk_cfg_list = cfg["plugins"].get("risk", []) or []
    if risk_cfg_list:
        merged_risk_params: dict[str, Any] = {}
        for r in risk_cfg_list:
            merged_risk_params.update(r.get("params", {}))
        pipeline_params["_risk_cfg"] = merged_risk_params

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
    )

    # --- Validation plugins (post-backtest) ---
    validation_cfg = cfg["plugins"].get("validation", []) or []
    if validation_cfg and mode == "backtest":
        validation_results = []
        for v_cfg in validation_cfg:
            v_name = v_cfg["name"]
            if v_name not in registry.validations:
                logger.warning("Validation plugin '%s' not found, skipping", v_name)
                continue
            v_cls = registry.validations[v_name]
            v_plugin = v_cls(**v_cfg.get("params_init", {}))
            # Load returns and weights from artifacts
            returns_path = result.artifacts.get("returns", "")
            weights_path = result.artifacts.get("weights_history", "")
            returns_df = pd.read_parquet(returns_path) if returns_path else pd.DataFrame()
            weights_df = pd.read_parquet(weights_path) if weights_path else pd.DataFrame()
            benchmark_df = None
            v_result = v_plugin.validate(returns_df, weights_df, benchmark_df, v_cfg.get("params", {}))
            validation_results.append({"plugin": v_name, **v_result})
        if validation_results:
            validation_path = store.put_json("validation", validation_results)
            result.artifacts["validation"] = validation_path
            result.notes["validation"] = validation_results

    # --- Monitor plugins (paper/live) ---
    monitor_cfg = cfg["plugins"].get("monitors", []) or []
    if monitor_cfg and mode in ("paper", "live"):
        all_alerts = []
        for m_cfg in monitor_cfg:
            m_name = m_cfg["name"]
            if m_name not in registry.monitors:
                logger.warning("Monitor plugin '%s' not found, skipping", m_name)
                continue
            m_cls = registry.monitors[m_name]
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
        pub_cls = registry.publishers[p["name"]]
        pub: PublisherPlugin = pub_cls(**p.get("params_init", {}))
        pub.publish(result, p.get("params", {}))

    # LLM-friendly manifest (single file to understand the run)
    config_path_obj = Path(config_path).resolve() if config_path is not None else None
    plugin_versions = {
        "pipeline": _plugin_meta(pipeline, pipe_name),
        "data": _plugin_meta(data, data_name),
        "broker": _plugin_meta(broker, broker_block["name"]) if broker_block and broker else None,
        "risk": [_plugin_meta(plugin) for plugin in risk_plugins],
        "strategies": [_plugin_meta(plugin) for plugin in strategy_plugins or []],
        "aggregator": _plugin_meta(aggregator) if aggregator else None,
        "rebalancer": _plugin_meta(rebalancer) if rebalancer else None,
    }
    manifest = {
        "run_id": result.run_id,
        "asof": result.asof,
        "mode": result.mode,
        "pipeline": result.pipeline_name,
        "config_hash": cfg_hash,
        "config": {
            "path": str(config_path_obj) if config_path_obj else None,
            "sha256": cfg_hash_full,
            "file_sha256": _sha256_file(config_path_obj) if config_path_obj else None,
            "git_blob_sha": (
                _git_value(["hash-object", str(config_path_obj)], Path.cwd()) if config_path_obj else None
            ),
        },
        "git": _git_info(Path.cwd()),
        "plugins": {
            "pipeline": getattr(getattr(pipeline, "meta", None), "name", pipe_name),
            "data": getattr(getattr(data, "meta", None), "name", data_name),
            "broker": getattr(getattr(broker, "meta", None), "name", broker_block["name"]) if broker else None,
        },
        "plugin_versions": plugin_versions,
        "dataset": _dataset_block(data),
        "capability_results": _run_capability_checks(data, run_ctx=None),
        "artifacts": result.artifacts,
        "metrics": result.metrics,
        "warnings": [],
    }

    # Validate artifacts against JSON schemas when available (best-effort)
    schema_dir = repo_root() / "schemas"
    for logical, path in (result.artifacts or {}).items():
        schema_path = schema_dir / f"{logical}.schema.json"
        if schema_path.exists() and path.endswith(".parquet"):
            try:
                df = pd.read_parquet(path)
                schema = load_schema(schema_path)
                manifest["warnings"].extend([f"{logical}:{w}" for w in validate_table(df, schema)])
            except Exception as e:
                manifest["warnings"].append(f"{logical}:schema_check_error:{e}")

    strict_mode = bool(cfg.get("run", {}).get("strict")) or result.mode == "promotion"
    if strict_mode:
        if manifest["dataset"]["tier"] == "raw":
            store.put_json("run_manifest", manifest)
            raise RuntimeError(
                "strict mode rejects Tier-0 raw ingest — see "
                "quantbox-qute/docs/decisions/0004-quantbox-dataset-plugin-tiers.md"
            )
        failures = [c for c, r in manifest["capability_results"].items() if not r["passed"]]
        if failures:
            store.put_json("run_manifest", manifest)
            raise RuntimeError(f"strict mode capability failures: {failures}")

    store.put_json("run_manifest", manifest)
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
