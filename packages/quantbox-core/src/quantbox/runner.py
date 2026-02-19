from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

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
from .validate import validate_config

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _hash_config(cfg: dict[str, Any]) -> str:
    b = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:12]


def _run_id(asof: str, pipeline_name: str, cfg_hash: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = pipeline_name.replace(".", "_")
    return f"{asof}__{safe}__{cfg_hash}__{ts}"


def run_from_config(cfg: dict[str, Any], registry) -> RunResult:
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

    # --- Strategy plugins (new) ---
    strategy_plugins: list[StrategyPlugin] | None = None
    strategies_cfg = cfg["plugins"].get("strategies", [])
    if strategies_cfg:
        strategy_plugins = []
        for s in strategies_cfg:
            cls = registry.strategies[s["name"]]
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
    manifest = {
        "run_id": result.run_id,
        "asof": result.asof,
        "mode": result.mode,
        "pipeline": result.pipeline_name,
        "config_hash": cfg_hash,
        "plugins": {
            "pipeline": getattr(getattr(pipeline, "meta", None), "name", pipe_name),
            "data": getattr(getattr(data, "meta", None), "name", data_name),
            "broker": getattr(getattr(broker, "meta", None), "name", broker_block["name"]) if broker else None,
        },
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
                import pandas as pd

                df = pd.read_parquet(path)
                schema = load_schema(schema_path)
                manifest["warnings"].extend([f"{logical}:{w}" for w in validate_table(df, schema)])
            except Exception as e:
                manifest["warnings"].append(f"{logical}:schema_check_error:{e}")

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
