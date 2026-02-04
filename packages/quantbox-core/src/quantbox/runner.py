from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .contracts import Mode, RunResult, PipelinePlugin, BrokerPlugin, DataPlugin, PublisherPlugin, RiskPlugin
from .store import FileArtifactStore

def _hash_config(cfg: Dict[str, Any]) -> str:
    b = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:12]

def _run_id(asof: str, pipeline_name: str, cfg_hash: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = pipeline_name.replace(".", "_")
    return f"{asof}__{safe}__{cfg_hash}__{ts}"

def run_from_config(cfg: Dict[str, Any], registry) -> RunResult:
    run_cfg = cfg["run"]
    mode: Mode = run_cfg["mode"]
    asof: str = run_cfg["asof"]
    pipeline_key: str = run_cfg["pipeline"]

    cfg_hash = _hash_config(cfg)
    run_id = _run_id(asof, pipeline_key, cfg_hash)

    store = FileArtifactStore(cfg["artifacts"]["root"], run_id)

    pipe_name = cfg["plugins"]["pipeline"]["name"]
    pipeline_cls = registry.pipelines[pipe_name]
    pipeline: PipelinePlugin = pipeline_cls(**cfg["plugins"]["pipeline"].get("params_init", {}))

    data_name = cfg["plugins"]["data"]["name"]
    data_cls = registry.data[data_name]
    data: DataPlugin = data_cls(**cfg["plugins"]["data"].get("params_init", {}))

    broker: Optional[BrokerPlugin] = None
    broker_block = cfg["plugins"].get("broker")
    if broker_block and getattr(pipeline, "kind", None) == "trading" and mode in ("paper", "live"):
        broker_cls = registry.brokers[broker_block["name"]]
        broker = broker_cls(**broker_block.get("params_init", {}))

    risk_plugins: List[RiskPlugin] = []
    for r in cfg["plugins"].get("risk", []) or []:
        risk_cls = registry.risk[r["name"]]
        risk_plugins.append(risk_cls(**r.get("params_init", {})))

    result = pipeline.run(
        mode=mode,
        asof=asof,
        params=cfg["plugins"]["pipeline"].get("params", {}),
        data=data,
        store=store,
        broker=broker,
        risk=risk_plugins,
    )

    store.put_json("run_meta", {
        "run_id": result.run_id,
        "pipeline": result.pipeline_name,
        "mode": result.mode,
        "asof": result.asof,
        "config_hash": cfg_hash,
        "artifacts": result.artifacts,
        "metrics": result.metrics,
        "notes": result.notes,
    })

    for p in cfg["plugins"].get("publishers", []) or []:
        pub_cls = registry.publishers[p["name"]]
        pub: PublisherPlugin = pub_cls(**p.get("params_init", {}))
        pub.publish(result, p.get("params", {}))

    return result
