"""Programmatic agent tools for QuantBox.

Provides QuantBoxAgent — a single class that wraps all framework operations
into clean, LLM-friendly methods. Each method returns structured dicts
designed for agent consumption.

Usage:
    from quantbox.agents import QuantBoxAgent

    agent = QuantBoxAgent()

    # Explore
    plugins = agent.list_plugins()
    info = agent.plugin_info("strategy.crypto_trend.v1")

    # Build & validate
    config = agent.build_config(
        mode="backtest",
        pipeline="backtest.pipeline.v1",
        strategy="strategy.crypto_trend.v1",
        data="binance.live_data.v1",
    )
    issues = agent.validate_config(config)

    # Execute
    result = agent.run(config)
    artifacts = agent.inspect_run(result["run_id"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class QuantBoxAgent:
    """Agent interface to QuantBox framework.

    All methods return plain dicts/lists — no DataFrames or complex objects.
    Errors are returned as structured dicts with 'error' key, never raised.
    """

    def __init__(self) -> None:
        self._registry = None

    @property
    def registry(self):
        if self._registry is None:
            from quantbox.registry import PluginRegistry
            self._registry = PluginRegistry.discover()
        return self._registry

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_plugins(self, kind: str | None = None) -> dict[str, list[dict[str, str]]]:
        """List all registered plugins, optionally filtered by kind.

        Args:
            kind: Filter by plugin type (pipeline, strategy, data, broker,
                  risk, rebalancing, publisher). None = all types.

        Returns:
            {"pipeline": [{"name": "...", "description": "..."}], ...}
        """
        from quantbox.introspect import describe_plugin_class

        result: dict[str, list[dict[str, str]]] = {}
        kind_map = {
            "pipeline": "pipelines",
            "strategy": "strategies",
            "data": "data",
            "broker": "brokers",
            "risk": "risk",
            "rebalancing": "rebalancing",
            "publisher": "publishers",
        }

        for k, attr in kind_map.items():
            if kind and k != kind:
                continue
            plugins = getattr(self.registry, attr, {})
            entries = []
            for name, cls in sorted(plugins.items()):
                meta = getattr(cls, "meta", None)
                entries.append({
                    "name": name,
                    "description": getattr(meta, "description", ""),
                    "tags": list(getattr(meta, "tags", ())),
                })
            if entries:
                result[k] = entries

        return result

    def plugin_info(self, name: str) -> dict[str, Any]:
        """Get detailed info about a specific plugin.

        Args:
            name: Plugin ID (e.g. "strategy.crypto_trend.v1")

        Returns:
            Full plugin description including params_schema, defaults, methods.
        """
        from quantbox.introspect import describe_plugin_class

        for attr in ("pipelines", "strategies", "data", "brokers", "risk", "rebalancing", "publishers"):
            plugins = getattr(self.registry, attr, {})
            if name in plugins:
                return describe_plugin_class(plugins[name])

        return {
            "error": f"Plugin '{name}' not found",
            "available": self._all_plugin_names(),
        }

    def search_plugins(self, query: str) -> list[dict[str, str]]:
        """Search plugins by name, description, or tags.

        Args:
            query: Search term (case-insensitive)

        Returns:
            List of matching plugins with name, description, tags.
        """
        query_lower = query.lower()
        matches = []

        for attr in ("pipelines", "strategies", "data", "brokers", "risk", "rebalancing", "publishers"):
            plugins = getattr(self.registry, attr, {})
            for name, cls in plugins.items():
                meta = getattr(cls, "meta", None)
                if not meta:
                    continue
                searchable = f"{name} {meta.description} {' '.join(meta.tags)}".lower()
                if query_lower in searchable:
                    matches.append({
                        "name": name,
                        "kind": meta.kind,
                        "description": meta.description,
                        "tags": list(meta.tags),
                    })

        return matches

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def build_config(
        self,
        *,
        mode: str = "backtest",
        asof: str = "2026-02-01",
        pipeline: str = "backtest.pipeline.v1",
        strategy: str | None = None,
        data: str = "binance.live_data.v1",
        broker: str | None = None,
        risk: str | None = None,
        strategy_params: dict | None = None,
        pipeline_params: dict | None = None,
        data_params_init: dict | None = None,
        broker_params_init: dict | None = None,
    ) -> dict[str, Any]:
        """Build a config dict from parameters.

        Returns a config dict ready for validate_config() and run().
        """
        config: dict[str, Any] = {
            "run": {
                "mode": mode,
                "asof": asof,
                "pipeline": pipeline,
            },
            "artifacts": {"root": "./artifacts"},
            "plugins": {
                "pipeline": {
                    "name": pipeline,
                    "params": pipeline_params or {},
                },
                "data": {
                    "name": data,
                    "params_init": data_params_init or {},
                },
            },
        }

        if strategy:
            config["plugins"]["strategies"] = [{
                "name": strategy,
                "weight": 1.0,
                "params": strategy_params or {},
            }]
            config["plugins"]["aggregator"] = {
                "name": "strategy.weighted_avg.v1",
                "params": {},
            }

        if broker:
            config["plugins"]["broker"] = {
                "name": broker,
                "params_init": broker_params_init or {},
            }

        if risk:
            config["plugins"]["risk"] = [{"name": risk, "params": {}}]
        else:
            config["plugins"]["risk"] = []

        config["plugins"]["publishers"] = []

        return config

    def validate_config(self, config: dict[str, Any] | str | Path) -> dict[str, Any]:
        """Validate a config dict or YAML file.

        Args:
            config: Config dict, YAML string, or path to YAML file.

        Returns:
            {"valid": True/False, "findings": [...]}
        """
        cfg = self._load_config(config)
        if "error" in cfg:
            return cfg

        from quantbox.validate import validate_config
        findings = validate_config(cfg)

        return {
            "valid": not any(f.level == "error" for f in findings),
            "findings": [
                {"level": f.level, "path": f.path, "message": f.message}
                for f in findings
            ],
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, config: dict[str, Any] | str | Path, *, dry_run: bool = False) -> dict[str, Any]:
        """Execute a pipeline from config.

        Args:
            config: Config dict, YAML string, or path to YAML file.
            dry_run: If True, validate and preview without executing.

        Returns:
            Run result dict with run_id, artifacts, metrics.
        """
        cfg = self._load_config(config)
        if "error" in cfg:
            return cfg

        if dry_run:
            validation = self.validate_config(cfg)
            if not validation.get("valid"):
                return {"error": "Config validation failed", "findings": validation["findings"]}
            return {
                "dry_run": True,
                "valid": True,
                "mode": cfg["run"]["mode"],
                "pipeline": cfg["run"]["pipeline"],
                "plugins": {k: v.get("name", v) if isinstance(v, dict) else v
                            for k, v in cfg.get("plugins", {}).items()
                            if k != "profile"},
            }

        try:
            from quantbox.runner import run_from_config
            result = run_from_config(cfg, self.registry)
            return {
                "run_id": result.run_id,
                "pipeline": result.pipeline_name,
                "mode": result.mode,
                "asof": result.asof,
                "artifacts": result.artifacts,
                "metrics": result.metrics,
                "notes": result.notes,
            }
        except Exception as e:
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, "details", {}),
            }

    def inspect_run(self, run_dir: str | Path) -> dict[str, Any]:
        """Inspect artifacts from a completed run.

        Args:
            run_dir: Path to artifacts/<run_id>/ directory.

        Returns:
            Dict with manifest, artifact list, and summary metrics.
        """
        run_path = Path(run_dir)
        if not run_path.exists():
            return {"error": f"Run directory not found: {run_dir}"}

        result: dict[str, Any] = {"run_dir": str(run_path)}

        # Read manifest
        manifest_path = run_path / "run_manifest.json"
        if manifest_path.exists():
            result["manifest"] = json.loads(manifest_path.read_text())

        # Read meta
        meta_path = run_path / "run_meta.json"
        if meta_path.exists():
            result["meta"] = json.loads(meta_path.read_text())

        # List artifacts
        artifacts = []
        for f in sorted(run_path.iterdir()):
            if f.is_file():
                info = {"name": f.name, "size_bytes": f.stat().st_size}
                if f.suffix == ".parquet":
                    try:
                        import pandas as pd
                        df = pd.read_parquet(f)
                        info["rows"] = len(df)
                        info["columns"] = list(df.columns)
                    except Exception:
                        pass
                elif f.suffix == ".json":
                    try:
                        info["preview"] = json.loads(f.read_text())
                    except Exception:
                        pass
                artifacts.append(info)

        result["artifacts"] = artifacts
        return result

    def read_artifact(self, run_dir: str | Path, artifact_name: str) -> dict[str, Any]:
        """Read a specific artifact from a run.

        Args:
            run_dir: Path to artifacts/<run_id>/ directory.
            artifact_name: File name (e.g. "target_weights.parquet")

        Returns:
            For Parquet: {"columns": [...], "rows": N, "data": [...first 20 rows...]}
            For JSON: the parsed JSON content.
        """
        path = Path(run_dir) / artifact_name
        if not path.exists():
            return {"error": f"Artifact not found: {path}"}

        if path.suffix == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(path)
                return {
                    "type": "parquet",
                    "rows": len(df),
                    "columns": list(df.columns),
                    "dtypes": {c: str(df[c].dtype) for c in df.columns},
                    "data": df.head(20).to_dict(orient="records"),
                }
            except Exception as e:
                return {"error": str(e)}

        if path.suffix in (".json", ".jsonl"):
            try:
                text = path.read_text()
                if path.suffix == ".jsonl":
                    lines = [json.loads(line) for line in text.strip().split("\n") if line.strip()]
                    return {"type": "jsonl", "entries": len(lines), "data": lines[:20]}
                return {"type": "json", "data": json.loads(text)}
            except Exception as e:
                return {"error": str(e)}

        return {"error": f"Unsupported format: {path.suffix}"}

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------

    def list_profiles(self) -> dict[str, Any]:
        """List available plugin profiles from manifest.yaml."""
        try:
            from quantbox.plugin_manifest import load_manifest
            manifest = load_manifest()
            profiles = manifest.get("profiles", {})
            return {
                name: {k: v.get("name", v) if isinstance(v, dict) else v for k, v in prof.items()}
                for name, prof in profiles.items()
            }
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_config(self, config: dict[str, Any] | str | Path) -> dict[str, Any]:
        """Load config from dict, YAML string, or file path."""
        if isinstance(config, dict):
            return config
        if isinstance(config, Path) or (isinstance(config, str) and Path(config).exists()):
            path = Path(config)
            return yaml.safe_load(path.read_text())
        if isinstance(config, str):
            try:
                return yaml.safe_load(config)
            except Exception:
                return {"error": f"Could not parse config: {config[:100]}"}
        return {"error": f"Unsupported config type: {type(config)}"}

    def _all_plugin_names(self) -> list[str]:
        names = []
        for attr in ("pipelines", "strategies", "data", "brokers", "risk", "rebalancing", "publishers"):
            names.extend(getattr(self.registry, attr, {}).keys())
        return sorted(names)
