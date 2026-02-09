from __future__ import annotations
import yaml
import typer

from .registry import PluginRegistry
from .runner import run_from_config
from .exceptions import PluginNotFoundError
from .validate import validate_config
from .plugin_manifest import load_manifest, resolve_profile

app = typer.Typer(name="quantbox", help="Quant research & trading CLI")


def _as_json(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)

def cmd_plugins_list(reg: PluginRegistry, as_json: bool = False):
    if as_json:
        payload = {
            "pipelines": sorted(list(reg.pipelines.keys())),
            "strategies": sorted(list(reg.strategies.keys())),
            "brokers": sorted(list(reg.brokers.keys())),
            "data": sorted(list(reg.data.keys())),
            "rebalancing": sorted(list(reg.rebalancing.keys())),
            "publishers": sorted(list(reg.publishers.keys())),
            "risk": sorted(list(reg.risk.keys())),
        }
        print(_as_json(payload))
        return

    def show(title, d):
        print(title + ":")
        for k in sorted(d): print("  -", k)
    show("Pipelines", reg.pipelines)
    show("Strategies", reg.strategies)
    show("Brokers", reg.brokers)
    show("Data", reg.data)
    show("Rebalancing", reg.rebalancing)
    show("Publishers", reg.publishers)
    show("Risk", reg.risk)

def cmd_plugins_info(reg: PluginRegistry, name: str, as_json: bool = False):
    # name can match any group
    groups = {
        "pipeline": reg.pipelines,
        "strategy": reg.strategies,
        "broker": reg.brokers,
        "data": reg.data,
        "rebalancing": reg.rebalancing,
        "publisher": reg.publishers,
        "risk": reg.risk,
    }
    for gname, d in groups.items():
        if name in d:
            cls = d[name]
            inst = cls() if callable(cls) else cls
            meta = getattr(inst, "meta", None)
            payload = {
                "group": gname,
                "name": name,
                "meta": meta.__dict__ if meta else None,
            }
            print(_as_json(payload) if as_json else payload)
            return
    all_names = sorted(set(
        k for d in groups.values() for k in d
    ))
    raise PluginNotFoundError(name, "any", all_names)

def cmd_plugins_doctor(as_json: bool = False, strict: bool = False):
    import importlib.metadata
    from .registry import ENTRYPOINT_GROUPS
    from .plugins.builtins import builtins as builtin_plugins
    from .plugin_manifest import repo_root

    results = []

    builtins = builtin_plugins()
    for group, mapping in builtins.items():
        for name in sorted(mapping.keys()):
            results.append({
                "source": "builtin",
                "group": group,
                "name": name,
                "status": "ok",
                "message": "",
            })

    # Optional dependency checks for built-in live brokers
    try:
        from .plugins.broker import ibkr as _ibkr_mod
        if getattr(_ibkr_mod, "IB", None) is None:
            results.append({
                "source": "builtin",
                "group": "broker",
                "name": "ibkr.live.v1",
                "status": "warn",
                "message": "optional dependency missing: ib_insync",
            })
    except Exception:
        pass

    try:
        from .plugins.broker import binance as _binance_mod
        if getattr(_binance_mod, "Client", None) is None:
            results.append({
                "source": "builtin",
                "group": "broker",
                "name": "binance.live.v1",
                "status": "warn",
                "message": "optional dependency missing: python-binance",
            })
    except Exception:
        pass

    # External entry points
    for group_name, ep_group in ENTRYPOINT_GROUPS.items():
        eps = importlib.metadata.entry_points(group=ep_group)
        for ep in eps:
            status = "ok"
            message = ""
            try:
                ep.load()
            except Exception as e:  # pragma: no cover
                status = "error"
                message = f"entrypoint_load_failed: {e}"

            if ep.name in builtins.get(group_name, {}):
                if status == "ok":
                    status = "warn"
                if message:
                    message = message + "; "
                message = message + "overrides built-in"

            results.append({
                "source": "entrypoint",
                "group": group_name,
                "name": ep.name,
                "status": status,
                "message": message,
            })

    # Schemas for built-in plugins
    schema_dir = repo_root() / "schemas"
    for group, mapping in builtins.items():
        for name, cls in mapping.items():
            meta = getattr(cls, "meta", None)
            if not meta:
                continue
            logicals = list(getattr(meta, "outputs", ()) or ()) + list(getattr(meta, "inputs", ()) or ())
            for logical in logicals:
                schema_path = schema_dir / f"{logical}.schema.json"
                if not schema_path.exists():
                    results.append({
                        "source": "schema",
                        "group": group,
                        "name": name,
                        "status": "warn",
                        "message": f"missing_schema:{logical}",
                    })

    # Config references
    try:
        reg = PluginRegistry.discover()
    except Exception:
        reg = None
    manifest = load_manifest()
    config_dir = repo_root() / "configs"
    if config_dir.exists():
        for cfg_path in sorted(config_dir.glob("*.yaml")):
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                plugins = cfg.get("plugins", {}) or {}
                profile = plugins.get("profile")
                prof = resolve_profile(str(profile), manifest) if profile else {}
                merged = dict(plugins)
                for key in ("pipeline", "data", "broker", "publishers", "risk"):
                    if key not in merged and key in prof:
                        merged[key] = prof[key]

                def _check(group: str, name: str | None):
                    if not name or not reg:
                        return
                    registry_map = {
                        "pipeline": reg.pipelines,
                        "data": reg.data,
                        "broker": reg.brokers,
                        "publisher": reg.publishers,
                        "risk": reg.risk,
                    }[group]
                    if name not in registry_map:
                        results.append({
                            "source": "config",
                            "group": group,
                            "name": name,
                            "status": "error",
                            "message": f"config_ref_not_found:{cfg_path.name}",
                        })

                _check("pipeline", (merged.get("pipeline") or {}).get("name"))
                _check("data", (merged.get("data") or {}).get("name"))
                _check("broker", (merged.get("broker") or {}).get("name"))
                for pub in (merged.get("publishers") or []):
                    _check("publisher", pub.get("name"))
                for rk in (merged.get("risk") or []):
                    _check("risk", rk.get("name"))
            except Exception as e:  # pragma: no cover
                results.append({
                    "source": "config",
                    "group": "config",
                    "name": cfg_path.name,
                    "status": "error",
                    "message": f"config_parse_failed:{e}",
                })

    if as_json:
        print(_as_json(results))
        if strict and any(r["status"] in ("warn", "error") for r in results):
            raise SystemExit(2)
        return

    print("Plugins doctor:")
    for r in results:
        msg = f" ({r['message']})" if r["message"] else ""
        print(f"- {r['source']} {r['group']} {r['name']}: {r['status']}{msg}")
    if strict and any(r["status"] in ("warn", "error") for r in results):
        raise SystemExit(2)


@app.command()
def plugins(
    action: str = typer.Argument(help="Action: list, info, or doctor"),
    name: str = typer.Option(None, help="Plugin name (required for 'info')"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
    strict: bool = typer.Option(False, help="Exit non-zero on warnings (doctor only)"),
):
    """List, inspect, or diagnose plugins."""
    reg = PluginRegistry.discover()
    if action == "list":
        cmd_plugins_list(reg, as_json=json)
    elif action == "info":
        if not name:
            raise typer.BadParameter("--name is required for 'plugins info'")
        cmd_plugins_info(reg, name, as_json=json)
    elif action == "doctor":
        cmd_plugins_doctor(as_json=json, strict=strict)
    else:
        raise typer.BadParameter(f"Unknown action: {action}. Use list, info, or doctor.")


@app.command()
def validate(
    config: str = typer.Option(..., "-c", "--config", help="Path to config YAML"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Validate a run config file."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    findings = validate_config(cfg)
    payload = [f.__dict__ for f in findings]
    if json:
        print(_as_json(payload))
    else:
        for f in findings:
            print(f.level.upper() + ":", f.message)
    if any(f.level == "error" for f in findings):
        raise SystemExit(2)


@app.command()
def run(
    config: str = typer.Option(..., "-c", "--config", help="Path to config YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without executing"),
):
    """Run a trading pipeline from config."""
    import json as json_mod

    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if dry_run:
        plugins_cfg = cfg.get("plugins", {}) or {}
        profile = plugins_cfg.get("profile")
        prof = resolve_profile(str(profile), load_manifest()) if profile else {}
        merged = dict(plugins_cfg)
        for key in ("pipeline", "data", "broker"):
            if key not in merged and key in prof:
                merged[key] = prof[key]
        plan = {
            "pipeline": (merged.get("pipeline") or {}).get("name"),
            "data": (merged.get("data") or {}).get("name"),
            "broker": (merged.get("broker") or {}).get("name"),
            "mode": cfg["run"]["mode"],
            "asof": cfg["run"]["asof"],
        }
        print(json_mod.dumps(plan, ensure_ascii=False, indent=2))
        return

    reg = PluginRegistry.discover()
    result = run_from_config(cfg, reg)
    print("RUN_ID:", result.run_id)
    print("PIPELINE:", result.pipeline_name)
    print("METRICS:", result.metrics)


@app.command()
def warehouse(
    action: str = typer.Argument(
        help="Action: init, tables, query, describe, ingest, register-dataset"
    ),
    root: str = typer.Option("./warehouse", "-r", "--root", help="Warehouse root directory"),
    sql: str = typer.Option(None, "-q", "--query", help="SQL query (for 'query' action)"),
    table: str = typer.Option(None, "-t", "--table", help="Table name (for 'describe')"),
    run_dir: str = typer.Option(None, "--run-dir", help="Artifact run directory (for 'ingest')"),
    name: str = typer.Option(None, "-n", "--name", help="Dataset name (for 'register-dataset')"),
    path: str = typer.Option(None, "-p", "--path", help="Dataset path (for 'register-dataset')"),
    output: str = typer.Option(None, "-o", "--output", help="Output file (for 'query')"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Interact with the warehouse (query, ingest, manage)."""
    import json as json_mod
    from .warehouse import Warehouse

    if action == "init":
        wh = Warehouse(root)
        wh.close()
        print(f"Warehouse initialized at {root}")
        return

    wh = Warehouse(root)
    try:
        if action == "tables":
            all_tables = wh.list_tables()
            if json_out:
                print(_as_json(all_tables))
            else:
                for section, items in all_tables.items():
                    print(f"{section}:")
                    for item in items:
                        print(f"  - {item}")
            return

        if action == "query":
            if not sql:
                raise typer.BadParameter("--query/-q is required for 'query' action")
            result = wh.query(sql)
            if output:
                result.to_parquet(output, index=False)
                print(f"Written {len(result)} rows to {output}")
            elif json_out:
                print(result.to_json(orient="records", indent=2))
            else:
                print(result.to_string())
            return

        if action == "describe":
            if not table:
                raise typer.BadParameter("--table/-t is required for 'describe' action")
            desc = wh.describe(table)
            if json_out:
                print(_as_json(desc))
            else:
                for col in desc:
                    print(f"  {col['column_name']:30s} {col['column_type']:15s} null={col['null']}")
            return

        if action == "ingest":
            if not run_dir:
                raise typer.BadParameter("--run-dir is required for 'ingest' action")
            from pathlib import Path
            from .store import FileArtifactStore
            from .warehouse.ingestion import ingest_run
            run_path = Path(run_dir)
            store = FileArtifactStore(
                str(run_path.parent), run_path.name, _readonly=True
            )
            results = ingest_run(wh, store)
            if json_out:
                print(_as_json(results))
            else:
                for tbl, rows in results.items():
                    print(f"  {tbl}: {rows} rows")
            return

        if action == "register-dataset":
            if not name or not path:
                raise typer.BadParameter(
                    "--name/-n and --path/-p are required for 'register-dataset'"
                )
            views = wh.register_dataset(name, path)
            if json_out:
                print(_as_json({"views": views}))
            else:
                print(f"Registered {len(views)} view(s):")
                for v in views:
                    print(f"  - {v}")
            return

        raise typer.BadParameter(
            f"Unknown action: {action}. "
            "Use init, tables, query, describe, ingest, or register-dataset."
        )
    finally:
        wh.close()


def main():
    app()

if __name__ == "__main__":
    main()
