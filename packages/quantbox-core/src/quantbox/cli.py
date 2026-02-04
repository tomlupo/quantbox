from __future__ import annotations
import argparse
import yaml

from .registry import PluginRegistry
from .runner import run_from_config
from .validate import validate_config

def _as_json(obj):
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)

def cmd_plugins_list(reg: PluginRegistry, as_json: bool = False):
    if as_json:
        payload = {
            "pipelines": sorted(list(reg.pipelines.keys())),
            "brokers": sorted(list(reg.brokers.keys())),
            "data": sorted(list(reg.data.keys())),
            "publishers": sorted(list(reg.publishers.keys())),
            "risk": sorted(list(reg.risk.keys())),
        }
        print(_as_json(payload))
        return

    def show(title, d):
        print(title + ":")
        for k in sorted(d): print("  -", k)
    show("Pipelines", reg.pipelines)
    show("Brokers", reg.brokers)
    show("Data", reg.data)
    show("Publishers", reg.publishers)
    show("Risk", reg.risk)

def cmd_plugins_info(reg: PluginRegistry, name: str, as_json: bool = False):
    # name can match any group
    groups = {
        "pipeline": reg.pipelines,
        "broker": reg.brokers,
        "data": reg.data,
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
    raise SystemExit(f"plugin_not_found: {name}")

def cmd_plugins_doctor(as_json: bool = False):
    import importlib.metadata
    from .registry import ENTRYPOINT_GROUPS
    from .plugins.builtins import builtins as builtin_plugins

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

    if as_json:
        print(_as_json(results))
        return

    print("Plugins doctor:")
    for r in results:
        msg = f" ({r['message']})" if r["message"] else ""
        print(f"- {r['source']} {r['group']} {r['name']}: {r['status']}{msg}")

def main():
    ap = argparse.ArgumentParser(prog="quantbox")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plugins")
    sp.add_argument("action", choices=["list","info","doctor"])
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--name", default=None)
    vp = sub.add_parser("validate")
    vp.add_argument("-c","--config", required=True)
    vp.add_argument("--json", action="store_true")

    rp = sub.add_parser("run")
    rp.add_argument("-c", "--config", required=True)
    rp.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    reg = PluginRegistry.discover()

    if args.cmd == "plugins" and args.action == "list":
        cmd_plugins_list(reg, as_json=args.json)
        return

    if args.cmd == "plugins" and args.action == "info":
        if not args.name:
            raise SystemExit("--name is required for plugins info")
        cmd_plugins_info(reg, args.name, as_json=args.json)
        return
    if args.cmd == "plugins" and args.action == "doctor":
        cmd_plugins_doctor(as_json=args.json)
        return

    if args.cmd == "validate":
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        findings = validate_config(cfg)
        payload = [f.__dict__ for f in findings]
        if args.json:
            import json
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for f in findings:
                print(f.level.upper() + ":", f.message)
        # non-zero exit on error
        if any(f.level == "error" for f in findings):
            raise SystemExit(2)
        return

    if args.cmd == "run":
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if args.dry_run:
            # resolve plugins + show plan
            pipe = cfg["plugins"]["pipeline"]["name"]
            data = cfg["plugins"]["data"]["name"]
            broker = cfg["plugins"].get("broker", {}).get("name")
            plan = {"pipeline": pipe, "data": data, "broker": broker, "mode": cfg["run"]["mode"], "asof": cfg["run"]["asof"]}
            import json
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        result = run_from_config(cfg, reg)
        print("RUN_ID:", result.run_id)
        print("PIPELINE:", result.pipeline_name)
        print("METRICS:", result.metrics)

if __name__ == "__main__":
    main()
