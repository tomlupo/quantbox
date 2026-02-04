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

def main():
    ap = argparse.ArgumentParser(prog="quantbox")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plugins")
    sp.add_argument("action", choices=["list","info"])
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
