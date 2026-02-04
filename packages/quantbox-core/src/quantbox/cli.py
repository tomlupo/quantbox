from __future__ import annotations
import argparse
import yaml

from .registry import PluginRegistry
from .runner import run_from_config

def cmd_plugins_list(reg: PluginRegistry):
    def show(title, d):
        print(title + ":")
        for k in sorted(d): print("  -", k)
    show("Pipelines", reg.pipelines)
    show("Brokers", reg.brokers)
    show("Data", reg.data)
    show("Publishers", reg.publishers)
    show("Risk", reg.risk)

def main():
    ap = argparse.ArgumentParser(prog="quantbox")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plugins")
    sp.add_argument("action", choices=["list"])

    rp = sub.add_parser("run")
    rp.add_argument("-c", "--config", required=True)

    args = ap.parse_args()
    reg = PluginRegistry.discover()

    if args.cmd == "plugins" and args.action == "list":
        cmd_plugins_list(reg)
        return

    if args.cmd == "run":
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        result = run_from_config(cfg, reg)
        print("RUN_ID:", result.run_id)
        print("PIPELINE:", result.pipeline_name)
        print("METRICS:", result.metrics)

if __name__ == "__main__":
    main()
