from __future__ import annotations

import contextlib
import sys
from importlib.resources import files as _res_files
from pathlib import Path
from typing import Any

import typer
import yaml

from .exceptions import PluginNotFoundError
from .gates_cli import gates_app
from .plugin_manifest import load_manifest, resolve_profile
from .registry import PluginRegistry
from .runner import run_from_config
from .validate import validate_config

app = typer.Typer(name="quantbox", help="Quant research & trading CLI")
app.add_typer(gates_app, name="gates")


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
            "overlays": sorted(list(reg.overlays.keys())),
        }
        print(_as_json(payload))
        return

    def show(title, d):
        print(title + ":")
        for k in sorted(d):
            print("  -", k)

    show("Pipelines", reg.pipelines)
    show("Strategies", reg.strategies)
    show("Brokers", reg.brokers)
    show("Data", reg.data)
    show("Rebalancing", reg.rebalancing)
    show("Publishers", reg.publishers)
    show("Risk", reg.risk)
    show("Overlays", reg.overlays)


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
        "overlay": reg.overlays,
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
    all_names = sorted(set(k for d in groups.values() for k in d))
    raise PluginNotFoundError(name, "any", all_names)


def cmd_plugins_schema(reg: PluginRegistry, name: str | None = None, as_json: bool = False):
    """Every registered plugin with id, status and its params JSON Schema (TOM-1350)."""
    from .params_schema import catalog

    payload = catalog(reg)
    if name:
        payload["plugins"] = [p for p in payload["plugins"] if p["id"] == name]
        if not payload["plugins"]:
            all_names = sorted({p["id"] for p in catalog(reg)["plugins"]})
            raise PluginNotFoundError(name, "any", all_names)
    if as_json:
        print(_as_json(payload))
        return
    for p in payload["plugins"]:
        print(f"{p['id']}  [{p['group']}, {p['status']}]")
        if p["params"] is None:
            print("  (no params_schema declared)")
            continue
        for row in p["params"]:
            print(f"  - {row['name']}: {row['type']} = {row['default']!r}  {row['description']}")


def cmd_plugins_doctor(as_json: bool = False, strict: bool = False):
    import importlib.metadata

    from .plugins.builtins import builtins as builtin_plugins
    from .registry import ENTRYPOINT_GROUPS

    results = []

    builtins = builtin_plugins()
    for group, mapping in builtins.items():
        for name in sorted(mapping.keys()):
            results.append(
                {
                    "source": "builtin",
                    "group": group,
                    "name": name,
                    "status": "ok",
                    "message": "",
                }
            )

    # Optional dependency checks for built-in live brokers
    try:
        from .plugins.broker import ibkr as _ibkr_mod

        if getattr(_ibkr_mod, "IB", None) is None:
            results.append(
                {
                    "source": "builtin",
                    "group": "broker",
                    "name": "ibkr.live.v1",
                    "status": "warn",
                    "message": "optional dependency missing: ib_insync",
                }
            )
    except Exception:
        pass

    try:
        from .plugins.broker import binance as _binance_mod

        if getattr(_binance_mod, "Client", None) is None:
            results.append(
                {
                    "source": "builtin",
                    "group": "broker",
                    "name": "binance.live.v1",
                    "status": "warn",
                    "message": "optional dependency missing: python-binance",
                }
            )
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

            results.append(
                {
                    "source": "entrypoint",
                    "group": group_name,
                    "name": ep.name,
                    "status": status,
                    "message": message,
                }
            )

    # Schemas for built-in plugins
    schema_dir = Path(str(_res_files("quantbox").joinpath("artifact_schemas")))
    for group, mapping in builtins.items():
        for name, cls in mapping.items():
            meta = getattr(cls, "meta", None)
            if not meta:
                continue
            logicals = list(getattr(meta, "outputs", ()) or ()) + list(getattr(meta, "inputs", ()) or ())
            for logical in logicals:
                schema_path = schema_dir / f"{logical}.schema.json"
                if not schema_path.exists():
                    results.append(
                        {
                            "source": "schema",
                            "group": group,
                            "name": name,
                            "status": "warn",
                            "message": f"missing_schema:{logical}",
                        }
                    )

    # Config references
    try:
        reg = PluginRegistry.discover()
    except Exception:
        reg = None
    manifest = load_manifest()
    config_dir = Path.cwd() / "cookbook" / "configs"
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
                        results.append(
                            {
                                "source": "config",
                                "group": group,
                                "name": name,
                                "status": "error",
                                "message": f"config_ref_not_found:{cfg_path.name}",  # noqa: B023
                            }
                        )

                _check("pipeline", (merged.get("pipeline") or {}).get("name"))
                _check("data", (merged.get("data") or {}).get("name"))
                _check("broker", (merged.get("broker") or {}).get("name"))
                for pub in merged.get("publishers") or []:
                    _check("publisher", pub.get("name"))
                for rk in merged.get("risk") or []:
                    _check("risk", rk.get("name"))
            except Exception as e:  # pragma: no cover
                results.append(
                    {
                        "source": "config",
                        "group": "config",
                        "name": cfg_path.name,
                        "status": "error",
                        "message": f"config_parse_failed:{e}",
                    }
                )

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
    action: str = typer.Argument(help="Action: list, info, schema, or doctor"),
    name: str = typer.Option(None, help="Plugin name (required for 'info', optional filter for 'schema')"),
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
    elif action == "schema":
        cmd_plugins_schema(reg, name, as_json=json)
    elif action == "doctor":
        cmd_plugins_doctor(as_json=json, strict=strict)
    else:
        raise typer.BadParameter(f"Unknown action: {action}. Use list, info, schema, or doctor.")


@app.command()
def validate(
    config: str = typer.Option(..., "-c", "--config", help="Path to config YAML"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Validate a run config file."""
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    findings = validate_config(cfg)
    payload = [f.__dict__ for f in findings]
    if json:
        print(_as_json(payload))
    else:
        for f in findings:
            print(f.level.upper() + ":", f.message)
    n_errors = sum(1 for f in findings if f.level == "error")
    if n_errors:
        if not json:
            print(f"INVALID: {config} has {n_errors} error(s)")
        raise SystemExit(2)
    if not json:
        print(f"OK: {config} is valid")


@app.command()
def run(
    config: str = typer.Option(..., "-c", "--config", help="Path to config YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without executing"),
    as_json: bool = typer.Option(
        False, "--json", help="Print only the run manifest (quantbox/run@1) on stdout; everything else goes to stderr"
    ),
    summary_out: str | None = typer.Option(None, "--summary-out", help="Also write the run manifest to this path"),
):
    """Run a trading pipeline from config."""
    import json as json_mod

    with open(config, encoding="utf-8") as f:
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

    # --json: stdout carries ONE JSON document, so anything printed during the
    # run (and every human line below) goes to stderr instead.
    out = sys.stderr if as_json else sys.stdout
    with contextlib.redirect_stdout(out):
        result = run_from_config(cfg, PluginRegistry.discover(), config_path=config)

    manifest_text = (Path(cfg["artifacts"]["root"]) / result.run_id / "run_manifest.json").read_text(encoding="utf-8")
    if summary_out:
        Path(summary_out).write_text(manifest_text, encoding="utf-8")
    if as_json:
        print(manifest_text)
    else:
        print("RUN_ID:", result.run_id)
        print("PIPELINE:", result.pipeline_name)
        print("METRICS:", result.metrics)
        execution = (result.notes or {}).get("execution")
        if execution:
            print("EXECUTION:", execution["description"])
        # Part of the summary, never only a log line above it (TOM-1529).
        report = ((result.notes or {}).get("reports") or {}).get("finding_report")
        if report:
            print(
                "FINDING REPORT:",
                report["file"] if report["produced"] else f"NOT PRODUCED — {report['error']}",
            )

    # Dead-man detection (quantbox#120): a rebalancer freeze (every intended
    # order suppressed, book stuck on stale positions) previously exited 0 --
    # the run "succeeded" while silently not trading. `rebalance_frozen` is
    # already computed by trading_pipeline.py; the missing piece was the CLI
    # never acting on it. Fail the job so cron/CI surfaces it instead of
    # swallowing it.
    if result.metrics.get("rebalance_frozen"):
        print(
            "REBALANCER FROZEN: all intended orders were suppressed this run "
            "-- portfolio not rebalanced, holding stale positions. "
            "See run notes['freeze_reasons'] for detail.",
            file=out,
        )
        raise SystemExit(1)


def _dataset_frame(dataset: Any, name: str) -> Any:
    """One wide frame of a quantbox-datasets Dataset: the public property when it has one
    (prices, volume, market_cap, funding_rates), else its reader (high, low, ...)."""
    try:
        return getattr(dataset, name)
    except AttributeError:
        return dataset._read(name)


@app.command()
def sweep(
    config: str = typer.Option(..., "-c", "--config", help="Path to sweep config YAML"),
):
    """Run a parameter-grid sweep from a YAML config.

    Reads a YAML config describing a strategy, base/sweep params, market data
    location, and heatmap settings. Iterates rebalancing bands, runs each grid
    combination through the vbt backtest engine, and saves heatmap PNGs +
    grid.parquet to the configured output directory.

    Expected YAML schema (relative paths resolve from the config file; the dataset
    is loaded by name at the build pinned in the ``datasets.lock`` nearest the config):

        strategy: strategy.crypto_regime_trend.v1
        data:
          dataset: crypto-spot-daily
          frames: [prices, volume, market_cap]
          align_to: prices
        base_params: { ... strategy kwargs ... }
        sweep_params: { window_pairs: [...] }
        bands: [0.0, 0.05]
        heatmap:
          index: window_pairs
          columns: [vol_target, tranches]
          metrics: [sharpe_ratio, ...]
        backtest:
          engine: vectorbt   # or rsims — the engine seam (docs/adr/0008)
          fees: 0.005
          rebalancing_freq: 1D
        execution:
          lag_bars: 1        # default; same convention as `quantbox run`
                             # (backtest.shift_signal is a deprecated alias)
        output_dir: heatmaps

    ``strategy`` may also be ``{source: strategy.py:MyStrategy}`` (path relative to
    the config). Writes ``<output_dir>/grid.parquet`` and
    ``<output_dir>/sweep_manifest.json`` (``quantbox/sweep@1``: strategy, execution
    timing, n_trials = grid rows).
    """
    from .analysis import DEFAULT_METRICS, run_grid
    from .analysis.parameter_grid import align_market_data

    config_path = Path(config).resolve()
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # The `execution:` BLOCK goes through the same resolver as `quantbox run`,
    # before any work: a typo (`lag_bar: 0`, `execution: 0`) is refused, never
    # defaulted. Absent block -> None -> run_grid resolves the default / alias.
    from .execution import resolve_lag_bars

    if isinstance(cfg.get("execution"), dict) and "same_bar" in cfg["execution"]:
        raise ValueError(
            "quantbox sweep does not take execution.same_bar: a grid of same-bar numbers is the "
            "multiple-testing search the override must never feed (docs/adr/0006). Sweep next-bar."
        )
    if isinstance(cfg.get("execution"), dict) and "calendar" in cfg["execution"]:
        raise ValueError(
            "quantbox sweep does not take execution.calendar: the sweep engine trades the bars it is given; "
            "the execution calendar is a `quantbox run` (backtest pipeline) setting (docs/adr/0007)."
        )
    sweep_lag_bars = resolve_lag_bars(cfg["execution"]) if "execution" in cfg else None

    config_dir = config_path.parent
    strategy_spec, strategy_cls = _sweep_strategy(cfg["strategy"], config_dir)
    data_cfg = cfg.get("data", {}) or {}
    if "dataset" not in data_cfg:
        raise typer.BadParameter("sweep config needs data.dataset: <quantbox-datasets name>")
    try:
        from quantbox_datasets.lock import find_lock, load
    except ImportError as exc:  # quantbox does not depend on quantbox-datasets
        raise typer.BadParameter(
            "sweep needs quantbox-datasets installed (it carries quantbox_datasets.lock); "
            "install it from its clone and point QUANTBOX_DATASETS_ROOT at <clone>/datasets"
        ) from exc

    # The lock nearest the config wins; with none there, load() searches from cwd.
    dataset = load(data_cfg["dataset"], lock=find_lock(config_dir))
    align_to = data_cfg.get("align_to", "prices")
    names = [*data_cfg.get("frames", ["prices", "volume", "market_cap"]), align_to]
    market_data = align_market_data({name: _dataset_frame(dataset, name) for name in dict.fromkeys(names)}, align_to)

    output_dir = (config_dir / cfg.get("output_dir", "heatmaps")).resolve()
    heatmap = cfg.get("heatmap", {}) or {}
    backtest = cfg.get("backtest", {}) or {}

    grid = run_grid(
        strategy_cls=strategy_cls,
        base_params=cfg.get("base_params", {}) or {},
        sweep_params=cfg.get("sweep_params", {}) or {},
        market_data=market_data,
        bands=cfg.get("bands", [0.0, 0.05]),
        output_dir=output_dir,
        heatmap_index=heatmap.get("index"),
        heatmap_columns=heatmap.get("columns"),
        metrics=tuple(heatmap.get("metrics") or DEFAULT_METRICS),
        fees=float(backtest.get("fees", 0.005)),
        rebalancing_freq=backtest.get("rebalancing_freq", "1D"),
        lag_bars=sweep_lag_bars,
        shift_signal=backtest.get("shift_signal"),  # deprecated alias of execution.lag_bars
        engine=backtest.get("engine"),
    )
    # The sweep's own manifest: the timing every row was simulated with, and the
    # honest trial count (one per grid row), so a gate never counts by hand.
    from .engine.registry import get_engine
    from .execution import execution_record, resolve_sweep_lag_bars

    sweep_manifest = {
        "schema": "quantbox/sweep@1",
        "config": str(config_path),
        "strategy": strategy_spec,
        "engine": get_engine(backtest.get("engine"), require_installed=False).name,
        "execution": execution_record(resolve_sweep_lag_bars(sweep_lag_bars, backtest.get("shift_signal"))),
        "n_trials": len(grid),
        "grid": "grid.parquet",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep_manifest.json").write_text(_as_json(sweep_manifest), encoding="utf-8")
    print(f"SWEEP: {len(grid)} rows  ->  {output_dir}")


def _sweep_strategy(spec: Any, config_dir: Path) -> tuple[dict[str, str], type]:
    """A sweep's ``strategy:`` — a registry name, ``{name: ...}`` or ``{source: file.py:Class}``.

    A relative ``source`` path resolves from the sweep config, like every other sweep path.
    """
    from .runner import _resolve_plugin_cls

    if isinstance(spec, str):
        spec = {"name": spec}
    if not isinstance(spec, dict) or ("name" in spec) == ("source" in spec):
        raise typer.BadParameter("sweep `strategy:` is a registry name, {name: ...} or {source: path.py:Class}")
    resolved = dict(spec)
    if "source" in spec:
        file_part, sep, cls_name = str(spec["source"]).rpartition(":")
        if sep and not Path(file_part).is_absolute():
            resolved["source"] = f"{config_dir / file_part}:{cls_name}"
    registry = PluginRegistry.discover().strategies if "name" in spec else {}
    cls = _resolve_plugin_cls(resolved, registry, "strategy", mode="backtest")
    return {k: str(v) for k, v in spec.items()}, cls


@app.command()
def arms(
    config: str = typer.Option(..., "-c", "--config", help="Path to the arms YAML (base + overrides or grid)"),
    max_workers: int | None = typer.Option(
        None, "--max-workers", help="Arms run at once (overrides parallel.max_workers)"
    ),
    memory_budget_gb: float | None = typer.Option(
        None, "--memory-budget-gb", help="Total memory for concurrent arms (overrides parallel.memory_budget_gb)"
    ),
    as_json: bool = typer.Option(False, "--json", help="Print only the batch summary (quantbox/arms@1) on stdout"),
):
    """Run every arm of an arms file in parallel; write arms_summary.json.

    Exits 1 when any arm failed, naming it; the other arms' manifests are kept and linked.
    File format: the `quantbox.arms` module docstring.
    """
    import json as json_mod

    from .arms import load_arms, run_arms

    out = sys.stderr if as_json else sys.stdout
    with contextlib.redirect_stdout(out):
        summary = run_arms(load_arms(config), max_workers=max_workers, memory_budget_gb=memory_budget_gb)
    if as_json:
        print(json_mod.dumps(summary, indent=2))
    else:
        par = summary["parallel"]
        print(f"ARMS: {len(summary['arms'])} arms, n_trials={summary['n_trials']}, workers={par['workers']}")
        for arm in summary["arms"]:
            print(f"  {arm['status']:6s} {arm['name']}  {arm['manifest'] or arm['error']}")
        print("SUMMARY:", summary["path"])
    if summary["failed"]:
        print(f"ARMS FAILED: {', '.join(summary['failed'])}", file=out)
        raise SystemExit(1)


@app.command()
def warehouse(
    action: str = typer.Argument(help="Action: init, tables, query, describe, ingest, register-dataset"),
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
            store = FileArtifactStore(str(run_path.parent), run_path.name, _readonly=True)
            results = ingest_run(wh, store)
            if json_out:
                print(_as_json(results))
            else:
                for tbl, rows in results.items():
                    print(f"  {tbl}: {rows} rows")
            return

        if action == "register-dataset":
            if not name or not path:
                raise typer.BadParameter("--name/-n and --path/-p are required for 'register-dataset'")
            views = wh.register_dataset(name, path)
            if json_out:
                print(_as_json({"views": views}))
            else:
                print(f"Registered {len(views)} view(s):")
                for v in views:
                    print(f"  - {v}")
            return

        raise typer.BadParameter(
            f"Unknown action: {action}. Use init, tables, query, describe, ingest, or register-dataset."
        )
    finally:
        wh.close()


@app.command()
def approve(
    run_dir: str = typer.Option(..., "--run-dir", help="Path to artifacts/<run_id>/"),
    who: str = typer.Option("human", "--who", help="Approver identity"),
    note: str = typer.Option("approved", "--note", help="Approval note"),
):
    """Write an approval file for the orders in a run directory."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    run_path = Path(run_dir)
    od = run_path / "orders_digest.json"
    if not od.exists():
        raise typer.BadParameter(f"orders_digest.json not found in {run_dir}")

    digest = json.loads(od.read_text(encoding="utf-8"))["orders_digest"]
    out_dir = Path("approvals")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{digest}.json"

    payload = {
        "approved": True,
        "orders_digest": digest,
        "who": who,
        "when": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote approval:", out)


dataset_app = typer.Typer(help="Datasets read by name: pinned by datasets.lock, rooted by $QUANTBOX_DATASETS_ROOT.")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("resolve")
def dataset_resolve(
    name: str = typer.Argument(help="Dataset name, as a config's data.params_init.dataset names it"),
    lock: str = typer.Option(None, "--lock", help="datasets.lock to read (overrides --config)"),
    config: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Resolve as `run -c <config>` would: its data.params_init.dataset_lock, else the lock nearest the config",
    ),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """What a run would read for a dataset: path, pinned sha256, market, funding file, match.

    With --config, the lock is the one `run -c <config>` binds (the nearest above the
    config). Without it, the nearest datasets.lock above the cwd is used, which matches
    a run only when no lock sits above the config.

    Exits 1 when the bytes are not the pinned build, or the dataset cannot be resolved.
    """
    from .dataset_lock import DatasetResolveError, lock_for_config, resolve_dataset

    if lock is None and config is not None:
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
        params_init = ((cfg.get("plugins") or {}).get("data") or {}).get("params_init") or {}
        explicit = params_init.get("dataset_lock")
        found = lock_for_config(config)
        lock = explicit if explicit is not None else (str(found) if found is not None else None)

    try:
        resolved = resolve_dataset(name, lock=lock)
    except DatasetResolveError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc
    if json:
        print(_as_json(resolved))
    else:
        for key, value in resolved.items():
            print(f"{key}: {value}")
    if resolved["matches"] is False:
        raise SystemExit(1)


new_app = typer.Typer(help="Scaffold new things (a research line).")
app.add_typer(new_app, name="new")
line_app = typer.Typer(help="Research lines (Model C): move a line's engine pin.")
app.add_typer(line_app, name="line")


def _line_report(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(_as_json(result))
        return
    for key, value in result.items():
        print(f"{key}: {value}")


@new_app.command("line")
def new_line(
    slug: str = typer.Argument(help="Line name: lowercase letters, digits, '-' and '_'"),
    directory: str = typer.Option(None, "--dir", help="Where to create it (default: ./<slug>); must be empty"),
    quantbox_ref: str = typer.Option(
        None, "--quantbox-ref", help="quantbox tag, branch or SHA (default: latest v* tag)"
    ),
    quantbox_url: str = typer.Option(None, "--quantbox-url", help="quantbox git URL (default: GitHub)"),
    datasets_ref: str = typer.Option(None, "--datasets-ref", help="quantbox-datasets ref (default: its HEAD)"),
    datasets_url: str = typer.Option(None, "--datasets-url", help="quantbox-datasets git URL (default: GitHub)"),
    extras: str = typer.Option(None, "--extras", help="quantbox extras to install (default: full; '' for none)"),
    python: str = typer.Option(None, "--python", help="Python minor version the line runs on (default: 3.12)"),
    dataset: str = typer.Option(None, "--dataset", help="Dataset the base config reads (default: crypto-spot-daily)"),
    question: str = typer.Option(None, "--question", help="One sentence: what this line investigates"),
    no_lock: bool = typer.Option(False, "--no-lock", help="Write the files only; skip `uv lock`"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Create a Model C research line: pinned pyproject + lock, README with prereg, datasets.lock, arms, repro test.

    quantbox is pinned to the commit the tag names, every transitive dependency exactly.
    Then: `cd <slug> && uv sync && uv run quantbox run -c config.yaml`.
    """
    from . import line

    kwargs: dict[str, Any] = {
        "directory": directory,
        "quantbox_ref": quantbox_ref,
        "datasets_ref": datasets_ref,
        "question": question,
        "lock": not no_lock,
    }
    defaults = {
        "quantbox_url": (quantbox_url, line.QUANTBOX_URL),
        "datasets_url": (datasets_url, line.DATASETS_URL),
        "extras": (extras, line.DEFAULT_EXTRAS),
        "python": (python, line.DEFAULT_PYTHON),
        "dataset": (dataset, line.DEFAULT_DATASET),
    }
    kwargs.update({k: (v if v is not None else d) for k, (v, d) in defaults.items()})
    try:
        result = line.new_line(slug, **kwargs)
    except line.LineError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc
    _line_report(result, as_json)
    if result["dataset"]["sha256"] is None:
        typer.echo(
            f"WARNING: datasets.lock does not pin {result['dataset']['name']} yet: "
            f"{result['dataset']['unpinned_reason']}",
            err=True,
        )


@line_app.command("repin")
def line_repin(
    path: str = typer.Argument(".", help="The line directory"),
    ref: str = typer.Option(None, "--ref", help="quantbox tag, branch or SHA (default: latest v* tag)"),
    quantbox_url: str = typer.Option(None, "--quantbox-url", help="quantbox git URL (default: the one the line pins)"),
    no_lock: bool = typer.Option(False, "--no-lock", help="Rewrite pyproject.toml only; skip `uv lock`"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Move a line's quantbox pin to REF, re-derive every exact pin from it, and refresh uv.lock.

    Then: `uv sync && uv run pytest -m reproduction` — red means the engine moved a number.
    """
    from . import line

    try:
        result = line.repin(path, ref=ref, quantbox_url=quantbox_url, lock=not no_lock)
    except line.LineError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc
    _line_report(result, as_json)


report_app = typer.Typer(help="Report data exported from run directories.")
app.add_typer(report_app, name="report")


@report_app.command("export")
def report_export(
    path: str = typer.Argument(help="A run directory, or a directory of arms (one run each)"),
    fmt: str = typer.Option(..., "--format", help="Export format: qute-research/finding-report@1"),
    out: str = typer.Option(None, "--out", "-o", help="Write here instead of stdout"),
    primary: str = typer.Option(None, "--primary", help="The arm the hero cards report (default: the first)"),
):
    """Export a run's returns, drawdowns, metrics, robustness across arms and provenance.

    The qute-research /finding-report renderer reads the result with --data; it owns
    the page and the contract. Exits 1 when there is no run under PATH, 2 on an
    unknown --format.
    """
    from .finding_export import FORMATS, dumps, export_finding_report

    if fmt not in FORMATS:
        raise typer.BadParameter(f"unknown format {fmt!r}; supported: {', '.join(FORMATS)}", param_hint="--format")
    try:
        payload = export_finding_report(path, primary=primary)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc
    text = dumps(payload)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


config_app = typer.Typer(help="What the runner does with a config.")
app.add_typer(config_app, name="config")


@config_app.command("explain")
def config_explain(
    config: str = typer.Argument(help="Path to config YAML"),
    json: bool = typer.Option(False, "--json", help="Print only the plan (quantbox/explain@1) on stdout"),
):
    """Resolve a config exactly as `quantbox run` would, without running it.

    Reports pipeline, engine, dataset (name, sha256, source, market), funding,
    execution.lag_bars, shorts and max leverage, strategies with resolved params,
    whether every plugin id resolves, and the artifact root — under run@1's field
    names. Exits 1 with the reason when a plugin, the dataset or the params do not resolve.
    """
    from .explain import explain_config

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    with contextlib.redirect_stdout(sys.stderr):  # stdout carries ONE JSON document
        doc = explain_config(cfg, PluginRegistry.discover(), config_path=config)
    if json:
        print(_as_json(doc))
    else:
        for key, value in doc.items():
            print(f"{key}: {value}")
    for err in doc["errors"]:
        typer.echo(f"ERROR: {err}", err=True)
    if not doc["ok"]:
        raise SystemExit(1)


def main():
    app()


if __name__ == "__main__":
    main()
