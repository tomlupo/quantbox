"""``quantbox config explain``: what the runner will do with a config, without running it (TOM-1362).

The output is ``quantbox/explain@1`` (``artifact_schemas/config_explain.schema.json``,
next to ``run@1``). It uses the run@1 manifest's field names, so a plan and the record of
the run it plans compare key by key on :data:`SHARED_FIELDS`.

Nothing here decides what a config MEANS — every fact comes from the code a run takes:

- :func:`quantbox.runner.prepare_config` / :func:`~quantbox.runner.resolve_run` — profile
  merge, validation, plugin lookup and instantiation, the datasets.lock binding;
- ``pipeline.plan(params)`` — engine, execution timing, venue, frequency, costs, the
  variants guards and data-load params (:meth:`BacktestPipeline.plan`, which ``run()``
  itself reads, so every refusal on params alone happens there, for both);
- ``data.planned_paths(load_params)`` — the files the data plugin will read, a by-name
  dataset resolved against its lock (never loaded), then ``pipeline.check_planned_data``
  (a backtest refuses a missing prices file, as its ``run()`` does);
- :func:`quantbox.runner.strict_refusal` — the ``run.strict`` dataset-tier refusal;
- :mod:`quantbox.run_manifest` — the same functions that fill run@1's ``engine``,
  ``dataset`` and ``funding``; :func:`quantbox.overlays.overlay_record` — its ``overlays``.

The contract (``tests/test_config_explain.py``): a config the run refuses, explain refuses
(``ok`` false), from the same check. The one run-time refusal explain cannot see is
data-dependent (an empty window, no ticker overlap, a strict-mode capability check).

One fact is a plan, not a record: ``funding.modelled`` is true when the engine charges
funding and a funding file exists; the run records true only if that file has rows in the
backtest window.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from functools import lru_cache
from importlib.resources import files as _res_files
from pathlib import Path
from typing import Any

from . import run_manifest as _rm
from .execution import run_record
from .overlays import overlay_record
from .runner import (
    _config_block,
    _dataset_block,
    _hash_config,
    _plugins_block,
    _resolve_plugin_cls,
    plugin_refs,
    prepare_config,
    resolve_run,
    strict_refusal,
)

SCHEMA_ID = "quantbox/explain@1"

#: Top-level fields explain shares with run@1 (plus ``config.sha256`` and ``dataset``'s
#: ``name``/``sha256``/``source``/``tier``); a plan and its run agree on every one.
SHARED_FIELDS: tuple[str, ...] = (
    "asof",
    "mode",
    "pipeline",
    "config_hash",
    "plugins",
    "engine",
    "funding",
    "execution",
    "run",
    "venue",
    "overlays",
    "n_trials",
)


@lru_cache(maxsize=1)
def load_explain_schema() -> dict[str, Any]:
    path = _res_files("quantbox").joinpath("artifact_schemas").joinpath("config_explain.schema.json")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_explain(doc: dict[str, Any]) -> list[str]:
    """Every way *doc* fails explain@1, as messages; ``[]`` means it validates."""
    import jsonschema
    from referencing import Registry, Resource

    # explain@1 borrows run@1's definitions by reference, so the shared fields cannot drift.
    run_schema = _rm.load_run_schema()
    registry = Registry().with_resource(run_schema["$id"], Resource.from_contents(run_schema))
    validator = jsonschema.Draft202012Validator(load_explain_schema(), registry=registry)
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in validator.iter_errors(doc)
    ]


def _jsonable(obj: Any) -> Any:
    return _rm.json_safe(json.loads(json.dumps(obj, default=str)))


def _init_params(plugin: Any) -> dict[str, Any]:
    """The constructor params a plugin instance ended up with, defaults filled."""
    if dataclasses.is_dataclass(plugin):
        return {f.name: getattr(plugin, f.name) for f in dataclasses.fields(plugin) if f.init}
    return {}


def _strategy(plugin: Any, spec: dict[str, Any], **extra: Any) -> dict[str, Any]:
    meta = getattr(plugin, "meta", None)
    return {
        "name": getattr(meta, "name", spec.get("name")),
        "version": getattr(meta, "version", None),
        **extra,
        "params_init": _init_params(plugin),
        "params": spec.get("params") or {},
    }


def _variant_spec(variant: dict[str, Any]) -> dict[str, Any]:
    """A variant's strategy spec; a bare registry id is ``{"name": id}``, as resolve_run reads it."""
    spec = variant.get("strategy") or {}
    return spec if isinstance(spec, dict) else {"name": str(spec)}


def explain_config(
    cfg: dict[str, Any],
    registry: Any,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """The explain@1 document for *cfg*; ``ok`` false and ``errors`` say why a run could not start."""
    cfg = copy.deepcopy(cfg)
    doc: dict[str, Any] = {"schema": SCHEMA_ID, "ok": False, "errors": []}
    errors: list[str] = doc["errors"]

    try:
        prepare_config(cfg)
        n_trials = _rm.n_trials(cfg)
    except Exception as exc:  # noqa: BLE001 — every refusal is reported, not raised
        errors.append(f"config: {exc}")
        return doc
    mode = cfg["run"]["mode"]

    plugin_ids = []
    for role, group, attr, spec in plugin_refs(cfg):
        entry: dict[str, Any] = {"role": role, "group": group, "name": spec.get("name") or spec.get("source")}
        try:
            _resolve_plugin_cls(spec, getattr(registry, attr), group, mode=mode)
            entry["resolved"] = True
        except Exception as exc:  # noqa: BLE001
            entry["resolved"] = False
            entry["error"] = str(exc)
            errors.append(f"{role}: {exc}")
        plugin_ids.append(entry)
    doc["plugins_resolved"] = all(p["resolved"] for p in plugin_ids)
    doc["plugin_ids"] = plugin_ids
    if not doc["plugins_resolved"]:
        return doc

    try:
        resolved = resolve_run(cfg, registry, config_path=config_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"plugins: {exc}")
        return doc
    pipeline, data = resolved.pipeline, resolved.data

    try:
        plan_fn = getattr(pipeline, "plan", None)
        plan = plan_fn(resolved.pipeline_params) if callable(plan_fn) else None
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pipeline: {exc}")
        return doc
    try:
        planned_paths = getattr(data, "planned_paths", None)
        if callable(planned_paths):
            # What load_market_data will record as loaded_paths; run@1's dataset/funding read it.
            data.loaded_paths = planned_paths((plan or {}).get("load_params") or {})
            check = getattr(pipeline, "check_planned_data", None)
            if callable(check):  # the same refusal run() makes before it loads anything
                check(data, data.loaded_paths)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"dataset: {exc}")
        return doc

    dataset = _dataset_block(data)
    refusal = strict_refusal(cfg, resolved.mode, dataset.get("tier"))
    if refusal:
        errors.append(f"strict: {refusal}")
        return doc
    dataset.update(_rm.dataset_fields(data, dataset))
    resolution = getattr(data, "dataset_resolution", None)
    dataset["market"] = resolution.get("market") if isinstance(resolution, dict) else None

    funding_path = _rm._effective_path(data, "funding_rates", "funding_rates_path")
    modelled = bool(plan and plan.get("charges_funding") and funding_path and Path(funding_path).is_file())

    doc.update(
        {
            "ok": True,
            "asof": resolved.asof,
            "mode": resolved.mode,
            "pipeline": getattr(getattr(pipeline, "meta", None), "name", resolved.pipe_name),
            "config_hash": _hash_config(cfg),
            "config": _config_block(cfg, config_path),
            "plugins": _plugins_block(resolved),
            "engine": _rm.engine_block({"engine": plan["engine"]} if plan else {}),
            "dataset": dataset,
            "funding": _rm.funding_block(data, {"funding": {"modelled": modelled}}),
            "n_trials": n_trials,
            "strategies": [
                _strategy(p, s, weight=float(s.get("weight", 1.0)))
                for p, s in zip(resolved.strategy_plugins or [], cfg["plugins"].get("strategies") or [], strict=True)
            ]
            + [
                _strategy(resolved.variant_plugins[str(v["name"])], _variant_spec(v), variant=str(v["name"]))
                for v in resolved.pipeline_params.get("variants") or []
            ],
            "artifacts_root": str(Path(cfg["artifacts"]["root"]).resolve()),
        }
    )
    if plan:
        doc["execution"] = plan["execution"]
        doc["run"] = run_record(plan["execution"])
        doc["venue"] = plan["venue"]
    if getattr(pipeline, "accepts_overlays", False):
        # A pipeline that applies overlays records the chain, empty or not (run@1 minor 1).
        doc["overlays"] = overlay_record(resolved.overlay_chain)
    return _jsonable(doc)
