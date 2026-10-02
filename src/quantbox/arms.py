"""Arms as data: one base config plus named overrides or a grid, run in parallel.

A research line declares its arms ONCE instead of copy-pasting near-identical
YAMLs (TOM-1363)::

    # arms.yaml
    base: h32-base.yaml              # relative to this file, or an inline mapping
    overrides:                       # named arms: dotted path -> value
      minp-63:  {plugins.strategies.0.params.min_periods: 63}
      minp-150: {plugins.strategies.0.params.min_periods: 150}
    # ...or instead of `overrides`, a Cartesian grid:
    # grid:
    #   plugins.strategies.0.params.min_periods: [63, 100, 126, 150, 180]
    execution: {lag_bars: 1}         # optional: THE timing of every arm (same block as `quantbox sweep`)
    n_trials: 11                     # optional: the honest count when more were tried than listed
    parallel: {max_workers: 4, memory_budget_gb: 4, arm_memory_gb: 1.0}
    artifacts: {root: ./artifacts/h32-minp}   # optional; defaults to the base's artifacts.root

Each arm is an ordinary :func:`quantbox.runner.run_from_config` run and writes
its own ``quantbox/run@1`` manifest. The batch writes ``arms_summary.json``
(``quantbox/arms@1``): the arm list, ``n_trials``, the timing, and a link to
every arm's manifest. n_trials is stamped into every arm's manifest too, so the
DSR gate reads it instead of a hand count.

Batch-level settings are refused inside an arm: timing (``execution``),
``run.n_trials`` and ``artifacts`` — arms that differ in timing are not
comparable, and a trial count that varies per arm is not a count.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .execution import resolve_execution, timing_record

ARMS_SCHEMA_ID = "quantbox/arms@1"
SUMMARY_FILE = "arms_summary.json"
DEFAULT_ARM_MEMORY_GB = 1.0  # one backtest arm held ~0.9 GB on the H32/H08 lines

_TOP_KEYS = frozenset({"base", "overrides", "grid", "n_trials", "execution", "parallel", "artifacts"})
_PARALLEL_KEYS = frozenset({"max_workers", "memory_budget_gb", "arm_memory_gb"})
# Dotted-path prefixes an arm may not touch: they are properties of the BATCH.
_BATCH_LEVEL = {
    (
        "plugins",
        "pipeline",
        "params",
        "execution",
    ): "execution timing is batch-level: set `execution:` in the arms file",
    ("run", "n_trials"): "n_trials is batch-level: set `n_trials:` in the arms file",
    ("artifacts",): "artifacts is batch-level: set `artifacts:` in the arms file",
}


@dataclass
class ArmsSpec:
    path: Path
    base: dict[str, Any]
    base_path: Path | None = None
    overrides: dict[str, dict[str, Any]] | None = None
    grid: dict[str, list[Any]] | None = None
    n_trials: int | None = None
    parallel: dict[str, Any] = field(default_factory=dict)
    artifacts_root: str | None = None


def _pipeline_params(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.setdefault("plugins", {}).setdefault("pipeline", {}).setdefault("params", {})


def load_arms(path: str | Path) -> ArmsSpec:
    """Read and validate an arms file. Nothing runs; every refusal is a ``ValueError``."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: an arms file is a mapping")
    unknown = sorted(set(raw) - _TOP_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown}; allowed: {sorted(_TOP_KEYS)}")

    base_ref = raw.get("base")
    base_path: Path | None = None
    if isinstance(base_ref, str):
        base_path = (path.parent / base_ref).resolve()
        base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    elif isinstance(base_ref, Mapping):
        base = copy.deepcopy(dict(base_ref))
    else:
        raise ValueError(f"{path}: `base:` must be a config path or an inline config mapping")

    overrides, grid = raw.get("overrides"), raw.get("grid")
    if (overrides is None) == (grid is None):
        raise ValueError(f"{path}: declare exactly one of `overrides:` (named arms) or `grid:` (a Cartesian grid)")
    if overrides is not None and not (
        isinstance(overrides, Mapping) and overrides and all(isinstance(v, Mapping) for v in overrides.values())
    ):
        raise ValueError(f"{path}: `overrides:` maps arm names to {{dotted.path: value}} mappings")
    if grid is not None and not (
        isinstance(grid, Mapping) and grid and all(isinstance(v, list) and v for v in grid.values())
    ):
        raise ValueError(f"{path}: `grid:` maps dotted paths to non-empty lists of values")

    parallel = dict(raw.get("parallel") or {})
    bad = sorted(set(parallel) - _PARALLEL_KEYS)
    if bad:
        raise ValueError(f"{path}: parallel: unknown key(s) {bad}; allowed: {sorted(_PARALLEL_KEYS)}")

    # ONE timing for the batch. The block goes through the same resolver as
    # `quantbox run` and `quantbox sweep`; it must not contradict the base.
    if "execution" in raw:
        timing = resolve_execution(raw["execution"])
        params = _pipeline_params(base)
        if "execution" in params and resolve_execution(params["execution"]) != timing:
            raise ValueError(
                f"{path}: execution {dict(raw['execution'])!r} contradicts the base config's "
                f"plugins.pipeline.params.execution {params['execution']!r}"
            )
        # the whole block: a same-bar override (docs/adr/0006) travels with its lag
        params["execution"] = copy.deepcopy(dict(raw["execution"]))

    n_trials = raw.get("n_trials")
    if n_trials is not None and (not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1):
        raise ValueError(f"{path}: n_trials must be a positive integer, got {n_trials!r}")

    artifacts_root = (raw.get("artifacts") or {}).get("root")
    return ArmsSpec(
        path=path,
        base=base,
        base_path=base_path,
        overrides={str(k): dict(v) for k, v in overrides.items()} if overrides is not None else None,
        grid={str(k): list(v) for k, v in grid.items()} if grid is not None else None,
        n_trials=n_trials,
        parallel=parallel,
        artifacts_root=artifacts_root,
    )


def _split(dotted: str) -> tuple[str, ...]:
    return tuple(dotted.split("."))


def _set_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``a.b.0.c`` in ``cfg``. Intermediate keys must exist (a typo is refused);
    integer segments index lists."""
    parts = _split(dotted)
    for prefix, why in _BATCH_LEVEL.items():
        if parts[: len(prefix)] == prefix:
            raise ValueError(f"override {dotted!r}: {why}")
    node: Any = cfg
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        where = ".".join(parts[: i + 1])
        if isinstance(node, list):
            if not part.isdigit() or int(part) >= len(node):
                raise ValueError(f"override {dotted!r}: list index {part} out of range at {where!r} (len {len(node)})")
            if last:
                node[int(part)] = copy.deepcopy(value)
            else:
                node = node[int(part)]
        elif isinstance(node, dict):
            if last:
                node[part] = copy.deepcopy(value)
            elif part not in node:
                raise ValueError(f"override {dotted!r}: no key {part!r} at {where!r} in the base config")
            else:
                node = node[part]
        else:
            raise ValueError(f"override {dotted!r}: {where!r} is a scalar, not a mapping or list")


def _check_name(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"arm name {name!r} is not usable as a directory name")
    return name


def _label(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))


def _arm_table(spec: ArmsSpec) -> list[tuple[str, dict[str, Any]]]:
    """``[(arm name, {dotted path: value})]`` in declaration order."""
    if spec.overrides is not None:
        return [(_check_name(name), ov) for name, ov in spec.overrides.items()]
    assert spec.grid is not None
    keys = list(spec.grid)
    leaves = [_split(k)[-1] for k in keys]
    labels = leaves if len(set(leaves)) == len(leaves) else keys
    table = []
    for combo in itertools.product(*(spec.grid[k] for k in keys)):
        name = ",".join(f"{lbl}={_label(v)}" for lbl, v in zip(labels, combo, strict=True))
        table.append((_check_name(name), dict(zip(keys, combo, strict=True))))
    return table


def resolve_n_trials(spec: ArmsSpec, n_arms: int) -> int:
    """The honest trial count: the arms file's ``n_trials``, else the number of arms
    (never below a count the base config already states). Never fewer than the arms run."""
    if spec.n_trials is not None:
        if spec.n_trials < n_arms:
            raise ValueError(f"n_trials={spec.n_trials} is fewer than the {n_arms} arms this file runs")
        return spec.n_trials
    stated = (spec.base.get("run") or {}).get("n_trials")
    return max(n_arms, stated if isinstance(stated, int) and not isinstance(stated, bool) else 0)


def expand_arms(spec: ArmsSpec) -> list[tuple[str, dict[str, Any]]]:
    """Every arm as ``(name, full run config)``, with n_trials stamped in. Validates all
    arms before returning, so a bad override fails before the first backtest."""
    table = _arm_table(spec)
    names = [name for name, _ in table]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate arm names {dupes}")
    n_trials = resolve_n_trials(spec, len(table))
    out = []
    for name, ov in table:
        cfg = copy.deepcopy(spec.base)
        for dotted, value in ov.items():
            _set_path(cfg, dotted, value)
        cfg.setdefault("run", {})["n_trials"] = n_trials
        out.append((name, cfg))
    return out


# ----------------------------------------------------------------------
# Parallelism
# ----------------------------------------------------------------------


def _available_memory_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return None
    return None


def plan_workers(n_arms: int, *, max_workers: int, memory_budget_gb: float, arm_memory_gb: float) -> int:
    """How many arms run at once: bounded by ``max_workers``, the arm count, and
    ``floor(memory_budget_gb / arm_memory_gb)``. A budget below one arm is refused."""
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    if arm_memory_gb <= 0:
        raise ValueError(f"arm_memory_gb must be > 0, got {arm_memory_gb}")
    by_memory = int(memory_budget_gb // arm_memory_gb)
    if by_memory < 1:
        raise ValueError(
            f"memory budget {memory_budget_gb} GB cannot hold one arm ({arm_memory_gb} GB); "
            "raise memory_budget_gb or lower arm_memory_gb"
        )
    return max(1, min(max_workers, n_arms, by_memory))


def _pool_kwargs(workers: int) -> dict[str, Any]:
    """Executor settings for a parallel batch.

    spawn: no inherited threads or locks. One arm per child on Python 3.11+, so
    an arm's memory goes back to the OS before the next arm starts in that slot;
    3.10 has no ``max_tasks_per_child``, so there a worker serves arms in turn.
    """
    import multiprocessing

    kwargs: dict[str, Any] = {"max_workers": workers, "mp_context": multiprocessing.get_context("spawn")}
    if sys.version_info >= (3, 11):
        kwargs["max_tasks_per_child"] = 1
    return kwargs


def _run_one(name: str, cfg: dict[str, Any], config_path: str | None) -> dict[str, Any]:
    """Run one arm; never raises, so one arm's failure cannot hide another's result.

    Top-level so a spawned worker can import it. stdout goes to stderr: the
    parent's stdout may be carrying ``--json``.
    """
    try:
        from .registry import PluginRegistry
        from .runner import run_from_config

        with contextlib.redirect_stdout(sys.stderr):
            result = run_from_config(cfg, PluginRegistry.discover(), config_path=config_path)
        return {"name": name, "status": "ok", "run_id": result.run_id, "error": None}
    except Exception as exc:  # noqa: BLE001 — the batch reports it by arm name
        return {"name": name, "status": "failed", "run_id": None, "error": f"{type(exc).__name__}: {exc}"}


def run_arms(
    spec: ArmsSpec,
    *,
    max_workers: int | None = None,
    memory_budget_gb: float | None = None,
) -> dict[str, Any]:
    """Run every arm, write ``arms_summary.json`` and return it (with its ``path``).

    CLI arguments win over the file's ``parallel:`` block. With no budget anywhere,
    the budget is the memory available right now (``MemAvailable``); when that cannot
    be read, one arm runs at a time.

    Parallel arms run in ``spawn`` worker processes, so a SCRIPT that calls this
    with more than one worker needs the ``if __name__ == "__main__":`` guard;
    without it every arm fails (by name, ``BrokenProcessPool``). The CLI needs nothing.
    """
    arms = expand_arms(spec)
    par = spec.parallel
    arm_mem = float(par.get("arm_memory_gb", DEFAULT_ARM_MEMORY_GB))
    max_w = int(max_workers if max_workers is not None else par.get("max_workers", os.cpu_count() or 1))
    budget = memory_budget_gb if memory_budget_gb is not None else par.get("memory_budget_gb")
    budget_source = "declared"
    if budget is None:
        budget, budget_source = _available_memory_gb(), "available"
        if budget is None:
            budget, budget_source = arm_mem, "unknown"
    workers = plan_workers(len(arms), max_workers=max_w, memory_budget_gb=float(budget), arm_memory_gb=arm_mem)

    root = Path(spec.artifacts_root or (spec.base.get("artifacts") or {}).get("root") or "artifacts")
    if not root.is_absolute():
        root = Path.cwd() / root
    batch_id = f"{spec.path.stem}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    batch_dir = root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)

    jobs = []
    for name, cfg in arms:
        cfg["artifacts"] = {**(cfg.get("artifacts") or {}), "root": str(batch_dir / name)}
        jobs.append((name, cfg))
    config_path = str(spec.base_path or spec.path)

    results: dict[str, dict[str, Any]] = {}
    if workers == 1:
        for name, cfg in jobs:
            results[name] = _run_one(name, cfg, config_path)
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(**_pool_kwargs(workers)) as pool:
            futures = {name: pool.submit(_run_one, name, cfg, config_path) for name, cfg in jobs}
            for name, fut in futures.items():
                try:
                    results[name] = fut.result()
                except Exception as exc:  # noqa: BLE001 — e.g. a worker killed by the OOM killer
                    results[name] = {
                        "name": name,
                        "status": "failed",
                        "run_id": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

    overrides = dict(_arm_table(spec))
    arm_rows = []
    for name, _ in jobs:
        r = results[name]
        manifest_rel = metrics = None
        if r["status"] == "ok":
            manifest_rel = f"{name}/{r['run_id']}/run_manifest.json"
            metrics = json.loads((batch_dir / manifest_rel).read_text(encoding="utf-8")).get("metrics")
        arm_rows.append(
            {
                "name": name,
                "overrides": overrides[name],
                "status": r["status"],
                "run_id": r["run_id"],
                "manifest": manifest_rel,
                "metrics": metrics,
                "error": r["error"],
            }
        )
    failed = [row["name"] for row in arm_rows if row["status"] != "ok"]
    summary_path = batch_dir / SUMMARY_FILE
    summary = {
        "schema": ARMS_SCHEMA_ID,
        "batch_id": batch_id,
        "path": str(summary_path),
        "arms_file": str(spec.path),
        "base": str(spec.base_path) if spec.base_path else None,
        "status": "failed" if failed else "ok",
        "failed": failed,
        "n_trials": arms[0][1]["run"]["n_trials"],
        "execution": timing_record(resolve_execution(_pipeline_params(copy.deepcopy(spec.base)).get("execution"))),
        "parallel": {
            "workers": workers,
            "max_workers": max_w,
            "memory_budget_gb": float(budget),
            "memory_budget_source": budget_source,
            "arm_memory_gb": arm_mem,
        },
        "arms": arm_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False, default=str), encoding="utf-8")
    return summary
