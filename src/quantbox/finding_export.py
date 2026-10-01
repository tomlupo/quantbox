"""Export a run (or a directory of arms) as ``qute-research/finding-report@1`` data.

quantbox emits the data; the qute-research ``/finding-report`` renderer owns the
page and its contract (``skills/finding-report/references/adapter-contract.md``
in qute-plugins — not restated here). This module is the adapter that contract
says each lab must write, written once in quantbox so no lab writes its own.

It reads runs through their ``quantbox/run@1`` manifest only (``files.returns``,
``files.metrics``) plus, for a variants run, ``variant_returns.parquet`` and
``variant_metrics.parquet``. An *arm* is one return series:

* a run directory (holds ``run_manifest.json``) is one arm — or one arm per
  variant when it is a variants run;
* any other directory is a directory of arms: each child that holds a run (or
  runs: the last by name, run ids ending in a UTC timestamp) is one arm, named
  after the child.

It never recomputes an engine metric: Sharpe, CAGR, drawdown in ``kpis`` and
``metrics`` are the engine's numbers. Equity, drawdown and the per-year returns
in ``robustness`` are compounded from the run's own returns file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .parquet_io import read_parquet
from .run_manifest import SCHEMA_ID as RUN_SCHEMA_ID

SCHEMA_ID = "qute-research/finding-report@1"
FORMATS = (SCHEMA_ID,)
FILENAME = "finding_report.json"

#: (metric key in metrics.json, label, renderer format) — rows of the metrics table, in order.
_METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("total_return", "Total return", "pct"),
    ("cagr", "CAGR", "pct"),
    ("sharpe", "Sharpe", "num"),
    ("sortino", "Sortino", "num"),
    ("calmar", "Calmar", "num"),
    ("annual_volatility", "Volatility", "pct"),
    ("max_drawdown", "Max drawdown", "pct"),
    ("win_rate", "Win rate", "pct"),
    ("traded_mean_gross_exposure", "Mean gross exposure", "num"),
    ("traded_mean_turnover", "Mean turnover", "num"),
)


@dataclass
class Arm:
    name: str
    returns: pd.Series  # indexed by timestamp
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    run_dir: Path


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def _manifest(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != RUN_SCHEMA_ID:
        raise ValueError(f"{run_dir}: manifest schema {manifest.get('schema')!r} is not {RUN_SCHEMA_ID!r}")
    return manifest


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    s = frame.set_index("date")[column].astype(float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _run_arms(run_dir: Path, name: str) -> list[Arm]:
    manifest = _manifest(run_dir)
    files = manifest.get("files") or {}
    variant_returns = run_dir / "variant_returns.parquet"
    if variant_returns.exists():
        wide = read_parquet(variant_returns)
        metrics = read_parquet(run_dir / "variant_metrics.parquet").set_index("variant")
        variants = [c for c in wide.columns if c != "date"]
        prefix = "" if name == "" else f"{name}/"
        return [
            Arm(
                f"{prefix}{v}",
                _series(wide, v),
                {k: val for k, val in metrics.loc[v].items() if k != "strategy"},
                manifest,
                run_dir,
            )
            for v in variants
        ]
    if not files.get("returns") or not files.get("metrics"):
        raise ValueError(f"{run_dir}: the manifest lists no returns/metrics file — not a backtest run")
    returns = _series(read_parquet(run_dir / files["returns"]), "returns")
    metrics = json.loads((run_dir / files["metrics"]).read_text(encoding="utf-8"))
    return [Arm(name or "Strategy", returns, metrics, manifest, run_dir)]


def _latest_run(directory: Path) -> Path | None:
    if (directory / "run_manifest.json").is_file():
        return directory
    runs = sorted(p.parent for p in directory.glob("*/run_manifest.json"))
    return runs[-1] if runs else None


def load_arms(path: str | Path) -> list[Arm]:
    """The arms under ``path``: a run directory, or a directory of arm directories."""
    path = Path(path)
    if (path / "run_manifest.json").is_file():
        return _run_arms(path, "")
    arms: list[Arm] = []
    for child in sorted(p for p in path.iterdir() if p.is_dir()) if path.is_dir() else []:
        run = _latest_run(child)
        if run is not None:
            arms.extend(_run_arms(run, child.name))
    if not arms:
        raise FileNotFoundError(f"{path}: no run found (no run_manifest.json in it or in any child directory)")
    return arms


# ---------------------------------------------------------------------------
# building the payload
# ---------------------------------------------------------------------------


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _round(v: Any) -> float | None:
    """A chart point: 6 decimals of a growth-of-1 curve is far below a pixel, and keeps the file small."""
    f = _num(v)
    return None if f is None else round(f, 6)


def _cell(v: Any) -> Any:
    """A table value the renderer accepts: number, string or null — never a boolean."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return v if not isinstance(v, float) or math.isfinite(v) else None
    return str(v)


def _date_labels(index: pd.DatetimeIndex) -> list[str]:
    naive = index.tz_convert(None) if index.tz is not None else index
    if (naive == naive.normalize()).all():
        return [d.strftime("%Y-%m-%d") for d in naive]
    return [d.isoformat() for d in naive]


def _series_block(arms: list[Arm]) -> dict[str, Any]:
    frame = pd.concat({a.name: a.returns for a in arms}, axis=1).sort_index()
    lines = []
    for arm in arms:
        r = frame[arm.name]
        live = r.notna()
        equity = (1 + r.fillna(0)).cumprod().where(live)
        # drawdown is given explicitly: the renderer cannot derive one from an
        # equity that ever touches zero (a liquidated book)
        drawdown = (equity / equity.cummax() - 1).where(live)
        lines.append(
            {
                "name": arm.name,
                "equity": [_round(v) for v in equity],
                "drawdown": [_round(v) for v in drawdown],
                "benchmark": "bench" in arm.name.lower(),
            }
        )
    return {"title": "Equity (growth of 1)", "dates": _date_labels(pd.DatetimeIndex(frame.index)), "lines": lines}


def _kpis(arm: Arm) -> list[dict[str, Any]]:
    m = arm.metrics
    kpis = []
    for key, label, fmt, finding_key in (
        ("sharpe", "Sharpe", "num", "backtest_sharpe"),
        ("cagr", "CAGR", "pct", None),
        ("max_drawdown", "Max drawdown", "pct", None),
    ):
        value = _num(m.get(key))
        if value is None:
            continue
        kpi = {"label": label, "value": value, "format": fmt, "note": arm.name}
        if finding_key:
            kpi["key"] = finding_key
        kpis.append(kpi)
    return kpis


def _metrics_block(arms: list[Arm]) -> dict[str, Any] | None:
    rows = []
    for key, label, fmt in _METRIC_ROWS:
        values = [_num(a.metrics.get(key)) for a in arms]
        if any(v is not None for v in values):
            rows.append({"label": label, "format": fmt, "values": values})
    if not rows:
        return None
    return {"title": "Metrics", "note": "as the engine reported them", "columns": [a.name for a in arms], "rows": rows}


def _robustness_block(arms: list[Arm]) -> dict[str, Any] | None:
    cells = []
    years: set[int] = set()
    for arm in arms:
        r = arm.returns.dropna()
        for year, chunk in r.groupby(r.index.year):
            value = float((1 + chunk).prod() - 1)
            years.add(int(year))
            cells.append(
                {
                    "row": arm.name,
                    "col": str(year),
                    "metric": "Return",
                    "value": value if math.isfinite(value) else None,
                    "status": "na" if not math.isfinite(value) else ("pass" if value > 0 else "fail"),
                    "sub": f"{len(chunk)} bars",
                }
            )
    if not cells:
        return None
    return {
        "intro": "Compounded return per calendar year, per arm. pass = positive year; a partial first "
        "or last year counts only its bars.",
        "rows": [a.name for a in arms],
        "cols": [str(y) for y in sorted(years)],
        "cells": cells,
    }


_PROVENANCE: tuple[tuple[str, Any], ...] = (
    ("run_id", lambda m: m.get("run_id")),
    ("pipeline", lambda m: m.get("pipeline")),
    (
        "engine",
        lambda m: (
            " ".join(str(x) for x in (m["engine"].get("name"), m["engine"].get("version")) if x)
            if m.get("engine")
            else None
        ),
    ),
    ("execution", lambda m: (m.get("execution") or {}).get("description")),
    ("execution.lag_bars", lambda m: (m.get("execution") or {}).get("lag_bars")),
    ("venue.allow_shorts", lambda m: (m.get("venue") or {}).get("allow_shorts")),
    ("venue.max_leverage", lambda m: (m.get("venue") or {}).get("max_leverage")),
    ("dataset", lambda m: (m.get("dataset") or {}).get("name")),
    ("dataset.source", lambda m: (m.get("dataset") or {}).get("source")),
    ("dataset.sha256", lambda m: (m.get("dataset") or {}).get("sha256")),
    ("funding.modelled", lambda m: (m.get("funding") or {}).get("modelled")),
    ("funding.sha256", lambda m: (m.get("funding") or {}).get("sha256")),
    ("n_trials", lambda m: m.get("n_trials")),
    ("config.sha256", lambda m: (m.get("config") or {}).get("sha256")),
    ("git.commit", lambda m: (m.get("git") or {}).get("commit")),
    ("git.dirty", lambda m: (m.get("git") or {}).get("dirty")),
)


def _provenance_table(arms: list[Arm]) -> dict[str, Any]:
    # one column per RUN: the variants of one run share its manifest
    runs: dict[Path, tuple[str, dict[str, Any]]] = {}
    for arm in arms:
        if arm.run_dir not in runs:
            runs[arm.run_dir] = (arm.name.split("/")[0], arm.manifest)
    labels = ["value"] if len(runs) == 1 else [label for label, _ in runs.values()]
    rows = [[field, *(_cell(get(m)) for _, m in runs.values())] for field, get in _PROVENANCE]
    return {
        "title": "Run provenance",
        "note": "from each run's quantbox/run@1 manifest",
        "columns": ["field", *labels],
        "rows": rows,
    }


def export_finding_report(path: str | Path, *, primary: str | None = None) -> dict[str, Any]:
    """The ``qute-research/finding-report@1`` payload for the run or arms under ``path``.

    ``primary`` names the arm the hero cards report (default: the first);
    the finding page refuses a Sharpe card that disagrees with the finding's
    ``backtest_sharpe``, so it must be the arm the finding is about.
    """
    arms = load_arms(path)
    names = [a.name for a in arms]
    if primary is None:
        hero = arms[0]
    elif primary in names:
        hero = arms[names.index(primary)]
    else:
        raise ValueError(f"no arm named {primary!r}; arms are {names}")

    payload: dict[str, Any] = {"schema": SCHEMA_ID, "lab": "quantbox"}
    if (Path(path) / "run_manifest.json").is_file():
        payload["run_id"] = arms[0].manifest.get("run_id")
    kpis = _kpis(hero)
    if kpis:
        payload["kpis"] = kpis
    payload["series"] = _series_block(arms)
    for key, block in (("metrics", _metrics_block(arms)), ("robustness", _robustness_block(arms))):
        if block is not None:
            payload[key] = block
    payload["tables"] = [_provenance_table(arms)]
    return payload


def dumps(payload: dict[str, Any]) -> str:
    """Strict, compact JSON (NaN/Infinity refused) — the export's serialised form."""
    return json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"


def write_finding_report(run_dir: str | Path) -> Path:
    """Write the export beside a run as ``finding_report.json`` (strict JSON); return its path."""
    out = Path(run_dir) / FILENAME
    out.write_text(dumps(export_finding_report(run_dir)), encoding="utf-8")
    return out
