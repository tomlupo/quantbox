"""Export a run (or a directory of arms) as ``qute-research/finding-report@1`` data.

quantbox emits the data; the qute-research ``/finding-report`` renderer owns the
page and its contract (``skills/finding-report/references/adapter-contract.md``
in qute-plugins — not restated here). This module is the adapter that contract
says each lab must write, written once in quantbox so no lab writes its own.

It reads runs through their ``quantbox/run@1`` manifest only (``files.returns``,
``files.metrics``) plus, for a variants run, ``variant_returns.parquet`` (long:
``date, variant, returns``) and ``variant_metrics.parquet``. An *arm* is one return series:

* a run directory (holds ``run_manifest.json``) is one arm — or one arm per
  variant when it is a variants run;
* any other directory is a directory of arms: each child that holds a run (or
  runs: the newest by the UTC start timestamp that ends each run id, never by
  name) is one arm, named after the child.

It never recomputes an engine metric: Sharpe, CAGR, drawdown in ``kpis`` and
``metrics`` are the engine's numbers. A drawdown is reported under both names
(TOM-1627): "Max drawdown" is ``max_drawdown``, signed (<= 0); "Max drawdown
(depth)" is ``max_drawdown_abs`` (>= 0) — for a run written before run@1 minor 8,
the magnitude of its recorded ``max_drawdown``. The ``series`` drawdown lines are
signed fractions, as the renderer's contract defines them. Each ``metrics`` row
names the bar its statistic was computed at — "Sharpe (daily bars)" — read from
the arm's own return dates (TOM-3718); a Sharpe of daily returns is not a Sharpe
of monthly ones.

A RESEARCH run (``run.kind: research`` — same-bar under the explicit override,
docs/adr/0006) is never exported as a backtest: its hero cards do not report
the finding's ``backtest_sharpe`` and are toned ``bad``, the chart title and
provenance say research, and ``audit.axes`` carries a failed "Execution timing"
axis with the recorded reason. Equity, drawdown and the per-year returns
in ``robustness`` are compounded from the run's own returns file.

What the runs cannot carry, a line's own analysis can: ``extra`` names a JSON file
the line saved (its results), holding some of ``lab``, ``kpis``, ``metrics``,
``robustness``, ``audit`` and ``tables`` in the payload's own shapes. Its tables
come before the run provenance table; each other section replaces the engine's.
The series and the provenance always come from the runs, and a RESEARCH run keeps
what marks it: its failed "Execution timing" axes lead the line's audit, and the
line's kpis are refused. So a line renders its page from this export and its saved
results, with no adapter of its own.

Honest labels (TOM-3764). "Total return" stays the number the engine reported (format
``pct``: 303.58 renders +30357.94 %), and the row after it is the wealth multiple,
``1 + total_return`` (304.58), so a bare 303.58 is never read as a multiple; both
are numbers, never a display string. Text outputs (summary.md, the HTML report) print
:func:`quantbox.metrics.total_return_label` instead ("30358% (304.58x wealth)"). A promotion gate's status is never
copied from the results file when it can be checked: a gate with a number ``value``
and a one-comparator ``threshold`` (``>= 0.95``) gets ``pass`` or ``fail`` from that
comparison, and its threshold is shown as the comparator that HOLDS (``< 0.95`` when
the bar is missed, the bar kept in the note). A threshold that is not one comparator
keeps the line's status and the note says it was not checked. For a directory of arms
written by ``quantbox arms``, a "Fair baseline" table is filled from its
``arms_summary.json``: the baseline is the ``benchmark`` arm (a ``arm/variant`` name
names its arm), else the one arm the series draws as the benchmark
(:func:`_is_benchmark_name`); with none (or two) there is no table rather than a guess.
"""

from __future__ import annotations

import json
import math
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import compute_drawdown_series, sampling_frequency, total_return_label
from .parquet_io import read_parquet
from .run_history import run_started_at
from .run_manifest import SCHEMA_ID as RUN_SCHEMA_ID
from .run_manifest import research_note, run_kind

SCHEMA_ID = "qute-research/finding-report@1"
#: The sections a line's results file may give (``extra=``).
EXTRA_SECTIONS = ("lab", "kpis", "metrics", "robustness", "audit", "tables")
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
    ("max_drawdown", "Max drawdown", "pct"),  # signed, <= 0
    ("max_drawdown_abs", "Max drawdown (depth)", "pct"),  # the same drawdown, >= 0 (TOM-1627)
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


def _drawdown_named(metrics: dict[str, Any]) -> dict[str, Any]:
    """*metrics* with both drawdown names (TOM-1627): ``max_drawdown`` signed, ``max_drawdown_abs`` its depth.

    A run written before run@1 minor 8 recorded only the signed ``max_drawdown``; its
    depth is that number's magnitude, not a recomputed metric.
    """
    signed = metrics.get("max_drawdown")
    if "max_drawdown_abs" in metrics or not isinstance(signed, (int, float)) or isinstance(signed, bool):
        return metrics
    return {**metrics, "max_drawdown_abs": abs(float(signed))}


def _manifest(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != RUN_SCHEMA_ID:
        raise ValueError(f"{run_dir}: manifest schema {manifest.get('schema')!r} is not {RUN_SCHEMA_ID!r}")
    return manifest


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    # A run written before TOM-1529 on a prices index with no name has its date column
    # as reset_index() named it, "index"; never a variant or a metric.
    date = "date" if "date" in frame.columns or "index" not in frame.columns else "index"
    s = frame.set_index(date)[column].astype(float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _run_arms(run_dir: Path, name: str) -> list[Arm]:
    manifest = _manifest(run_dir)
    files = manifest.get("files") or {}
    variant_returns = run_dir / "variant_returns.parquet"
    if variant_returns.exists():
        # long (date, variant, returns): a variant may be named anything, "date" included
        long = read_parquet(variant_returns)
        metrics = read_parquet(run_dir / "variant_metrics.parquet").set_index("variant")
        variants = list(dict.fromkeys(long["variant"]))
        prefix = "" if name == "" else f"{name}/"
        return [
            Arm(
                f"{prefix}{v}",
                _series(long[long["variant"] == v], "returns"),
                _drawdown_named({k: val for k, val in metrics.loc[v].items() if k != "strategy"}),
                manifest,
                run_dir,
            )
            for v in variants
        ]
    if not files.get("returns") or not files.get("metrics"):
        raise ValueError(f"{run_dir}: the manifest lists no returns/metrics file — not a backtest run")
    returns = _series(read_parquet(run_dir / files["returns"]), "returns")
    metrics = _drawdown_named(json.loads((run_dir / files["metrics"]).read_text(encoding="utf-8")))
    return [Arm(name or "Strategy", returns, metrics, manifest, run_dir)]


def _latest_run(directory: Path) -> Path | None:
    """The run in ``directory``: itself, or its newest child run by the run id's start timestamp.

    Never by name: a run id orders by config hash before its timestamp. Two or
    more runs that cannot be told apart in time are refused, not guessed.
    """
    if (directory / "run_manifest.json").is_file():
        return directory
    runs = sorted(p.parent for p in directory.glob("*/run_manifest.json"))
    if len(runs) <= 1:
        return runs[0] if runs else None
    started = {r: run_started_at(r.name) for r in runs}
    undated = [r.name for r, ts in started.items() if ts is None]
    if undated:
        raise ValueError(f"{directory}: cannot tell the latest run, no start timestamp in run id(s) {undated}")
    newest = max(started.values())
    latest = [r for r, ts in started.items() if ts == newest]
    if len(latest) > 1:
        raise ValueError(f"{directory}: cannot tell the latest run, {[r.name for r in latest]} started together")
    return latest[0]


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


def _is_benchmark_name(name: str) -> bool:
    """An arm that is a benchmark by its name alone: it says "bench", or it is buy-and-hold.

    The ONE rule for both the series' benchmark line and the fair-baseline table.
    """
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return "bench" in key or "buyhold" in key or "buyandhold" in key


def _series_block(arms: list[Arm], benchmark: str | None = None) -> dict[str, Any]:
    frame = pd.concat({a.name: a.returns for a in arms}, axis=1).sort_index()
    lines = []
    for arm in arms:
        r = frame[arm.name]
        live = r.notna()
        equity = (1 + r.fillna(0)).cumprod().where(live)
        # drawdown is given explicitly: the renderer cannot derive one from an
        # equity that ever touches zero (a liquidated book)
        drawdown = compute_drawdown_series(equity).where(live)
        lines.append(
            {
                "name": arm.name,
                "equity": [_round(v) for v in equity],
                "drawdown": [_round(v) for v in drawdown],
                "benchmark": arm.name == benchmark if benchmark else _is_benchmark_name(arm.name),
            }
        )
    title = "Equity (growth of 1)"
    if any(run_kind(a.manifest) == "research" for a in arms):
        title += " — RESEARCH run, same-bar fills, not a backtest"
    return {"title": title, "dates": _date_labels(pd.DatetimeIndex(frame.index)), "lines": lines}


def _kpis(arm: Arm) -> list[dict[str, Any]]:
    m = arm.metrics
    research = run_kind(arm.manifest) == "research"
    kpis = []
    for key, label, fmt, finding_key in (
        ("sharpe", "Sharpe", "num", "backtest_sharpe"),
        ("cagr", "CAGR", "pct", None),
        ("max_drawdown", "Max drawdown", "pct", None),
        ("max_drawdown_abs", "Max drawdown (depth)", "pct", None),
    ):
        value = _num(m.get(key))
        if value is None:
            continue
        if research:
            # Not "Sharpe": the renderer reads that label as the finding's backtest_sharpe.
            kpis.append(
                {
                    "label": f"Same-bar {label} (research)",
                    "value": value,
                    "format": fmt,
                    "tone": "bad",
                    "note": f"{arm.name} — RESEARCH run, same-bar fills, not a backtest",
                }
            )
            continue
        kpi = {"label": label, "value": value, "format": fmt, "note": arm.name}
        if finding_key:
            kpi["key"] = finding_key
        kpis.append(kpi)
    return kpis


def _audit_block(arms: list[Arm]) -> dict[str, Any] | None:
    """A failed "Execution timing" axis per research run: the page must not read as a backtest."""
    axes, seen = [], set()
    for arm in arms:
        note = research_note(arm.manifest)
        if note is None or arm.run_dir in seen:
            continue
        seen.add(arm.run_dir)
        axes.append({"name": "Execution timing", "status": "fail", "note": f"{arm.name.split('/')[0]}: {note}"})
    return {"axes": axes} if axes else None


def _bars(arms: list[Arm]) -> str:
    """The bar every engine statistic was computed at, as a label suffix (TOM-3718).

    The engine computes each one from the arm's per-bar returns, the series this
    export reads, so the bar is named from that series' own dates
    (:func:`quantbox.metrics.sampling_frequency`). Arms on different bars name each
    distinct bar once, in first-seen order (not one per column).
    """
    bars = list(dict.fromkeys(sampling_frequency(a.returns.dropna().index) for a in arms))
    return "bar unknown" if bars == ["unknown"] else f"{' / '.join(bars)} bars"


def _metrics_block(arms: list[Arm]) -> dict[str, Any] | None:
    """The engine's statistics, one row each, its label naming the bar it was sampled at (TOM-3718).

    The renderer refuses a row key it does not know, so the bar goes in the label:
    "Sharpe (daily bars)". The values are the engine's, never recomputed.
    """
    rows = []
    bars = _bars(arms)
    for key, label, fmt in _METRIC_ROWS:
        values = [_num(a.metrics.get(key)) for a in arms]
        if any(v is not None for v in values):
            rows.append({"label": _with_bars(label, bars), "format": fmt, "values": values})
            if key == "total_return":  # the wealth multiple beside it, so 303.58 never reads as one
                wealth = [None if v is None else 1 + v for v in values]
                rows.append(
                    {"label": _with_bars("Wealth multiple (1 + total return)", bars), "format": "num", "values": wealth}
                )
    if not rows:
        return None
    return {"title": "Metrics", "note": "as the engine reported them", "columns": [a.name for a in arms], "rows": rows}


def _with_bars(label: str, bars: str) -> str:
    """``Max drawdown (depth)`` + ``daily bars`` -> ``Max drawdown (depth, daily bars)``."""
    return f"{label[:-1]}, {bars})" if label.endswith(")") else f"{label} ({bars})"


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
    ("run.kind", run_kind),
    ("execution", lambda m: (m.get("execution") or {}).get("description")),
    ("execution.lag_bars", lambda m: (m.get("execution") or {}).get("lag_bars")),
    ("execution.same_bar", lambda m: (m.get("execution") or {}).get("same_bar")),
    ("execution.same_bar_reason", lambda m: (m.get("execution") or {}).get("same_bar_reason")),
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


#: A gate threshold quantbox can check against the value: ONE comparator and a number.
_COMPARATOR = re.compile(r"^\s*(>=|<=|>|<|≥|≤)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
_OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt}
#: The comparator that holds when the bar's does not: value < bar when value >= bar fails.
_NEGATED = {">=": "<", ">": "<=", "<=": ">", "<": ">="}


def _checked_gate(gate: dict[str, Any]) -> dict[str, Any]:
    """*gate* with its status derived from ``value`` against ``threshold`` (module docstring).

    Pass when the comparator holds, fail when it does not; the threshold shown is the
    comparator that holds and the note keeps the bar and the status the line wrote,
    when it differs. A gate that cannot be checked is returned with a note saying so.
    """
    threshold, value = gate.get("threshold"), gate.get("value")
    match = _COMPARATOR.match(threshold) if isinstance(threshold, str) else None
    number = None if isinstance(value, bool) else _num(value)
    note = gate.get("note")
    if match is None or number is None:
        why = f"threshold {threshold!r} is not one comparator" if match is None else f"value {value!r} is not a number"
        said = f"status as the line wrote it ({why}; not checked)"
        return {**gate, "note": f"{said}. {note}" if note else said}
    op = {"≥": ">=", "≤": "<="}.get(match.group(1), match.group(1))
    bar = match.group(2)
    holds = _OPS[op](number, float(bar))
    status = "pass" if holds else "fail"
    said = [] if holds else [f"needs {op} {bar}"]
    if str(gate.get("status", "")).strip().lower() != status:
        said.append(f"the line wrote {gate.get('status')!r}")
    prefix = "; ".join(said)
    shown = f"{op} {bar}" if holds else f"{_NEGATED[op]} {bar}"
    out = {**gate, "status": status, "threshold": shown}
    if prefix:
        out["note"] = f"{prefix}. {note}" if note else prefix
    return out


def _baseline_name(names: list[str], benchmark: str | None) -> str | None:
    """The fair-baseline arm of *names* (arms_summary.json's), else None.

    *benchmark* when given: an arm, or an ``arm/variant`` as :func:`load_arms` names a
    run with variants, which names its arm. Otherwise the ONE arm that is a benchmark
    by name (:func:`_is_benchmark_name`, the series' rule).
    """
    if benchmark is not None:
        arm = benchmark if benchmark in names else benchmark.split("/", 1)[0]
        return arm if arm in names else None
    hits = [n for n in names if _is_benchmark_name(n)]
    return hits[0] if len(hits) == 1 else None


def _baseline_table(path: Path, benchmark: str | None) -> dict[str, Any] | None:
    """The "Fair baseline" table from an arms batch's ``arms_summary.json``; None when there is none.

    One row per arm the batch ran (``status: ok``), the baseline's role named, with the
    engine's own Sharpe, CAGR, max drawdown and total return as ``quantbox arms`` recorded them.
    """
    summary_path = path / "arms_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    arms = [a for a in summary.get("arms") or [] if a.get("status") == "ok" and isinstance(a.get("metrics"), dict)]
    baseline = _baseline_name([a["name"] for a in arms], benchmark)
    if baseline is None or len(arms) < 2:
        return None

    def pct(v: Any) -> str | None:
        f = _num(v)
        return None if f is None else f"{f * 100:.2f}%"

    rows = []
    for a in arms:
        m = a["metrics"]
        tr = _num(m.get("total_return"))
        rows.append(
            [
                a["name"],
                "fair baseline" if a["name"] == baseline else "candidate",
                _num(m.get("sharpe")),
                pct(m.get("cagr")),
                pct(m.get("max_drawdown")),
                None if tr is None else total_return_label(tr),
            ]
        )
    return {
        "title": f"Fair baseline: {baseline}",
        "note": f"from {summary.get('batch_id') or path.name}/arms_summary.json, as the engine reported each arm; "
        "the baseline is the --benchmark arm, else the one arm named as a benchmark (bench, buy-and-hold)",
        "columns": ["arm", "role", "Sharpe", "CAGR", "Max drawdown", "Total return"],
        "rows": rows,
    }


def _load_extra(path: str | Path) -> dict[str, Any]:
    """A line's results file: some of :data:`EXTRA_SECTIONS`, none empty; anything else is refused."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a results file is a JSON object of payload sections")
    unknown = sorted(set(data) - set(EXTRA_SECTIONS))
    if unknown:
        raise ValueError(
            f"{path}: {unknown} cannot come from a results file; it gives only {list(EXTRA_SECTIONS)}"
            " (series and provenance come from the runs)"
        )
    empty = sorted(k for k, v in data.items() if not v)
    if empty:
        raise ValueError(f"{path}: empty section(s) {empty}; omit them instead")
    if "tables" in data and not isinstance(data["tables"], list):
        raise ValueError(f"{path}: tables is a list of tables")
    return data


def export_finding_report(
    path: str | Path,
    *,
    primary: str | None = None,
    benchmark: str | None = None,
    extra: str | Path | None = None,
) -> dict[str, Any]:
    """The ``qute-research/finding-report@1`` payload for the run or arms under ``path``.

    ``primary`` names the arm the hero cards report (default: the first);
    the finding page refuses a Sharpe card that disagrees with the finding's
    ``backtest_sharpe``, so it must be the arm the finding is about.
    ``benchmark`` names the arm drawn as the benchmark (default: any arm whose
    name contains "bench"). ``extra`` is a line's results file (module docstring).
    """
    arms = load_arms(path)
    names = [a.name for a in arms]
    for label, name in (("primary", primary), ("benchmark", benchmark)):
        if name is not None and name not in names:
            raise ValueError(f"no arm named {name!r} ({label}); arms are {names}")
    hero = arms[names.index(primary)] if primary else arms[0]
    line = _load_extra(extra) if extra is not None else {}

    payload: dict[str, Any] = {"schema": SCHEMA_ID, "lab": "quantbox"}
    if (Path(path) / "run_manifest.json").is_file():
        payload["run_id"] = arms[0].manifest.get("run_id")
    kpis = _kpis(hero)
    if kpis:
        payload["kpis"] = kpis
    payload["series"] = _series_block(arms, benchmark)
    for key, block in (
        ("metrics", _metrics_block(arms)),
        ("robustness", _robustness_block(arms)),
        ("audit", _audit_block(arms)),
    ):
        if block is not None:
            payload[key] = block
    baseline = _baseline_table(Path(path), benchmark)
    payload["tables"] = [*([baseline] if baseline else []), *line.pop("tables", []), _provenance_table(arms)]
    research_axes = (payload.get("audit") or {}).get("axes", [])
    if research_axes and "kpis" in line:
        raise ValueError(f"{extra}: a RESEARCH run's hero cards are the engine's; drop kpis from the results file")
    if research_axes and "audit" in line:
        line["audit"] = {**line["audit"], "axes": [*research_axes, *line["audit"].get("axes", [])]}
    payload.update(line)
    gates = (payload.get("audit") or {}).get("gates")
    if gates:
        payload["audit"] = {
            **payload["audit"],
            "gates": [_checked_gate(g) if isinstance(g, dict) else g for g in gates],
        }
    return payload


def dumps(payload: dict[str, Any]) -> str:
    """Strict, compact JSON (NaN/Infinity refused) — the export's serialised form."""
    return json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"


def write_finding_report(run_dir: str | Path) -> Path:
    """Write the export beside a run as ``finding_report.json`` (strict JSON); return its path."""
    out = Path(run_dir) / FILENAME
    out.write_text(dumps(export_finding_report(run_dir)), encoding="utf-8")
    return out
