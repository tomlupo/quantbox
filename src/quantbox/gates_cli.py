"""``quantbox gates dsr|nw|factor|bootstrap|episode|predreg|replicate`` — the acceptance gates as a CLI.

The maths lives in :mod:`quantbox.gates`; this module only reads files,
calls the gate and maps the result to an exit code:

  0  the gate PASSED          1  the gate ran and FAILED
  2  the gate COULD NOT be computed (unreadable or degenerate input, NaNs not
     opted into dropping, a refused argument) — never "no verdict, carry on"

With ``--json`` the verdict is one JSON object on stdout; a 2 prints
``{"error": ...}`` on stderr. Inputs are per-period return series as ``.csv`` or
``.parquet``. Paired inputs (``factor``, ``bootstrap``, ``episode --baseline``, ``predreg``)
are date-indexed — the first column is the date — and are inner-joined on it, in
date order; a numeric first column is refused rather than paired by row position.
A single-series file that carries a date first column is read in date order too.
A date that appears twice in any input file is refused (exit 2), never joined.

A returns file that sits in a run directory whose ``run_manifest.json`` says
``run.kind: research`` (same-bar under the explicit override, docs/adr/0006) is
marked: ``run_kind: "research"`` and ``research_note`` in ``--json``, and a
RESEARCH line under the verdict. The gate maths is unchanged; what the series
is not — a backtest — is said where the verdict is read.

``--period ME|QE|YE`` compounds every RETURN series a gate reads (the returns, a
baseline, the factor panel and its rf column) through
:func:`quantbox.metrics.period_returns` before the gate, and ``--start/--end`` keep
an inclusive window of dates; the file must then be dated. The predreg predictor
is a level, not a return, so ``--period`` is refused there (TOM-3698). The JSON
carries the ``window`` it applied.

``episode --episodes 1929-09:1932-06[,START:END...]`` removes those windows instead of the
largest drawdown (TOM-3700). A bound is a year, a month or a day and covers all of it
(``1932-06`` keeps June in the window); the file must be dated. ``ex_episode.nw`` is the
HAC t-stat on the kept rows with its lags restarted at each window, ``full.nw`` the
unmasked one beside it.

``dsr`` and ``nw`` take ``--rf dataset:<name>:<column>`` (TOM-3764): that column of a
pinned by-name RETURNS dataset (``quantbox.dataset.served_returns_column``: the
lock's bytes, the returns as stored) is the per-period risk-free rate. It gets the
same ``--period/--start/--end`` as the returns and is subtracted on the returns' own
dates, period t from period t (the bill's return over the period the strategy
earned; nothing later). The returns file must then be dated; a return date with no
rate is refused (exit 2), never dropped or filled. A tz-naive date on either side is
read as UTC. ``--json`` carries ``rf`` (the source, the dataset's pin and the periods).

``replicate`` (TOM-3699) has three verdicts: PASS (exit 0), FAIL and DEFECT (both
exit 1; ``verdict`` in ``--json`` and on the text line tells them apart). Its
``--period`` is the declared frequency of the correlation: both series are
compounded to it, and a date or period NaN on either side is masked, not refused.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import typer

gates_app = typer.Typer(
    name="gates",
    help=(
        "Acceptance gates on a return series: dsr, nw, factor, bootstrap, episode, predreg, replicate. "
        "Exit 0 pass, 1 fail, 2 could not compute."
    ),
    no_args_is_help=True,
)

RETURN_COLUMNS = ("returns", "return", "ret", "rets", "pnl", "r")

# the one field a human reads first, per gate (the --json payload carries everything)
HEADLINE = {
    "dsr": ("dsr_conservative", "threshold"),
    "nw": ("nw_tstat", "t_threshold"),
    "factor": ("alpha_tstat", "t_threshold"),
    "bootstrap": ("probability", "min_probability"),
    "episode": ("ex_episode", "threshold"),
    "predreg": ("pvalue_adjusted_min", "alpha"),
    "replicate": ("rho", "min_corr"),
}


def _input_error(msg: str) -> Exception:
    from quantbox.gates import GateInputError

    return GateInputError(msg)


def _read_frame(path: str, *, indexed: bool):
    import pandas as pd

    p = Path(path)
    if not p.is_file():
        raise _input_error(f"file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        from quantbox.parquet_io import read_parquet

        frame = read_parquet(p)
        if indexed and isinstance(frame.index, pd.RangeIndex) and frame.shape[1] > 1:
            frame = frame.set_index(frame.columns[0])
    elif suffix in (".csv", ".txt"):
        frame = pd.read_csv(p, encoding="utf-8", index_col=0 if indexed else None)
    else:
        raise _input_error(f"unsupported file type {suffix!r} (need .csv or .parquet): {p}")
    if not indexed:
        return _in_date_order(frame, p)
    # A numeric index is a row number or a return, never a date: to_datetime would read it
    # as epoch nanoseconds and the two files would pair POSITIONALLY on fabricated 1970 dates.
    if not isinstance(frame.index, pd.DatetimeIndex) and pd.api.types.is_numeric_dtype(frame.index):
        raise _input_error(
            f"{p}: a paired or windowed gate needs dates — the first column (or the parquet index) must be a date, "
            f"got a numeric {type(frame.index).__name__} ({frame.index.dtype})"
        )
    # one spelling of a date per row, whatever the file wrote ("2020-01-01" vs "... 00:00:00")
    try:
        frame.index = pd.to_datetime(frame.index, format="mixed")
    except (ValueError, TypeError) as exc:
        raise _input_error(f"{p}: the first column must be a date index for a paired or windowed gate ({exc})") from exc
    _refuse_duplicate_dates(frame.index, p)
    return frame


def _refuse_duplicate_dates(dates, path: Path) -> None:
    """Two rows for one date is a malformed series, not something a join may resolve.

    An inner join pairs the single row of one side against both rows of the other
    (or crashes, depending on the pandas version), and an unpaired gate counts the
    period twice; either way the verdict would be about data nobody wrote.
    """
    import pandas as pd

    dup = pd.Index(dates).duplicated()
    if dup.any():
        first = pd.Index(dates)[dup][0]
        shown = first.isoformat() if hasattr(first, "isoformat") else str(first)
        raise _input_error(
            f"{path}: duplicate date {shown} ({int(dup.sum())} duplicate row(s)) — "
            "one row per period is required; de-duplicate the file first"
        )


def _in_date_order(frame, path: Path):
    """An unpaired read in DATE order when the file carries dates, else in file order.

    HAC and the drawdown episode both depend on row order, so a dated file written
    newest-first (or shuffled) must not be read as if it were chronological.
    """
    import numpy as np
    import pandas as pd

    if isinstance(frame.index, pd.DatetimeIndex):
        _refuse_duplicate_dates(frame.index, path)
        return frame.sort_index(kind="stable")
    if frame.shape[1] == 0 or pd.api.types.is_numeric_dtype(frame.iloc[:, 0]):
        return frame
    try:
        dates = pd.to_datetime(frame.iloc[:, 0], format="mixed")
    except (ValueError, TypeError):
        return frame
    _refuse_duplicate_dates(dates, path)
    return frame.iloc[np.argsort(dates.to_numpy(), kind="stable")]


def _pick_column(frame, column: str | None, path: str) -> str:
    if column is not None:
        if column not in frame.columns:
            raise _input_error(f"column {column!r} not in {list(frame.columns)} ({path})")
        return column
    numeric = frame.select_dtypes("number")
    if numeric.shape[1] == 1:
        return str(numeric.columns[0])
    for name in RETURN_COLUMNS:
        for col in frame.columns:
            if str(col).lower() == name:
                return col
    raise _input_error(
        f"cannot tell which column of {path} holds the returns among {list(frame.columns)} — "
        "refusing to guess; pass --column"
    )


class _Window(NamedTuple):
    """``--period``, ``--start``, ``--end`` as one command gave them; all None = the file as read."""

    period: str | None = None
    start: str | None = None
    end: str | None = None

    @property
    def active(self) -> bool:
        return any(v is not None for v in self)

    def apply(self, s):
        """A return series compounded to ``period`` inside the window, or only cut to it."""
        if not self.active:
            return s
        from quantbox.metrics import Window, period_returns

        try:
            if self.period is None:
                return Window(self.start, self.end).clip(s)
            return period_returns(s, self.period, self.start, self.end)
        except ValueError as exc:
            raise _input_error(f"--period/--start/--end: {exc}") from exc


_NO_WINDOW = _Window()


def _series(path: str, column: str | None, *, indexed: bool, window: _Window = _NO_WINDOW):
    # a window needs the dates, so a windowed read is a dated read, in date order
    frame = _read_frame(path, indexed=indexed or window.active)
    if window.active:
        frame = frame.sort_index(kind="stable")
    col = _pick_column(frame, column, path)
    try:
        s = frame[col].astype(float)
    except (ValueError, TypeError) as exc:
        raise _input_error(f"column {col!r} of {path} is not numeric ({exc})") from exc
    return window.apply(s)


def _joined(returns: str, column: str | None, baseline: str, baseline_column: str | None, window: _Window = _NO_WINDOW):
    import pandas as pd

    c = _series(returns, column, indexed=True, window=window).rename("c")
    b = _series(baseline, baseline_column, indexed=True, window=window).rename("b")
    joined = pd.concat([c, b], axis=1, join="inner").sort_index()
    if joined.empty:
        raise _input_error(f"{returns} and {baseline} share no dates — nothing to pair")
    return joined


def _research_notes(*paths: str | None) -> list[str]:
    """The research note of every input that is a research run's returns file (docs/adr/0006)."""
    from quantbox.run_manifest import research_note

    notes = []
    for path in paths:
        if not path:
            continue
        manifest_path = Path(path).resolve().parent / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _input_error(f"{manifest_path}: unreadable run manifest beside {path} ({exc})") from exc
        note = research_note(manifest) if isinstance(manifest, dict) else None
        if note:
            notes.append(f"{path}: {note}")
    return notes


def _emit(
    gate: str, as_json: bool, compute: Callable[[], dict], *inputs: str | None, window: _Window = _NO_WINDOW
) -> None:
    from quantbox.gates import GateInputError

    try:
        notes = _research_notes(*inputs)
        out = compute()
    except GateInputError as exc:
        _fail(gate, as_json, str(exc))
    except Exception as exc:  # noqa: BLE001 — exit 1 MEANS "ran and failed"; a crash is "could not compute"
        _fail(gate, as_json, f"{type(exc).__name__}: {exc}")
    if notes:
        out["run_kind"] = "research"
        out["research_note"] = " | ".join(notes)
    if window.active:
        out["window"] = window._asdict()
    if as_json:
        typer.echo(json.dumps(out, indent=2, default=_jsonable))
    else:
        key, bar = HEADLINE[gate]
        value = out[key]["value"] if isinstance(out[key], dict) else out[key]
        # replicate names a third verdict, DEFECT; the exit code still reads gate_pass
        verdict = out.get("verdict") or ("PASS" if out["gate_pass"] else "FAIL")
        typer.echo(f"{gate}: {verdict} ({key}={value:.6g}, {bar}={out[bar]})")
        for note in notes:
            typer.echo(f"  {note}")
        if isinstance(out.get("rf"), dict):
            f = out["rf"]
            pin = (f.get("sha256") or "unpinned")[:12]
            typer.echo(
                f"  rf: {f['source']} subtracted on {f['n_periods']} dates, {f['first']}..{f['last']} (pin {pin})"
            )
        if window.active:
            typer.echo("  window: " + " ".join(f"{k}={v}" for k, v in window._asdict().items() if v is not None))
    raise typer.Exit(0 if out["gate_pass"] else 1)


def _fail(gate: str, as_json: bool, msg: str) -> None:
    typer.echo(json.dumps({"gate": gate, "error": msg}) if as_json else f"{gate}: CANNOT COMPUTE — {msg}", err=True)
    raise typer.Exit(2)


def _jsonable(obj: Any):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


_RETURNS = typer.Option(..., "--returns", help="per-period return series, .csv or .parquet")
_COLUMN = typer.Option(None, "--column", help="return column (else the only numeric one, or a known name)")
_DROP = typer.Option(False, "--allow-nonfinite-drop", help="drop NaN/Inf rows instead of refusing; count recorded")
_JSON = typer.Option(False, "--json", help="print the full verdict as JSON")
_LAGS = typer.Option(None, "--lags", help="HAC lags (default floor(4*(n/100)^(2/9)))")
_METRIC = typer.Option(
    "sharpe",
    "--metric",
    help="sharpe (per period) | mean | max_drawdown (leg = positive depth; JSON drawdowns name both signs)",
)
_COMPARE = typer.Option("diff", "--compare", help="diff (candidate - baseline) | ratio (candidate / baseline)")
_PASS_IF = typer.Option("above", "--pass-if", help="above | below the threshold, STRICTLY")
_BASELINE_COLUMN = typer.Option(None, "--baseline-column", help="return column of the baseline file")
_PERIOD = typer.Option(
    None,
    "--period",
    help="compound per-bar returns to ME | QE | YE first; a period counts only when the data shows both its ends",
)
_START = typer.Option(None, "--start", help="first date kept, inclusive (the file must be dated)")
_END = typer.Option(None, "--end", help="last date kept, inclusive (the file must be dated)")

_RF = typer.Option(
    None,
    "--rf",
    help="risk-free per period, dataset:<name>:<column> of a pinned returns dataset (e.g. dataset:sbbi:TBill_30d); "
    "subtracted from --returns on its own dates before the gate (the file must be dated)",
)


def _utc(index):
    """*index* as tz-aware UTC dates; a naive date is read as UTC (quantbox's convention)."""
    import pandas as pd

    index = pd.DatetimeIndex(index)
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


def _excess(returns: str, column: str | None, rf: str, window: _Window):
    """The returns minus the ``--rf`` dataset column on the same dates, and what was subtracted."""
    parts = rf.split(":")
    if len(parts) != 3 or parts[0] != "dataset" or not parts[1] or not parts[2]:
        raise _input_error(f"--rf must be dataset:<name>:<column>, got {rf!r}")
    from quantbox.dataset import served_returns_column

    # date order, not file order: the HAC standard error weights NEIGHBOURING rows
    r = _series(returns, column, indexed=True, window=window).sort_index(kind="stable")
    try:
        rate, resolution = served_returns_column(parts[1], parts[2])
    except Exception as exc:  # noqa: BLE001 — a dataset that cannot be read is "could not compute" (exit 2)
        raise _input_error(f"--rf {rf}: {type(exc).__name__}: {exc}") from exc
    rate.index = _utc(rate.index)
    rate = window.apply(rate)
    aligned = rate.reindex(_utc(r.index))
    missing = aligned.isna().to_numpy()
    if missing.any():
        dates = [d.date().isoformat() for d in aligned.index[missing][:5]]
        raise _input_error(
            f"--rf {rf}: no rate on {int(missing.sum())} of {len(r)} return dates (first {dates}); "
            "a return is never paired with another period's rate — cut the window or compound with --period"
        )
    excess = r - aligned.to_numpy()
    info = {
        "source": rf,
        "dataset": parts[1],
        "column": parts[2],
        "sha256": resolution.get("sha256"),
        "n_periods": len(excess),
        "first": aligned.index[0].date().isoformat() if len(excess) else None,
        "last": aligned.index[-1].date().isoformat() if len(excess) else None,
    }
    return excess, info


@gates_app.command("dsr")
def dsr(
    returns: str = typer.Option(None, "--returns", help="per-period return series, .csv or .parquet"),
    column: str = _COLUMN,
    allow_nonfinite_drop: bool = _DROP,
    sharpe: float = typer.Option(None, "--sharpe", help="ANNUALISED Sharpe (summary path, needs --periods)"),
    skew: float = typer.Option(None, "--skew", help="per-period skew (summary path)"),
    kurtosis: float = typer.Option(None, "--kurtosis", help="per-period PEARSON kurtosis, 3.0 = normal"),
    n_obs: int = typer.Option(None, "--n-obs", help="observation count (summary path)"),
    n_trials: str = typer.Option("1,5,10,20,50,100", "--n-trials", "--trials", help="trial counts; verdict at MAX"),
    trials_sr_std: float = typer.Option(None, "--trials-sr-std", help="per-period std of the trials' Sharpes"),
    periods: int = typer.Option(None, "--periods", help="periods per year; REQUIRED with --sharpe"),
    threshold: float = typer.Option(0.95, "--threshold", help="DSR acceptance bar, in (0,1)"),
    period: str = _PERIOD,
    start: str = _START,
    end: str = _END,
    rf: str = _RF,
    as_json: bool = _JSON,
):
    """Deflated Sharpe Ratio across a trial-count range (verdict at the most deflated end)."""

    def compute() -> dict:
        from quantbox.gates import dsr_gate, dsr_gate_from_returns

        summary = [sharpe, skew, kurtosis, n_obs]
        kw = dict(n_trials=n_trials, periods=periods, threshold=threshold, trials_sr_std=trials_sr_std)
        if returns is not None:
            if any(v is not None for v in summary):
                raise _input_error("--returns is exclusive with --sharpe/--skew/--kurtosis/--n-obs")
            if rf is not None:
                r, rf_info = _excess(returns, column, rf, window)
                out = dsr_gate_from_returns(r.to_numpy(), allow_nonfinite_drop=allow_nonfinite_drop, **kw)
                return {**out, "rf": rf_info}
            r = _series(returns, column, indexed=False, window=window)
            return dsr_gate_from_returns(r.to_numpy(), allow_nonfinite_drop=allow_nonfinite_drop, **kw)
        if rf is not None:
            raise _input_error("--rf is subtracted from a return series: it needs --returns, not --sharpe")
        if window.active:
            raise _input_error("--period/--start/--end shape a return series: they need --returns, not --sharpe")
        if sharpe is None:
            raise _input_error("give --returns <file>, or --sharpe with --skew, --kurtosis, --n-obs and --periods")
        missing = [n for n, v in zip(("--skew", "--kurtosis", "--n-obs"), summary[1:], strict=True) if v is None]
        if missing:
            raise _input_error(
                f"--sharpe needs {', '.join(missing)} too — there is no default skew/kurtosis "
                "(assuming a normal is exactly the bug a DSR exists to avoid)"
            )
        if periods is None:
            raise _input_error(
                "--sharpe needs --periods: the annualised Sharpe is divided by sqrt(periods), so the value "
                "moves the verdict (daily equities 252, daily crypto 365, weekly 52, monthly 12)"
            )
        if periods <= 0:
            raise _input_error(f"--periods must be positive, got {periods}")
        return dsr_gate(sr=sharpe / math.sqrt(periods), T=n_obs, skew=skew, kurtosis=kurtosis, **kw)

    window = _Window(period, start, end)
    _emit("dsr", as_json, compute, returns, window=window)


@gates_app.command("nw")
def nw(
    returns: str = _RETURNS,
    column: str = _COLUMN,
    allow_nonfinite_drop: bool = _DROP,
    t_threshold: float = typer.Option(2.0, "--t-threshold"),
    min_oos_periods: int = typer.Option(252, "--min-oos-periods"),
    oos_periods: int = typer.Option(None, "--oos-periods", help="test only the LAST N finite observations"),
    lags: int = _LAGS,
    period: str = _PERIOD,
    start: str = _START,
    end: str = _END,
    rf: str = _RF,
    as_json: bool = _JSON,
):
    """Newey-West HAC t-stat on the mean, plus a minimum out-of-sample window."""

    def compute() -> dict:
        from quantbox.gates import nw_gate

        rf_info = None
        if rf is not None:
            r, rf_info = _excess(returns, column, rf, window)
        else:
            r = _series(returns, column, indexed=False, window=window)
        out = nw_gate(
            r.to_numpy(),
            lags=lags,
            t_threshold=t_threshold,
            min_oos_periods=min_oos_periods,
            oos_periods=oos_periods,
            allow_nonfinite_drop=allow_nonfinite_drop,
        )
        return {**out, "rf": rf_info} if rf_info else out

    window = _Window(period, start, end)
    _emit("nw", as_json, compute, returns, window=window)


@gates_app.command("factor")
def factor(
    returns: str = _RETURNS,
    factors: str = typer.Option(..., "--factors", help="factor return panel, date-indexed"),
    column: str = _COLUMN,
    factor_columns: str = typer.Option(None, "--factor-columns", help="comma-separated (else every numeric one)"),
    rf: str = typer.Option(
        None, "--rf", help="per-period risk-free number, or a factor-panel column. Omit ONLY for excess returns"
    ),
    allow_nonfinite_drop: bool = _DROP,
    t_threshold: float = typer.Option(2.0, "--t-threshold"),
    min_obs: int = typer.Option(60, "--min-obs"),
    lags: int = _LAGS,
    period: str = _PERIOD,
    start: str = _START,
    end: str = _END,
    as_json: bool = _JSON,
):
    """Jensen's alpha after factor controls, HAC standard error, one-sided."""

    def compute() -> dict:
        import pandas as pd

        from quantbox.gates import factor_gate
        from quantbox.inference import require_finite

        strat = _series(returns, column, indexed=True, window=window).rename("__y__")
        panel = _read_frame(factors, indexed=True)
        if factor_columns is not None:
            names = [c.strip() for c in factor_columns.split(",") if c.strip()]
            missing = [c for c in names if c not in panel.columns]
            if missing:
                raise _input_error(f"factor columns {missing} not in {list(panel.columns)}")
        else:
            names = [str(c) for c in panel.select_dtypes("number").columns]
        rf_value, rf_column = None, None
        if rf is not None:
            try:
                rf_value = float(rf)
            except ValueError:
                if rf not in panel.columns:
                    raise _input_error(
                        f"--rf {rf!r} is neither a number nor a column of the factor panel {list(panel.columns)}"
                    ) from None
                rf_column = rf
            if rf_value is not None and not math.isfinite(rf_value):
                raise _input_error(f"--rf must be finite, got {rf!r}")
        if rf_column is not None:
            if factor_columns is not None and rf_column in names:
                raise _input_error(f"--rf column {rf_column!r} is also listed as a factor")
            names = [c for c in names if c != rf_column]
        if window.active:  # factor and rf columns are returns too: the same periods as the strategy
            shaped = names + ([rf_column] if rf_column is not None else [])
            panel = pd.concat({c: window.apply(panel[c].astype(float)) for c in shaped}, axis=1)
        parts = [strat, panel[names]]
        if rf_column is not None:
            parts.append(panel[[rf_column]].rename(columns={rf_column: "__rf__"}))
        # date order, not file order: the HAC standard error weights NEIGHBOURING rows
        joined = pd.concat(parts, axis=1, join="inner").sort_index(kind="stable")
        rows, dropped = require_finite(
            joined.to_numpy(dtype=float), allow_nonfinite_drop=allow_nonfinite_drop, what="aligned"
        )
        joined = pd.DataFrame(rows, columns=joined.columns)
        rf_arg = joined["__rf__"].to_numpy(dtype=float) if rf_column is not None else rf_value
        out = factor_gate(
            joined["__y__"].to_numpy(dtype=float),
            joined[names].to_numpy(dtype=float),
            names,
            rf=rf_arg,
            lags=lags,
            t_threshold=t_threshold,
            min_obs=min_obs,
            n_dropped=dropped,
        )
        if rf_column is not None:
            out["rf"] = rf_column
        return out

    window = _Window(period, start, end)
    _emit("factor", as_json, compute, returns, window=window)


@gates_app.command("bootstrap")
def bootstrap(
    returns: str = _RETURNS,
    baseline: str = typer.Option(..., "--baseline", help="baseline return series, date-indexed"),
    column: str = _COLUMN,
    baseline_column: str = _BASELINE_COLUMN,
    metric: str = _METRIC,
    compare: str = _COMPARE,
    pass_if: str = _PASS_IF,
    threshold: float = typer.Option(0.0, "--threshold"),
    draws: int = typer.Option(2000, "--draws"),
    mean_block: float = typer.Option(20.0, "--mean-block", help="mean block length, in periods"),
    seed: int = typer.Option(0, "--seed"),
    min_probability: float = typer.Option(0.95, "--min-probability", help="P(leg holds) that counts as support"),
    allow_nonfinite_drop: bool = _DROP,
    period: str = _PERIOD,
    start: str = _START,
    end: str = _END,
    as_json: bool = _JSON,
):
    """Paired stationary block bootstrap: P(candidate-vs-baseline leg holds) over resamples."""

    def compute() -> dict:
        from quantbox.gates import paired_block_bootstrap

        j = _joined(returns, column, baseline, baseline_column, window)
        out = paired_block_bootstrap(
            j["c"].to_numpy(),
            j["b"].to_numpy(),
            metric=metric,
            compare=compare,
            pass_if=pass_if,
            threshold=threshold,
            draws=draws,
            mean_block=mean_block,
            seed=seed,
            min_probability=min_probability,
            allow_nonfinite_drop=allow_nonfinite_drop,
        )
        out["first_date"], out["last_date"] = j.index[0], j.index[-1]
        return out

    window = _Window(period, start, end)
    _emit("bootstrap", as_json, compute, returns, baseline, window=window)


def _episode_windows(raw: str) -> list[tuple[str, object, object]]:
    """``"1929-09:1932-06,2008"`` -> ``[(text, first day, last day), ...]``; a bound covers its whole year/month/day."""
    import pandas as pd

    out = []
    for text in (t.strip() for t in raw.split(",")):
        if not text:
            continue
        parts = text.split(":")
        if len(parts) != 2 or not all(p.strip() for p in parts):
            raise _input_error(f"--episodes: {text!r} is not START:END (e.g. 1929-09:1932-06)")
        try:
            lo, hi = pd.Period(parts[0].strip()).start_time, pd.Period(parts[1].strip()).end_time.normalize()
        except (ValueError, TypeError) as exc:
            raise _input_error(f"--episodes: {text!r} has a bound that is not a date ({exc})") from exc
        out.append((text, lo, hi))
    if not out:
        raise _input_error("--episodes is empty: name at least one START:END window")
    return out


def _episode_positions(dates, windows) -> tuple[list[tuple[int, int]], list[dict]]:
    """Each window as the (first, last) row position of the dated rows inside it, with the dates it removes."""
    import numpy as np
    import pandas as pd

    from quantbox.metrics import Window

    rows = pd.Series(np.arange(len(dates)), index=dates)
    positions, named = [], []
    for text, lo, hi in windows:
        try:
            inside = Window(lo, hi).clip(rows)
        except ValueError as exc:
            raise _input_error(f"--episodes {text!r}: {exc}") from exc
        if inside.empty:
            raise _input_error(f"--episodes {text!r}: no row of the series is dated inside it")
        positions.append((int(inside.iloc[0]), int(inside.iloc[-1])))
        named.append({"window": text, "start_date": inside.index[0], "end_date": inside.index[-1]})
    return positions, named


@gates_app.command("episode")
def episode(
    returns: str = _RETURNS,
    baseline: str = typer.Option(None, "--baseline", help="baseline series (date-indexed); the episode is ITS"),
    column: str = _COLUMN,
    baseline_column: str = _BASELINE_COLUMN,
    episodes: str = typer.Option(
        None,
        "--episodes",
        help="START:END date windows to remove instead of the largest drawdown, comma-separated, inclusive, "
        "non-overlapping "
        "(1929-09:1932-06 removes Sep 1929 through Jun 1932); the file must be dated",
    ),
    metric: str = _METRIC,
    compare: str = _COMPARE,
    pass_if: str = _PASS_IF,
    threshold: float = typer.Option(0.0, "--threshold"),
    lags: int = _LAGS,
    allow_nonfinite_drop: bool = _DROP,
    period: str = _PERIOD,
    start: str = _START,
    end: str = _END,
    as_json: bool = _JSON,
):
    """The leg re-evaluated with the largest drawdown episode (or --episodes) excluded; HAC t-stat beside it."""

    def compute() -> dict:
        import numpy as np

        from quantbox.gates import episode_gate

        kw = dict(
            metric=metric,
            compare=compare,
            pass_if=pass_if,
            threshold=threshold,
            lags=lags,
            allow_nonfinite_drop=allow_nonfinite_drop,
        )
        windows = _episode_windows(episodes) if episodes is not None else None
        if baseline is None:
            r = _series(returns, column, indexed=windows is not None, window=window)
            frame, args = r.to_frame(), (r.to_numpy(),)
        else:
            frame = _joined(returns, column, baseline, baseline_column, window)
            args = (frame["c"].to_numpy(), frame["b"].to_numpy())
        # the gate's row positions count the FINITE rows (a dropped row is gone from both series)
        finite = np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)
        named = None
        if windows is not None:
            kw["episodes"], named = _episode_positions(frame.index[finite], windows)
        out = episode_gate(*args, **kw)
        if named is not None:
            for item, dated in zip(out["episodes"], named, strict=True):
                item.update(dated)
        ep = out["episode"]
        if baseline is not None and ep is not None and not out["n_nonfinite_dropped"]:
            for key in ("start", "trough", "end"):
                ep[f"{key}_date"] = frame.index[ep[key]]
        return out

    window = _Window(period, start, end)
    _emit("episode", as_json, compute, returns, baseline, window=window)


@gates_app.command("predreg")
def predreg(
    returns: str = _RETURNS,
    predictor: str = typer.Option(
        ..., "--predictor", help="predictor series (e.g. log D/P), date-indexed; may be the --returns file"
    ),
    column: str = _COLUMN,
    predictor_column: str = typer.Option(None, "--predictor-column", help="predictor column of the --predictor file"),
    rho_bound: float = typer.Option(
        0.9999, "--rho-bound", help="Lewellen's bound on the predictor's AR(1) rho (the true rho is at most this)"
    ),
    alpha: float = typer.Option(0.05, "--alpha", help="family-wise level over the bias-adjusted tests (Bonferroni)"),
    min_obs: int = typer.Option(60, "--min-obs", help="minimum (r_t, x_t-1) pairs"),
    allow_nonfinite_drop: bool = _DROP,
    period: str = _PERIOD,
    start: str = _START,
    end: str = _END,
    as_json: bool = _JSON,
):
    """Predictive regression r_t on x_t-1: Stambaugh bias-adjusted and Lewellen conditional slopes, one-sided.

    Rows must be consecutive periods: a date one file lacks inside the shared span is refused, but a
    period missing from BOTH files cannot be seen, and the lag would pair across it.
    """

    def compute() -> dict:
        import pandas as pd

        from quantbox.gates import predreg_gate

        if period is not None:
            raise _input_error(
                "--period compounds returns, and the predictor is a level (log D/P), not a return: "
                "give predreg per-period files; --start/--end alone cut both"
            )
        r = _series(returns, column, indexed=True, window=window)
        panel = _read_frame(predictor, indexed=True)
        numeric = [str(c) for c in panel.select_dtypes("number").columns]
        if predictor_column is None and len(numeric) != 1:
            # never the return-name guess: it would pick the RETURN column of a shared file
            raise _input_error(
                f"cannot tell which column of {predictor} is the predictor among {numeric} — pass --predictor-column"
            )
        pcol = _pick_column(panel, predictor_column, predictor) if predictor_column is not None else numeric[0]
        if Path(predictor).resolve() == Path(returns).resolve() and pcol == r.name:
            raise _input_error(f"--predictor column {pcol!r} is the return column itself")
        try:
            x = panel[pcol].astype(float)
        except (ValueError, TypeError) as exc:
            raise _input_error(f"column {pcol!r} of {predictor} is not numeric ({exc})") from exc
        x = window.apply(x)
        j = pd.concat([r.rename("c"), x.rename("b")], axis=1, join="inner").sort_index()
        if j.empty:
            raise _input_error(f"{returns} and {predictor} share no dates — nothing to pair")
        # The lag pairs NEIGHBOURING rows, so a period one file has and the other lacks inside
        # the shared span would silently pair a return with a predictor two periods old.
        lo, hi = j.index[0], j.index[-1]
        for path, s in ((returns, r), (predictor, x)):
            inside = s.index[(s.index >= lo) & (s.index <= hi)]
            missing = inside.difference(j.index)
            if len(missing):
                raise _input_error(
                    f"{path}: {len(missing)} date(s) inside the shared span have no partner in the other file "
                    f"(first {missing[0].isoformat()}) — the lag would pair the wrong periods; align the files first"
                )
        out = predreg_gate(
            j["c"].to_numpy(),
            j["b"].to_numpy(),
            rho_bound=rho_bound,
            alpha=alpha,
            min_obs=min_obs,
            allow_nonfinite_drop=allow_nonfinite_drop,
        )
        out["first_date"], out["last_date"] = j.index[0], j.index[-1]
        return out

    window = _Window(period, start, end)
    _emit("predreg", as_json, compute, returns, predictor, window=window)


@gates_app.command("replicate")
def replicate(
    returns: str = _RETURNS,
    reference: str = typer.Option(
        ..., "--reference", help="reference return series (a paper's, an index's), date-indexed"
    ),
    column: str = _COLUMN,
    reference_column: str = typer.Option(None, "--reference-column", help="return column of the reference file"),
    min_corr: float = typer.Option(0.85, "--min-corr", help="PASS at rho >= this"),
    floor: float = typer.Option(0.70, "--floor", help="DEFECT at rho < this (a broken replication)"),
    min_obs: int = typer.Option(12, "--min-obs", help="minimum overlapping periods; fewer is FAIL"),
    period: str = typer.Option(
        None,
        "--period",
        help="the correlation's frequency, ME | QE | YE: both series compounded to it (else the rows as read)",
    ),
    start: str = _START,
    end: str = _END,
    as_json: bool = _JSON,
):
    """Correlation with a reference series: PASS, FAIL, or DEFECT below the floor (pairwise NaN masking)."""

    def compute() -> dict:
        from quantbox.gates import replication_gate

        # unwindowed dated reads: the gate applies --period/--start/--end through metrics.correlation
        r = _series(returns, column, indexed=True)
        ref = _series(reference, reference_column, indexed=True)
        return replication_gate(
            r, ref, freq=period, start=start, end=end, min_corr=min_corr, floor=floor, min_obs=min_obs
        )

    window = _Window(period, start, end)
    _emit("replicate", as_json, compute, returns, reference, window=window)
