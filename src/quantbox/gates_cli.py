"""``quantbox gates dsr|nw|factor|bootstrap|episode`` — the acceptance gates as a CLI.

The maths lives in :mod:`quantbox.analysis.gates`; this module only reads files,
calls the gate and maps the result to an exit code:

  0  the gate PASSED          1  the gate ran and FAILED
  2  the gate COULD NOT be computed (unreadable or degenerate input, NaNs not
     opted into dropping, a refused argument) — never "no verdict, carry on"

With ``--json`` the verdict is one JSON object on stdout; a 2 prints
``{"error": ...}`` on stderr. Inputs are per-period return series as ``.csv`` or
``.parquet``. Paired inputs (``factor``, ``bootstrap``, ``episode --baseline``)
are date-indexed — the first column is the date — and are inner-joined on it, in
date order; a numeric first column is refused rather than paired by row position.
A single-series file that carries a date first column is read in date order too.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

gates_app = typer.Typer(
    name="gates",
    help="Acceptance gates on a return series. Exit 0 pass, 1 fail, 2 could not compute.",
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
}


def _input_error(msg: str) -> Exception:
    from quantbox.analysis.gates import GateInputError

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
        return _in_date_order(frame)
    # A numeric index is a row number or a return, never a date: to_datetime would read it
    # as epoch nanoseconds and the two files would pair POSITIONALLY on fabricated 1970 dates.
    if not isinstance(frame.index, pd.DatetimeIndex) and pd.api.types.is_numeric_dtype(frame.index):
        raise _input_error(
            f"{p}: a paired gate needs dates — the first column (or the parquet index) must be a date, "
            f"got a numeric {type(frame.index).__name__} ({frame.index.dtype})"
        )
    # one spelling of a date per row, whatever the file wrote ("2020-01-01" vs "... 00:00:00")
    try:
        frame.index = pd.to_datetime(frame.index)
    except (ValueError, TypeError) as exc:
        raise _input_error(f"{p}: the first column must be a date index for a paired gate ({exc})") from exc
    return frame


def _in_date_order(frame):
    """An unpaired read in DATE order when the file carries dates, else in file order.

    HAC and the drawdown episode both depend on row order, so a dated file written
    newest-first (or shuffled) must not be read as if it were chronological.
    """
    import numpy as np
    import pandas as pd

    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.sort_index(kind="stable")
    if frame.shape[1] == 0 or pd.api.types.is_numeric_dtype(frame.iloc[:, 0]):
        return frame
    try:
        dates = pd.to_datetime(frame.iloc[:, 0], format="mixed")
    except (ValueError, TypeError):
        return frame
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


def _series(path: str, column: str | None, *, indexed: bool):
    frame = _read_frame(path, indexed=indexed)
    col = _pick_column(frame, column, path)
    try:
        return frame[col].astype(float)
    except (ValueError, TypeError) as exc:
        raise _input_error(f"column {col!r} of {path} is not numeric ({exc})") from exc


def _joined(returns: str, column: str | None, baseline: str, baseline_column: str | None):
    import pandas as pd

    c = _series(returns, column, indexed=True).rename("c")
    b = _series(baseline, baseline_column, indexed=True).rename("b")
    joined = pd.concat([c, b], axis=1, join="inner").sort_index()
    if joined.empty:
        raise _input_error(f"{returns} and {baseline} share no dates — nothing to pair")
    return joined


def _emit(gate: str, as_json: bool, compute: Callable[[], dict]) -> None:
    from quantbox.analysis.gates import GateInputError

    try:
        out = compute()
    except GateInputError as exc:
        _fail(gate, as_json, str(exc))
    except Exception as exc:  # noqa: BLE001 — exit 1 MEANS "ran and failed"; a crash is "could not compute"
        _fail(gate, as_json, f"{type(exc).__name__}: {exc}")
    if as_json:
        typer.echo(json.dumps(out, indent=2, default=_jsonable))
    else:
        key, bar = HEADLINE[gate]
        value = out[key]["value"] if isinstance(out[key], dict) else out[key]
        verdict = "PASS" if out["gate_pass"] else "FAIL"
        typer.echo(f"{gate}: {verdict} ({key}={value:.6g}, {bar}={out[bar]})")
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
_METRIC = typer.Option("sharpe", "--metric", help="sharpe (per period) | mean | max_drawdown (positive fraction)")
_COMPARE = typer.Option("diff", "--compare", help="diff (candidate - baseline) | ratio (candidate / baseline)")
_PASS_IF = typer.Option("above", "--pass-if", help="above | below the threshold, STRICTLY")
_BASELINE_COLUMN = typer.Option(None, "--baseline-column", help="return column of the baseline file")


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
    as_json: bool = _JSON,
):
    """Deflated Sharpe Ratio across a trial-count range (verdict at the most deflated end)."""

    def compute() -> dict:
        from quantbox.analysis.gates import dsr_gate, dsr_gate_from_returns

        summary = [sharpe, skew, kurtosis, n_obs]
        kw = dict(n_trials=n_trials, periods=periods, threshold=threshold, trials_sr_std=trials_sr_std)
        if returns is not None:
            if any(v is not None for v in summary):
                raise _input_error("--returns is exclusive with --sharpe/--skew/--kurtosis/--n-obs")
            r = _series(returns, column, indexed=False)
            return dsr_gate_from_returns(r.to_numpy(), allow_nonfinite_drop=allow_nonfinite_drop, **kw)
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

    _emit("dsr", as_json, compute)


@gates_app.command("nw")
def nw(
    returns: str = _RETURNS,
    column: str = _COLUMN,
    allow_nonfinite_drop: bool = _DROP,
    t_threshold: float = typer.Option(2.0, "--t-threshold"),
    min_oos_periods: int = typer.Option(252, "--min-oos-periods"),
    oos_periods: int = typer.Option(None, "--oos-periods", help="test only the LAST N finite observations"),
    lags: int = _LAGS,
    as_json: bool = _JSON,
):
    """Newey-West HAC t-stat on the mean, plus a minimum out-of-sample window."""

    def compute() -> dict:
        from quantbox.analysis.gates import nw_gate

        r = _series(returns, column, indexed=False)
        return nw_gate(
            r.to_numpy(),
            lags=lags,
            t_threshold=t_threshold,
            min_oos_periods=min_oos_periods,
            oos_periods=oos_periods,
            allow_nonfinite_drop=allow_nonfinite_drop,
        )

    _emit("nw", as_json, compute)


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
    as_json: bool = _JSON,
):
    """Jensen's alpha after factor controls, HAC standard error, one-sided."""

    def compute() -> dict:
        import numpy as np
        import pandas as pd

        from quantbox.analysis.gates import factor_gate

        strat = _series(returns, column, indexed=True).rename("__y__")
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
        parts = [strat, panel[names]]
        if rf_column is not None:
            parts.append(panel[[rf_column]].rename(columns={rf_column: "__rf__"}))
        # date order, not file order: the HAC standard error weights NEIGHBOURING rows
        joined = pd.concat(parts, axis=1, join="inner").sort_index(kind="stable")
        finite = np.isfinite(joined.to_numpy(dtype=float)).all(axis=1)
        dropped = int((~finite).sum())
        if dropped and not allow_nonfinite_drop:
            raise _input_error(
                f"{dropped} of {len(joined)} aligned rows carry NaN/Inf — refusing to drop them silently; "
                "pass --allow-nonfinite-drop to drop them and record the count"
            )
        joined = joined[finite]
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

    _emit("factor", as_json, compute)


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
    as_json: bool = _JSON,
):
    """Paired stationary block bootstrap: P(candidate-vs-baseline leg holds) over resamples."""

    def compute() -> dict:
        from quantbox.analysis.gates import paired_block_bootstrap

        j = _joined(returns, column, baseline, baseline_column)
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

    _emit("bootstrap", as_json, compute)


@gates_app.command("episode")
def episode(
    returns: str = _RETURNS,
    baseline: str = typer.Option(None, "--baseline", help="baseline series (date-indexed); the episode is ITS"),
    column: str = _COLUMN,
    baseline_column: str = _BASELINE_COLUMN,
    metric: str = _METRIC,
    compare: str = _COMPARE,
    pass_if: str = _PASS_IF,
    threshold: float = typer.Option(0.0, "--threshold"),
    allow_nonfinite_drop: bool = _DROP,
    as_json: bool = _JSON,
):
    """The leg re-evaluated with the largest drawdown episode excluded."""

    def compute() -> dict:
        from quantbox.analysis.gates import episode_gate

        kw = dict(
            metric=metric,
            compare=compare,
            pass_if=pass_if,
            threshold=threshold,
            allow_nonfinite_drop=allow_nonfinite_drop,
        )
        if baseline is None:
            r = _series(returns, column, indexed=False)
            return episode_gate(r.to_numpy(), **kw)
        j = _joined(returns, column, baseline, baseline_column)
        out = episode_gate(j["c"].to_numpy(), j["b"].to_numpy(), **kw)
        ep = out["episode"]
        if ep is not None and not out["n_nonfinite_dropped"]:
            for key in ("start", "trough", "end"):
                ep[f"{key}_date"] = j.index[ep[key]]
        return out

    _emit("episode", as_json, compute)
