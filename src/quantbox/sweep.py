"""Parameter-grid sweep + heatmap rendering — the library half of ``quantbox sweep``.

Lived at ``quantbox.analysis.parameter_grid`` until TOM-1618: a sweep RUNS a
grid of backtests, it does not analyse a result, so it sits next to the sweep
door rather than with the statistics. The old path still resolves, with a
``DeprecationWarning``.

Enumerate a Cartesian grid of strategy parameter values, run each combination
through the engine seam (:mod:`quantbox.engine`; vectorbt by default, ``engine: rsims``
in ``backtest_kwargs`` for the other adapter), and return a tidy DataFrame of per-cell
metrics. Strategies that natively produce multi-slice weights (e.g.
``CryptoRegimeTrendStrategy`` returning ``(vol_target, tranches, ticker)``
MultiIndex columns) get those slices auto-expanded into rows of the result —
one outer-sweep iteration delivers N slice rows for free, no extra backtest
runs.

Reproduces the Robuxio TrendCatcher v2 notebook cells 121 / 128 heatmaps but
also works for any StrategyPlugin-compatible class.

Example::

    from quantbox.sweep import sweep, plot_heatmaps
    from quantbox.plugins.strategies.crypto_regime_trend import CryptoRegimeTrendStrategy

    grid = sweep(
        strategy_cls=CryptoRegimeTrendStrategy,
        base_params={
            "use_ensemble": True, "long_max": 10, "coins_to_trade": 30,
            "vol_targets": ["off", 0.25, 0.5, 1.0],
            "tranches": [1, 2, 5],
            ...
        },
        sweep_params={"window_pairs": [[[10, 25]], [[20, 50]], [[40, 100]], [[100, 250]]]},
        data={"prices": prices, "volume": volume, "market_cap": mcap},
        backtest_kwargs={"fees": 0.005, "threshold": 0.05, "rebalancing_freq": "1D"},
    )
    plot_heatmaps(grid, save_dir="research/heatmaps", index="window_pair",
                  columns=["vol_target", "tranches"])
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quantbox.execution import resolve_execution, resolve_sweep_lag_bars
from quantbox.parquet_io import read_parquet

__all__ = [
    "DEFAULT_METRICS",
    "align_market_data",
    "load_parquet_market_data",
    "plot_heatmaps",
    "run_grid",
    "sweep",
]

logger = logging.getLogger(__name__)

DEFAULT_METRICS: tuple[str, ...] = (
    "total_return",
    "sharpe_ratio",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "calmar_ratio",
)


def _parse_vbt_slice_label(label: Any) -> dict[str, Any]:
    """Parse a vbt label like ``vol_target-50_tranches-5`` into a dict.

    Created by :func:`quantbox.plugins.backtesting.vectorbt_engine.create_labels`
    as ``"_".join(f"{name}-{value}" for ...)``. We reverse that here. Returns
    ``{"slice": <label>}`` if parsing fails.
    """
    if not isinstance(label, str) or "-" not in label:
        return {"slice": label}
    parsed: dict[str, Any] = {}
    for token in label.split("_"):
        if "-" not in token:
            return {"slice": label}
        key, _, val = token.partition("-")
        # Coerce numeric-looking values to float
        try:
            parsed[key] = float(val) if val.replace(".", "", 1).lstrip("-").isdigit() else val
        except (ValueError, AttributeError):
            parsed[key] = val
    return parsed


def sweep(
    strategy_cls: type,
    base_params: Mapping[str, Any],
    sweep_params: Mapping[str, Sequence[Any]],
    data: Mapping[str, pd.DataFrame],
    backtest_kwargs: Mapping[str, Any] | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    shift_signal: int | None = None,
    lag_bars: int | None = None,
    funding: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a strategy across a Cartesian product of parameter values.

    For each ``(key, [v1, v2, ...])`` entry in ``sweep_params``, runs the
    strategy with that key overridden to each value while holding
    ``base_params`` constant. Backtests each resulting weights frame and
    collects per-strategy-slice metrics.

    Parameters
    ----------
    strategy_cls
        Strategy class (e.g. ``CryptoRegimeTrendStrategy``). Must accept the
        union of ``base_params`` and ``sweep_params`` keys as kwargs and
        expose ``run(data) -> {"weights": DataFrame, ...}``.
    base_params
        Fixed strategy parameters.
    sweep_params
        Parameter names mapped to lists of values to enumerate.
    data
        Market data dict passed to strategy.run() (must contain ``"prices"``).
    backtest_kwargs
        ``engine`` (default vectorbt), costs (``fees``, ``fixed_fees``, ``slippage``),
        ``rebalancing_freq``, ``threshold`` (the seam's schedule, every engine), ``schedule``
        (``calendar``, the default, or ``bars``) and ``max_leverage`` (the decision's gross cap,
        default 1, TOM-1525); any other key is the engine adapter's
        own parameter (vectorbt: ``use_numba``, ...), refused when it does not own it.
    metrics
        Metric names, answered by the engine adapter (vectorbt: ``pf`` attribute names;
        rsims: the same names, from ``compute_backtest_metrics``).
    lag_bars
        Execution lag in bars — the SAME convention as ``execution.lag_bars``
        in ``quantbox run`` (:mod:`quantbox.execution`). Default 1: weights
        decided on bar t fill at the close of bar t+1; also the minimum, 0
        (same-bar) raises ``ValueError``.
    shift_signal
        DEPRECATED alias of ``lag_bars`` (emits ``DeprecationWarning``).
    funding
        ``{"ignore": True, "reason": ...}``, as in ``backtest()``: ``data["funding_rates"]``
        goes to the engine, and on one that does not charge funding it is refused
        without this escape (:mod:`quantbox.funding_guard`, TOM-1619).

    Returns
    -------
    pd.DataFrame
        Tidy long format. One row per (sweep_combo × strategy_slice).
        Columns: sweep keys, slice-decoded keys (e.g. ``vol_target``,
        ``tranches``), then the requested ``metrics``.
    """
    from quantbox.decision import DecisionRules, final_book, gross_cap
    from quantbox.engine import Costs, get_engine, simulate
    from quantbox.financing import DEFAULT_LEVERAGE
    from quantbox.funding_guard import check_series

    backtest_kwargs = dict(backtest_kwargs or {})
    # The engine and the schedule are the seam's; costs are Costs; anything else is the adapter's own
    # parameter, refused by it when it does not own it. Resolved before anything else, so a
    # missing [vectorbt] extra raises MissingExtraError naming it first.
    adapter = get_engine(backtest_kwargs.pop("engine", None))
    prices = data["prices"]
    costs = Costs(
        **{k: float(backtest_kwargs.pop(k)) for k in ("fees", "fixed_fees", "slippage") if k in backtest_kwargs}
    )
    rebalancing_freq = backtest_kwargs.pop("rebalancing_freq", 1)
    threshold = backtest_kwargs.pop("threshold", None)
    schedule = backtest_kwargs.pop("schedule", "calendar")  # execution.schedule: calendar | bars
    # The decision's gross cap, risk.max_leverage: default 1, as every door (TOM-1525).
    max_leverage = gross_cap({"max_leverage": backtest_kwargs.pop("max_leverage", None)})
    engine_params = adapter.check_params(backtest_kwargs)
    timing = resolve_execution({"lag_bars": resolve_sweep_lag_bars(lag_bars, shift_signal), "schedule": schedule})
    funding_rates = data.get("funding_rates")
    check_series(adapter, funding_rates, funding)  # the seam's funding guard, before any strategy runs

    # Defensive: strip index.freq so vbt's wrapper.freq lookup doesn't trip on
    # a `<Day>` offset (vbt + recent pandas can't convert it to a Timedelta).
    # Real-world parquet-backed indices have freq=None, so this is a no-op for
    # production callers and only matters for synthetic test fixtures.
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.freq is not None:
        prices = prices.copy()
        prices.index = pd.DatetimeIndex(prices.index.values)

    keys = list(sweep_params.keys())
    value_lists = [list(sweep_params[k]) for k in keys]

    rows: list[dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        sweep_kwargs = dict(zip(keys, combo, strict=False))
        # Use string repr of values that don't survive as dict keys (e.g. list-of-tuple
        # window_pairs); the raw value is preserved in `_value` for joins if needed.
        sweep_labels = {f"{k}": _label_value(v) for k, v in sweep_kwargs.items()}

        params = {**base_params, **sweep_kwargs}
        logger.info("parameter_grid.sweep: %s", sweep_labels)

        strat = strategy_cls(**params)
        out = strat.run(data)
        weights = out["weights"]
        # The warm-up (leading rows where no slice has decided anything) is not part of the book, as the
        # sweep always had it; the slices are one batch through the one book function (docs/adr/0008).
        decided_rows = weights.notna().any(axis=1).to_numpy()
        weights = weights.iloc[int(decided_rows.argmax()) :] if decided_rows.any() else weights.iloc[:0]
        if len(weights.index.intersection(prices.index)) < 2:
            logger.warning("parameter_grid.sweep: insufficient overlap for %s", sweep_labels)
            continue
        # The decision (TOM-1520): each slice capped at max_leverage gross, then normalised to net 1 on
        # its decided rows (the default venue.leverage; schedule: bars only measures it) — the seam
        # executes final targets.
        weights, _ = final_book(
            weights,
            DecisionRules(
                max_leverage=max_leverage, leverage="none" if timing.schedule == "bars" else DEFAULT_LEVERAGE
            ),
        )
        book = simulate(
            prices,
            weights,
            engine=adapter,
            timing=timing,
            costs=costs,
            rebalancing_freq=rebalancing_freq,
            threshold=threshold,
            funding=funding_rates,
            funding_ignore=funding,
            engine_params=engine_params,
        )

        # A strategy may return MultiIndex columns (strategy slices, the ticker last): one row per slice.
        slice_level_names = list(weights.columns.names[:-1]) if isinstance(weights.columns, pd.MultiIndex) else []

        # The adapter answers the metric names, per slice keyed by the original level values.
        slice_metrics = adapter.stats(book, metrics)

        for slice_key, mdict in slice_metrics.items():
            if slice_key == ("_single_",):
                slice_dict: dict[str, Any] = {}
            elif slice_level_names and len(slice_key) == len(slice_level_names):
                slice_dict = dict(zip(slice_level_names, slice_key, strict=False))
            else:
                # Fall back to string parsing for the legacy single-string slice id.
                only = slice_key[0]
                slice_dict = _parse_vbt_slice_label(only) if isinstance(only, str) else {"slice": only}
            entry: dict[str, Any] = {**sweep_labels, **slice_dict, **mdict}
            rows.append(entry)

    return pd.DataFrame(rows)


def _label_value(v: Any) -> Any:
    """Render a parameter value as a stable, hashable label for the grid."""
    if isinstance(v, (list, tuple)):
        return str(v)
    return v


def plot_heatmaps(
    grid: pd.DataFrame,
    index: str | list[str],
    columns: str | list[str],
    metrics: Sequence[str] | None = None,
    save_dir: str | Path | None = None,
    title_prefix: str = "",
    filename_suffix: str = "",
    cmap: str = "RdYlGn",
    fmt: str = ".2f",
) -> dict[str, Any]:
    """Pivot a tidy grid DataFrame to heatmaps, one per metric.

    Requires matplotlib + seaborn (optional dependencies of ``quantbox[viz]``).
    Pivots ``grid`` on (index, columns) and renders each metric as a
    colour-mapped heatmap with cell annotations.

    Returns a dict keyed by metric name. If ``save_dir`` is given, each entry
    is the saved PNG path; otherwise, the matplotlib Axes.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "plot_heatmaps requires matplotlib + seaborn; install with `pip install matplotlib seaborn`"
        ) from exc

    if metrics is None:
        metrics = [c for c in grid.columns if c not in _sweep_axis_columns(grid, index, columns)]

    save_path = Path(save_dir) if save_dir is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for metric in metrics:
        if metric not in grid.columns:
            logger.warning("plot_heatmaps: metric %r not in grid columns", metric)
            continue
        pivot = grid.pivot_table(index=index, columns=columns, values=metric, aggfunc="mean")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 0.9), max(4, pivot.shape[0] * 0.7)))
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar=True, linewidths=0.5)
        ax.set_title(f"{title_prefix}{metric}".strip())
        plt.tight_layout()
        if save_path is not None:
            stem = _slugify(metric)
            if filename_suffix:
                stem = f"{stem}{filename_suffix}"
            out = save_path / f"{stem}.png"
            fig.savefig(out, dpi=120)
            results[metric] = out
            plt.close(fig)
        else:
            results[metric] = ax
    return results


def _sweep_axis_columns(grid: pd.DataFrame, *axes: Any) -> set[str]:
    cols: set[str] = set()
    for a in axes:
        if isinstance(a, str):
            cols.add(a)
        elif isinstance(a, (list, tuple)):
            cols.update(a)
    return cols


def _slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def load_parquet_market_data(
    root: str | Path,
    names: Sequence[str] = ("prices", "volume", "market_cap"),
    align_to: str = "prices",
) -> dict[str, pd.DataFrame]:
    """Load named ``<name>.parquet`` files from a dataset directory and align
    them all to the index + columns of the ``align_to`` frame.

    Useful for strategy backtests where every input frame must share the same
    date axis and ticker universe (otherwise the strategy hits NaN dropouts on
    misaligned columns). Returns a dict keyed by the supplied ``names``.
    """
    root = Path(root)
    return align_market_data(
        {name: read_parquet(root / f"{name}.parquet") for name in dict.fromkeys([*names, align_to])}, align_to
    )


def align_market_data(frames: Mapping[str, pd.DataFrame], align_to: str = "prices") -> dict[str, pd.DataFrame]:
    """Align every frame to the index + columns of ``frames[align_to]``, keeping only the
    columns present in all of them."""
    anchor = frames[align_to]
    out: dict[str, pd.DataFrame] = {align_to: anchor}
    for name, df in frames.items():
        if name != align_to:
            out[name] = df.reindex(index=anchor.index, columns=anchor.columns)
    common_cols = anchor.columns
    for df in out.values():
        common_cols = common_cols.intersection(df.columns)
    return {name: df[common_cols] for name, df in out.items()}


def run_grid(
    strategy_cls: type,
    base_params: Mapping[str, Any],
    sweep_params: Mapping[str, Sequence[Any]],
    market_data: Mapping[str, pd.DataFrame],
    bands: Sequence[float] = (0.0, 0.05),
    output_dir: str | Path | None = None,
    heatmap_index: str | list[str] = None,
    heatmap_columns: str | list[str] = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    fees: float = 0.005,
    rebalancing_freq: int | str = "1D",
    shift_signal: int | None = None,
    cmap: str = "RdYlGn",
    fmt: str = ".3f",
    lag_bars: int | None = None,
    engine: str | None = None,
    funding: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Orchestrate a parameter-grid sweep across rebalancing bands.

    For each value in ``bands``, runs :func:`sweep` once and (if both
    ``output_dir`` and ``heatmap_index`` are set) renders one heatmap PNG per
    metric, suffixed with the band setting. Saves the combined tidy grid to
    ``<output_dir>/grid.parquet``. Returns the combined grid.

    This is the strategy-agnostic orchestrator used by per-research scripts —
    they supply ``strategy_cls``, base/sweep params and a market_data dict,
    and everything else (iteration, naming, saving) is centralised here.
    ``engine`` picks the engine adapter (default vectorbt; :mod:`quantbox.engine`);
    ``funding`` is the funding escape, forwarded to :func:`sweep`.
    """
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    lag = resolve_sweep_lag_bars(lag_bars, shift_signal)

    all_grids: list[pd.DataFrame] = []
    for band in bands:
        logger.info("run_grid: bands=%s", band)
        grid = sweep(
            strategy_cls=strategy_cls,
            base_params=base_params,
            sweep_params=sweep_params,
            data=market_data,
            backtest_kwargs={
                "fees": fees,
                "threshold": band,
                "rebalancing_freq": rebalancing_freq,
                **({"engine": engine} if engine is not None else {}),
            },
            metrics=metrics,
            lag_bars=lag,
            funding=funding,
        )
        grid["bands"] = f"{int(band * 100)}%"
        grid["lag_bars"] = lag  # execution timing travels with the numbers
        all_grids.append(grid)

        if output is not None and heatmap_index is not None and heatmap_columns is not None:
            plot_heatmaps(
                grid,
                index=heatmap_index,
                columns=heatmap_columns,
                metrics=list(metrics),
                save_dir=output,
                title_prefix=f"Bands={int(band * 100)}% — ",
                filename_suffix=f"_bands{int(band * 100)}",
                cmap=cmap,
                fmt=fmt,
            )

    combined = pd.concat(all_grids, ignore_index=True)
    if output is not None:
        combined.to_parquet(output / "grid.parquet")
    return combined
