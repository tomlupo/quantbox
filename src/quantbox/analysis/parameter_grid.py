"""Parameter-grid sweep + heatmap rendering.

Enumerate a Cartesian grid of strategy parameter values, run each combination
through the vectorbt backtest engine, and return a tidy DataFrame of per-cell
metrics. Strategies that natively produce multi-slice weights (e.g.
``CryptoRegimeTrendStrategy`` returning ``(vol_target, tranches, ticker)``
MultiIndex columns) get those slices auto-expanded into rows of the result —
one outer-sweep iteration delivers N slice rows for free, no extra backtest
runs.

Reproduces the Robuxio TrendCatcher v2 notebook cells 121 / 128 heatmaps but
also works for any StrategyPlugin-compatible class.

Example::

    from quantbox.analysis import sweep, plot_heatmaps
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

from quantbox.plugins.backtesting.vectorbt_engine import run as _run_backtest

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
    shift_signal: int = 1,
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
        Forwarded to ``vectorbt_engine.run`` (fees, threshold, etc.).
    metrics
        vbt stats column names to collect.
    shift_signal
        Lag applied to weights before backtest (default 1 = use yesterday's
        signal for today's allocation; matches notebook convention).

    Returns
    -------
    pd.DataFrame
        Tidy long format. One row per (sweep_combo × strategy_slice).
        Columns: sweep keys, slice-decoded keys (e.g. ``vol_target``,
        ``tranches``), then the requested ``metrics``.
    """
    prices = data["prices"]
    backtest_kwargs = dict(backtest_kwargs or {})

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
        if shift_signal:
            weights = weights.shift(shift_signal)

        # Align to common date range
        common_idx = weights.dropna(how="all").index.intersection(prices.index)
        if len(common_idx) < 2:
            logger.warning("parameter_grid.sweep: insufficient overlap for %s", sweep_labels)
            continue
        w = weights.reindex(common_idx)
        p = prices.reindex(common_idx)

        # Restrict prices to columns weights actually reference
        ticker_level = -1
        if isinstance(w.columns, pd.MultiIndex):
            tickers = w.columns.get_level_values(ticker_level).unique()
            slice_level_names = [n for i, n in enumerate(w.columns.names) if i != len(w.columns.names) + ticker_level]
            slice_multi = w.columns.droplevel(ticker_level).unique()
            # vbt's create_labels joins "name-value" pairs with "_". Reproduce
            # that mapping here so we can recover original level values from
            # the stats index without lossy string parsing.
            label_map: dict[str, tuple] = {}
            for tup in slice_multi:
                tup_t = tup if isinstance(tup, tuple) else (tup,)
                label = "_".join(f"{name}-{val}" for name, val in zip(slice_level_names, tup_t, strict=False))
                label_map[label] = tup_t
        else:
            tickers = w.columns
            slice_level_names = []
            label_map = {}
        p = p.reindex(columns=tickers)

        pf = _run_backtest(p.ffill().bfill(), w, **backtest_kwargs)
        # vbt's wrapper.freq accessor calls pd.Timedelta(<Day>) which raises on
        # recent pandas. Pre-emptively force the wrapper to skip the broken
        # auto-inference by handing it a string freq. ``stats_defaults`` is a
        # property — patch the underlying ArrayWrapper.freq class-level
        # attribute by stashing a known-good value before invocation.
        if pf.wrapper.index.freq is not None:
            # Strip the offending Day offset from the wrapper's index.
            pf.wrapper.index.freq = None

        # Fetch metrics via pf.deep_getattr(name) — matches the notebook's
        # cell 121 idiom. Returns a scalar (single-slice) OR a Series indexed
        # by ``(strategy_label, *original_level_values)`` MultiIndex when
        # multi-slice. We key by the tuple of original level values, so the
        # final grid rows carry the unaltered MultiIndex levels.
        slice_metrics: dict[tuple, dict[str, Any]] = {}
        for m in metrics:
            try:
                v = pf.deep_getattr(m)
            except (AttributeError, KeyError) as e:
                logger.warning("parameter_grid.sweep: metric %r unavailable: %s", m, e)
                continue
            if isinstance(v, pd.Series):
                if isinstance(v.index, pd.MultiIndex):
                    # First level is the vbt label string; remaining levels are
                    # the original strategy MultiIndex levels.
                    for idx_tuple, val in v.items():
                        slice_key = tuple(idx_tuple[1:]) if len(idx_tuple) > 1 else (idx_tuple[0],)
                        slice_metrics.setdefault(slice_key, {})[m] = _coerce(val)
                else:
                    for slice_id, val in v.items():
                        slice_metrics.setdefault((slice_id,), {})[m] = _coerce(val)
            else:
                slice_metrics.setdefault(("_single_",), {})[m] = _coerce(v)

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


def _coerce(v: Any) -> Any:
    """Coerce numpy scalars to Python natives so the grid serialises cleanly."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, TypeError):
            pass
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
    anchor = pd.read_parquet(root / f"{align_to}.parquet")
    out: dict[str, pd.DataFrame] = {align_to: anchor}
    for name in names:
        if name == align_to:
            continue
        df = pd.read_parquet(root / f"{name}.parquet")
        out[name] = df.reindex(index=anchor.index, columns=anchor.columns)
    # Restrict the anchor to columns present in every loaded frame.
    common_cols = anchor.columns
    for name, df in out.items():
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
    shift_signal: int = 1,
    cmap: str = "RdYlGn",
    fmt: str = ".3f",
) -> pd.DataFrame:
    """Orchestrate a parameter-grid sweep across rebalancing bands.

    For each value in ``bands``, runs :func:`sweep` once and (if both
    ``output_dir`` and ``heatmap_index`` are set) renders one heatmap PNG per
    metric, suffixed with the band setting. Saves the combined tidy grid to
    ``<output_dir>/grid.parquet``. Returns the combined grid.

    This is the strategy-agnostic orchestrator used by per-research scripts —
    they supply ``strategy_cls``, base/sweep params and a market_data dict,
    and everything else (iteration, naming, saving) is centralised here.
    """
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    all_grids: list[pd.DataFrame] = []
    for band in bands:
        logger.info("run_grid: bands=%s", band)
        grid = sweep(
            strategy_cls=strategy_cls,
            base_params=base_params,
            sweep_params=sweep_params,
            data=market_data,
            backtest_kwargs={"fees": fees, "threshold": band, "rebalancing_freq": rebalancing_freq},
            metrics=metrics,
            shift_signal=shift_signal,
        )
        grid["bands"] = f"{int(band * 100)}%"
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
