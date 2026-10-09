"""The ONE strategy runner, shared by backtest and trading (TOM-1448).

Both ``backtest.pipeline.v1`` and ``trade.full_pipeline.v1`` build a
:class:`~quantbox.contracts.StrategyContext` with :func:`build_strategy_context`
and run their strategies with :func:`run_strategies`. A strategy therefore sees
the same bars-per-year, as-of date, calendar and frequency in its backtest and
in paper/live; only ``mode`` differs. TOM-1338 fixed the symptom (live handed
252 where the backtest handed 365); this removes the cause, two runners.

The runner returns DECIDED weights. Both pipelines then turn them into the
final TARGET weights with the one decision, :func:`quantbox.decision.final_targets`
(short clip, gross cap, group limits, ``venue.leverage`` normalisation;
TOM-1520), before any rebalancing policy reads them.

A strategy reads its annualisation through :func:`resolve_annualize`, the one
annualisation block: an explicit per-strategy value wins, else the context's
``bars_per_year``. The ``_pipeline_annualize`` param that the pipelines used to
inject survives only as a shim for strategies whose ``run`` takes no
``context`` (third-party and lab strategies), with a DeprecationWarning.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Any

import pandas as pd

from quantbox._lazy import load
from quantbox.contracts import StrategyContext
from quantbox.frequency import resolve_pipeline_frequency

__all__ = [
    "FALLBACK_BARS_PER_YEAR",
    "LEGACY_ANNUALIZE_KEY",
    "build_strategy_context",
    "call_strategy",
    "resolve_annualize",
    "run_strategies",
]

logger = logging.getLogger(__name__)

LEGACY_ANNUALIZE_KEY = "_pipeline_annualize"
# The historical equity default a strategy fell back to with nothing handed in.
FALLBACK_BARS_PER_YEAR = 252.0


def build_strategy_context(
    mode: str, asof: Any, params: dict[str, Any], prices_params: dict[str, Any]
) -> StrategyContext:
    """The run's StrategyContext, from the pipeline params and the ``prices`` block.

    The frequency comes from :func:`quantbox.frequency.resolve_pipeline_frequency`,
    the one resolver both pipelines already use for their metrics.
    """
    freq = resolve_pipeline_frequency(params, prices_params)
    return StrategyContext(
        bars_per_year=float(freq.bars_per_year()),
        mode=str(mode),
        asof=str(asof),
        calendar=str(freq.calendar),
        frequency=freq.to_dict()["bar_size"],
    )


def resolve_annualize(
    explicit: float | None,
    params: dict[str, Any] | None,
    context: StrategyContext | None,
    *,
    owner: str,
) -> float:
    """A strategy's annualisation factor — the one block every strategy calls.

    Order: the strategy's explicit field (``annualize`` / ``trading_days``), else
    an explicit ``_pipeline_annualize`` in its params (deprecated; warns), else
    ``context.bars_per_year``, else 252.0 (a direct L3 call with no context).
    An explicit field that disagrees with the run's derived value by more than
    one bar logs a warning: it is allowed, but rarely meant.
    """
    legacy = (params or {}).get(LEGACY_ANNUALIZE_KEY)
    if legacy is not None and context is not None:
        warnings.warn(
            f"{owner}: `{LEGACY_ANNUALIZE_KEY}` in params is deprecated — the run's "
            "StrategyContext carries bars_per_year; set the strategy's own field to override it.",
            DeprecationWarning,
            stacklevel=2,
        )
    derived = legacy if legacy is not None else (context.bars_per_year if context is not None else None)
    if explicit is None:
        return float(derived) if derived is not None else FALLBACK_BARS_PER_YEAR
    value = float(explicit)
    if derived is not None and abs(value - float(derived)) > 1:
        logger.warning(
            "%s=%s overrides pipeline-derived %.1f. If intentional, ignore; otherwise drop the explicit value.",
            owner,
            value,
            float(derived),
        )
    return value


def _accepts_context(fn: Any) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "context" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def call_strategy(
    run: Any, data: dict[str, Any], params: dict[str, Any] | None, context: StrategyContext, *, name: str
) -> dict[str, Any]:
    """Call one strategy's ``run`` with the context — or, for a legacy one, the deprecated param."""
    strat_params = dict(params or {})
    if _accepts_context(run):
        return run(data=data, params=strat_params, context=context)
    warnings.warn(
        f"strategy {name!r}: run() takes no `context`, so it is handed the deprecated "
        f"`{LEGACY_ANNUALIZE_KEY}` param instead. Add `context: StrategyContext | None = None` "
        "to run() and read context.bars_per_year (quantbox.strategy_runner.resolve_annualize); "
        f"`{LEGACY_ANNUALIZE_KEY}` is removed in a later minor version.",
        DeprecationWarning,
        stacklevel=2,
    )
    # An explicit value already in the strategy's params wins; the config is never mutated.
    strat_params.setdefault(LEGACY_ANNUALIZE_KEY, context.bars_per_year)
    return run(data=data, params=strat_params)


def run_strategies(
    market_data: dict[str, Any],
    strategies_cfg: list[dict[str, Any]],
    context: StrategyContext,
    plugins: list[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run every strategy of a run and collect ``{name: {"result", "weight"}}``.

    ``plugins`` are injected StrategyPlugin instances, paired with
    ``strategies_cfg`` by position. Without them each config entry names a
    module under ``quantbox.plugins.strategies`` whose module-level ``run`` is
    called. Multi-level weight columns are collapsed to one column per ticker.
    """
    if plugins:
        calls = [
            (strat.meta.name, strat.run, strategies_cfg[i] if i < len(strategies_cfg) else {})
            for i, strat in enumerate(plugins)
        ]
    else:
        calls = []
        for cfg in strategies_cfg:
            name = cfg["name"]
            try:
                module = load(f"quantbox.plugins.strategies.{name}")
            except ImportError:
                logger.error("Could not import strategy '%s'", name)
                raise
            calls.append((name, module.run, cfg))

    results: dict[str, dict[str, Any]] = {}
    for name, run, cfg in calls:
        weight = float(cfg.get("weight", 1.0))
        result = call_strategy(run, market_data, cfg.get("params", {}), context, name=name)

        weights_df = result.get("weights", pd.DataFrame())
        if isinstance(weights_df, pd.DataFrame) and weights_df.columns.nlevels > 1:
            if weights_df.droplevel("ticker", axis=1).columns.unique().shape[0] > 1:
                logger.warning(
                    "Strategy %s has multiple weights columns: %s",
                    name,
                    weights_df.droplevel("ticker", axis=1).columns.unique().tolist(),
                )
            result["weights"] = weights_df.T.groupby("ticker").sum().T

        results[name] = {"result": result, "weight": weight}
        logger.info("Strategy '%s' completed (weight=%.2f)", name, weight)
    return results
