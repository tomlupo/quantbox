"""Weighted-average meta-strategy aggregator.

Composes multiple sub-strategy results into a single set of weights
using a weighted average.  This is a StrategyPlugin so it can be
injected via config just like any other strategy.

Ported from ``TradingPipeline._aggregate_strategies()``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from quantbox.contracts import PluginMeta

logger = logging.getLogger(__name__)


@dataclass
class WeightedAverageAggregator:
    """Aggregate sub-strategy weights via weighted average.

    Expects ``data["strategy_results"]`` to be a dict of::

        { strategy_name: {"result": {..., "weights": DataFrame}, "weight": float} }

    Returns a dict with:
    - ``weights``: aggregated weights as a DataFrame (date x ticker)
    - ``simple_weights``: last-row weights as ``{ticker: float}``
    - ``details``: per-strategy contribution info
    """

    meta = PluginMeta(
        name="strategy.weighted_avg.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Weighted-average meta-strategy aggregator",
        tags=("aggregator", "meta-strategy"),
    )

    def run(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        strategy_results: Dict[str, Dict[str, Any]] = data.get("strategy_results", {})
        weight_overrides = params.get("strategy_weights", {})

        names = list(strategy_results.keys())
        if not names:
            return {"weights": pd.DataFrame(), "simple_weights": {}, "details": {}}

        weight_dfs = []
        account_weights = []
        valid_names = []
        for sname in names:
            sinfo = strategy_results[sname]
            w_df = sinfo["result"].get("weights", pd.DataFrame())
            if w_df is None or (isinstance(w_df, pd.DataFrame) and w_df.empty):
                continue
            weight_dfs.append(w_df)
            w = float(weight_overrides.get(sname, sinfo["weight"]))
            account_weights.append(w)
            valid_names.append(sname)

        if not weight_dfs:
            return {"weights": pd.DataFrame(), "simple_weights": {}, "details": {}}

        if len(weight_dfs) == 1:
            df = weight_dfs[0]
            scaled = df * account_weights[0]
            last = scaled.iloc[-1] if isinstance(scaled, pd.DataFrame) else scaled
            simple = {str(k): float(v) for k, v in last.items()}
            return {"weights": scaled, "simple_weights": simple, "details": {}}

        # Multi-strategy aggregation
        try:
            combined = pd.concat(
                weight_dfs, axis=1,
                keys=valid_names, names=["strategy"],
            )
            acct_w = pd.Series(
                account_weights,
                index=pd.Index(valid_names, name="strategy"),
            )
            weighted = combined.mul(acct_w, level="strategy")
            flat = weighted.droplevel(0, axis=1)
            # Sum duplicate column names (same ticker from different strategies)
            if isinstance(flat.columns, pd.MultiIndex) or flat.columns.duplicated().any():
                flat = flat.T.groupby(level=0).sum().T
            aggregated_df = flat
            last_row = aggregated_df.iloc[-1]
        except Exception:
            logger.warning("Concat aggregation failed, using manual fallback")
            agg: Dict[str, float] = {}
            for i, df in enumerate(weight_dfs):
                last = df.iloc[-1] if isinstance(df, pd.DataFrame) else df
                w = account_weights[i]
                for ticker, val in last.items():
                    ticker_str = str(ticker)
                    agg[ticker_str] = agg.get(ticker_str, 0.0) + float(val) * w
            return {
                "weights": pd.DataFrame(),
                "simple_weights": agg,
                "details": {},
            }

        simple = {str(k): float(v) for k, v in last_row.items()}
        return {
            "weights": aggregated_df,
            "simple_weights": simple,
            "details": {
                "strategies": valid_names,
                "account_weights": account_weights,
            },
        }
