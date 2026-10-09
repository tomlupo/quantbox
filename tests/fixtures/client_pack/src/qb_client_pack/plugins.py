"""The client pack's two plugins. They import quantbox CORE only (``quantbox.contracts``).

- :class:`InMemoryDataPlugin` builds a seeded random-walk price panel in
  memory: no file, no network, no extra.
- :class:`TopMomentumStrategy` holds the ``top_n`` symbols with the best
  ``lookback``-bar return, equal weight, long only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta, StrategyContext


@dataclass
class InMemoryDataPlugin:
    meta = PluginMeta(
        name="client_pack.synthetic_data.v1",
        kind="data",
        version="0.0.1",
        core_compat=">=0.11",
        description="Seeded random-walk spot prices built in memory (client-pack fixture).",
        capabilities=("backtest",),
        params_schema={
            "type": "object",
            "properties": {
                "n_days": {"type": "integer", "description": "Bars in the panel."},
                "seed": {"type": "integer", "description": "Random seed."},
            },
        },
    )

    n_days: int = 400
    seed: int = 7

    _SYMBOLS = ("AAA", "BBB", "CCC", "DDD")

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"symbol": list(self._SYMBOLS)})

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        symbols = [str(s) for s in universe["symbol"]] if isinstance(universe, pd.DataFrame) else list(universe)
        rng = np.random.default_rng(self.seed)
        drift = np.linspace(-0.0005, 0.001, len(symbols))
        steps = rng.normal(drift, 0.02, size=(self.n_days, len(symbols)))
        index = pd.date_range(end=pd.Timestamp(asof), periods=self.n_days, freq="D")
        prices = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=symbols)
        volume = pd.DataFrame(1e7, index=index, columns=symbols)
        return {"prices": prices, "volume": volume}

    def load_fx(self, asof: str, params: dict[str, Any]) -> pd.DataFrame | None:
        return None

    def planned_funding(self) -> bool:
        # A spot panel: it carries no funding series, and none is owed.
        return False


@dataclass
class TopMomentumStrategy:
    meta = PluginMeta(
        name="client_pack.top_momentum.v1",
        kind="strategy",
        version="0.0.1",
        core_compat=">=0.11",
        description="Equal weight in the top_n symbols by lookback return (client-pack fixture).",
        params_schema={
            "type": "object",
            "properties": {
                "lookback": {"type": "integer", "description": "Return window in bars."},
                "top_n": {"type": "integer", "description": "Symbols held."},
            },
        },
    )

    lookback: int = 20
    top_n: int = 2

    def run(
        self, data: dict[str, Any], params: dict[str, Any] | None = None, context: StrategyContext | None = None
    ) -> dict[str, Any]:
        params = params or {}
        lookback = int(params.get("lookback", self.lookback))
        top_n = int(params.get("top_n", self.top_n))
        prices: pd.DataFrame = data["prices"]
        momentum = prices.pct_change(lookback)
        rank = momentum.rank(axis=1, ascending=False)
        held = (rank <= top_n) & momentum.notna()
        weights = held.astype(float).div(held.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        return {"weights": weights, "details": {"strategy": self.meta.name}}
