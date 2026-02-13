"""Synthetic market data plugin using Monte Carlo simulation.

Generates realistic multi-asset price data from stochastic models,
useful for strategy research, stress-testing, and CI pipelines that
need deterministic synthetic data without external API calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta
from quantbox.simulation import (
    GBM,
    GBMParams,
    JumpDiffusion,
    JumpDiffusionParams,
    MarketSimulator,
    MeanReversion,
    MeanReversionParams,
    SimulationConfig,
    generate_random_correlation_matrix,
)

logger = logging.getLogger(__name__)


_MODEL_BUILDERS = {
    "gbm": lambda p: GBM(
        GBMParams(
            mu=p.get("mu", 0.08),
            sigma=p.get("sigma", 0.20),
        )
    ),
    "jump_diffusion": lambda p: JumpDiffusion(
        JumpDiffusionParams(
            mu=p.get("mu", 0.08),
            sigma=p.get("sigma", 0.20),
            jump_intensity=p.get("jump_intensity", 5.0),
            jump_mean=p.get("jump_mean", -0.02),
            jump_std=p.get("jump_std", 0.03),
        )
    ),
    "mean_reversion": lambda p: MeanReversion(
        MeanReversionParams(
            mu=p.get("mu", 0.05),
            sigma=p.get("sigma", 0.15),
            theta=p.get("theta", 0.5),
            long_term_mean=p.get("long_term_mean", 100.0),
        )
    ),
}


@dataclass
class SyntheticDataPlugin:
    """Generate synthetic market data via Monte Carlo simulation.

    Produces wide-format DataFrames (date index x symbol columns) matching
    the standard DataPlugin contract.
    """

    meta = PluginMeta(
        name="data.synthetic.v1",
        kind="data",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Synthetic market data generator using stochastic models "
            "(GBM, jump diffusion, mean reversion). Useful for strategy "
            "research, stress-testing, and CI pipelines."
        ),
        tags=("synthetic", "simulation", "research"),
        capabilities=("backtest",),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "n_assets": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 10,
                    "description": "Number of synthetic assets to generate.",
                },
                "n_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 252,
                    "description": "Number of trading days to simulate.",
                },
                "model": {
                    "type": "string",
                    "enum": ["gbm", "jump_diffusion", "mean_reversion"],
                    "default": "gbm",
                    "description": "Stochastic model for price dynamics.",
                },
                "model_params": {
                    "type": "object",
                    "default": {},
                    "description": "Parameters forwarded to the model (mu, sigma, etc).",
                },
                "correlation": {
                    "type": "string",
                    "enum": ["identity", "random", "stressed"],
                    "default": "random",
                    "description": "Correlation structure between assets.",
                },
                "initial_price": {
                    "type": "number",
                    "minimum": 0,
                    "default": 100.0,
                    "description": "Starting price for all assets.",
                },
                "random_state": {
                    "type": "integer",
                    "default": 42,
                    "description": "Random seed for reproducibility.",
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit symbol names. If omitted, generates SYN_001, SYN_002, ...",
                },
            },
        },
        examples=(
            "plugins:\n  data:\n    name: data.synthetic.v1\n    params:\n"
            "      n_assets: 20\n      n_steps: 504\n      model: jump_diffusion\n"
            "      correlation: random\n      random_state: 42",
        ),
    )

    def load_universe(self, params: dict[str, Any]) -> list[str]:
        """Return list of synthetic symbol names."""
        symbols = params.get("symbols")
        if symbols:
            return list(symbols)
        n_assets = int(params.get("n_assets", 10))
        return [f"SYN_{i + 1:03d}" for i in range(n_assets)]

    def load_market_data(
        self,
        universe: list[str],
        asof: str,
        params: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        """Generate synthetic price/volume data.

        Returns wide-format DataFrames backdated from *asof*.
        """
        n_assets = len(universe)
        n_steps = int(params.get("n_steps", 252))
        model_name = params.get("model", "gbm")
        model_params = params.get("model_params", {})
        correlation_type = params.get("correlation", "random")
        initial_price = float(params.get("initial_price", 100.0))
        random_state = params.get("random_state", 42)
        random_state = int(random_state) if random_state is not None else None

        rng = np.random.default_rng(random_state)

        builder = _MODEL_BUILDERS.get(model_name)
        if builder is None:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(_MODEL_BUILDERS.keys())}")

        sim = MarketSimulator()
        for symbol in universe:
            per_asset_params = dict(model_params)
            # add mild variation per asset so they don't all look identical
            per_asset_params.setdefault("mu", 0.08)
            per_asset_params["mu"] = per_asset_params["mu"] + rng.normal(0, 0.02)
            per_asset_params.setdefault("sigma", 0.20)
            per_asset_params["sigma"] = abs(per_asset_params["sigma"] + rng.normal(0, 0.03))
            sim.add_asset(symbol, builder(per_asset_params), initial_price=initial_price)

        # Correlation matrix
        if correlation_type == "identity":
            corr = np.eye(n_assets)
        elif correlation_type == "stressed":
            corr = generate_random_correlation_matrix(n_assets, eigenvalue_concentration=0.9, random_state=random_state)
        else:  # "random"
            corr = generate_random_correlation_matrix(n_assets, eigenvalue_concentration=0.5, random_state=random_state)

        sim.set_correlation_matrix(corr)

        config = SimulationConfig(
            n_paths=1,  # single median path for DataPlugin output
            n_steps=n_steps,
            dt=1 / 252,
            random_state=random_state,
        )
        result = sim.simulate(config)

        # Build date index backdated from asof
        # Use calendar days to guarantee exact length match with price array
        asof_date = pd.Timestamp(asof)
        n_dates = result.prices.shape[2]  # n_steps + 1
        dates = pd.date_range(end=asof_date, periods=n_dates, freq="D")

        # Extract single path (path index 0) for each asset
        prices_dict = {}
        volume_dict = {}
        for i, symbol in enumerate(universe):
            prices_dict[symbol] = result.prices[i, 0, :]
            # Synthetic volume: baseline scaled by price change magnitude
            base_vol = rng.uniform(1e6, 1e8)
            if n_steps > 0:
                returns = np.diff(result.prices[i, 0, :]) / result.prices[i, 0, :-1]
                vol_series = np.concatenate([[base_vol], base_vol * (1 + np.abs(returns) * 5)])
            else:
                vol_series = np.array([base_vol])
            volume_dict[symbol] = vol_series

        prices_df = pd.DataFrame(prices_dict, index=dates)
        volume_df = pd.DataFrame(volume_dict, index=dates)

        logger.info(
            "Generated synthetic data: %d assets, %d steps, model=%s, corr=%s",
            n_assets,
            n_steps,
            model_name,
            correlation_type,
        )

        return {"prices": prices_df, "volume": volume_df}
