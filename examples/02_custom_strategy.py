"""Create a minimal custom strategy plugin.

Shows the minimum code needed to implement a StrategyPlugin
that can be used with any pipeline.

Usage:
    uv run python examples/02_custom_strategy.py
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta, StrategyPlugin


@dataclass
class EqualWeightStrategy:
    """Assign equal weight to all assets in the universe."""

    meta = PluginMeta(
        name="strategy.equal_weight.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1",
        description="Equal-weight allocation across all assets.",
        tags=("simple", "benchmark"),
        outputs=("weights",),
        examples=(
            "strategies:\n"
            "  - name: strategy.equal_weight.v1\n"
            "    weight: 1.0\n"
            "    params: {}",
        ),
    )

    def run(self, data: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Compute equal weights for all assets.

        Args:
            data: Dict with "prices" (wide DataFrame), "universe", etc.
            params: Strategy parameters (unused for equal-weight).

        Returns:
            Dict with "weights" key -> wide DataFrame (dates x symbols).
        """
        prices: pd.DataFrame = data["prices"]

        # Equal weight: 1/N for each asset with a valid price
        n_assets = prices.notna().sum(axis=1)
        weights = prices.notna().astype(float).div(n_assets, axis=0)

        return {"weights": weights}


# -- Quick test --
if __name__ == "__main__":
    # Verify the plugin satisfies the protocol
    strategy: StrategyPlugin = EqualWeightStrategy()
    print(f"Plugin:  {strategy.meta.name}")
    print(f"Kind:    {strategy.meta.kind}")

    # Test with fake data
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {"BTC": [95000, 96000, 94000, 97000, 95500], "ETH": [3200, 3300, 3100, 3400, 3250]},
        index=dates,
    )
    result = strategy.run({"prices": prices}, {})
    print("\nWeights:")
    print(result["weights"])
