"""
Trading Pipeline Plugin

Thin wrapper around quantlab's trading.py - NO NEW LOGIC.
Just exposes the existing pipeline via quantbox plugin interface.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add quantlab paths for imports
QUANTLAB_DIR = Path(__file__).parent / "quantlab"
QUANTLAB_SRC = QUANTLAB_DIR  # utils, trading_bot are here

# Ensure paths are set
for p in [QUANTLAB_DIR, QUANTLAB_SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@dataclass
class TradingPipelinePlugin:
    """
    Quantbox plugin wrapper for quantlab trading pipeline.

    This is a 1:1 copy of quantlab/workflow/trading.py wrapped in plugin interface.
    NO NEW LOGIC - just calls the original run() function.

    Usage:
        plugin = TradingPipelinePlugin(account_name='binance')
        result = plugin.run()
    """

    # Config
    account_name: str = "binance"
    quantlab_root: str = field(default_factory=lambda: str(Path(__file__).parent / "quantlab"))
    use_strategy_configs: bool = True

    # Plugin metadata
    name: str = "trading.pipeline.v1"
    version: str = "1.0.0"

    def describe(self) -> dict[str, Any]:
        """Describe the plugin for LLM agents."""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Quantlab trading pipeline - runs strategies and generates orders",
            "account": self.account_name,
            "config_source": self.quantlab_root,
            "inputs": ["account_name", "use_strategy_configs"],
            "outputs": ["final_asset_weights", "portfolio_orders", "execution_report"],
        }

    def run(self, **kwargs) -> dict[str, Any]:
        """
        Run the full trading pipeline.

        This calls quantlab's trading.run() directly - no modifications.

        Returns:
            Dict with keys:
                - final_asset_weights: Dict[str, float]
                - portfolio_orders: Dict
                - execution_report: Dict
                - strategy_results: Dict
                - artifact_payload: Dict
        """
        # Import here to avoid circular imports
        from .quantlab import trading

        # Load the main config (quantlab expects this)
        config = {}  # trading.py loads its own config

        # Call the original run function
        result = trading.run(
            account_name=self.account_name,
            config=config,
            use_strategy_configs=self.use_strategy_configs,
        )

        return result


# Convenience function
def run_pipeline(account_name: str = "binance", **kwargs) -> dict[str, Any]:
    """Run the trading pipeline for an account."""
    plugin = TradingPipelinePlugin(account_name=account_name, **kwargs)
    return plugin.run()
