"""ETH Mean Reversion Strategy (24h lookback).

Entry: Enter LONG when ETH/USDT 24-hour rolling return <= -5%.
Exit: +3% take profit, 48h timeout, or -10% stop-loss.
Position sizing: 10% of portfolio.
Risk: Max 1 concurrent position, 24h cooldown after exit.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class EthMeanReversion24h:
    """ETH mean reversion strategy based on 24-hour rolling returns.

    Enters long when ETH drops >= 5% over 24h, exits on:
    - Take profit: +3% gain from entry
    - Stop loss: -10% loss from entry
    - Timeout: 48 hours since entry

    Includes 24h cooldown after exit to avoid whipsaws.
    """

    meta: PluginMeta = PluginMeta(
        name="strategy.eth_mean_reversion_24h.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1.0",
        description="ETH mean reversion strategy: long when 24h return <= -5%, exit on +3%/-10%/48h",
        tags=("crypto", "mean-reversion", "eth", "hourly"),
        capabilities=("backtest", "paper"),
        params_schema={
            "type": "object",
            "properties": {
                "entry_threshold": {
                    "type": "number",
                    "default": -0.05,
                    "description": "24h return threshold for entry (negative, e.g., -0.05 = -5%)",
                },
                "take_profit": {
                    "type": "number",
                    "default": 0.03,
                    "description": "Take profit threshold as fraction (e.g., 0.03 = +3%)",
                },
                "stop_loss": {
                    "type": "number",
                    "default": -0.10,
                    "description": "Stop loss threshold as fraction (e.g., -0.10 = -10%)",
                },
                "timeout_hours": {
                    "type": "integer",
                    "default": 48,
                    "description": "Exit position after this many hours",
                },
                "lookback_hours": {
                    "type": "integer",
                    "default": 24,
                    "description": "Hours for rolling return calculation",
                },
                "position_pct": {
                    "type": "number",
                    "default": 0.10,
                    "description": "Portfolio allocation per trade (e.g., 0.10 = 10%)",
                },
                "cooldown_hours": {
                    "type": "integer",
                    "default": 24,
                    "description": "Hours to wait after exit before new entry",
                },
                "symbol": {
                    "type": "string",
                    "default": "ETHUSDT",
                    "description": "Trading symbol",
                },
            },
        },
        inputs=("prices",),
        outputs=("weights",),
        examples=(
            """
strategy:
  plugin: strategy.eth_mean_reversion_24h.v1
  params:
    entry_threshold: -0.05
    take_profit: 0.03
    stop_loss: -0.10
    timeout_hours: 48
    lookback_hours: 24
    position_pct: 0.10
    cooldown_hours: 24
    symbol: ETHUSDT
""",
        ),
    )

    def run(self, data: dict, params: dict) -> dict:
        """Compute target weights based on mean reversion signals.

        Args:
            data: Dict containing 'prices' DataFrame (date index x symbol columns)
            params: Strategy parameters

        Returns:
            Dict with 'weights' DataFrame (date index x symbol columns)
        """
        # Extract parameters with defaults
        entry_threshold = params.get("entry_threshold", -0.05)
        take_profit = params.get("take_profit", 0.03)
        stop_loss = params.get("stop_loss", -0.10)
        timeout_hours = params.get("timeout_hours", 48)
        lookback_hours = params.get("lookback_hours", 24)
        position_pct = params.get("position_pct", 0.10)
        cooldown_hours = params.get("cooldown_hours", 24)
        symbol = params.get("symbol", "ETHUSDT")

        prices_df = data["prices"]

        # Find the symbol column (case-insensitive match)
        symbol_col = None
        for col in prices_df.columns:
            if col.upper() == symbol.upper():
                symbol_col = col
                break

        if symbol_col is None:
            # Return zero weights if symbol not found
            weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
            return {"weights": weights, "trades": [], "error": f"Symbol {symbol} not found"}

        prices = prices_df[symbol_col].dropna()

        if len(prices) < lookback_hours + 1:
            weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
            return {"weights": weights, "trades": [], "error": "Insufficient data"}

        # Calculate 24h rolling returns
        rolling_return = prices.pct_change(periods=lookback_hours, fill_method=None)

        # State tracking for position management
        weights_series = pd.Series(0.0, index=prices.index)
        trades = []

        in_position = False
        entry_price = None
        entry_idx = None
        last_exit_idx = None

        for i, (idx, price) in enumerate(prices.items()):
            if i < lookback_hours:
                continue

            ret_24h = rolling_return.loc[idx]

            if in_position:
                # Check exit conditions
                pnl_pct = (price - entry_price) / entry_price
                hours_held = i - entry_idx

                exit_reason = None
                if pnl_pct >= take_profit:
                    exit_reason = "take_profit"
                elif pnl_pct <= stop_loss:
                    exit_reason = "stop_loss"
                elif hours_held >= timeout_hours:
                    exit_reason = "timeout"

                if exit_reason:
                    trades.append(
                        {
                            "entry_time": prices.index[entry_idx],
                            "exit_time": idx,
                            "entry_price": entry_price,
                            "exit_price": price,
                            "pnl_pct": pnl_pct,
                            "hours_held": hours_held,
                            "exit_reason": exit_reason,
                        }
                    )
                    in_position = False
                    entry_price = None
                    last_exit_idx = i
                    weights_series.loc[idx] = 0.0
                else:
                    # Stay in position
                    weights_series.loc[idx] = position_pct
            else:
                # Check entry conditions
                cooldown_ok = last_exit_idx is None or (i - last_exit_idx) >= cooldown_hours

                if cooldown_ok and not np.isnan(ret_24h) and ret_24h <= entry_threshold:
                    # Entry signal
                    in_position = True
                    entry_price = price
                    entry_idx = i
                    weights_series.loc[idx] = position_pct
                else:
                    weights_series.loc[idx] = 0.0

        # Build full weights DataFrame with all symbols (zeros for non-ETH)
        weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
        weights[symbol_col] = weights_series.reindex(prices_df.index, fill_value=0.0)

        return {
            "weights": weights,
            "trades": trades,
            "n_trades": len(trades),
            "symbol": symbol_col,
        }
