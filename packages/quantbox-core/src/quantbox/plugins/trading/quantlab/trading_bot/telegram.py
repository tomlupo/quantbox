# --- Imports ---

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Any, Dict
from tabulate import tabulate
from ..utils import get_logger
from ..utils.messaging import send_telegram_message

logger = get_logger()

def format_execution_summary(summary: dict, paper_trading: bool, run_time: Any, account_name: str) -> str:
    """
    Formats the execution summary message.
    Args:
        summary: Dict with keys 'total_executed', 'total_failed', 'total_value', 'total_cost'.
        paper_trading: Whether in paper trading mode.
        run_time: Timestamp of run completion.
        account_name: Name of the trading account.
    Returns:
        Formatted summary string.
    """
    return (
        "🤖 <b>Trading Bot Report</b>\n"
        f"Account: <b>{account_name}</b>\n\n"
        "📊 Execution Summary\n"
        f"Total Executed: {summary.get('total_executed', 0)}\n"
        f"Total Failed: {summary.get('total_failed', 0)}\n"
        f"Total Value: ${summary.get('total_value', 0):,.2f}\n"
        f"Total Cost: ${summary.get('total_cost', 0):,.2f}\n"
        f"Mode: {'📄 Paper Trading' if paper_trading else '💰 Live Trading'}\n"
        f"Run completed at: {run_time}"
    )

def format_portfolio_rebalancing(rebalancing_df: pd.DataFrame, account_name: str) -> str:
    """
    Formats the portfolio rebalancing message to match the specified template.
    Args:
        rebalancing_df: DataFrame with rebalancing info.
        account_name: Name of the trading account.
    Returns:
        Formatted rebalancing string.
    """
    header = (
        "🤖 <b>Trading Bot Report</b>\n"
        f"Account: <b>{account_name}</b>\n\n"
        "🔄 Portfolio Rebalancing\n"
        "\n"
        "<pre>"
        "ASSET  CURRENT  TARGET   DELTA\n"
        "-------------------------------\n"
    )
    rows = []
    for _, row in rebalancing_df.iterrows():
        if row['Current Weight'] > 0 or row['Target Weight'] > 0:
            asset = f"{row.Asset:6}"
            curr = f"{row['Current Weight']*100:6.1f}%"
            tgt = f"{row['Target Weight']*100:7.1f}%"
            delta_val = row['Weight Delta']*100
            delta = f"{delta_val:+7.1f}%"
            rows.append(f"{asset}{curr}{tgt}{delta}")
    # Calculate totals
    total_curr = rebalancing_df['Current Weight'].sum() * 100
    total_tgt = rebalancing_df['Target Weight'].sum() * 100
    total_delta = rebalancing_df['Weight Delta'].sum() * 100
    total_row = f"TOTAL  {total_curr:6.1f}%  {total_tgt:7.1f}%{total_delta:+7.1f}%"
    sep = "-------------------------------\n"
    return header + "\n".join(rows) + "\n" + sep + total_row + "\n</pre>"

def format_executed_orders(executed_orders: list, account_name: str) -> str:
    """
    Formats the executed orders message to match the specified template.
    Args:
        executed_orders: List of order dicts.
        account_name: Name of the trading account.
    Returns:
        Formatted executed orders string.
    """
    header = (
        "🤖 <b>Trading Bot Report</b>\n"
        f"Account: <b>{account_name}</b>\n\n"
        "💡 Executed Orders\n"
        "<pre>"
    )
    rows = []
    for order in executed_orders:
        action = order.get('Action', '').upper()
        symbol = order.get('Symbol', '')
        qty = order.get('executed_quantity', 0)
        price = order.get('executed_price', 0)
        # Compute spread in bps for display
        spread_pct = order.get('spread_pct')
        if spread_pct is not None:
            spread_bps = spread_pct * 10000
        else:
            ref_price = order.get('reference_price') or order.get('executed_price') or 0
            raw_spread = order.get('spread', 0)
            spread_bps = (raw_spread / ref_price * 10000) if ref_price else 0
        # Format: ACTION: SYMBOL QTY @ PRICE (SPREAD)
        rows.append(f"{action}: {symbol} {qty:g} @ {price:g} ({spread_bps:.1f})")
    return header + "\n".join(rows) + "\n</pre>\n\n<i>quantity @ price (spread in bps)</i>\n"

def format_failed_orders(failed_orders: list, account_name: str) -> str:
    """
    Formats the failed orders message to match the specified template.
    Args:
        failed_orders: List of order dicts.
        account_name: Name of the trading account.
    Returns:
        Formatted failed orders string.
    """
    header = (
        "🤖 <b>Trading Bot Report</b>\n"
        f"Account: <b>{account_name}</b>\n\n"
        "❌ Failed Orders\n"
        "<pre>"
    )
    rows = []
    for order in failed_orders:
        action = order.get('Action', '').upper()
        symbol = order.get('Symbol', '')
        qty = order.get('Adjusted Quantity', order.get('executed_quantity', 0))
        reason = order.get('Reason', order.get('status', 'Unknown'))
        # Format: ACTION: SYMBOL QTY (REASON)
        rows.append(f"{action}: {symbol} {qty:g} ({reason})")
    return header + "\n".join(rows) + "\n</pre>\n\n<i>quantity (reason)</i>\n"


def send_telegram_notifications(portfolio_orders: Dict, execution_report: Dict, account_config: Dict) -> None:
    """Send Telegram notifications: Execution Summary, Portfolio Rebalancing, Executed Orders, Failed Orders."""
    telegram_token = account_config.get('telegram_token') or os.environ.get('TELEGRAM_TOKEN')
    telegram_chat_id = account_config.get('telegram_chat_id') or os.environ.get('TELEGRAM_CHAT_ID')
    account_name = account_config.get('name') or account_config.get('account_name') or 'N/A'
    
    if not telegram_token or not telegram_chat_id:
        logger.warning("Telegram credentials not configured, skipping notifications")
        return
    try:
        rebalancing_df = portfolio_orders.get('rebalancing')
        orders_details = execution_report.get('orders_details', [])
        summary = execution_report.get('summary', {})
        paper_trading = execution_report.get('paper_trading', True)
        run_time = pd.Timestamp.now()

        # --- Execution Summary ---
        summary_msg = format_execution_summary(summary, paper_trading, run_time, account_name)
        send_telegram_message(telegram_token, telegram_chat_id, summary_msg)

        # --- Portfolio Rebalancing ---
        rebalancing_msg = format_portfolio_rebalancing(rebalancing_df, account_name)
        send_telegram_message(telegram_token, telegram_chat_id, rebalancing_msg)

        # --- Executed Orders ---
        executed_orders = [o for o in orders_details if o.get('status') == 'FILLED']
        if executed_orders:
            executed_msg = format_executed_orders(executed_orders, account_name)
            send_telegram_message(telegram_token, telegram_chat_id, executed_msg)

        # --- Failed Orders ---
        failed_orders = [o for o in orders_details if o.get('status') != 'FILLED']
        if failed_orders:
            failed_msg = format_failed_orders(failed_orders, account_name)
            send_telegram_message(telegram_token, telegram_chat_id, failed_msg)

    except Exception as e:
        logger.error(f"Error sending Telegram notifications: {e}")