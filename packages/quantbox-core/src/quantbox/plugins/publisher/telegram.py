"""Telegram publisher plugin.

Sends trading pipeline results to a Telegram chat.  Ported from
quantlab ``trading_bot/telegram.py`` formatting functions and
``utils/messaging.py:send_telegram_message()``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

from quantbox.contracts import PluginMeta, RunResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Formatting helpers (ported from quantlab telegram.py)
# ------------------------------------------------------------------


def _format_execution_summary(
    summary: dict[str, Any],
    paper_trading: bool,
    pipeline_name: str,
    mode: str,
    portfolio_value: float,
) -> str:
    mode_label = "Paper Trading" if paper_trading else "Live Trading"
    return (
        "<b>Trading Bot Report</b>\n"
        f"Pipeline: <b>{pipeline_name}</b>\n\n"
        "Execution Summary\n"
        f"Total Executed: {summary.get('total_executed', 0)}\n"
        f"Total Failed: {summary.get('total_failed', 0)}\n"
        f"Total Value Traded: ${summary.get('total_value_traded', 0):,.2f}\n"
        f"Total Cost: ${summary.get('total_cost', 0):,.4f}\n"
        f"Portfolio Value: ${portfolio_value:,.2f}\n"
        f"Mode: {mode_label} ({mode})\n"
        f"Run completed at: {pd.Timestamp.now()}"
    )


def _format_portfolio_rebalancing(
    rebalancing: list[dict[str, Any]],
    pipeline_name: str,
) -> str:
    header = (
        "<b>Trading Bot Report</b>\n"
        f"Pipeline: <b>{pipeline_name}</b>\n\n"
        "Portfolio Rebalancing\n"
        "\n"
        "<pre>"
        "ASSET  CURRENT  TARGET   DELTA\n"
        "-------------------------------\n"
    )
    rows: list[str] = []
    total_curr = 0.0
    total_tgt = 0.0
    total_delta = 0.0

    for entry in rebalancing:
        curr_w = float(entry.get("current_weight", 0))
        tgt_w = float(entry.get("target_weight", 0))
        delta_w = float(entry.get("weight_delta", 0))
        if curr_w != 0 or tgt_w != 0:
            asset = str(entry.get("asset", ""))[:6].ljust(6)
            rows.append(f"{asset}{curr_w * 100:6.1f}%{tgt_w * 100:7.1f}%{delta_w * 100:+7.1f}%")
            total_curr += curr_w
            total_tgt += tgt_w
            total_delta += delta_w

    total_row = f"TOTAL  {total_curr * 100:6.1f}%  {total_tgt * 100:7.1f}%{total_delta * 100:+7.1f}%"
    sep = "-------------------------------\n"
    return header + "\n".join(rows) + "\n" + sep + total_row + "\n</pre>"


def _format_executed_orders(
    orders: list[dict[str, Any]],
    pipeline_name: str,
) -> str:
    header = f"<b>Trading Bot Report</b>\nPipeline: <b>{pipeline_name}</b>\n\nExecuted Orders\n<pre>"
    rows: list[str] = []
    for o in orders:
        action = str(o.get("action", "")).upper()
        symbol = str(o.get("symbol", ""))
        qty = float(o.get("quantity", o.get("executed_quantity", 0)))
        price = float(o.get("executed_price", 0))
        spread_bps = float(o.get("spread_bps", 0))
        rows.append(f"{action}: {symbol} {qty:g} @ {price:g} ({spread_bps:.1f})")
    return header + "\n".join(rows) + "\n</pre>\n\n<i>quantity @ price (spread in bps)</i>\n"


def _format_failed_orders(
    orders: list[dict[str, Any]],
    pipeline_name: str,
) -> str:
    header = f"<b>Trading Bot Report</b>\nPipeline: <b>{pipeline_name}</b>\n\nFailed Orders\n<pre>"
    rows: list[str] = []
    for o in orders:
        action = str(o.get("action", "")).upper()
        symbol = str(o.get("symbol", ""))
        qty = float(o.get("quantity", o.get("adjusted_qty", 0)))
        reason = str(o.get("error", o.get("reason", "Unknown")))
        rows.append(f"{action}: {symbol} {qty:g} ({reason})")
    return header + "\n".join(rows) + "\n</pre>\n\n<i>quantity (reason)</i>\n"


def _convert_md_bold_to_html(text: str) -> str:
    """Convert **bold** markdown to <b>bold</b> HTML for Telegram."""
    import re

    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _send_telegram_message(
    token: str,
    chat_id: str,
    message: str,
    parse_mode: str = "HTML",
) -> dict[str, Any]:
    """Post a message to the Telegram Bot API via httpx."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = httpx.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": parse_mode},
            timeout=30,
        )
        return resp.json()
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)
        return {"ok": False, "error": str(exc)}


# ------------------------------------------------------------------
# Plugin
# ------------------------------------------------------------------


@dataclass
class TelegramPublisher:
    meta = PluginMeta(
        name="telegram.publisher.v1",
        kind="publisher",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Send trading results to Telegram (execution summary, rebalancing, orders).",
        tags=("telegram", "notifications"),
        capabilities=("paper", "live"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "telegram_token_env": {
                    "type": "string",
                    "default": "TELEGRAM_TOKEN",
                    "description": "Env var name holding the bot token.",
                },
                "telegram_chat_id_env": {
                    "type": "string",
                    "default": "TELEGRAM_CHAT_ID",
                    "description": "Env var name holding the chat id.",
                },
                "telegram_token": {
                    "type": "string",
                    "description": "Explicit bot token (overrides env).",
                },
                "telegram_chat_id": {
                    "type": "string",
                    "description": "Explicit chat id (overrides env).",
                },
            },
        },
        examples=(
            "plugins:\n  publishers:\n    - name: telegram.publisher.v1\n      params:\n        telegram_token_env: TELEGRAM_TOKEN\n        telegram_chat_id_env: TELEGRAM_CHAT_ID",
        ),
    )

    def publish(self, result: RunResult, params: dict[str, Any]) -> None:
        """Send trading result notifications to Telegram."""
        token = params.get("telegram_token") or os.environ.get(params.get("telegram_token_env", "TELEGRAM_TOKEN"), "")
        chat_id = params.get("telegram_chat_id") or os.environ.get(
            params.get("telegram_chat_id_env", "TELEGRAM_CHAT_ID"), ""
        )
        if not token or not chat_id:
            logger.warning("Telegram credentials not configured, skipping publish")
            return

        notes = result.notes or {}
        pipeline_name = result.pipeline_name
        mode = result.mode
        paper_trading = mode == "paper"

        # Extract data from notes (populated by TradingPipeline)
        artifact_payload = notes.get("artifact_payload", {})
        portfolio_value = float(
            artifact_payload.get("portfolio_value", result.metrics.get("portfolio_value_usd_pre", 0))
        )
        exec_summary = artifact_payload.get("execution_summary", {})
        rebalancing = artifact_payload.get("rebalancing_table", [])
        executed = exec_summary.get("executed_orders", [])
        failed = exec_summary.get("failed_orders", [])

        # 1. Execution summary
        msg = _format_execution_summary(exec_summary, paper_trading, pipeline_name, mode, portfolio_value)
        _send_telegram_message(token, chat_id, msg)

        # 2. Portfolio rebalancing
        if rebalancing:
            msg = _format_portfolio_rebalancing(rebalancing, pipeline_name)
            _send_telegram_message(token, chat_id, msg)

        # 3. Executed orders
        if executed:
            msg = _format_executed_orders(executed, pipeline_name)
            _send_telegram_message(token, chat_id, msg)

        # 4. Failed orders
        if failed:
            msg = _format_failed_orders(failed, pipeline_name)
            _send_telegram_message(token, chat_id, msg)

        # 5. New token alerts (from TokenPolicy)
        new_token_alert = notes.get("new_token_alert")
        if new_token_alert:
            alert_msg = _convert_md_bold_to_html(new_token_alert)
            _send_telegram_message(token, chat_id, alert_msg, parse_mode="HTML")

        logger.info("Telegram notifications sent for %s", result.run_id)
