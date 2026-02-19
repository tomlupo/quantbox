"""QuantBox agent tools for LLM-driven workflows.

This module provides two levels of agent integration:

## 1. Programmatic API (any agent framework)

    from quantbox.agents import QuantBoxAgent

    agent = QuantBoxAgent()
    agent.list_plugins()                         # browse registry
    agent.plugin_info("strategy.crypto_trend.v1")  # inspect a plugin
    agent.validate_config("configs/my.yaml")     # validate
    agent.run("configs/my.yaml")                 # execute
    agent.inspect_run("artifacts/<run_id>")       # read results

## 2. Claude Agent SDK (subagents)

    from quantbox.agents import research_agent, backtest_agent

    import asyncio
    asyncio.run(research_agent("Find the best momentum strategy for crypto"))
    asyncio.run(backtest_agent("Backtest crypto trend on BTC ETH SOL"))

## 3. MCP Server (Claude Code, Cursor, etc.)

    # In .claude/settings.json:
    # {"mcpServers": {"quantbox": {"command": "uv", "args": ["run", "quantbox-mcp"]}}}

## 4. Subagent definitions (for custom orchestrators)

    from quantbox.agents import QUANTBOX_AGENTS
    # Dict of agent name -> {description, prompt, tools}
"""

from .tools import QuantBoxAgent

# Lazy imports for optional Claude Agent SDK
def __getattr__(name):
    if name in ("research_agent", "backtest_agent", "monitor_agent",
                "plugin_builder_agent", "QUANTBOX_AGENTS"):
        from . import claude_agents
        return getattr(claude_agents, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "QuantBoxAgent",
    "research_agent",
    "backtest_agent",
    "monitor_agent",
    "plugin_builder_agent",
    "QUANTBOX_AGENTS",
]
