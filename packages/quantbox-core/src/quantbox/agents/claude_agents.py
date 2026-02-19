"""QuantBox agents built on the Claude Agent SDK.

Pre-built agent configurations for common quant workflows:
research, backtesting, paper trading, and portfolio monitoring.

These agents use the quantbox MCP server for programmatic access to the
framework, combined with Claude's built-in tools for code editing and
shell commands.

Requirements:
    pip install claude-agent-sdk

Usage:
    import asyncio
    from quantbox.agents.claude_agents import research_agent, backtest_agent

    asyncio.run(research_agent("Find the best momentum strategy for crypto"))
    asyncio.run(backtest_agent("Backtest crypto trend on BTC ETH SOL from 2025"))

Environment:
    ANTHROPIC_API_KEY must be set.
"""

from __future__ import annotations

from typing import Any

# Agent definitions — these are the subagent configs, not instantiated agents.
# They can be used directly with the Claude Agent SDK's `agents` parameter.

QUANTBOX_AGENTS: dict[str, dict[str, Any]] = {
    "quant-researcher": {
        "description": (
            "Quant research agent. Explores strategies, backtests ideas, "
            "analyzes data, and recommends portfolio configurations. "
            "Has access to the full quantbox framework via MCP tools."
        ),
        "prompt": (
            "You are a quantitative research agent with access to the QuantBox "
            "framework. Use the quantbox MCP tools to:\n"
            "1. Browse available strategies with quantbox_list_plugins\n"
            "2. Inspect strategy details with quantbox_plugin_info\n"
            "3. Build configs with quantbox_build_config\n"
            "4. Run backtests with quantbox_run\n"
            "5. Inspect results with quantbox_inspect_run\n\n"
            "Always validate configs before running. Explain your reasoning "
            "and summarize results with key metrics."
        ),
        "tools": ["Read", "Bash", "Glob", "Grep"],
    },
    "backtest-runner": {
        "description": (
            "Backtest execution agent. Takes a strategy idea, builds a config, "
            "validates it, runs the backtest, and summarizes results."
        ),
        "prompt": (
            "You are a backtest execution agent. Your workflow:\n"
            "1. Use quantbox_search_plugins to find relevant strategies\n"
            "2. Use quantbox_build_config to create a backtest config\n"
            "3. Use quantbox_validate_config to check for errors\n"
            "4. Use quantbox_run to execute the backtest\n"
            "5. Use quantbox_inspect_run to read artifacts\n"
            "6. Summarize: Sharpe ratio, max drawdown, total return\n\n"
            "If the backtest fails, diagnose the error and retry with fixes."
        ),
        "tools": ["Read", "Bash"],
    },
    "trading-monitor": {
        "description": (
            "Portfolio monitoring agent. Inspects recent runs, checks positions, "
            "and flags any issues or risk violations."
        ),
        "prompt": (
            "You are a portfolio monitoring agent. Your tasks:\n"
            "1. List recent run artifacts in the artifacts/ directory\n"
            "2. Use quantbox_inspect_run to read the latest run manifest\n"
            "3. Check for warnings in run_manifest.json\n"
            "4. Report positions, PnL, and any risk violations\n"
            "5. Suggest corrective actions if needed\n\n"
            "Be concise. Flag only actionable items."
        ),
        "tools": ["Read", "Bash", "Glob"],
    },
    "plugin-builder": {
        "description": (
            "Plugin development agent. Creates new quantbox plugins following "
            "the framework's conventions and protocols."
        ),
        "prompt": (
            "You are a plugin development agent for QuantBox. When asked to "
            "create a plugin:\n"
            "1. Use quantbox_plugin_info on a similar existing plugin as reference\n"
            "2. Read the relevant protocol from contracts.py\n"
            "3. Create the plugin module following the @dataclass + PluginMeta pattern\n"
            "4. Register in builtins.py and __init__.py\n"
            "5. Create an example YAML config\n"
            "6. Write a test\n"
            "7. Verify with: uv run quantbox plugins list\n\n"
            "meta must be a CLASS attribute. Use describe() for LLM introspection."
        ),
        "tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    },
}


async def research_agent(prompt: str, **kwargs) -> str:
    """Run the quant research agent.

    Args:
        prompt: Research question (e.g. "What crypto strategies are available
                and which has the best risk-adjusted returns?")

    Returns:
        Agent's research findings as text.
    """
    return await _run_agent(prompt, agent_hint="quant-researcher", **kwargs)


async def backtest_agent(prompt: str, **kwargs) -> str:
    """Run the backtest agent.

    Args:
        prompt: Backtest request (e.g. "Backtest crypto trend following
                on BTC, ETH, SOL for the last year")

    Returns:
        Backtest results summary.
    """
    return await _run_agent(prompt, agent_hint="backtest-runner", **kwargs)


async def monitor_agent(prompt: str = "Check the latest run for issues", **kwargs) -> str:
    """Run the portfolio monitoring agent.

    Returns:
        Portfolio status report.
    """
    return await _run_agent(prompt, agent_hint="trading-monitor", **kwargs)


async def plugin_builder_agent(prompt: str, **kwargs) -> str:
    """Run the plugin builder agent.

    Args:
        prompt: Plugin specification (e.g. "Create a mean reversion strategy
                plugin that buys oversold assets")

    Returns:
        Summary of created files and verification results.
    """
    return await _run_agent(prompt, agent_hint="plugin-builder", **kwargs)


async def _run_agent(prompt: str, *, agent_hint: str, **kwargs) -> str:
    """Internal: run a quantbox agent with the Claude Agent SDK."""
    try:
        from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition
    except ImportError:
        raise ImportError(
            "Claude Agent SDK not installed. Install with:\n"
            "  pip install claude-agent-sdk\n"
            "  # or: uv add claude-agent-sdk"
        )

    # Build agent definitions
    agents = {}
    for name, config in QUANTBOX_AGENTS.items():
        agents[name] = AgentDefinition(
            description=config["description"],
            prompt=config["prompt"],
            tools=config["tools"],
        )

    # The orchestrator prompt references the specific subagent
    orchestrator_prompt = (
        f"Use the {agent_hint} agent to handle this request:\n\n{prompt}"
    )

    result_text = ""
    async for message in query(
        prompt=orchestrator_prompt,
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Bash", "Glob", "Grep", "Task"],
            agents=agents,
            mcp_servers={
                "quantbox": {
                    "command": "uv",
                    "args": ["run", "--extra", "agents", "python", "-m", "quantbox.agents.mcp_server"],
                }
            },
            **kwargs,
        ),
    ):
        if hasattr(message, "result"):
            result_text = message.result

    return result_text
