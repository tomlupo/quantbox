"""QuantBox MCP Server — exposes framework operations as agent tools.

Run standalone:
    uv run python -m quantbox.agents.mcp_server

Configure in Claude Code (.claude/settings.json):
    {
      "mcpServers": {
        "quantbox": {
          "command": "uv",
          "args": ["run", "python", "-m", "quantbox.agents.mcp_server"]
        }
      }
    }

Tools exposed:
    - quantbox_list_plugins: Browse all registered plugins
    - quantbox_plugin_info: Get detailed info about a plugin
    - quantbox_search_plugins: Search by name/description/tags
    - quantbox_validate_config: Validate a YAML config
    - quantbox_run: Execute a pipeline
    - quantbox_dry_run: Preview execution without side effects
    - quantbox_inspect_run: Read artifacts from a completed run
    - quantbox_build_config: Construct a config from parameters
    - quantbox_list_profiles: List available plugin profiles
"""

from __future__ import annotations

import json
import sys


def create_server():
    """Create and configure the MCP server."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "quantbox",
        instructions=(
            "QuantBox quant trading framework. Use these tools to browse plugins, "
            "build configs, validate, run pipelines, and inspect results. "
            "Start with quantbox_list_plugins to see available plugins, then "
            "quantbox_build_config to construct a config, quantbox_validate_config "
            "to validate, and quantbox_run to execute."
        ),
    )

    # Lazy-init the agent to avoid import cost on tool definition
    _agent = None

    def get_agent():
        nonlocal _agent
        if _agent is None:
            from quantbox.agents.tools import QuantBoxAgent
            _agent = QuantBoxAgent()
        return _agent

    @mcp.tool()
    def quantbox_list_plugins(kind: str | None = None) -> str:
        """List all registered QuantBox plugins.

        Args:
            kind: Filter by type (pipeline, strategy, data, broker, risk,
                  rebalancing, publisher). Omit for all types.

        Returns:
            JSON with plugin names and descriptions grouped by type.
        """
        result = get_agent().list_plugins(kind)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def quantbox_plugin_info(name: str) -> str:
        """Get detailed info about a specific plugin.

        Args:
            name: Plugin ID (e.g. "strategy.crypto_trend.v1").
                  Use quantbox_list_plugins to find valid names.

        Returns:
            JSON with description, parameter schema, defaults, methods, examples.
        """
        result = get_agent().plugin_info(name)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def quantbox_search_plugins(query: str) -> str:
        """Search plugins by name, description, or tags.

        Args:
            query: Search term (e.g. "trend", "futures", "binance").

        Returns:
            JSON list of matching plugins.
        """
        result = get_agent().search_plugins(query)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def quantbox_build_config(
        mode: str = "backtest",
        asof: str = "2026-02-01",
        pipeline: str = "backtest.pipeline.v1",
        strategy: str | None = None,
        data: str = "binance.live_data.v1",
        broker: str | None = None,
        risk: str | None = None,
        strategy_params: str | None = None,
        pipeline_params: str | None = None,
    ) -> str:
        """Build a QuantBox config from parameters.

        Args:
            mode: Execution mode (backtest, paper, live).
            asof: Reference date (YYYY-MM-DD).
            pipeline: Pipeline plugin ID.
            strategy: Strategy plugin ID (optional).
            data: Data plugin ID.
            broker: Broker plugin ID (required for paper/live).
            risk: Risk plugin ID (optional).
            strategy_params: JSON string of strategy parameters.
            pipeline_params: JSON string of pipeline parameters.

        Returns:
            YAML config string ready for quantbox_validate_config and quantbox_run.
        """
        import yaml

        kwargs: dict = {
            "mode": mode,
            "asof": asof,
            "pipeline": pipeline,
            "data": data,
        }
        if strategy:
            kwargs["strategy"] = strategy
        if broker:
            kwargs["broker"] = broker
        if risk:
            kwargs["risk"] = risk
        if strategy_params:
            kwargs["strategy_params"] = json.loads(strategy_params)
        if pipeline_params:
            kwargs["pipeline_params"] = json.loads(pipeline_params)

        config = get_agent().build_config(**kwargs)
        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    @mcp.tool()
    def quantbox_validate_config(config_yaml: str) -> str:
        """Validate a QuantBox YAML config.

        Args:
            config_yaml: YAML config string or path to a YAML file.

        Returns:
            JSON with {"valid": true/false, "findings": [...]}.
        """
        result = get_agent().validate_config(config_yaml)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def quantbox_run(config_yaml: str) -> str:
        """Execute a QuantBox pipeline.

        Args:
            config_yaml: YAML config string or path to a YAML file.

        Returns:
            JSON with run_id, artifacts, metrics, or error details.
        """
        result = get_agent().run(config_yaml)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def quantbox_dry_run(config_yaml: str) -> str:
        """Preview a pipeline execution without side effects.

        Args:
            config_yaml: YAML config string or path to a YAML file.

        Returns:
            JSON with validation results and execution plan.
        """
        result = get_agent().run(config_yaml, dry_run=True)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def quantbox_inspect_run(run_dir: str) -> str:
        """Inspect artifacts from a completed pipeline run.

        Args:
            run_dir: Path to artifacts/<run_id>/ directory.

        Returns:
            JSON with manifest, artifact list, and summaries.
        """
        result = get_agent().inspect_run(run_dir)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def quantbox_list_profiles() -> str:
        """List available plugin profiles (preset configurations).

        Returns:
            JSON with profile names and their plugin selections.
        """
        result = get_agent().list_profiles()
        return json.dumps(result, indent=2, default=str)

    return mcp


def main():
    mcp = create_server()
    mcp.run()


if __name__ == "__main__":
    main()
