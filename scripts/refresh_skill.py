#!/usr/bin/env python
"""Refresh quantbox agent skill from live codebase.

Introspects the plugin registry, contracts, and manifest to regenerate
auto-generated sections of skill reference files. Manually-authored content
(gotchas, patterns, editorial) is preserved.

Sections between <!-- BEGIN AUTO-GENERATED --> and <!-- END AUTO-GENERATED -->
markers are replaced; everything else is untouched.

Usage:
    uv run python scripts/refresh_skill.py          # refresh all
    uv run python scripts/refresh_skill.py --check  # check if up to date (exit 1 if stale)
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
import textwrap
from dataclasses import fields
from pathlib import Path
from typing import Any

# Ensure quantbox is importable
ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / ".claude" / "skills" / "quantbox"

sys.path.insert(0, str(ROOT / "packages" / "quantbox-core" / "src"))

from quantbox.contracts import (  # noqa: E402
    ArtifactStore,
    BrokerPlugin,
    DataPlugin,
    PipelinePlugin,
    PluginMeta,
    PublisherPlugin,
    RebalancingPlugin,
    RiskPlugin,
    RunResult,
    StrategyPlugin,
)
from quantbox.exceptions import (  # noqa: E402
    BrokerExecutionError,
    ConfigValidationError,
    DataLoadError,
    PluginLoadError,
    PluginNotFoundError,
)
from quantbox.plugins.builtins import builtins  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MARKER_START = "<!-- BEGIN AUTO-GENERATED -->"
MARKER_END = "<!-- END AUTO-GENERATED -->"

# Map from builtins() key to skill reference directory
KIND_TO_DIR = {
    "pipeline": "pipelines",
    "data": "data",
    "broker": "brokers",
    "strategy": "strategies",
    "risk": "risk",
    "rebalancing": "pipelines",  # rebalancing docs live in pipelines/
    "publisher": "pipelines",    # publisher docs live in pipelines/
}

# Map from kind to friendly category name
KIND_LABELS = {
    "pipeline": "Pipelines",
    "data": "Data Sources",
    "broker": "Brokers",
    "strategy": "Strategies",
    "risk": "Risk",
    "rebalancing": "Rebalancing",
    "publisher": "Publishers",
}


def get_plugin_meta(cls: type) -> PluginMeta | None:
    """Extract PluginMeta from a plugin class."""
    meta = getattr(cls, "meta", None)
    if isinstance(meta, PluginMeta):
        return meta
    return None


def get_all_plugins() -> dict[str, list[tuple[str, PluginMeta, type]]]:
    """Return {kind: [(name, meta, cls), ...]} for all built-in plugins."""
    result: dict[str, list[tuple[str, PluginMeta, type]]] = {}
    for kind, plugins in builtins().items():
        entries = []
        for name, cls in sorted(plugins.items()):
            meta = get_plugin_meta(cls)
            if meta:
                entries.append((name, meta, cls))
        result[kind] = entries
    return result


def get_dataclass_params(cls: type) -> list[tuple[str, str, Any]]:
    """Return [(field_name, type_str, default)] for dataclass constructor params."""
    params = []
    for f in fields(cls):
        if f.name.startswith("_"):
            continue
        type_str = str(f.type) if f.type else "Any"
        default = f.default if f.default is not f.default_factory else "..."
        params.append((f.name, type_str, default))
    return params


def replace_auto_section(content: str, new_section: str) -> str:
    """Replace content between AUTO-GENERATED markers."""
    pattern = re.compile(
        re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
        re.DOTALL,
    )
    replacement = f"{MARKER_START}\n{new_section}\n{MARKER_END}"
    if pattern.search(content):
        return pattern.sub(replacement, content)
    # No markers found — append
    return content.rstrip() + "\n\n" + replacement + "\n"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_skill_product_index(all_plugins: dict) -> str:
    """Generate the Product Index table for SKILL.md."""
    lines = ["| Category | Count | Plugin IDs |", "|---|---|---|"]
    total = 0
    for kind in ["pipeline", "strategy", "data", "broker", "risk", "rebalancing", "publisher"]:
        entries = all_plugins.get(kind, [])
        total += len(entries)
        ids = ", ".join(f"`{e[0]}`" for e in entries)
        lines.append(f"| {KIND_LABELS[kind]} | {len(entries)} | {ids} |")
    lines.append(f"\nTotal: **{total}** built-in plugins.")
    return "\n".join(lines)


def generate_strategy_catalog(plugins: list) -> str:
    """Generate the strategy catalog table."""
    lines = ["| ID | Description | Tags |", "|---|---|---|"]
    for name, meta, cls in plugins:
        tags = ", ".join(meta.tags) if meta.tags else "-"
        lines.append(f"| `{name}` | {meta.description} | {tags} |")
    return "\n".join(lines)


def generate_broker_catalog(plugins: list) -> str:
    """Generate the broker catalog table."""
    lines = ["| ID | Description | Tags |", "|---|---|---|"]
    for name, meta, cls in plugins:
        tags = ", ".join(meta.tags) if meta.tags else "-"
        lines.append(f"| `{name}` | {meta.description} | {tags} |")
    return "\n".join(lines)


def generate_data_catalog(plugins: list) -> str:
    """Generate the data source catalog table."""
    lines = ["| ID | Description | Tags |", "|---|---|---|"]
    for name, meta, cls in plugins:
        tags = ", ".join(meta.tags) if meta.tags else "-"
        lines.append(f"| `{name}` | {meta.description} | {tags} |")
    return "\n".join(lines)


def generate_pipeline_catalog(pipelines: list, rebalancing: list, publishers: list) -> str:
    """Generate pipeline-domain catalogs (pipelines + rebalancing + publishers)."""
    sections = []

    sections.append("### Pipelines\n")
    lines = ["| ID | Description | Kind |", "|---|---|---|"]
    for name, meta, cls in pipelines:
        kind = getattr(cls, "kind", "-")
        lines.append(f"| `{name}` | {meta.description} | {kind} |")
    sections.append("\n".join(lines))

    sections.append("\n### Rebalancing\n")
    lines = ["| ID | Description |", "|---|---|"]
    for name, meta, cls in rebalancing:
        lines.append(f"| `{name}` | {meta.description} |")
    sections.append("\n".join(lines))

    sections.append("\n### Publishers\n")
    lines = ["| ID | Description |", "|---|---|"]
    for name, meta, cls in publishers:
        lines.append(f"| `{name}` | {meta.description} |")
    sections.append("\n".join(lines))

    return "\n".join(sections)


def generate_risk_catalog(plugins: list) -> str:
    """Generate the risk plugin catalog table."""
    lines = ["| ID | Description | Tags |", "|---|---|---|"]
    for name, meta, cls in plugins:
        tags = ", ".join(meta.tags) if meta.tags else "-"
        lines.append(f"| `{name}` | {meta.description} | {tags} |")
    return "\n".join(lines)


def generate_strategy_decision_tree(plugins: list) -> str:
    """Generate strategy selection decision tree rows."""
    lines = ["| Strategy | Description | Tags |", "|---|---|---|"]
    for name, meta, cls in plugins:
        tags = ", ".join(meta.tags) if meta.tags else "-"
        lines.append(f"| `{name}` | {meta.description} | {tags} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File updaters
# ---------------------------------------------------------------------------


def update_file(path: Path, new_content: str, *, check_only: bool = False) -> bool:
    """Update auto-generated section in a file. Returns True if changed."""
    if not path.exists():
        print(f"  SKIP {path.relative_to(ROOT)} (not found)")
        return False

    old = path.read_text()
    updated = replace_auto_section(old, new_content)

    if old == updated:
        print(f"  OK   {path.relative_to(ROOT)}")
        return False

    if check_only:
        print(f"  STALE {path.relative_to(ROOT)}")
        return True

    path.write_text(updated)
    print(f"  UPDATED {path.relative_to(ROOT)}")
    return True


def refresh_all(*, check_only: bool = False) -> bool:
    """Refresh all skill files. Returns True if any were stale/updated."""
    all_plugins = get_all_plugins()
    changed = False

    # --- SKILL.md: Product Index ---
    print("Refreshing SKILL.md product index...")
    product_index = generate_skill_product_index(all_plugins)
    changed |= update_file(SKILL_DIR / "SKILL.md", product_index, check_only=check_only)

    # --- strategies/README.md ---
    print("Refreshing strategies catalog...")
    strat_catalog = generate_strategy_catalog(all_plugins.get("strategy", []))
    changed |= update_file(
        SKILL_DIR / "references" / "strategies" / "README.md",
        strat_catalog,
        check_only=check_only,
    )

    # --- brokers/README.md ---
    print("Refreshing brokers catalog...")
    broker_catalog = generate_broker_catalog(all_plugins.get("broker", []))
    changed |= update_file(
        SKILL_DIR / "references" / "brokers" / "README.md",
        broker_catalog,
        check_only=check_only,
    )

    # --- data/README.md ---
    print("Refreshing data catalog...")
    data_catalog = generate_data_catalog(all_plugins.get("data", []))
    changed |= update_file(
        SKILL_DIR / "references" / "data" / "README.md",
        data_catalog,
        check_only=check_only,
    )

    # --- pipelines/README.md ---
    print("Refreshing pipelines catalog...")
    pipeline_catalog = generate_pipeline_catalog(
        all_plugins.get("pipeline", []),
        all_plugins.get("rebalancing", []),
        all_plugins.get("publisher", []),
    )
    changed |= update_file(
        SKILL_DIR / "references" / "pipelines" / "README.md",
        pipeline_catalog,
        check_only=check_only,
    )

    # --- risk/README.md ---
    print("Refreshing risk catalog...")
    risk_catalog = generate_risk_catalog(all_plugins.get("risk", []))
    changed |= update_file(
        SKILL_DIR / "references" / "risk" / "README.md",
        risk_catalog,
        check_only=check_only,
    )

    return changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Refresh quantbox skill references")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if files are up to date (exit 1 if stale)",
    )
    args = parser.parse_args()

    print(f"Skill directory: {SKILL_DIR}")
    print(f"Mode: {'check' if args.check else 'refresh'}\n")

    stale = refresh_all(check_only=args.check)

    if args.check and stale:
        print("\nSkill docs are STALE. Run: uv run python scripts/refresh_skill.py")
        sys.exit(1)
    elif not stale:
        print("\nAll skill docs are up to date.")
    else:
        print("\nSkill docs refreshed.")


if __name__ == "__main__":
    main()
