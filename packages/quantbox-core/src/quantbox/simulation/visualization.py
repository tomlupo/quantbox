"""Visualization utilities for market simulation results.

Optional module — requires matplotlib and seaborn.
Install with: ``uv pip install 'quantbox[viz]'``

Ported from quantlabnew/src/market-simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

_VIZ_ERROR = (
    "matplotlib and seaborn are required for visualization. "
    "Install with: uv pip install 'quantbox[viz]'"
)


def _require_viz():
    if not HAS_VIZ:
        raise ImportError(_VIZ_ERROR)


class SimulationPlotter:
    """Visualization utilities for market simulation results."""

    def __init__(self, figsize: tuple = (12, 8), style: str = "seaborn-v0_8-whitegrid"):
        _require_viz()
        self.figsize = figsize
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn" in plt.style.available else "ggplot")
        self.colors = plt.cm.tab10.colors

    def plot_simulation_paths(
        self, prices: np.ndarray, n_paths_to_show: int = 100,
        percentiles: list[int] | None = None,
        title: str = "Simulated Price Paths", save_path: Optional[str] = None,
    ):
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        fig, ax = plt.subplots(figsize=self.figsize)
        n_paths, n_steps = prices.shape
        t = np.arange(n_steps)

        indices = np.random.choice(n_paths, min(n_paths_to_show, n_paths), replace=False)
        for i in indices:
            ax.plot(t, prices[i], alpha=0.1, color="gray", linewidth=0.5)

        colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(percentiles) // 2 + 1))
        for i, (lower_p, upper_p) in enumerate(zip(percentiles[: len(percentiles) // 2], percentiles[len(percentiles) // 2 + 1 :][::-1])):
            lower = np.percentile(prices, lower_p, axis=0)
            upper = np.percentile(prices, upper_p, axis=0)
            ax.fill_between(t, lower, upper, alpha=0.3, color=colors[i], label=f"{lower_p}th-{upper_p}th percentile")

        median = np.percentile(prices, 50, axis=0)
        ax.plot(t, median, "b-", linewidth=2, label="Median")
        ax.set_xlabel("Time Steps"); ax.set_ylabel("Price"); ax.set_title(title)
        ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_return_distribution(
        self, returns: np.ndarray, title: str = "Return Distribution",
        bins: int = 100, show_normal: bool = True, save_path: Optional[str] = None,
    ):
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1] * 0.6))
        ax1 = axes[0]
        ax1.hist(returns, bins=bins, density=True, alpha=0.7, color=self.colors[0], edgecolor="white")
        if show_normal:
            from scipy import stats
            x = np.linspace(returns.min(), returns.max(), 200)
            normal = stats.norm.pdf(x, np.mean(returns), np.std(returns))
            ax1.plot(x, normal, "r-", linewidth=2, label="Normal")
            ax1.legend()
        ax1.set_xlabel("Return"); ax1.set_ylabel("Density"); ax1.set_title(f"{title} - Histogram")

        from scipy import stats
        ax2 = axes[1]
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title(f"{title} - Q-Q Plot")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_correlation_matrix(
        self, correlation: np.ndarray, asset_names: list[str] | None = None,
        title: str = "Correlation Matrix", annot: bool = True, save_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=(self.figsize[0] * 0.8, self.figsize[1] * 0.8))
        if asset_names is None:
            asset_names = [f"Asset {i}" for i in range(len(correlation))]
        mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
        sns.heatmap(correlation, mask=mask, annot=annot, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    xticklabels=asset_names, yticklabels=asset_names, ax=ax)
        ax.set_title(title); plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_correlation_dynamics(
        self, correlation_history: np.ndarray, asset_pairs: list[tuple[int, int]] | None = None,
        asset_names: list[str] | None = None, title: str = "Correlation Dynamics", save_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
        n_times, n_assets, _ = correlation_history.shape
        t = np.arange(n_times)
        if asset_names is None:
            asset_names = [f"Asset {i}" for i in range(n_assets)]
        if asset_pairs is None:
            asset_pairs = [(i, j) for i in range(n_assets) for j in range(i + 1, n_assets)]
        for i, (a1, a2) in enumerate(asset_pairs[:10]):
            corr_series = correlation_history[:, a1, a2]
            ax.plot(t, corr_series, label=f"{asset_names[a1]}-{asset_names[a2]}", color=self.colors[i % len(self.colors)])
        ax.set_xlabel("Time"); ax.set_ylabel("Correlation"); ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3); ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_stress_test_comparison(
        self, results_df: pd.DataFrame, metric: str = "portfolio_impact",
        title: str = "Stress Test Comparison", save_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.6))
        scenarios = results_df.index.tolist()
        values = results_df[metric].values
        colors = ["red" if v < 0 else "green" for v in values]
        bars = ax.barh(scenarios, values, color=colors, alpha=0.7)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel(metric.replace("_", " ").title()); ax.set_title(title)
        for bar, val in zip(bars, values):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", ha="left" if val >= 0 else "right")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_fan_chart(
        self, fan_data: pd.DataFrame, title: str = "Return Forecast Fan Chart", save_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
        horizons = fan_data["horizon"].values
        percentile_cols = sorted([c for c in fan_data.columns if c.startswith("p")], key=lambda x: int(x[1:]))
        n_bands = len(percentile_cols) // 2
        colors = plt.cm.Blues(np.linspace(0.2, 0.8, n_bands))
        for i in range(n_bands):
            lower_col, upper_col = percentile_cols[i], percentile_cols[-(i + 1)]
            ax.fill_between(horizons, fan_data[lower_col].values, fan_data[upper_col].values, alpha=0.5, color=colors[i], label=f"{lower_col[1:]}-{upper_col[1:]}th")
        if "p50" in fan_data.columns:
            ax.plot(horizons, fan_data["p50"].values, "b-", linewidth=2, label="Median")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Horizon (days)"); ax.set_ylabel("Cumulative Return"); ax.set_title(title)
        ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_term_structure(
        self, term_data: pd.DataFrame, metrics: list[str] | None = None,
        title: str = "Term Structure of Expected Returns", save_path: Optional[str] = None,
    ):
        if metrics is None:
            metrics = ["annualized_return", "annualized_volatility"]
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(self.figsize[0], self.figsize[1] * 0.5))
        if n_metrics == 1:
            axes = [axes]
        horizons = term_data["horizon_days"].values
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.plot(horizons, term_data[metric].values, "b-o", linewidth=2, markersize=6)
            ax.set_xlabel("Horizon (days)"); ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(metric.replace("_", " ").title()); ax.grid(True, alpha=0.3)
            if "return" in metric.lower():
                ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        plt.suptitle(title, y=1.02); plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_risk_metrics(
        self, var_values: dict[float, float], cvar_values: dict[float, float] | None = None,
        title: str = "Risk Metrics", save_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=(self.figsize[0] * 0.7, self.figsize[1] * 0.6))
        levels = list(var_values.keys())
        x = np.arange(len(levels))
        width = 0.35
        var_vals = [var_values[l] for l in levels]
        bars1 = ax.bar(x - width / 2, var_vals, width, label="VaR", color=self.colors[0])
        if cvar_values:
            cvar_vals = [cvar_values[l] for l in levels]
            bars2 = ax.bar(x + width / 2, cvar_vals, width, label="CVaR", color=self.colors[1])
        ax.set_xlabel("Confidence Level"); ax.set_ylabel("Loss"); ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels([f"{int(l * 100)}%" for l in levels])
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2%}", ha="center", va="bottom", fontsize=9)
        if cvar_values:
            for bar in bars2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2%}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_drawdown_distribution(
        self, max_drawdowns: np.ndarray, title: str = "Maximum Drawdown Distribution", save_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=(self.figsize[0] * 0.8, self.figsize[1] * 0.6))
        ax.hist(max_drawdowns, bins=50, density=True, alpha=0.7, color=self.colors[3], edgecolor="white")
        for p, color in [(5, "red"), (50, "blue"), (95, "green")]:
            val = np.percentile(max_drawdowns, p)
            ax.axvline(x=val, color=color, linestyle="--", linewidth=2, label=f"{p}th: {val:.1%}")
        ax.set_xlabel("Maximum Drawdown"); ax.set_ylabel("Density"); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def create_summary_dashboard(
        self, simulation_result, stress_results: pd.DataFrame | None = None, save_path: Optional[str] = None,
    ):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        prices = simulation_result.prices[0]
        n_show = min(100, len(prices))
        for i in range(n_show):
            ax1.plot(prices[i], alpha=0.1, color="gray", linewidth=0.5)
        ax1.plot(np.percentile(prices, 50, axis=0), "b-", linewidth=2)
        ax1.set_title(f"Price Paths - {simulation_result.asset_names[0]}")
        ax1.set_xlabel("Time"); ax1.set_ylabel("Price")

        ax2 = fig.add_subplot(gs[0, 1])
        terminal_returns = simulation_result.get_terminal_returns().iloc[:, 0].values
        ax2.hist(terminal_returns, bins=50, density=True, alpha=0.7)
        ax2.axvline(np.mean(terminal_returns), color="red", linestyle="--", label=f"Mean: {np.mean(terminal_returns):.1%}")
        ax2.set_title("Terminal Return Distribution"); ax2.set_xlabel("Return"); ax2.legend()

        ax3 = fig.add_subplot(gs[1, 0])
        if simulation_result.correlation_matrix is not None:
            sns.heatmap(simulation_result.correlation_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax3,
                        xticklabels=simulation_result.asset_names, yticklabels=simulation_result.asset_names)
        ax3.set_title("Correlation Matrix")

        ax4 = fig.add_subplot(gs[1, 1])
        stats = simulation_result.get_path_statistics()
        ax4.axis("off")
        table_data = stats[["mean_return", "std_return", "var_95", "sharpe_ratio"]].round(4)
        table = ax4.table(cellText=table_data.values, rowLabels=table_data.index, colLabels=table_data.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.5)
        ax4.set_title("Path Statistics")

        if stress_results is not None:
            ax5 = fig.add_subplot(gs[2, :])
            scenarios = stress_results.index.tolist()
            impacts = stress_results["portfolio_impact"].values
            colors = ["red" if v < 0 else "green" for v in impacts]
            ax5.barh(scenarios, impacts, color=colors, alpha=0.7)
            ax5.axvline(x=0, color="black", linestyle="-")
            ax5.set_title("Stress Test Impacts"); ax5.set_xlabel("Portfolio Impact")
        else:
            ax5 = fig.add_subplot(gs[2, :])
            stats = simulation_result.get_path_statistics()
            ax5.bar(stats.index, stats["max_drawdown_mean"], color=self.colors[3], alpha=0.7)
            ax5.set_title("Mean Maximum Drawdown by Asset"); ax5.set_ylabel("Max Drawdown")

        plt.suptitle("Market Simulation Dashboard", fontsize=14, fontweight="bold")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


__all__ = ["SimulationPlotter"]
