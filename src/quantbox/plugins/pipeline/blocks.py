"""Report block registry — reusable units of the research report.

A *block* is a named, self-describing unit of report content (typically one
chart, sometimes a table or a multi-panel composite). Every block has:

  - **name** — stable key used in HTML ids and config references
  - **title** — display title rendered above the chart
  - **description** — one-line summary (for the cookbook + autodocs)
  - **payload_schema** — informal contract describing the payload shape
  - **section** — which report section the block belongs to
  - **tags** — categorisation (e.g. ``trend``, ``crypto``, ``generic``)
  - **builder** — callable that returns a Plotly figure dict

Block kinds (split by data source)
----------------------------------

**generic blocks** compute from standard pipeline outputs (``returns``,
``portfolio_daily``, ``weights_history``, ``bt_prices``) and auto-render for
every variant — no strategy opt-in needed. Use these to build the simple
"baseline" report.

**diagnostic blocks** are opt-in: a strategy's ``run()`` returns
``details["diagnostics"][block_name] = {...payload}`` and the block renders
only for that variant. Use these for strategy-specific deep dives.

**cross-variant blocks** compare variants and only render in multi-variant
mode (e.g. equity overlay, per-variant heatmap).

How to add a new block
----------------------

1. Write a builder ``def _build_<name>_chart(payload, **ctx) -> dict | None``.
   ``ctx`` carries the report-level inputs (``bt_prices``, ``weights_history``,
   ``returns``, ``portfolio_daily``, ``variant_results``). Return ``None`` to
   suppress the block.
2. Add a :class:`ReportBlock` entry to :data:`BLOCKS` with metadata.
3. Document it in ``docs/playbooks/report-blocks.md``.
4. For strategy-specific blocks, register it under your strategy's tags so the
   cookbook can group it. The reporter's auto-dispatch picks it up by name.

Strategy-specific blocks
------------------------

Domain-specific blocks (Donchian channels for trend-following, factor
loadings for stat-arb, etc.) belong in the consuming repo (e.g.
``quantbox-lab``) so the engine stays domain-agnostic. The lab can extend the
registry at import time::

    from quantbox.plugins.pipeline.blocks import register_block, ReportBlock

    register_block(ReportBlock(
        name="my_strategy_diagnostic",
        title="Factor loadings",
        ...
    ))
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

BuilderFn = Callable[..., dict | None]
Section = Literal["framework", "diagnostics", "comparison", "appendix"]


@dataclass(frozen=True)
class ReportBlock:
    """Metadata + builder for a single report unit.

    The :data:`BLOCKS` registry is keyed by ``name``. Builders receive their
    payload (the strategy-emitted dict for diagnostic blocks, or ``None`` for
    generic/cross-variant blocks) and a ``ctx`` dict with shared inputs::

        ctx = {
            "bt_prices": pd.DataFrame,
            "weights_history": pd.DataFrame,
            "returns": pd.Series,
            "portfolio_daily": pd.DataFrame,
            "variant_results": dict[str, dict],
        }

    The builder returns a Plotly figure dict (``{"data": [...], "layout": {...}}``)
    or ``None`` to skip rendering.
    """

    name: str
    title: str
    description: str
    builder: BuilderFn
    section: Section = "diagnostics"
    payload_schema: str = ""
    tags: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    """Which ``ctx`` keys the block needs to be non-empty. Used to skip
    blocks gracefully when inputs are missing (e.g. ``weights_history``)."""


# ---------------------------------------------------------------------------
# Public registry — strategy/lab code can mutate this via ``register_block()``.
# ---------------------------------------------------------------------------

BLOCKS: dict[str, ReportBlock] = {}


def register_block(block: ReportBlock) -> None:
    """Register a block in the global registry.

    Overwrites silently if a block with the same name already exists — caller's
    responsibility to namespace (e.g. ``ssrn.donchian_overlay``).
    """
    BLOCKS[block.name] = block


def get_block(name: str) -> ReportBlock | None:
    """Lookup helper used by the reporter dispatch."""
    return BLOCKS.get(name)


def diagnostic_block_names() -> list[str]:
    """All block names whose section is ``diagnostics`` — used by the report
    dispatch to know which strategy-emitted keys to render."""
    return [b.name for b in BLOCKS.values() if b.section == "diagnostics"]


# ---------------------------------------------------------------------------
# Generic blocks — compute from standard pipeline outputs only.
# ---------------------------------------------------------------------------


def _build_return_distribution_chart(_payload: dict | None, *, returns: pd.Series, **_ctx) -> dict | None:
    """Daily return histogram + stat badge (mean / std / skew / excess kurt).

    Builds from ``returns`` alone — no strategy opt-in required.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if returns is None or len(returns) == 0:
        return None
    r = pd.Series(returns).dropna()
    if r.empty:
        return None

    r_pct = r * 100
    mean = float(r_pct.mean())
    std = float(r_pct.std())
    try:
        skew = float(r_pct.skew())
        kurt = float(r_pct.kurt())
    except Exception:
        skew = float("nan")
        kurt = float("nan")
    q05 = float(r_pct.quantile(0.05))
    q95 = float(r_pct.quantile(0.95))

    # Histogram + normal overlay
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Histogram(
            x=r_pct.values,
            nbinsx=60,
            marker=dict(color="#3a4055", line=dict(color="#15192c", width=0.5)),
            opacity=0.85,
            name="daily returns",
            histnorm="probability density",
        )
    )
    # Theoretical normal density
    if std > 0:
        xs = np.linspace(r_pct.min(), r_pct.max(), 200)
        ys = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((xs - mean) ** 2) / (2 * std**2))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=f"Normal(μ={mean:.2f}, σ={std:.2f})",
                line=dict(color="#b85427", width=2),
                mode="lines",
            )
        )
    # Quantile rules
    fig.add_vline(x=q05, line=dict(color="#8b2a1f", width=1, dash="dash"))
    fig.add_vline(x=q95, line=dict(color="#1f5d3a", width=1, dash="dash"))

    badge = (
        f"  mean={mean:.2f}%  ·  std={std:.2f}%  ·  skew={skew:.2f}  ·"
        f"  exc-kurt={kurt:.2f}  ·  q05={q05:.2f}%  ·  q95={q95:.2f}%"
    )
    fig.update_layout(
        title=dict(text=f"<b>Daily return distribution</b>{badge}", font=dict(size=12)),
        xaxis_title="daily return (%)",
        yaxis_title="density",
        height=320,
        template="plotly_white",
        showlegend=True,
    )
    return _fig_to_dict(fig)


def _build_rolling_metrics_chart(_payload: dict | None, *, returns: pd.Series, **_ctx) -> dict | None:
    """Rolling Sharpe + rolling annualised vol, two stacked panels.

    Window is chosen as max(63, len/8) so the chart stays readable across
    backtest lengths. Annualisation uses 365 (crypto default); for equities
    pass a custom payload override (future work).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if returns is None or len(returns) == 0:
        return None
    r = pd.Series(returns).dropna()
    if len(r) < 60:
        return None
    if not isinstance(r.index, pd.DatetimeIndex):
        with contextlib.suppress(Exception):
            r.index = pd.DatetimeIndex(r.index)
    annualise = np.sqrt(365.0)
    win = max(63, len(r) // 8)
    rs = (r.rolling(win).mean() / r.rolling(win).std()) * annualise
    rv = r.rolling(win).std() * annualise
    rs = rs.dropna()
    rv = rv.dropna()
    if rs.empty:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"Rolling annualised Sharpe ({win}-day window)",
            f"Rolling annualised volatility ({win}-day window)",
        ),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(x=rs.index, y=rs.values, line=dict(color="#1f5d3a", width=1.6), name="Sharpe"),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="#6a6f80", width=0.5, dash="dash"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=rv.index, y=rv.values * 100, line=dict(color="#b85427", width=1.6), name="ann vol"),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Sharpe", row=1, col=1)
    fig.update_yaxes(title_text="ann vol (%)", row=2, col=1)
    fig.update_layout(
        height=420,
        template="plotly_white",
        showlegend=False,
        title=dict(text="<b>Rolling risk-adjusted performance</b>", font=dict(size=13)),
    )
    return _fig_to_dict(fig)


def _build_turnover_timeline_chart(_payload: dict | None, *, weights_history: pd.DataFrame, **_ctx) -> dict | None:
    """Daily portfolio turnover (sum of absolute weight changes).

    Useful for assessing rebalancing cost vs strategy churn. Cumulative
    turnover (right axis) gives a sense of total trading volume.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if weights_history is None or weights_history.empty:
        return None
    wh = weights_history.select_dtypes(include="number")
    if not isinstance(wh.index, pd.DatetimeIndex):
        with contextlib.suppress(Exception):
            wh.index = pd.DatetimeIndex(wh.index)
    # Sum of |Δweight| per day
    turnover = wh.diff().abs().sum(axis=1).dropna()
    if turnover.empty or turnover.sum() == 0:
        return None
    cumulative = turnover.cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=turnover.index,
            y=turnover.values * 100,
            mode="lines",
            line=dict(color="#3a4055", width=0, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(58,64,85,0.4)",
            name="daily turnover (%)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative.values * 100,
            line=dict(color="#b85427", width=1.8),
            name="cumulative (%)",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="daily turnover (%)", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="cumulative turnover (%)", secondary_y=True, rangemode="tozero")
    fig.update_layout(
        height=320,
        template="plotly_white",
        title=dict(
            text=f"<b>Turnover</b>  ·  total = {cumulative.iloc[-1] * 100:,.0f}%  ·  mean daily = {turnover.mean() * 100:.2f}%",
            font=dict(size=13),
        ),
    )
    return _fig_to_dict(fig)


def _fig_to_dict(fig) -> dict:
    """Serialise a Plotly figure to a JSON-friendly dict."""
    import json

    return json.loads(fig.to_json())


# ---------------------------------------------------------------------------
# Diagnostic blocks (strategy opt-in) — wrappers around the legacy builders.
# Builders themselves live in _report.py for now to keep the diff small. They
# are imported lazily at registry-build time to avoid circular imports.
# ---------------------------------------------------------------------------


def _lazy_legacy_builder(builder_name: str) -> BuilderFn:
    """Return a thunk that resolves the named builder from ``_report`` on first call."""

    def _resolved(payload: dict | None, **ctx) -> dict | None:
        from quantbox.plugins.pipeline import _report as _r

        fn = getattr(_r, builder_name, None)
        if fn is None:
            return None
        try:
            return fn(payload, **ctx) if payload is not None else fn(**ctx)
        except TypeError:
            # Builder doesn't take payload kwarg
            try:
                return fn(**ctx)
            except Exception:
                return None

    _resolved.__name__ = builder_name
    return _resolved


def _populate_default_blocks() -> None:
    """Register the built-in blocks. Called once at import."""

    # ── Generic blocks (auto-on, no strategy opt-in needed) ──────────────
    register_block(
        ReportBlock(
            name="return_distribution",
            title="Daily return distribution",
            description=(
                "Histogram of daily returns with Normal overlay and a stat badge "
                "(mean, std, skew, excess kurtosis, 5%/95% quantiles)."
            ),
            builder=_build_return_distribution_chart,
            section="framework",
            payload_schema="generic — derives from ctx['returns']",
            tags=("generic",),
            requires=("returns",),
        )
    )
    register_block(
        ReportBlock(
            name="rolling_metrics",
            title="Rolling Sharpe and volatility",
            description=(
                "Two-panel: rolling annualised Sharpe (top) and rolling annualised "
                "volatility (bottom). Window auto-scales to backtest length."
            ),
            builder=_build_rolling_metrics_chart,
            section="framework",
            payload_schema="generic — derives from ctx['returns']",
            tags=("generic",),
            requires=("returns",),
        )
    )
    register_block(
        ReportBlock(
            name="turnover_timeline",
            title="Turnover timeline",
            description=(
                "Daily portfolio turnover (sum of absolute weight changes) on the "
                "left axis, cumulative turnover on the right. Reads strategy churn "
                "and gives a back-of-envelope cost estimate."
            ),
            builder=_build_turnover_timeline_chart,
            section="framework",
            payload_schema="generic — derives from ctx['weights_history']",
            tags=("generic",),
            requires=("weights_history",),
        )
    )

    # ── Diagnostic blocks (strategy opt-in via details["diagnostics"]) ──
    register_block(
        ReportBlock(
            name="regime_overlay",
            title="Regime & trend overlay",
            description=(
                "Reference-ticker price with two moving averages (fast + slow). "
                "Used by trend-following strategies to show the regime filter."
            ),
            builder=_lazy_legacy_builder("_build_regime_overlay_chart"),
            section="diagnostics",
            payload_schema='{"ref_ticker": str, "fast_window": int, "slow_window": int, "log_y": bool?}',
            tags=("trend", "regime"),
        )
    )
    register_block(
        ReportBlock(
            name="signal_count",
            title="Active signals over time",
            description=(
                "Daily count of active long signals across the universe, with an "
                "optional dashed cap line. Filled step area to show signal density."
            ),
            builder=_lazy_legacy_builder("_build_signal_count_chart"),
            section="diagnostics",
            payload_schema='{"series": pd.Series[int], "cap": int|float?, "label": str?}',
            tags=("trend", "signal"),
        )
    )
    register_block(
        ReportBlock(
            name="donchian_overlay",
            title="Donchian channel + trailing stop",
            description=(
                "Two-panel: price with Donchian high/low/mid bands + trailing stop + "
                "entry/exit markers (top), binary signal step (bottom). Crypto-trend specific."
            ),
            builder=_lazy_legacy_builder("_build_donchian_overlay_chart"),
            section="diagnostics",
            payload_schema='{"ref_ticker": str, "window": int, "price": Series, "high": Series, "low": Series, "mid": Series, "trailing_stop": Series, "signal": Series}',
            tags=("trend", "donchian", "crypto"),
        )
    )
    register_block(
        ReportBlock(
            name="per_window_signal",
            title="Per-window Donchian signal density",
            description=(
                "Heatmap of per-window 0/1 signal averaged monthly. Surfaces which "
                "lookback windows the ensemble is firing on across time."
            ),
            builder=_lazy_legacy_builder("_build_per_window_heatmap"),
            section="diagnostics",
            payload_schema='{"ref_ticker": str, "signals": {window: pd.Series}}',
            tags=("trend", "ensemble"),
        )
    )
    register_block(
        ReportBlock(
            name="vol_overview",
            title="Realized volatility + scalers",
            description=(
                "Two-panel: top-10 most-covered tickers + universe mean for realized vol, "
                "vol scalers (mean across universe) below. Y-clip at the 99th percentile."
            ),
            builder=_lazy_legacy_builder("_build_vol_overview_chart"),
            section="diagnostics",
            payload_schema='{"realized_vol": DataFrame, "scalers": {label: DataFrame}, "vol_lookback": int}',
            tags=("vol-target",),
        )
    )


_populate_default_blocks()


__all__ = [
    "ReportBlock",
    "BLOCKS",
    "register_block",
    "get_block",
    "diagnostic_block_names",
]
