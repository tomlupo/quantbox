"""Backtest report generation — summary.md, report_data.json, and report.html."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "research_report.html"


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np

            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _f2(v: float) -> str:
    return f"{v:.2f}"


def generate_summary_md(
    run_id: str,
    asof: str,
    metrics: dict[str, Any],
    strategy_names: list[str],
    period_start: str,
    period_end: str,
) -> str:
    m = metrics
    strategies_str = ", ".join(strategy_names) if strategy_names else "—"
    rows = [
        ("Total Return", _pct(m.get("total_return", 0))),
        ("CAGR", _pct(m.get("cagr", 0))),
        ("Sharpe Ratio", _f2(m.get("sharpe", 0))),
        ("Sortino Ratio", _f2(m.get("sortino", 0))),
        ("Max Drawdown", _pct(m.get("max_drawdown", 0))),
        ("Max DD Duration", f"{m.get('max_drawdown_duration_days', 0):.0f} days"),
        ("Calmar Ratio", _f2(m.get("calmar", 0))),
        ("Annual Volatility", _pct(m.get("annual_volatility", 0))),
        ("Win Rate", _pct(m.get("win_rate", 0))),
        ("Profit Factor", _f2(m.get("profit_factor", 0))),
        ("VaR 95%", _pct(m.get("var_95", 0))),
        ("CVaR 95%", _pct(m.get("cvar_95", 0))),
    ]
    table = "\n".join(f"| {name} | {val} |" for name, val in rows)
    n_assets = int(m.get("n_assets", 0))
    n_dates = int(m.get("n_dates", 0))
    return f"""# Backtest Summary

**Run:** `{run_id}`
**As of:** {asof}
**Period:** {period_start} → {period_end}
**Strategies:** {strategies_str}

## Performance

| Metric | Value |
|---|---|
{table}

## Universe

| | |
|---|---|
| Assets | {n_assets} |
| Bars | {n_dates} |
"""


def _fig_to_dict(fig) -> dict:
    """Convert a plotly Figure/FigureWidget to a clean JSON-serializable dict."""
    return json.loads(fig.to_json())


def _build_portfolio_chart_manual(
    portfolio_daily: pd.DataFrame,
    returns: pd.Series,
    bt_prices: pd.DataFrame,
) -> dict:
    """Fallback portfolio chart — 4-subplot manual figure."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pv_col = "portfolio_value" if "portfolio_value" in portfolio_daily.columns else portfolio_daily.columns[0]
    pv_idx = (
        portfolio_daily.index
        if isinstance(portfolio_daily.index, pd.DatetimeIndex)
        else pd.DatetimeIndex(portfolio_daily["date"])
    )
    pv = pd.Series(portfolio_daily[pv_col].values, index=pv_idx)
    pv = pv * (100.0 / pv.iloc[0])
    drawdown = (pv - pv.cummax()) / pv.cummax() * 100

    ret = returns.copy()
    if not isinstance(ret.index, pd.DatetimeIndex):
        ret.index = pd.DatetimeIndex(ret.index)
    if len(ret) > 1:
        step_secs = max((ret.index[1] - ret.index[0]).total_seconds(), 60)
        bars_per_day = int(86400 / step_secs)
    else:
        bars_per_day = 1
    roll_window = max(10, min(30 * bars_per_day, len(ret) // 4))
    annualise = np.sqrt(365 * bars_per_day)
    roll_sharpe = ret.rolling(roll_window).mean() / ret.rolling(roll_window).std() * annualise

    btc_col = next((c for c in bt_prices.columns if c in ("BTC", "BTCUSDT")), None)
    btc_norm = None
    if btc_col is not None:
        btc_raw = bt_prices[btc_col].dropna()
        btc_norm = btc_raw * (100.0 / btc_raw.iloc[0])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.40, 0.25, 0.35],
        subplot_titles=["Portfolio Value (base 100)", "Drawdown %", f"Rolling Sharpe ({roll_window} bars)"],
        vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Scatter(x=pv.index, y=pv.round(2).values, name="Strategy", line=dict(color="#1f77b4", width=2)),
        row=1,
        col=1,
    )
    if btc_norm is not None:
        fig.add_trace(
            go.Scatter(
                x=btc_norm.index,
                y=btc_norm.round(2).values,
                name="BTC B&H",
                line=dict(color="#ff7f0e", width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.round(2).values,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#d62728", width=1),
            fillcolor="rgba(214,39,40,0.15)",
        ),
        row=2,
        col=1,
    )
    roll_clean = roll_sharpe.dropna()
    fig.add_trace(
        go.Scatter(
            x=roll_clean.index,
            y=roll_clean.round(3).values,
            name="Rolling Sharpe",
            line=dict(color="#2ca02c", width=1.5),
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.25)", row=3, col=1)
    fig.update_layout(
        height=750,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(t=80, b=40),
    )
    return _fig_to_dict(fig)


def _build_monthly_chart(returns: pd.Series) -> dict:
    import plotly.graph_objects as go

    ret = returns.copy()
    if not isinstance(ret.index, pd.DatetimeIndex):
        ret.index = pd.DatetimeIndex(ret.index)
    monthly_pct = ((1 + ret).resample("ME").prod() - 1) * 100
    monthly_df = pd.DataFrame({"ret": monthly_pct, "year": monthly_pct.index.year, "month": monthly_pct.index.month})
    pivot = monthly_df.pivot(index="year", columns="month", values="ret")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[c - 1] for c in pivot.columns]

    z = [[float(v) if pd.notna(v) else None for v in row] for row in pivot.values]
    text = [[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in pivot.values]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index.tolist()],
            colorscale="RdYlGn",
            zmid=0,
            text=text,
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Year",
        height=max(220, 32 * len(pivot) + 120),
        margin=dict(t=20, b=40),
        template="plotly_white",
    )
    return _fig_to_dict(fig)


def _build_variant_framework_charts(
    variant: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the per-variant framework charts (portfolio, monthly, contrib,
    weights, position_stack) from a single variant_results entry. Returns a
    dict keyed by chart name (no variant prefix — caller adds that).

    Each builder is wrapped in try/except so one broken chart doesn't take
    down the whole section.
    """
    out: dict[str, dict[str, Any]] = {}
    portfolio_daily = variant.get("portfolio_daily")
    returns = variant.get("returns")
    weights_history = variant.get("weights_history")
    bt_prices = variant.get("bt_prices")
    vbt_pf = variant.get("vbt_portfolio")

    if vbt_pf is not None:
        try:
            out["portfolio"] = _fig_to_dict(vbt_pf.plot())
        except Exception:
            if portfolio_daily is not None and returns is not None and bt_prices is not None:
                out["portfolio"] = _build_portfolio_chart_manual(portfolio_daily, returns, bt_prices)
    elif portfolio_daily is not None and returns is not None and bt_prices is not None:
        out["portfolio"] = _build_portfolio_chart_manual(portfolio_daily, returns, bt_prices)

    if returns is not None:
        with contextlib.suppress(Exception):
            out["monthly"] = _build_monthly_chart(returns)

    if isinstance(weights_history, pd.DataFrame) and isinstance(bt_prices, pd.DataFrame):
        with contextlib.suppress(Exception):
            c = _build_contrib_chart(weights_history, bt_prices)
            if c is not None:
                out["contrib"] = c
        with contextlib.suppress(Exception):
            w = _build_weights_chart(weights_history)
            if w is not None:
                out["weights"] = w
        with contextlib.suppress(Exception):
            ps = _build_position_stack_chart(weights_history)
            if ps is not None:
                out["position_stack"] = ps
    return out


def _build_contrib_chart(weights_history: pd.DataFrame, bt_prices: pd.DataFrame) -> dict | None:
    import plotly.graph_objects as go

    wh_num = (
        weights_history.select_dtypes(include="number")
        if isinstance(weights_history.index, pd.DatetimeIndex)
        else pd.DataFrame()
    )
    if wh_num.empty or bt_prices.empty:
        return None

    price_ret = bt_prices.pct_change()
    wh_aligned = wh_num.reindex(price_ret.index).reindex(columns=price_ret.columns)
    attribution = (wh_aligned.shift(1) * price_ret).sum()
    attribution = attribution[attribution != 0].sort_values()
    # Single-asset strategies (e.g. BTC_BAH) collapse to one bar — not useful.
    if len(attribution) < 2:
        return None
    n_show = min(20, len(attribution))
    contrib_tickers = pd.concat([attribution.head(n_show // 2), attribution.tail(n_show - n_show // 2)])
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in contrib_tickers.values]

    fig = go.Figure(
        data=go.Bar(
            x=(contrib_tickers * 100).round(2).values,
            y=contrib_tickers.index.tolist(),
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        xaxis_title="Contribution (%)",
        height=max(300, 25 * len(contrib_tickers) + 100),
        margin=dict(t=20, b=40, l=80),
        template="plotly_white",
    )
    return _fig_to_dict(fig)


def _build_weights_chart(weights_history: pd.DataFrame) -> dict | None:
    import plotly.graph_objects as go

    wh_num = (
        weights_history.select_dtypes(include="number")
        if isinstance(weights_history.index, pd.DatetimeIndex)
        else pd.DataFrame()
    )
    if wh_num.empty:
        return None

    top_tickers = wh_num.sum().nlargest(20).index.tolist()
    wh_top = wh_num[top_tickers]
    wh_weekly = wh_top.resample("W").mean()

    fig = go.Figure(
        data=go.Heatmap(
            z=wh_weekly.T.values.tolist(),
            x=[str(d.date()) for d in wh_weekly.index],
            y=wh_weekly.columns.tolist(),
            colorscale="Blues",
            zmin=0,
            showscale=True,
            colorbar=dict(title="Weight"),
        )
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Ticker",
        height=500,
        margin=dict(t=20, b=60),
        template="plotly_white",
    )
    return _fig_to_dict(fig)


def _build_universe_size_chart(bt_prices: pd.DataFrame) -> dict | None:
    """Chart 1 — count of available tickers per day in the backtest universe."""
    import plotly.graph_objects as go

    if bt_prices.empty:
        return None
    coverage = bt_prices.notna().sum(axis=1)
    fig = go.Figure(
        data=go.Scatter(
            x=coverage.index,
            y=coverage.values,
            mode="lines",
            line=dict(color="#1f77b4", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.15)",
        )
    )
    fig.update_layout(
        yaxis_title="Tickers with data",
        height=260,
        margin=dict(t=20, b=40),
        template="plotly_white",
    )
    return _fig_to_dict(fig)


def _build_regime_overlay_chart(
    payload: dict[str, Any],
    *,
    bt_prices: pd.DataFrame,
    **_ctx,
) -> dict | None:
    """Strategy diagnostic — reference price with two MAs overlaid (e.g. trend + regime).

    Payload: {ref_ticker: str, fast_window: int, slow_window: int, label?: str, log_y?: bool}
    """
    import plotly.graph_objects as go

    ref_ticker = str(payload.get("ref_ticker", ""))
    fast_window = int(payload.get("fast_window", 0))
    slow_window = int(payload.get("slow_window", 0))
    if not ref_ticker or fast_window <= 0 or slow_window <= 0:
        return None
    col = next(
        (c for c in bt_prices.columns if c in (ref_ticker, ref_ticker + "USDT")),
        None,
    )
    if col is None:
        return None
    s = bt_prices[col].dropna()
    if s.empty:
        return None
    fast = s.rolling(fast_window).mean()
    slow = s.rolling(slow_window).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name=col, line=dict(color="#1a1a2e", width=1.2)))
    fig.add_trace(
        go.Scatter(
            x=fast.index,
            y=fast.values,
            name=f"MA{fast_window}",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=slow.index,
            y=slow.values,
            name=f"MA{slow_window}",
            line=dict(color="#d62728", width=1.5, dash="dash"),
        )
    )
    fig.update_layout(
        yaxis_title="Price",
        yaxis_type="log" if payload.get("log_y", True) else "linear",
        height=340,
        margin=dict(t=20, b=40),
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    return _fig_to_dict(fig)


def _build_signal_count_chart(
    payload: dict[str, Any],
    **_ctx,
) -> dict | None:
    """Strategy diagnostic — daily count of active signals, with optional cap line.

    Payload: {series: pd.Series, cap?: int|float, label?: str}
    """
    import plotly.graph_objects as go

    series = payload.get("series")
    if not isinstance(series, pd.Series):
        return None
    s = series.dropna()
    if s.empty:
        return None
    label = str(payload.get("label", "Signals"))
    cap = payload.get("cap")
    fig = go.Figure(
        data=go.Bar(
            x=s.index,
            y=s.values.astype(int),
            marker_color="#1f77b4",
            name=label,
        )
    )
    if cap is not None:
        fig.add_hline(
            y=float(cap),
            line_dash="dash",
            line_color="#d62728",
            annotation_text=f"cap = {cap}",
            annotation_position="top right",
        )
    fig.update_layout(
        yaxis_title="Signals (count)",
        height=260,
        margin=dict(t=20, b=40),
        template="plotly_white",
    )
    return _fig_to_dict(fig)


def _build_position_stack_chart(
    weights_history: pd.DataFrame,
    top_n: int = 25,
) -> dict | None:
    """Chart 5 — stacked area of position memberships over time (long only)."""
    import plotly.graph_objects as go

    wh = (
        weights_history.select_dtypes(include="number")
        if isinstance(weights_history.index, pd.DatetimeIndex)
        else pd.DataFrame()
    )
    if wh.empty:
        return None
    held = (wh > 0).astype(int)
    keep = held.sum().nlargest(top_n).index.tolist()
    held_top = held[keep]
    held_top = held_top.loc[:, held_top.sum() > 0]
    if held_top.empty or held_top.shape[1] < 2:
        # Single-asset strategy: stacked-membership view is just a flat line.
        return None

    fig = go.Figure()
    for col in held_top.columns:
        fig.add_trace(
            go.Scatter(
                x=held_top.index,
                y=held_top[col].values,
                name=str(col),
                stackgroup="positions",
                mode="lines",
                line=dict(width=0.3),
                hoverinfo="x+y+name",
            )
        )
    fig.update_layout(
        yaxis_title="Active long positions",
        height=420,
        margin=dict(t=20, b=40),
        template="plotly_white",
        legend=dict(orientation="v", y=1, x=1.02, font=dict(size=10)),
    )
    return _fig_to_dict(fig)


def _build_equity_overlay_chart(variant_results: dict[str, dict[str, Any]]) -> dict | None:
    """Multi-variant equity-curve overlay (notebook charts 9–13).

    Each variant's portfolio value normalised to base 100. Variants share an
    x-axis. Best-fit chart type for "BTC B&H vs strategy variants".
    """
    import plotly.graph_objects as go

    if not variant_results:
        return None

    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for i, (name, r) in enumerate(variant_results.items()):
        pd_df = r.get("portfolio_daily")
        if pd_df is None or pd_df.empty:
            continue
        col = "portfolio_value" if "portfolio_value" in pd_df.columns else pd_df.columns[0]
        idx = pd_df.index if isinstance(pd_df.index, pd.DatetimeIndex) else pd.DatetimeIndex(pd_df["date"])
        pv = pd.Series(pd_df[col].values, index=idx)
        if pv.iloc[0] == 0 or pd.isna(pv.iloc[0]):
            continue
        norm = pv * (100.0 / pv.iloc[0])
        fig.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm.round(2).values,
                name=name,
                line=dict(color=palette[i % len(palette)], width=1.6),
            )
        )

    fig.update_layout(
        yaxis_title="Equity (base 100)",
        yaxis_type="log",
        height=440,
        margin=dict(t=20, b=40),
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    return _fig_to_dict(fig)


def _build_per_variant_table_chart(variant_results: dict[str, dict[str, Any]]) -> dict | None:
    """Per-variant metrics table (chart 13 / summary table)."""
    import plotly.graph_objects as go

    if not variant_results:
        return None

    cols = [
        ("Variant", lambda n, r: n),
        ("Strategy", lambda n, r: r.get("strategy_name", "")),
        ("Total Return", lambda n, r: f"{r['metrics'].get('total_return', 0) * 100:.1f}%"),
        ("CAGR", lambda n, r: f"{r['metrics'].get('cagr', 0) * 100:.1f}%"),
        ("Sharpe", lambda n, r: f"{r['metrics'].get('sharpe', 0):.2f}"),
        ("Sortino", lambda n, r: f"{r['metrics'].get('sortino', 0):.2f}"),
        ("Max DD", lambda n, r: f"{r['metrics'].get('max_drawdown', 0) * 100:.1f}%"),
        ("Ann Vol", lambda n, r: f"{r['metrics'].get('annual_volatility', 0) * 100:.1f}%"),
        ("Calmar", lambda n, r: f"{r['metrics'].get('calmar', 0):.2f}"),
        ("Win Rate", lambda n, r: f"{r['metrics'].get('win_rate', 0) * 100:.1f}%"),
    ]
    headers = [c[0] for c in cols]
    rows = [[fn(n, r) for _, fn in cols] for n, r in variant_results.items()]
    columns = list(zip(*rows, strict=False)) if rows else [[] for _ in cols]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="#1a1a2e",
                    font=dict(color="white", size=12),
                    align="left",
                    height=30,
                ),
                cells=dict(
                    values=columns,
                    fill_color="#fafbfc",
                    align="left",
                    height=28,
                    font=dict(size=12),
                ),
            )
        ]
    )
    fig.update_layout(height=max(180, 28 * len(rows) + 70), margin=dict(t=10, b=10, l=0, r=0))
    return _fig_to_dict(fig)


def _build_gross_exposure_overlay_chart(
    variant_results: dict[str, dict[str, Any]],
) -> dict | None:
    """Per-variant gross exposure totals (chart 7: EW vs RW gross exposure)."""
    import plotly.graph_objects as go

    if not variant_results:
        return None
    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    plotted = 0
    for i, (name, r) in enumerate(variant_results.items()):
        wh = r.get("weights_history")
        if wh is None or wh.empty:
            continue
        gross = wh.abs().sum(axis=1)
        fig.add_trace(
            go.Scatter(
                x=gross.index,
                y=gross.round(4).values,
                name=name,
                line=dict(color=palette[i % len(palette)], width=1.4),
            )
        )
        plotted += 1
    if plotted == 0:
        return None
    fig.update_layout(
        yaxis_title="Gross exposure",
        height=300,
        margin=dict(t=20, b=40),
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    return _fig_to_dict(fig)


def _build_per_variant_symbol_weight_chart(
    variant_results: dict[str, dict[str, Any]],
    symbol: str = "BTC",
) -> dict | None:
    """Single-symbol weight time series across variants (chart 8: BTC EW vs RW)."""
    import plotly.graph_objects as go

    if not variant_results:
        return None
    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    plotted = 0
    for i, (name, r) in enumerate(variant_results.items()):
        wh = r.get("weights_history")
        if wh is None or wh.empty:
            continue
        col = next((c for c in wh.columns if c in (symbol, symbol + "USDT")), None)
        if col is None:
            continue
        s = wh[col]
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.round(4).values,
                name=name,
                line=dict(color=palette[i % len(palette)], width=1.4),
            )
        )
        plotted += 1
    if plotted == 0:
        return None
    fig.update_layout(
        yaxis_title=f"{symbol} weight",
        height=300,
        margin=dict(t=20, b=40),
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    return _fig_to_dict(fig)


# Registry of diagnostic-chart builders. Strategies opt in by emitting
# details["diagnostics"][type] = payload from their run() method. Add new types
# here and any strategy can use them by emitting the matching payload shape.
_DIAGNOSTIC_BUILDERS = {
    "regime_overlay": _build_regime_overlay_chart,
    "signal_count": _build_signal_count_chart,
}


def resolve_narrative(narrative_cfg: dict[str, Any] | None) -> dict[str, str]:
    """Resolve a narrative config block into a dict of {title, methodology, findings}.

    Each section can be supplied inline (key: text) or via a file
    (key_file: path). Files are resolved relative to cwd. Returns Markdown
    strings; the template handles markdown→HTML rendering.
    """
    out: dict[str, str] = {"title": "", "methodology": "", "findings": ""}
    if not narrative_cfg:
        return out
    for key in ("title", "methodology", "findings"):
        inline = narrative_cfg.get(key)
        file_key = f"{key}_file"
        path = narrative_cfg.get(file_key)
        if inline is not None:
            out[key] = str(inline)
            continue
        if path:
            try:
                out[key] = Path(path).read_text(encoding="utf-8")
            except OSError:
                out[key] = ""
    return out


def build_reproducibility(
    *,
    run_id: str,
    asof: str,
    pipeline_name: str,
    pipeline_version: str,
    params: dict[str, Any],
    period_start: str,
    period_end: str,
    variant_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the reproducibility appendix payload from in-process run state.

    Captures whatever's deterministically available at report time: engine
    config, dataset identity (best-effort), variant configs, git commit.
    """
    import subprocess

    git_commit = ""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            git_commit = out.stdout.strip()
    except Exception:
        pass

    repro: dict[str, Any] = {
        "run_id": run_id,
        "asof": asof,
        "period_start": period_start,
        "period_end": period_end,
        "pipeline": f"{pipeline_name} v{pipeline_version}",
        "engine_commit": git_commit,
        "engine_config": {
            "engine": params.get("engine"),
            "fees": params.get("fees"),
            "fixed_fees": params.get("fixed_fees"),
            "slippage": params.get("slippage"),
            "rebalancing_freq": params.get("rebalancing_freq"),
            "threshold": params.get("threshold"),
            "trading_days": params.get("trading_days"),
            "risk": params.get("risk"),
        },
        "data": params.get("data") or {},
        "universe": params.get("universe") or {},
        "prices": params.get("prices") or {},
    }
    if variant_results:
        repro["variants"] = {name: r.get("config", {}) for name, r in variant_results.items()}
    return repro


def generate_report_data(
    run_id: str,
    asof: str,
    metrics: dict[str, Any],
    portfolio_daily: pd.DataFrame,
    returns: pd.Series,
    weights_history: pd.DataFrame,
    bt_prices: pd.DataFrame,
    strategy_names: list[str],
    period_start: str = "",
    period_end: str = "",
    vbt_portfolio=None,
    strategy_details: dict[str, dict[str, Any]] | None = None,
    variant_results: dict[str, dict[str, Any]] | None = None,
    narrative: dict[str, str] | None = None,
    reproducibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    charts: dict[str, Any] = {}
    vr = variant_results or {}
    is_multi = len(vr) >= 2

    if is_multi:
        # Per-variant framework charts. Each variant gets its own portfolio /
        # monthly / contrib / weights / position_stack under namespaced keys.
        # The template renders these inside a per-variant section.
        for vname, v in vr.items():
            v_charts = _build_variant_framework_charts(v)
            for ck, cf in v_charts.items():
                charts[f"{vname}__{ck}"] = cf
    else:
        # Single-strategy flow: framework charts at the top level (legacy).
        if vbt_portfolio is not None:
            try:
                charts["portfolio"] = _fig_to_dict(vbt_portfolio.plot())
            except Exception:
                charts["portfolio"] = _build_portfolio_chart_manual(portfolio_daily, returns, bt_prices)
        else:
            charts["portfolio"] = _build_portfolio_chart_manual(portfolio_daily, returns, bt_prices)

        with contextlib.suppress(Exception):
            charts["monthly"] = _build_monthly_chart(returns)

        with contextlib.suppress(Exception):
            contrib = _build_contrib_chart(weights_history, bt_prices)
            if contrib is not None:
                charts["contrib"] = contrib

        with contextlib.suppress(Exception):
            wt = _build_weights_chart(weights_history)
            if wt is not None:
                charts["weights"] = wt

        if vbt_portfolio is not None:
            with contextlib.suppress(Exception):
                trades_fig = vbt_portfolio.trades.plot()
                charts["trades"] = _fig_to_dict(trades_fig)

        with contextlib.suppress(Exception):
            ps = _build_position_stack_chart(weights_history)
            if ps is not None:
                charts["position_stack"] = ps

    # Framework-level: universe coverage (cross-variant, always rendered)
    with contextlib.suppress(Exception):
        u = _build_universe_size_chart(bt_prices)
        if u is not None:
            charts["universe_size"] = u

    # Strategy-emitted diagnostics — dispatched via _DIAGNOSTIC_BUILDERS registry.
    # Contract: strategy.run() returns details["diagnostics"] = {type: payload, ...}.
    # Each type is rendered by the registered builder. When multiple strategies emit
    # the same type, the chart key is prefixed with the strategy name.
    sd = strategy_details or {}
    type_counts: dict[str, int] = {}
    for _sname, sdetails in sd.items():
        diagnostics = (sdetails or {}).get("diagnostics") or {}
        for dtype in diagnostics:
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
    ctx = {"bt_prices": bt_prices, "weights_history": weights_history}
    for sname, sdetails in sd.items():
        diagnostics = (sdetails or {}).get("diagnostics") or {}
        for dtype, payload in diagnostics.items():
            builder = _DIAGNOSTIC_BUILDERS.get(dtype)
            if builder is None or not isinstance(payload, dict):
                continue
            try:
                fig_dict = builder(payload, **ctx)
            except Exception:
                continue
            if fig_dict is None:
                continue
            key = f"{sname}__{dtype}" if type_counts.get(dtype, 0) > 1 else dtype
            charts[key] = fig_dict

    # Multi-variant charts (only when N variants are present)
    vr = variant_results or {}
    if len(vr) >= 2:
        for fn, key in (
            (_build_equity_overlay_chart, "equity_overlay"),
            (_build_per_variant_table_chart, "per_variant_table"),
            (_build_gross_exposure_overlay_chart, "gross_exposure_overlay"),
        ):
            try:
                fig_dict = fn(vr)
                if fig_dict is not None:
                    charts[key] = fig_dict
            except Exception:
                pass
        try:
            sym_chart = _build_per_variant_symbol_weight_chart(vr, symbol="BTC")
            if sym_chart is not None:
                charts["btc_weight_overlay"] = sym_chart
        except Exception:
            pass

    # Per-variant metrics for the template to render variant sections.
    # Empty in single-strategy flow so the template skips that section.
    variant_metrics: dict[str, dict[str, Any]] = {}
    if is_multi:
        for vname, v in vr.items():
            m = dict(v.get("metrics") or {})
            cfg = v.get("config") or {}
            m["_strategy_name"] = v.get("strategy_name", "")
            m["_fees"] = cfg.get("fees")
            variant_metrics[vname] = m

    return {
        "run_id": run_id,
        "asof": asof,
        "period_start": period_start,
        "period_end": period_end,
        "strategies": strategy_names,
        "metrics": metrics,
        "variant_metrics": variant_metrics,
        "charts": charts,
        "narrative": narrative or {},
        "reproducibility": reproducibility or {},
    }


def report_data_to_json(report_data: dict[str, Any]) -> str:
    return json.dumps(report_data, cls=_NumpyEncoder)


def generate_html_report(report_data: dict[str, Any]) -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    json_str = report_data_to_json(report_data)
    return template.replace("{{REPORT_DATA}}", json_str)
