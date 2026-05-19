"""Backtest report generation — summary.md, report_data.json, and report.html."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np
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
    # Log-y on the equity panel so both strategy and BTC B&H are visible
    # regardless of relative scale (BTC can 20x while a vol-targeted strategy
    # ~2x, making the strategy invisible on a linear scale).
    fig.update_yaxes(type="log", title_text="Equity (log)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    fig.update_yaxes(title_text="Rolling Sharpe", row=3, col=1)
    fig.update_layout(
        height=750,
        template="plotly_white",
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
        margin=dict(t=80, b=40),
        hovermode="x unified",
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

    # Add a yearly-total column (compound monthly returns within each year).
    yearly = ((1 + monthly_pct.fillna(0) / 100).groupby(monthly_pct.index.year).prod() - 1) * 100
    yearly = yearly.reindex(pivot.index)
    pivot_w_yr = pivot.copy()
    pivot_w_yr["YTD"] = yearly.values

    z = [[float(v) if pd.notna(v) else None for v in row] for row in pivot_w_yr.values]
    text = [[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in pivot_w_yr.values]
    # Coolwarm-like diverging palette around zero.
    cmap = [
        [0.00, "#3b4cc0"],
        [0.25, "#8fa9e8"],
        [0.50, "#dddddd"],
        [0.75, "#f3a08a"],
        [1.00, "#b40426"],
    ]
    # Symmetric z range so zero sits on the white midpoint
    zmax = float(_np_abs_max(pivot_w_yr.values))
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=pivot_w_yr.columns.tolist(),
            y=[str(y) for y in pivot_w_yr.index.tolist()],
            colorscale=cmap,
            zmid=0,
            zmin=-zmax,
            zmax=zmax,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#1a1a2e"),
            showscale=True,
            xgap=1,
            ygap=1,
            colorbar=dict(title="%", thickness=12, len=0.8),
            hovertemplate="<b>%{y} %{x}</b><br>return: %{text}<extra></extra>",
        )
    )
    # Visual separator for the YTD column.
    fig.add_vline(x=11.5, line=dict(color="white", width=2.5))
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Year",
        height=max(260, 36 * len(pivot_w_yr) + 120),
        margin=dict(t=40, b=40),
        template="plotly_white",
        title=dict(text="<b>Monthly returns (%)</b> + annual total (rightmost column)", font=dict(size=13)),
    )
    return _fig_to_dict(fig)


def _np_abs_max(values) -> float:
    """Robust |max| across a 2D nan-containing matrix used for symmetric colorbar."""
    import numpy as _np

    arr = _np.array(values, dtype=float)
    finite = arr[_np.isfinite(arr)]
    return float(_np.abs(finite).max()) if finite.size else 1.0


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

    # Prefer the manual 3-panel chart (equity + drawdown + rolling sharpe) —
    # vbt's default ``Portfolio.plot()`` doesn't normalise the benchmark, so
    # in multi-variant flows the BTC reference dominates and the strategy
    # curve collapses to a flat line. Fall back to vbt only if manual fails.
    if portfolio_daily is not None and returns is not None and bt_prices is not None:
        try:
            out["portfolio"] = _build_portfolio_chart_manual(portfolio_daily, returns, bt_prices)
        except Exception:
            if vbt_pf is not None:
                with contextlib.suppress(Exception):
                    out["portfolio"] = _fig_to_dict(vbt_pf.plot())

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

    # Generic blocks — auto-render from variant inputs without strategy opt-in.
    # See plugins/pipeline/blocks.py for the registry. Each block declares which
    # ctx fields it requires; we skip silently if the field is missing.
    ctx = {
        "returns": returns,
        "portfolio_daily": portfolio_daily,
        "weights_history": weights_history,
        "bt_prices": bt_prices,
    }
    for name, block in _GENERIC_BLOCKS.items():
        if any(ctx.get(k) is None or (hasattr(ctx.get(k), "empty") and ctx.get(k).empty) for k in block.requires):
            continue
        try:
            fig = block.builder(None, **ctx)
        except Exception:
            continue
        if fig is not None:
            out[name] = fig
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

    price_ret = bt_prices.ffill().pct_change(fill_method=None)
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
            line=dict(color="#1f77b4", width=2),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.18)",
            hovertemplate="%{x|%Y-%m-%d}: %{y} tickers<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="<b>Universe coverage</b> — number of tickers with non-NaN price per day", font=dict(size=13)),
        yaxis=dict(title="tickers with data", rangemode="tozero"),
        height=300,
        margin=dict(t=50, b=40),
        template="plotly_white",
        hovermode="x unified",
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
        data=go.Scatter(
            x=s.index,
            y=s.values.astype(int),
            mode="lines",
            line=dict(color="#1f77b4", width=0, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.35)",
            name=label,
            hovertemplate="%{x|%Y-%m-%d}: %{y} signals<extra></extra>",
        )
    )
    if cap is not None:
        fig.add_hline(
            y=float(cap),
            line_dash="dash",
            line_color="#d62728",
            line_width=1.5,
            annotation_text=f"cap = {cap}",
            annotation_position="top right",
            annotation_font=dict(color="#d62728"),
        )
    fig.update_layout(
        title=dict(text=f"<b>{label}</b>", font=dict(size=13)),
        yaxis=dict(title="signals (count)", rangemode="tozero"),
        height=300,
        margin=dict(t=50, b=40),
        template="plotly_white",
        hovermode="x unified",
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
        title=dict(text="<b>Equity curves by variant</b> (base = 100, log y)", font=dict(size=13)),
        yaxis=dict(title="Equity (base 100)", type="log", gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.08)"),
        height=480,
        margin=dict(t=60, b=40, l=60, r=20),
        template="plotly_white",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right"),
        hovermode="x unified",
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

    # Color the numeric cells using a pandas-styled-table-like coolwarm gradient.
    # Each numeric column is independently min/max-normalised and given a per-cell
    # background colour (notebook ``.style.background_gradient(cmap='coolwarm')``).
    import numpy as _np

    def _coolwarm(v: float) -> str:
        # Approximate matplotlib's coolwarm at value in [0, 1].
        # Blue (#3b4cc0) → white (#dddddd) → red (#b40426)
        v = max(0.0, min(1.0, v))
        if v <= 0.5:
            t = v * 2.0
            r = int(59 + (221 - 59) * t)
            g = int(76 + (221 - 76) * t)
            b = int(192 + (221 - 192) * t)
        else:
            t = (v - 0.5) * 2.0
            r = int(221 + (180 - 221) * t)
            g = int(221 + (4 - 221) * t)
            b = int(221 + (38 - 221) * t)
        return f"rgb({r},{g},{b})"

    numeric_col_meta = [
        ("Total Return", "higher"),
        ("CAGR", "higher"),
        ("Sharpe", "higher"),
        ("Sortino", "higher"),
        ("Max DD", "lower"),
        ("Ann Vol", "lower"),
        ("Calmar", "higher"),
        ("Win Rate", "higher"),
    ]
    metric_key_map = {
        "Total Return": "total_return",
        "CAGR": "cagr",
        "Sharpe": "sharpe",
        "Sortino": "sortino",
        "Max DD": "max_drawdown",
        "Ann Vol": "annual_volatility",
        "Calmar": "calmar",
        "Win Rate": "win_rate",
    }
    # Build fill_color per cell (one list per column).
    fill_colors_per_col: list[list[str]] = [["#fafbfc"] * len(rows) for _ in headers]
    fill_colors_per_col[0] = ["#f3f4f6"] * len(rows)  # variant — light grey
    fill_colors_per_col[1] = ["#fafbfc"] * len(rows)  # strategy — neutral
    for col_idx, header in enumerate(headers):
        if header not in metric_key_map:
            continue
        key = metric_key_map[header]
        direction = dict(numeric_col_meta)[header]
        vals = _np.array(
            [float((r.get("metrics") or {}).get(key, _np.nan)) for _, r in variant_results.items()],
            dtype=float,
        )
        finite = vals[_np.isfinite(vals)]
        if finite.size == 0:
            continue
        lo, hi = finite.min(), finite.max()
        if hi - lo < 1e-12:
            colors = ["#dddddd"] * len(vals)
        else:
            normed = (vals - lo) / (hi - lo)
            if direction == "lower":
                normed = 1.0 - normed
            colors = [_coolwarm(float(n)) if _np.isfinite(n) else "#dddddd" for n in normed]
        fill_colors_per_col[col_idx] = colors

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[1.6, 1.8] + [1] * (len(headers) - 2),
                header=dict(
                    values=[f"<b>{h}</b>" for h in headers],
                    fill_color="#1a1a2e",
                    font=dict(color="white", size=12, family="-apple-system, 'Segoe UI', sans-serif"),
                    align="left",
                    height=34,
                ),
                cells=dict(
                    values=columns,
                    fill_color=fill_colors_per_col,
                    align="left",
                    height=26,
                    font=dict(size=12, color="#1a1a2e", family="-apple-system, 'Segoe UI', sans-serif"),
                    line=dict(color="rgba(255,255,255,0.6)", width=1),
                ),
            )
        ]
    )
    fig.update_layout(height=max(180, 28 * len(rows) + 80), margin=dict(t=8, b=8, l=0, r=0))
    return _fig_to_dict(fig)


def _build_per_variant_stats_heatmap(
    variant_results: dict[str, dict[str, Any]],
) -> dict | None:
    """Cross-variant stats heatmap (notebook charts 14, 15, 16, 26, 29).

    Renders the canonical SSRN 5-metric matrix (sharpe, max_drawdown,
    annualized_volatility, calmar_ratio, annualized_return) over the
    configured variants. One row per metric, one column per variant.
    Colorscale is metric-aware (higher-is-better vs lower-is-better).
    """
    import plotly.graph_objects as go

    if not variant_results:
        return None

    metric_specs = [
        ("sharpe", "Sharpe", "higher"),
        ("calmar", "Calmar", "higher"),
        ("annual_volatility", "Ann. Vol", "lower"),
        ("max_drawdown", "Max DD", "lower"),
        ("cagr", "CAGR", "higher"),
    ]
    variants = list(variant_results.keys())
    z: list[list[float]] = []
    text: list[list[str]] = []
    rows: list[str] = []
    for key, label, _direction in metric_specs:
        row = []
        text_row = []
        any_value = False
        for v in variants:
            m = variant_results[v].get("metrics") or {}
            val = m.get(key)
            if val is None:
                row.append(np.nan)
                text_row.append("—")
            else:
                row.append(float(val))
                if key in ("annual_volatility", "max_drawdown", "cagr"):
                    text_row.append(f"{val * 100:.1f}%")
                else:
                    text_row.append(f"{val:.2f}")
                any_value = True
        if any_value:
            z.append(row)
            text.append(text_row)
            rows.append(label)
    if not z:
        return None

    # Row-normalise z so the colorscale is meaningful per-metric (different
    # metrics have very different scales — Sharpe ~0.5, Max DD ~-50%, CAGR ~20%).
    # For lower-is-better metrics (vol, max_drawdown), invert the normalised value
    # so green still means "good".
    import numpy as _np

    z_arr = _np.array(z, dtype=float)
    z_norm = _np.zeros_like(z_arr)
    for i, (_key, _, direction) in enumerate(metric_specs[: z_arr.shape[0]]):
        row_vals = z_arr[i]
        finite = row_vals[_np.isfinite(row_vals)]
        if finite.size == 0:
            continue
        lo, hi = finite.min(), finite.max()
        if hi - lo < 1e-12:
            z_norm[i] = 0.5
        else:
            norm = (row_vals - lo) / (hi - lo)
            if direction == "lower":
                norm = 1.0 - norm
            z_norm[i] = norm

    fig = go.Figure(
        data=go.Heatmap(
            z=z_norm.tolist(),
            x=variants,
            y=rows,
            text=text,
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=13, color="#1a1a2e"),
            # Matplotlib-coolwarm-equivalent. Blue → off-white → red, but here
            # green-direction (1.0 = "better") so we map blue→red as bad→good.
            colorscale=[
                [0.0, "#3b4cc0"],
                [0.25, "#809cf3"],
                [0.5, "#dddddd"],
                [0.75, "#f49a7b"],
                [1.0, "#b40426"],
            ],
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{y}</b> — variant <i>%{x}</i><br>value: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="<b>Per-variant metrics heatmap</b> (row-normalised; green = better)", font=dict(size=13)),
        height=110 + 56 * len(rows),
        margin=dict(t=60, b=50, l=110, r=30),
        template="plotly_white",
        xaxis=dict(side="bottom", tickangle=-25),
        yaxis=dict(autorange="reversed"),
    )
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


def _build_donchian_overlay_chart(
    payload: dict[str, Any],
    **_ctx,
) -> dict | None:
    """Strategy diagnostic — Donchian channel band + trailing stop + entry/exit + signal panel.

    Two-row figure (top: price + bands + trailing stop, bottom: 0/1 signal).
    Replicates notebook charts 3 + 4 + 6 + 22 in a single visual.
    """
    import plotly.graph_objects as go
    import plotly.subplots as sp

    ref_ticker = str(payload.get("ref_ticker", ""))
    price = payload.get("price")
    if not isinstance(price, pd.Series) or price.dropna().empty:
        return None
    high = payload.get("high")
    low = payload.get("low")
    mid = payload.get("mid")
    stop = payload.get("trailing_stop")
    signal = payload.get("signal")
    window = int(payload.get("window", 0))

    fig = sp.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.04,
        subplot_titles=(f"{ref_ticker} price + Donchian({window}) + trailing stop", "signal"),
    )

    # --- Top panel: bands as filled area, price, mid, trailing stop -------------
    if isinstance(high, pd.Series) and isinstance(low, pd.Series):
        # Filled DC band
        fig.add_trace(
            go.Scatter(
                x=high.index,
                y=high.values,
                name=f"DC high ({window})",
                line=dict(color="rgba(46,160,67,0.6)", width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=low.index,
                y=low.values,
                name=f"DC low ({window})",
                line=dict(color="rgba(218,54,51,0.6)", width=1),
                fill="tonexty",
                fillcolor="rgba(46,160,67,0.10)",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if isinstance(mid, pd.Series):
        fig.add_trace(
            go.Scatter(
                x=mid.index,
                y=mid.values,
                name="DC mid",
                line=dict(color="#9467bd", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price.values,
            name=f"{ref_ticker}",
            line=dict(color="#1a1a2e", width=1.6),
        ),
        row=1,
        col=1,
    )
    if isinstance(stop, pd.Series):
        fig.add_trace(
            go.Scatter(
                x=stop.index,
                y=stop.values,
                name="trailing stop",
                line=dict(color="#ff7f0e", width=1.8),
            ),
            row=1,
            col=1,
        )
    if isinstance(signal, pd.Series):
        sig = signal.fillna(0).astype(float)
        diff = sig.diff().fillna(0)
        entries = price[diff == 1.0]
        exits = price[diff == -1.0]
        if not entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries.index,
                    y=entries.values,
                    mode="markers",
                    marker=dict(color="#2ca02c", symbol="triangle-up", size=11, line=dict(color="white", width=1)),
                    name="entry",
                ),
                row=1,
                col=1,
            )
        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits.index,
                    y=exits.values,
                    mode="markers",
                    marker=dict(color="#d62728", symbol="triangle-down", size=11, line=dict(color="white", width=1)),
                    name="exit",
                ),
                row=1,
                col=1,
            )
        # --- Bottom panel: signal as filled step ---
        fig.add_trace(
            go.Scatter(
                x=sig.index,
                y=sig.values,
                mode="lines",
                line=dict(color="#1f77b4", width=0, shape="hv"),
                fill="tozeroy",
                fillcolor="rgba(31,119,180,0.35)",
                name="signal",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(type="log", title_text="Price (log)", row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], tickvals=[0, 1], title_text="signal", row=2, col=1)
    fig.update_layout(
        height=560,
        margin=dict(t=50, b=40, l=10, r=10),
        template="plotly_white",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right"),
        hovermode="x unified",
    )
    return _fig_to_dict(fig)


def _build_per_window_heatmap(
    payload: dict[str, Any],
    **_ctx,
) -> dict | None:
    """Per-window signal-activity heatmap for the regime ticker (notebook chart 9 stand-in).

    Payload: {ref_ticker, signals: {window: pd.Series of 0/1}}.

    Without re-running a backtest per window we can't compute the full
    sharpe/maxdd/calmar matrix from the notebook (cells 348–366). This builder
    surfaces the upstream signal *density* per window so the report still has
    a per-window view; the multi-variant orchestration (P3) layers the actual
    metric heatmaps via `_build_per_variant_table_chart`.
    """
    import plotly.graph_objects as go

    sigs = payload.get("signals")
    if not isinstance(sigs, dict) or not sigs:
        return None
    ref_ticker = str(payload.get("ref_ticker", ""))
    windows = sorted(sigs.keys())
    # Resample monthly mean for readability — bool→0..1
    rows = []
    months = None
    for w in windows:
        s = sigs[w]
        if not isinstance(s, pd.Series) or s.dropna().empty:
            continue
        m = s.fillna(0).astype(float).resample("MS").mean()
        if months is None:
            months = m.index
        rows.append(m.values)
    if not rows or months is None:
        return None
    z = np.array(rows)
    valid_windows = [w for w in windows if isinstance(sigs.get(w), pd.Series)]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[d.strftime("%Y-%m") for d in months],
            y=[f"w={w}" for w in valid_windows],
            colorscale=[
                [0.0, "#440154"],
                [0.25, "#3b528b"],
                [0.5, "#21918c"],
                [0.75, "#5ec962"],
                [1.0, "#fde725"],
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(title="signal-on", thickness=12, len=0.8),
            hovertemplate="month=%{x}<br>window=%{y}<br>signal-on rate=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"<b>Per-window Donchian signal density</b> — {ref_ticker}",
            font=dict(size=13),
        ),
        height=380,
        margin=dict(t=60, b=60, l=60, r=20),
        template="plotly_white",
        xaxis=dict(title="month", tickangle=-45, nticks=24),
        yaxis=dict(title="lookback window", autorange="reversed"),
    )
    return _fig_to_dict(fig)


def _build_vol_overview_chart(
    payload: dict[str, Any],
    **_ctx,
) -> dict | None:
    """Realized vol + scalers (notebook charts 12, 13, 18, 19).

    Payload: {realized_vol: DataFrame, scalers: {label: DataFrame}, vol_lookback: int}.
    """
    import plotly.graph_objects as go
    import plotly.subplots as sp

    rv = payload.get("realized_vol")
    scalers = payload.get("scalers") or {}
    if not isinstance(rv, pd.DataFrame) or rv.dropna(how="all").empty:
        return None
    look = int(payload.get("vol_lookback", 60))

    # Pick the 10 most-observed tickers (least-NaN) for per-line view.
    coverage = rv.notna().sum().sort_values(ascending=False)
    top_tickers = coverage.head(10).index.tolist()
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Robust y-clip: bad-data artifacts (delistings, stale prices) can blow
    # vol up to 1000%+ for a few days, dominating the y-scale and hiding the
    # actual ~25-150% range. Cap per-line view at the 99th percentile of
    # finite realized-vol values, bounded by [60%, 300%].
    import numpy as _np

    finite_vals = rv.values[_np.isfinite(rv.values)]
    if finite_vals.size > 0:
        clip_cap = float(_np.quantile(finite_vals, 0.99))
        clip_cap = max(0.6, min(3.0, clip_cap * 1.2))
    else:
        clip_cap = 3.0

    fig = sp.make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"Realized {look}d annualized vol — top-10 most-covered tickers + universe mean (clipped @ {clip_cap:.0%})",
            "Vol scalers (mean across universe)",
        ),
        vertical_spacing=0.14,
    )
    for i, t in enumerate(top_tickers):
        s = rv[t].dropna().clip(upper=clip_cap)
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                name=t,
                line=dict(color=palette[i % len(palette)], width=1),
                opacity=0.55,
                hovertemplate="%{x|%Y-%m-%d}<br>" + t + ": %{y:.1%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    mean_rv = rv.mean(axis=1).clip(upper=clip_cap)
    fig.add_trace(
        go.Scatter(
            x=mean_rv.index,
            y=mean_rv.values,
            name="UNIVERSE MEAN",
            line=dict(color="#000000", width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>mean: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for label, sc in scalers.items():
        if not isinstance(sc, pd.DataFrame):
            continue
        if (sc == 1.0).all().all():
            continue
        fig.add_trace(
            go.Scatter(x=sc.index, y=sc.mean(axis=1), name=f"scaler={label}", line=dict(width=1.6)),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="ann vol", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="scaler", row=2, col=1)
    fig.update_layout(
        height=560,
        template="plotly_white",
        margin=dict(t=50, b=40),
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        hovermode="x unified",
    )
    return _fig_to_dict(fig)


# Registry of diagnostic-chart builders. Strategies opt in by emitting
# details["diagnostics"][type] = payload from their run() method. Add new types
# here and any strategy can use them by emitting the matching payload shape.
# Legacy alias. The single source of truth is now ``blocks.BLOCKS`` (see
# blocks.py). Existing callers that import ``_DIAGNOSTIC_BUILDERS`` from this
# module keep working — the dict is rebuilt from the registry at import time.
# Strategy/lab code should register new diagnostic blocks via
# ``blocks.register_block`` and consume them via ``blocks.BLOCKS``.
from quantbox.plugins.pipeline.blocks import BLOCKS as _BLOCKS  # noqa: E402

_DIAGNOSTIC_BUILDERS = {name: b.builder for name, b in _BLOCKS.items() if b.section == "diagnostics"}

# Generic blocks (section="framework") auto-render from pipeline outputs even
# without strategy opt-in. The dispatch in generate_report_data fires them
# unconditionally when their required ctx fields are populated.
_GENERIC_BLOCKS = {name: b for name, b in _BLOCKS.items() if b.section == "framework"}


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

        # Generic blocks (registry-driven, auto-on with required inputs).
        # Same dispatch as the multi-variant path in _build_variant_framework_charts.
        single_ctx = {
            "returns": returns,
            "portfolio_daily": portfolio_daily,
            "weights_history": weights_history,
            "bt_prices": bt_prices,
        }
        for name, block in _GENERIC_BLOCKS.items():
            missing = any(
                single_ctx.get(k) is None or (hasattr(single_ctx.get(k), "empty") and single_ctx.get(k).empty)
                for k in block.requires
            )
            if missing:
                continue
            try:
                fig = block.builder(None, **single_ctx)
            except Exception:
                continue
            if fig is not None:
                charts[name] = fig

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
            (_build_per_variant_stats_heatmap, "per_variant_stats_heatmap"),
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
            # ``primary: true`` in the variant config marks it as the source
            # of the shared diagnostics section. When no variant is flagged,
            # the template falls back to picking the highest-Sharpe non-benchmark.
            if cfg.get("primary") is True:
                m["_primary"] = True
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
