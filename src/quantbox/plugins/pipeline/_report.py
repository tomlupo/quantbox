"""Backtest report generation — summary.md and report.html."""

from __future__ import annotations

from typing import Any

import pandas as pd


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


def generate_html_report(
    run_id: str,
    asof: str,
    metrics: dict[str, Any],
    portfolio_daily: pd.DataFrame,
    returns: pd.Series,
    weights_history: pd.DataFrame,
    bt_prices: pd.DataFrame,
    strategy_names: list[str],
) -> str:
    try:
        import numpy as np
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except ImportError:
        return "<html><body><p>plotly not available — install plotly to enable HTML reports</p></body></html>"

    strategy_str = " | ".join(strategy_names) if strategy_names else run_id

    # Portfolio value normalised to 100
    pv_col = "portfolio_value" if "portfolio_value" in portfolio_daily.columns else portfolio_daily.columns[0]
    pv_idx = (
        portfolio_daily.index
        if isinstance(portfolio_daily.index, pd.DatetimeIndex)
        else pd.DatetimeIndex(portfolio_daily["date"])
    )
    pv = pd.Series(portfolio_daily[pv_col].values, index=pv_idx)
    pv = pv * (100.0 / pv.iloc[0])

    # Drawdown
    drawdown = (pv - pv.cummax()) / pv.cummax() * 100

    # Monthly returns heatmap
    ret = returns.copy()
    if not isinstance(ret.index, pd.DatetimeIndex):
        ret.index = pd.DatetimeIndex(ret.index)
    monthly_pct = ((1 + ret).resample("ME").prod() - 1) * 100
    monthly_df = pd.DataFrame({"ret": monthly_pct, "year": monthly_pct.index.year, "month": monthly_pct.index.month})
    pivot = monthly_df.pivot(index="year", columns="month", values="ret")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[c - 1] for c in pivot.columns]

    # Rolling Sharpe — auto-detect bar spacing for annualisation
    if len(ret) > 1:
        step_secs = max((ret.index[1] - ret.index[0]).total_seconds(), 60)
        bars_per_day = int(86400 / step_secs)
    else:
        bars_per_day = 1
    roll_window = max(10, min(30 * bars_per_day, len(ret) // 4))
    annualise = np.sqrt(365 * bars_per_day)
    roll_sharpe = ret.rolling(roll_window).mean() / ret.rolling(roll_window).std() * annualise

    # Active positions count from weights_history
    wh = weights_history
    active_pos = (wh > 0).sum(axis=1) if isinstance(wh.index, pd.DatetimeIndex) else None

    # BTC benchmark
    btc_col = next((c for c in bt_prices.columns if c in ("BTC", "BTCUSDT")), None)
    if btc_col is not None:
        btc_raw = bt_prices[btc_col].dropna()
        btc_norm = btc_raw * (100.0 / btc_raw.iloc[0])
    else:
        btc_norm = None

    # Build main figure
    n_rows = 4 if active_pos is not None else 3
    row_heights = [0.35, 0.20, 0.20, 0.25] if n_rows == 4 else [0.40, 0.25, 0.35]
    titles = ["Portfolio Value (base 100)", "Drawdown %", f"Rolling Sharpe ({roll_window} bars)"]
    if n_rows == 4:
        titles.append("Active Positions")

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, row_heights=row_heights, subplot_titles=titles, vertical_spacing=0.06
    )

    fig.add_trace(
        go.Scatter(x=pv.index, y=pv.round(2).values, name="Strategy", line=dict(color="#1f77b4", width=2)), row=1, col=1
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

    if active_pos is not None:
        fig.add_trace(
            go.Scatter(
                x=active_pos.index,
                y=active_pos.values,
                name="Active Positions",
                fill="tozeroy",
                line=dict(color="#9467bd", width=1),
                fillcolor="rgba(148,103,189,0.15)",
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        title=dict(text=f"Backtest Report — {strategy_str}<br><sub>{run_id} | as of {asof}</sub>", font_size=15),
        height=900 if n_rows == 4 else 750,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(t=100, b=60),
    )

    # Metrics table
    m = metrics
    rows_data = [
        ("Total Return", f"{m.get('total_return', 0) * 100:.1f}%"),
        ("CAGR", f"{m.get('cagr', 0) * 100:.1f}%"),
        ("Sharpe", f"{m.get('sharpe', 0):.2f}"),
        ("Sortino", f"{m.get('sortino', 0):.2f}"),
        ("Max Drawdown", f"{m.get('max_drawdown', 0) * 100:.1f}%"),
        ("Max DD Duration", f"{m.get('max_drawdown_duration_days', 0):.0f} days"),
        ("Annual Vol", f"{m.get('annual_volatility', 0) * 100:.1f}%"),
        ("Calmar", f"{m.get('calmar', 0):.2f}"),
        ("Win Rate", f"{m.get('win_rate', 0) * 100:.1f}%"),
        ("Profit Factor", f"{m.get('profit_factor', 0):.2f}"),
        ("VaR 95%", f"{m.get('var_95', 0) * 100:.2f}%"),
        ("CVaR 95%", f"{m.get('cvar_95', 0) * 100:.2f}%"),
    ]
    names_col, vals_col = zip(*rows_data, strict=True)
    stripe = ["white" if i % 2 == 0 else "#f7f9ff" for i in range(len(rows_data))]
    metrics_fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metric", "Value"], fill_color="#1f77b4", font=dict(color="white", size=13), align="left"
                ),
                cells=dict(
                    values=[list(names_col), list(vals_col)], fill_color=[stripe, stripe], font_size=12, align="left"
                ),
            )
        ]
    )
    metrics_fig.update_layout(title="Performance Metrics", height=420, margin=dict(t=50, b=20), template="plotly_white")

    # Monthly heatmap
    z = [[float(v) if pd.notna(v) else None for v in row] for row in pivot.values]
    text = [[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in pivot.values]
    heat_fig = go.Figure(
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
    heat_fig.update_layout(
        title="Monthly Returns (%)",
        xaxis_title="Month",
        yaxis_title="Year",
        height=max(220, 80 * len(pivot) + 120),
        margin=dict(t=50, b=40),
        template="plotly_white",
    )

    # Weight heatmap — top 20 tickers by cumulative allocation
    wh_num = wh.select_dtypes(include="number") if isinstance(wh.index, pd.DatetimeIndex) else pd.DataFrame()
    weight_html = ""
    contrib_html = ""
    if not wh_num.empty and not bt_prices.empty:
        top_tickers = wh_num.sum().nlargest(20).index.tolist()
        wh_top = wh_num[top_tickers]
        # Resample to weekly mean for readability
        wh_weekly = wh_top.resample("W").mean()
        weight_fig = go.Figure(
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
        weight_fig.update_layout(
            title="Portfolio Weight Heatmap — Top 20 Tickers (weekly avg)",
            xaxis_title="Date",
            yaxis_title="Ticker",
            height=500,
            margin=dict(t=60, b=60),
            template="plotly_white",
        )
        weight_html = pio.to_html(weight_fig, full_html=False, include_plotlyjs=False)

        # Per-ticker return contribution
        price_ret = bt_prices.pct_change()
        wh_aligned = wh_num.reindex(price_ret.index).reindex(columns=price_ret.columns)
        attribution = (wh_aligned.shift(1) * price_ret).sum()
        attribution = attribution[attribution != 0].sort_values()
        n_show = min(20, len(attribution))
        contrib_tickers = pd.concat([attribution.head(n_show // 2), attribution.tail(n_show - n_show // 2)])
        colors = ["#d62728" if v < 0 else "#1f77b4" for v in contrib_tickers.values]
        contrib_fig = go.Figure(
            data=go.Bar(
                x=(contrib_tickers * 100).round(2).values,
                y=contrib_tickers.index.tolist(),
                orientation="h",
                marker_color=colors,
            )
        )
        contrib_fig.update_layout(
            title="Per-Ticker Return Contribution (%)",
            xaxis_title="Contribution (%)",
            height=max(300, 25 * len(contrib_tickers) + 100),
            margin=dict(t=60, b=40, l=80),
            template="plotly_white",
        )
        contrib_html = pio.to_html(contrib_fig, full_html=False, include_plotlyjs=False)

    main_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    metrics_html = pio.to_html(metrics_fig, full_html=False, include_plotlyjs=False)
    heat_html = pio.to_html(heat_fig, full_html=False, include_plotlyjs=False)

    weight_card = f'<div class="card">{weight_html}</div>' if weight_html else ""
    contrib_card = f'<div class="card">{contrib_html}</div>' if contrib_html else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Backtest Report — {strategy_str}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f6fa; margin: 0; padding: 24px; color: #222; }}
    h1 {{ margin: 0 0 4px; font-size: 1.6em; color: #1a1a2e; }}
    .meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
    .card {{ background: white; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.08); padding: 16px; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
    .grid .card {{ margin-bottom: 0; }}
  </style>
</head>
<body>
  <h1>Backtest Report</h1>
  <p class="meta">
    <strong>Run:</strong> {run_id} &nbsp;·&nbsp;
    <strong>As of:</strong> {asof} &nbsp;·&nbsp;
    <strong>Strategies:</strong> {strategy_str}
  </p>
  <div class="card">{main_html}</div>
  <div class="grid">
    <div class="card">{metrics_html}</div>
    <div class="card">{heat_html}</div>
  </div>
  {weight_card}
  {contrib_card}
</body>
</html>"""
