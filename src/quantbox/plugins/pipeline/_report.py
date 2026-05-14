"""Backtest report generation — summary.md, report_data.json, and report.html."""

from __future__ import annotations

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
        height=max(220, 80 * len(pivot) + 120),
        margin=dict(t=20, b=40),
        template="plotly_white",
    )
    return _fig_to_dict(fig)


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
) -> dict[str, Any]:
    charts: dict[str, Any] = {}

    # Portfolio chart — prefer vbt native figure
    if vbt_portfolio is not None:
        try:
            charts["portfolio"] = _fig_to_dict(vbt_portfolio.plot())
        except Exception:
            charts["portfolio"] = _build_portfolio_chart_manual(portfolio_daily, returns, bt_prices)
    else:
        charts["portfolio"] = _build_portfolio_chart_manual(portfolio_daily, returns, bt_prices)

    # Monthly returns heatmap
    try:
        charts["monthly"] = _build_monthly_chart(returns)
    except Exception:
        pass

    # Per-ticker contribution
    try:
        contrib = _build_contrib_chart(weights_history, bt_prices)
        if contrib is not None:
            charts["contrib"] = contrib
    except Exception:
        pass

    # Weight heatmap
    try:
        wt = _build_weights_chart(weights_history)
        if wt is not None:
            charts["weights"] = wt
    except Exception:
        pass

    # Trade log — vbt trades plot
    if vbt_portfolio is not None:
        try:
            trades_fig = vbt_portfolio.trades.plot()
            charts["trades"] = _fig_to_dict(trades_fig)
        except Exception:
            pass

    return {
        "run_id": run_id,
        "asof": asof,
        "period_start": period_start,
        "period_end": period_end,
        "strategies": strategy_names,
        "metrics": metrics,
        "charts": charts,
    }


def report_data_to_json(report_data: dict[str, Any]) -> str:
    return json.dumps(report_data, cls=_NumpyEncoder)


def generate_html_report(report_data: dict[str, Any]) -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    json_str = report_data_to_json(report_data)
    return template.replace("{{REPORT_DATA}}", json_str)
