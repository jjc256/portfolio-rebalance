"""Reporting module.

Builds human-readable summaries and Plotly figures for the web dashboard and CLI.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from portfolio_rebalance.risk import (
    portfolio_volatility,
    portfolio_stats,
    compute_correlation,
    TRADING_DAYS,
)


# ---------------------------------------------------------------------------
# Text / table helpers
# ---------------------------------------------------------------------------


def weights_table(rebalanced: pd.DataFrame) -> pd.DataFrame:
    """Return a display-friendly weights comparison table."""
    cols = ["ticker", "weight"]
    proposed_col = "proposed_weight"
    if proposed_col not in rebalanced.columns:
        return rebalanced[cols].copy()

    df = rebalanced[["ticker", "weight", proposed_col]].copy()
    df["change"] = df[proposed_col] - df["weight"]
    df.columns = ["Ticker", "Current Weight", "Proposed Weight", "Δ Weight"]
    for c in ["Current Weight", "Proposed Weight", "Δ Weight"]:
        df[c] = df[c].map("{:.2%}".format)
    return df


def stats_summary(
    current_weights: np.ndarray,
    proposed_weights: np.ndarray,
    returns: pd.DataFrame,
    cov: pd.DataFrame,
) -> pd.DataFrame:
    """Return a two-row summary DataFrame comparing current vs proposed stats."""
    current_stats = portfolio_stats(current_weights, returns, cov)
    proposed_stats = portfolio_stats(proposed_weights, returns, cov)

    rows = []
    for label, stats in [("Current", current_stats), ("Proposed", proposed_stats)]:
        rows.append(
            {
                "Portfolio": label,
                "Annualised Volatility": f"{stats['volatility']:.2%}",
                "Annualised Return": f"{stats['mean_return']:.2%}",
                "Sharpe Ratio": f"{stats['sharpe']:.2f}",
                "Max Drawdown": f"{stats['max_drawdown']:.2%}",
            }
        )
    return pd.DataFrame(rows).set_index("Portfolio")


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------


def fig_weights_bar(rebalanced: pd.DataFrame) -> go.Figure:
    """Side-by-side bar chart of current vs proposed weights."""
    tickers = rebalanced["ticker"].tolist()
    fig = go.Figure(
        data=[
            go.Bar(
                name="Current",
                x=tickers,
                y=rebalanced["weight"].tolist(),
                marker_color="#4C72B0",
            ),
            go.Bar(
                name="Proposed",
                x=tickers,
                y=rebalanced["proposed_weight"].tolist(),
                marker_color="#DD8452",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Portfolio Weights: Current vs Proposed",
        xaxis_title="Ticker",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", y=1.1),
        template="plotly_white",
    )
    return fig


def fig_volatility_gauge(
    current_vol: float, proposed_vol: float
) -> go.Figure:
    """Bullet / indicator chart comparing portfolio volatility."""
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=proposed_vol * 100,
            number={"suffix": "%", "valueformat": ".2f"},
            delta={
                "reference": current_vol * 100,
                "valueformat": ".2f",
                "suffix": "%",
                "relative": False,
                "decreasing": {"color": "green"},
                "increasing": {"color": "red"},
            },
            title={"text": "Proposed Volatility<br><span style='font-size:0.8em'>vs Current</span>"},
        )
    )
    fig.update_layout(template="plotly_white", height=200)
    return fig


def fig_correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of asset returns."""
    corr = compute_correlation(returns)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Asset Return Correlation Matrix",
    )
    fig.update_layout(template="plotly_white")
    return fig


def fig_cumulative_returns(
    returns: pd.DataFrame,
    current_weights: np.ndarray,
    proposed_weights: np.ndarray,
) -> go.Figure:
    """Cumulative return chart for current and proposed portfolios."""
    port_current = (returns.values @ current_weights)
    port_proposed = (returns.values @ proposed_weights)

    cum_current = (1 + port_current).cumprod()
    cum_proposed = (1 + port_proposed).cumprod()

    index = returns.index
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=index, y=cum_current, name="Current", line=dict(color="#4C72B0"))
    )
    fig.add_trace(
        go.Scatter(x=index, y=cum_proposed, name="Proposed", line=dict(color="#DD8452"))
    )
    fig.update_layout(
        title="Cumulative Return: Current vs Proposed",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        legend=dict(orientation="h", y=1.1),
        template="plotly_white",
    )
    return fig
