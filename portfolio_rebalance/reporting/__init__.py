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
    portfolio_stats_realized,
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
    eval_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a two-row summary DataFrame comparing current vs proposed stats.

    Parameters
    ----------
    current_weights, proposed_weights:
        Weight vectors aligned with ``returns`` columns.
    returns:
        Daily returns used for the **estimation** period (same data the covariance
        matrix and optimiser were fitted on).  Stats derived here are in-sample.
    cov:
        Annualised covariance matrix (from the estimation period).
    eval_returns:
        Optional **out-of-sample** daily returns.  When provided, performance
        statistics (return, Sharpe, drawdown, *and* realised volatility) are
        computed from this held-out period only, preventing look-ahead bias.
        Rows are then labelled ``"… (OOS)"`` instead of ``"… (In-Sample)"``.

    Notes
    -----
    When no ``eval_returns`` is supplied the stats are computed on the same data
    used to fit the covariance matrix and run the optimiser.  They will be
    biased — the proposed portfolio is guaranteed to look at least as good as
    the current one in terms of volatility because that is what the optimiser
    minimised.  Treat in-sample figures as a *historical fit*, not a forecast.
    """
    if eval_returns is not None:
        current_stats = portfolio_stats_realized(current_weights, eval_returns)
        proposed_stats = portfolio_stats_realized(proposed_weights, eval_returns)
        suffix = " (OOS)"
    else:
        current_stats = portfolio_stats(current_weights, returns, cov)
        proposed_stats = portfolio_stats(proposed_weights, returns, cov)
        suffix = " (In-Sample)"

    rows = []
    for label, stats in [
        (f"Current{suffix}", current_stats),
        (f"Proposed{suffix}", proposed_stats),
    ]:
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
    eval_returns: pd.DataFrame | None = None,
) -> go.Figure:
    """Cumulative return chart for current and proposed portfolios.

    Parameters
    ----------
    returns:
        Estimation-period daily returns (used to fit the covariance / optimiser).
    current_weights, proposed_weights:
        Weight vectors.
    eval_returns:
        Optional out-of-sample daily returns.  When provided the chart shows
        the estimation period followed by the evaluation period, with a vertical
        dashed line marking the boundary.  The OOS segment is shown with a
        lighter, dashed line style to distinguish it visually.
    """
    def _cum(ret_arr: np.ndarray, start_val: float = 1.0) -> np.ndarray:
        return start_val * np.cumprod(1.0 + ret_arr)

    port_current_is = returns.values @ current_weights
    port_proposed_is = returns.values @ proposed_weights
    cum_current_is = _cum(port_current_is)
    cum_proposed_is = _cum(port_proposed_is)
    idx_is = returns.index

    title = "Cumulative Return: Current vs Proposed (In-Sample Backtest)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idx_is, y=cum_current_is, name="Current",
        line=dict(color="#4C72B0"),
    ))
    fig.add_trace(go.Scatter(
        x=idx_is, y=cum_proposed_is, name="Proposed",
        line=dict(color="#DD8452"),
    ))

    if eval_returns is not None and not eval_returns.empty:
        port_current_oos = eval_returns.values @ current_weights
        port_proposed_oos = eval_returns.values @ proposed_weights
        # Chain OOS returns so the cumulative curve is continuous
        cum_current_oos = _cum(port_current_oos, start_val=cum_current_is[-1])
        cum_proposed_oos = _cum(port_proposed_oos, start_val=cum_proposed_is[-1])
        idx_oos = eval_returns.index

        fig.add_trace(go.Scatter(
            x=idx_oos, y=cum_current_oos, name="Current (OOS)",
            line=dict(color="#4C72B0", dash="dash"),
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=idx_oos, y=cum_proposed_oos, name="Proposed (OOS)",
            line=dict(color="#DD8452", dash="dash"),
            showlegend=True,
        ))
        # Vertical boundary line — use add_shape to avoid Plotly/Pandas
        # datetime arithmetic incompatibility with add_vline
        boundary_str = str(pd.Timestamp(idx_oos[0]).date())
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=boundary_str,
            x1=boundary_str,
            y0=0,
            y1=1,
            line=dict(color="grey", width=1.5, dash="dot"),
        )
        fig.add_annotation(
            x=boundary_str,
            yref="paper",
            y=1.01,
            text="OOS →",
            showarrow=False,
            font=dict(color="grey", size=11),
            xanchor="left",
        )
        title = "Cumulative Return: Current vs Proposed (solid = in-sample · dashed = OOS)"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        legend=dict(orientation="h", y=1.1),
        template="plotly_white",
    )
    return fig
