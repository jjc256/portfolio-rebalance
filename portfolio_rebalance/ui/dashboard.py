"""Streamlit web dashboard for portfolio-rebalance.

Run with:
    streamlit run portfolio_rebalance/ui/dashboard.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make sure the package root is importable when running the file directly
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio_rebalance.data import (
    load_portfolio_csv,
    load_portfolio_dict,
    load_data,
)
from portfolio_rebalance.risk import (
    compute_returns,
    compute_covariance,
    portfolio_volatility,
)
from portfolio_rebalance.optimizer import rebalance
from portfolio_rebalance.reporting import (
    weights_table,
    stats_summary,
    fig_weights_bar,
    fig_volatility_gauge,
    fig_correlation_heatmap,
    fig_cumulative_returns,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Rebalancer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Portfolio Rebalancer")
st.caption("Minimum-variance optimisation · Long-only · Free data (yfinance)")

# ---------------------------------------------------------------------------
# Sidebar – inputs & settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Portfolio Input")

    input_mode = st.radio(
        "Input method",
        ["Upload CSV", "Sample portfolio", "Manual entry"],
        index=1,
    )

    portfolio_df: pd.DataFrame | None = None

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader(
            "CSV with 'ticker' and 'weight' columns",
            type=["csv"],
        )
        if uploaded:
            try:
                portfolio_df = load_portfolio_csv(io.StringIO(uploaded.read().decode()))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not parse CSV: {exc}")

    elif input_mode == "Sample portfolio":
        _sample_path = _ROOT / "sample_portfolio.csv"
        if _sample_path.exists():
            portfolio_df = load_portfolio_csv(_sample_path)
        else:
            portfolio_df = load_portfolio_dict(
                {"AAPL": 0.20, "MSFT": 0.20, "GOOGL": 0.15, "AMZN": 0.15,
                 "NVDA": 0.10, "JPM": 0.08, "JNJ": 0.07, "XOM": 0.05}
            )

    else:  # Manual entry
        st.caption("Enter tickers and weights (will be normalised).")
        default_text = "AAPL=0.25\nMSFT=0.25\nGOOGL=0.25\nAMZN=0.25"
        raw = st.text_area("TICKER=weight (one per line)", value=default_text, height=160)
        try:
            holdings: dict[str, float] = {}
            for line in raw.strip().splitlines():
                line = line.strip()
                if not line or "=" not in line:
                    continue
                t, v = line.split("=", 1)
                holdings[t.strip().upper()] = float(v)
            if holdings:
                portfolio_df = load_portfolio_dict(holdings)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Parse error: {exc}")

    st.divider()
    st.header("Settings")

    period = st.selectbox(
        "Lookback period",
        ["1y", "2y", "3y", "5y"],
        index=2,
    )
    max_weight = st.slider(
        "Max position size",
        min_value=0.05,
        max_value=1.0,
        value=1.0,
        step=0.05,
        format="%.0%%",
    )
    use_turnover = st.checkbox("Turnover constraint", value=False)
    turnover_limit: float | None = None
    if use_turnover:
        turnover_limit = st.slider(
            "Max one-way turnover",
            min_value=0.05,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.0%%",
        )

    run = st.button("🚀 Run Optimisation", type="primary", width='stretch')

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if portfolio_df is not None:
    st.subheader("Current Holdings")
    display_current = portfolio_df.copy()
    display_current["weight"] = display_current["weight"].map("{:.2%}".format)
    st.dataframe(display_current, width='stretch', hide_index=True)

if run and portfolio_df is not None:
    with st.spinner("Downloading price data and optimising…"):
        try:
            portfolio_filtered, prices = load_data(portfolio_df, period=period)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Data download failed: {exc}")
            st.stop()

    if prices.empty or len(prices.columns) < 2:
        st.error("Not enough price data. Try a different period or check your tickers.")
        st.stop()

    returns = compute_returns(prices)
    cov = compute_covariance(returns, annualise=True)

    # Align portfolio to available tickers
    portfolio_filtered = portfolio_filtered[
        portfolio_filtered["ticker"].isin(cov.columns)
    ].reset_index(drop=True)

    rebalanced = rebalance(
        portfolio_filtered,
        cov,
        max_weight=max_weight,
        turnover_limit=turnover_limit,
    )

    current_w = rebalanced["weight"].values
    proposed_w = rebalanced["proposed_weight"].values
    current_vol = portfolio_volatility(current_w, cov)
    proposed_vol = portfolio_volatility(proposed_w, cov)

    # ---- KPI row ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Volatility", f"{current_vol:.2%}")
    with col2:
        st.metric(
            "Proposed Volatility",
            f"{proposed_vol:.2%}",
            delta=f"{proposed_vol - current_vol:.2%}",
            delta_color="inverse",
        )
    with col3:
        vol_reduction = (current_vol - proposed_vol) / current_vol * 100
        st.metric("Volatility Reduction", f"{vol_reduction:.1f}%")
    with col4:
        turnover = float(np.sum(np.abs(proposed_w - current_w)))
        st.metric("One-way Turnover", f"{turnover:.2%}")

    st.divider()

    # ---- Weights comparison ----
    st.subheader("Weights: Current vs Proposed")
    col_tbl, col_bar = st.columns([1, 2])
    with col_tbl:
        st.dataframe(weights_table(rebalanced), width='stretch', hide_index=True)
    with col_bar:
        st.plotly_chart(fig_weights_bar(rebalanced), width='stretch')

    st.divider()

    # ---- Performance stats ----
    st.subheader("Performance Statistics")
    st.dataframe(stats_summary(current_w, proposed_w, returns, cov), width='stretch')

    st.divider()

    # ---- Cumulative returns ----
    st.subheader("Cumulative Returns")
    st.plotly_chart(
        fig_cumulative_returns(returns, current_w, proposed_w),
        width='stretch',
    )

    st.divider()

    # ---- Correlation heatmap ----
    st.subheader("Asset Correlation Heatmap")
    st.plotly_chart(fig_correlation_heatmap(returns), width='stretch')

elif run and portfolio_df is None:
    st.warning("Please provide a portfolio before running optimisation.")

else:
    if portfolio_df is None:
        st.info("👈 Configure your portfolio in the sidebar, then click **Run Optimisation**.")
    else:
        st.info("👈 Click **Run Optimisation** in the sidebar to see results.")
