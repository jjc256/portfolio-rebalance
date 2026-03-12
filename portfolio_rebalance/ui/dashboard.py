"""Streamlit web dashboard for portfolio-rebalance.

Run with:
    streamlit run portfolio_rebalance/ui/dashboard.py
"""

from __future__ import annotations

import io
import inspect
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
    download_market_caps,
)
from portfolio_rebalance.risk import (
    compute_returns,
    compute_covariance,
    portfolio_volatility,
    split_returns,
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
    require_full_history = st.checkbox(
        "Preserve full lookback (drop late-start tickers)",
        value=False,
        help=(
            "When enabled, assets without complete price history over the selected "
            "period are removed so returns span the full window."
        ),
    )
    max_weight_pct = st.slider(
        "Max position size",
        min_value=5,
        max_value=100,
        value=100,
        step=5,
        format="%d%%",
    )
    max_weight = max_weight_pct / 100.0
    use_max_increase = st.checkbox(
        "Cap per-position increase",
        value=False,
        help=(
            "Limit how much any ticker can increase versus its current weight in "
            "one rebalance step."
        ),
    )
    max_increase: float | None = None
    if use_max_increase:
        max_increase_pct = st.slider(
            "Max increase per ticker",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            format="%d%%",
        )
        max_increase = max_increase_pct / 100.0

    use_turnover = st.checkbox("Turnover constraint", value=False)
    turnover_limit: float | None = None
    if use_turnover:
        turnover_pct = st.slider(
            "Max one-way turnover",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
            format="%d%%",
        )
        turnover_limit = turnover_pct / 100.0

    use_small_cap_cap = st.checkbox(
        "Market-cap risk cap",
        value=False,
        help=(
            "Apply a tighter max-weight to smaller-cap names as a simple risk "
            "proxy."
        ),
    )
    small_cap_threshold_b: float | None = None
    small_cap_max_weight: float | None = None
    if use_small_cap_cap:
        small_cap_threshold_b = st.slider(
            "Small-cap threshold (USD billions)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )
        small_cap_max_weight_pct = st.slider(
            "Max weight for small-cap names",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            format="%d%%",
        )
        small_cap_max_weight = small_cap_max_weight_pct / 100.0

    st.divider()
    st.header("Evaluation")
    use_oos = st.checkbox(
        "Out-of-sample evaluation",
        value=False,
        help=(
            "Split the historical data into an estimation window (used to fit the "
            "covariance and optimiser) and a held-out evaluation window (used only "
            "to report performance). Disabling this shows in-sample stats, which "
            "are biased because the proposed weights were optimised on the same data."
        ),
    )
    eval_frac: float | None = None
    if use_oos:
        eval_frac_pct = st.slider(
            "Evaluation window (% of data held out)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            format="%d%%",
        )
        eval_frac = eval_frac_pct / 100.0

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
            input_tickers = portfolio_df["ticker"].tolist()
            supports_full_history = (
                "require_full_history" in inspect.signature(load_data).parameters
            )
            if supports_full_history:
                portfolio_filtered, prices = load_data(
                    portfolio_df,
                    period=period,
                    require_full_history=require_full_history,
                )
            else:
                portfolio_filtered, prices = load_data(portfolio_df, period=period)
                if require_full_history:
                    st.warning(
                        "This runtime uses an older portfolio_rebalance build that "
                        "doesn't support full-history filtering yet. "
                        "Reinstall with 'pip install -e .' from the repo root to enable it."
                    )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Data download failed: {exc}")
            st.stop()

    if prices.empty or len(prices.columns) < 2:
        st.error("Not enough price data. Try a different period or check your tickers.")
        st.stop()

    dropped_tickers = sorted(set(input_tickers) - set(portfolio_filtered["ticker"].tolist()))
    if dropped_tickers:
        reason = (
            "without full-period history"
            if require_full_history
            else "with missing price data"
        )
        st.warning(
            f"Dropped {len(dropped_tickers)} ticker(s) {reason}: "
            f"{', '.join(dropped_tickers)}"
        )

    returns = compute_returns(prices)

    first_valid = prices.apply(lambda s: s.first_valid_index())
    latest_start = first_valid.max()
    first_price_date = prices.index.min()
    if latest_start > first_price_date:
        limiting = sorted(first_valid[first_valid == latest_start].index.tolist())
        st.info(
            "Common return history starts later because not all tickers have data "
            f"from the selected window start. Effective return range: "
            f"{returns.index.min().date()} to {returns.index.max().date()}. "
            f"Latest start date is {latest_start.date()} for: {', '.join(limiting)}."
        )

    # Split into estimation and (optional) evaluation windows
    eval_returns: pd.DataFrame | None = None
    if use_oos and eval_frac is not None:
        est_returns, eval_returns = split_returns(returns, eval_frac=eval_frac)
    else:
        est_returns = returns

    cov = compute_covariance(est_returns, annualise=True)

    # Align portfolio to available tickers
    portfolio_filtered = portfolio_filtered[
        portfolio_filtered["ticker"].isin(cov.columns)
    ].reset_index(drop=True)

    per_asset_max: pd.Series | None = None
    if use_small_cap_cap and small_cap_threshold_b is not None and small_cap_max_weight is not None:
        market_caps = download_market_caps(portfolio_filtered["ticker"].tolist())
        threshold_usd = float(small_cap_threshold_b) * 1e9
        per_asset_max = pd.Series(max_weight, index=portfolio_filtered["ticker"], dtype=float)

        small_mask = market_caps < threshold_usd
        small_tickers = sorted(market_caps[small_mask].index.tolist())
        if small_tickers:
            per_asset_max.loc[small_tickers] = min(max_weight, small_cap_max_weight)
            st.info(
                f"Applied market-cap cap to {len(small_tickers)} ticker(s) below "
                f"${small_cap_threshold_b:.0f}B."
            )

        missing_caps = sorted(market_caps[market_caps.isna()].index.tolist())
        if missing_caps:
            st.warning(
                "Market cap unavailable for "
                f"{len(missing_caps)} ticker(s); using global max position for those names."
            )

    try:
        rebalanced = rebalance(
            portfolio_filtered,
            cov,
            max_weight=max_weight,
            max_weights=per_asset_max,
            turnover_limit=turnover_limit,
            max_increase=max_increase,
        )
    except ValueError as exc:
        st.error(f"Constraint set is infeasible: {exc}")
        st.stop()

    current_w = rebalanced["weight"].values
    proposed_w = rebalanced["proposed_weight"].values
    current_vol = portfolio_volatility(current_w, cov)
    proposed_vol = portfolio_volatility(proposed_w, cov)

    # ---- OOS / in-sample notice ----
    if use_oos and eval_returns is not None:
        n_est = len(est_returns)
        n_oos = len(eval_returns)
        st.info(
            f"📊 **Out-of-sample mode**: covariance estimated on {n_est} days, "
            f"performance stats evaluated on the held-out {n_oos}-day OOS window."
        )
    else:
        st.caption(
            "⚠️ Performance stats below are **in-sample** — computed on the same "
            "historical data used to fit the covariance matrix and run the optimiser. "
            "Enable *Out-of-sample evaluation* in the sidebar for a bias-free comparison."
        )

    # ---- KPI row ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Volatility (in-sample)", f"{current_vol:.2%}")
    with col2:
        st.metric(
            "Proposed Volatility (in-sample)",
            f"{proposed_vol:.2%}",
            delta=f"{proposed_vol - current_vol:.2%}",
            delta_color="inverse",
        )
    with col3:
        vol_reduction = (current_vol - proposed_vol) / current_vol * 100
        st.metric("Volatility Reduction (in-sample)", f"{vol_reduction:.1f}%")
    with col4:
        turnover = float(np.sum(np.abs(proposed_w - current_w)))
        st.metric("One-way Turnover", f"{turnover:.2%}")

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
    st.dataframe(
        stats_summary(current_w, proposed_w, est_returns, cov, eval_returns=eval_returns),
        width='stretch',
    )

    st.divider()

    # ---- Cumulative returns ----
    st.subheader("Cumulative Returns")
    st.plotly_chart(
        fig_cumulative_returns(est_returns, current_w, proposed_w, eval_returns=eval_returns),
        width='stretch',
    )

    st.divider()

    # ---- Correlation heatmap ----
    st.subheader("Asset Correlation Heatmap")
    st.plotly_chart(fig_correlation_heatmap(est_returns), width='stretch')

elif run and portfolio_df is None:
    st.warning("Please provide a portfolio before running optimisation.")

else:
    if portfolio_df is None:
        st.info("👈 Configure your portfolio in the sidebar, then click **Run Optimisation**.")
    else:
        st.info("👈 Click **Run Optimisation** in the sidebar to see results.")
