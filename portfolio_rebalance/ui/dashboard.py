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
    download_prices,
    download_treasury_risk_free_rate,
)
from portfolio_rebalance.risk import (
    compute_returns,
    compute_covariance,
    estimate_expected_returns,
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
from portfolio_rebalance.validation import run_random_sp500_sharpe_test

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Rebalancer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Portfolio Rebalancer")
st.caption("Sharpe-first optimisation · Long-only · Free data (yfinance)")

# ---------------------------------------------------------------------------
# Sidebar – inputs & settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Workflow")
    workflow_mode = st.radio(
        "Mode",
        ["Portfolio Optimization", "Random S&P 500 Robustness Test"],
        index=0,
        help="Choose one workflow at a time.",
    )
    run_random_test = workflow_mode == "Random S&P 500 Robustness Test"

    portfolio_df: pd.DataFrame | None = None

    if not run_random_test:
        st.header("Portfolio Input")

        input_mode = st.radio(
            "Input method",
            ["Upload CSV", "Sample portfolio", "Manual entry"],
            index=1,
        )

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
    else:
        st.info(
            "Robustness mode selected: portfolio upload/input is disabled. "
            "Only random S&P 500 portfolios will be tested."
        )

    st.divider()
    st.header("Settings")

    period = st.selectbox(
        "Lookback period",
        ["1y", "2y", "3y", "5y"],
        index=2,
    )
    objective_label = st.selectbox(
        "Optimization objective",
        ["Maximize Sharpe ratio", "Minimize volatility"],
        index=0,
        help=(
            "Sharpe balances return and risk. Minimize volatility ignores expected "
            "returns and optimizes risk only."
        ),
    )
    objective = (
        "max_sharpe" if objective_label == "Maximize Sharpe ratio" else "min_variance"
    )

    risk_free_rate = 0.0
    risk_free_source = "manual"
    risk_free_tenor = "3m"
    mu_method = "shrinkage"
    mu_shrinkage = 0.5
    mu_ewm_span = 60
    if objective == "max_sharpe":
        risk_free_source_label = st.radio(
            "Risk-free source",
            ["Manual input", "Live U.S. Treasury"],
            index=0,
            help=(
                "Manual input uses a fixed annual rate. Live U.S. Treasury fetches "
                "the latest official daily yield online at run time."
            ),
        )
        if risk_free_source_label == "Manual input":
            risk_free_rate_pct = st.number_input(
                "Risk-free rate (annual %)",
                min_value=-5.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                format="%.2f",
            )
            risk_free_rate = risk_free_rate_pct / 100.0
        else:
            risk_free_source = "treasury"
            risk_free_tenor = st.selectbox(
                "Treasury tenor",
                ["1m", "2m", "3m", "6m", "1y", "2y", "5y", "10y", "30y"],
                index=2,
                help="Rate is the latest available Treasury daily yield for this tenor.",
            )

        mu_method = st.selectbox(
            "Expected return model",
            ["Shrinkage mean", "EWMA mean", "Sample mean"],
            index=0,
            help=(
                "Sharpe optimization is sensitive to return estimates. Shrinkage "
                "is more stable than a plain sample mean."
            ),
        )
        if mu_method == "Shrinkage mean":
            mu_method = "shrinkage"
            mu_shrinkage = st.slider(
                "Mean shrinkage intensity",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="0 uses sample means; 1 fully shrinks to the cross-sectional grand mean.",
            )
        elif mu_method == "EWMA mean":
            mu_method = "ewma"
            mu_ewm_span = st.slider(
                "EWMA span (trading days)",
                min_value=20,
                max_value=252,
                value=60,
                step=5,
                help="Lower values emphasize more recent returns.",
            )
        else:
            mu_method = "sample"

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

    use_distance_penalty = st.checkbox(
        "Distance penalty",
        value=False,
        help=(
            "Softly penalize large changes from current weights instead of using "
            "a hard turnover cap."
        ),
    )
    distance_penalty = 0.0
    if use_distance_penalty:
        distance_penalty = st.slider(
            "Distance penalty strength",
            min_value=0.1,
            max_value=20.0,
            value=5.0,
            step=0.1,
            format="%.1f",
            help="Higher values keep proposed weights closer to current weights.",
        )

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
    st.header("Benchmark")
    use_benchmark = st.checkbox(
        "Compare vs benchmark",
        value=True,
        help="Adds benchmark stats and cumulative-return lines for context.",
    )
    benchmark_ticker = st.text_input(
        "Benchmark ticker",
        value="^GSPC",
        help="S&P 500 index ticker is ^GSPC. You can also use SPY.",
    ).strip().upper()

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

    random_test_n = 50
    random_test_size = 15
    random_test_seed = 42
    if run_random_test:
        st.divider()
        st.header("Robustness Test")
        random_test_n = st.slider(
            "Number of random portfolios",
            min_value=10,
            max_value=300,
            value=50,
            step=10,
        )
        random_test_size = st.slider(
            "Stocks per random portfolio",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
        )
        random_test_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=1_000_000,
            value=42,
            step=1,
        )

    if run_random_test:
        run_sp500_test = st.button(
            "🧪 Run Random S&P 500 Test",
            type="primary",
            width='stretch',
        )
        run_optimize = False
    else:
        run_optimize = st.button("🚀 Run Optimisation", type="primary", width='stretch')
        run_sp500_test = False

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if portfolio_df is not None and not run_sp500_test:
    st.subheader("Current Holdings")
    display_current = portfolio_df.copy()
    display_current["weight"] = display_current["weight"].map("{:.2%}".format)
    st.dataframe(display_current, width='stretch', hide_index=True)

if run_sp500_test:
    st.subheader("Random S&P 500 Robustness Test")
    if use_oos and eval_frac is not None:
        st.info(
            "📊 **Out-of-sample mode** enabled for random test: each sampled "
            "portfolio is optimized on the estimation window and scored on the "
            "held-out evaluation window."
        )
    else:
        st.caption(
            "⚠️ Random-test Sharpe comparison is **in-sample**. Enable "
            "*Out-of-sample evaluation* for a held-out robustness check."
        )

    effective_risk_free_rate = risk_free_rate
    if objective == "max_sharpe" and risk_free_source == "treasury":
        try:
            effective_risk_free_rate, quote_date = download_treasury_risk_free_rate(
                tenor=risk_free_tenor
            )
            st.info(
                "Using live Treasury risk-free rate "
                f"({risk_free_tenor}): {effective_risk_free_rate:.2%} "
                f"as of {quote_date.date()}."
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(
                "Treasury risk-free fetch failed "
                f"({exc}). Falling back to manual rate {risk_free_rate:.2%}."
            )

    progress_bar = st.progress(0.0, text="Preparing random S&P 500 test...")
    progress_status = st.empty()

    def _on_random_test_progress(
        processed: int,
        total: int,
        completed_count: int,
        improved_count: int,
    ) -> None:
        frac = processed / total if total > 0 else 0.0
        skipped_count = processed - completed_count
        progress_bar.progress(
            frac,
            text=(
                f"Running random portfolios: {processed}/{total} processed | "
                f"{completed_count} completed | {skipped_count} skipped"
            ),
        )
        progress_status.caption(
            f"Interim improvement rate: "
            f"{(improved_count / completed_count):.1%}" if completed_count > 0 else "Interim improvement rate: n/a"
        )

    with st.spinner("Running random S&P 500 Sharpe robustness test..."):
        try:
            test_result = run_random_sp500_sharpe_test(
                n_portfolios=int(random_test_n),
                portfolio_size=int(random_test_size),
                period=period,
                require_full_history=require_full_history,
                cov_method="sample",
                objective=objective,
                mu_method=mu_method,
                mu_shrinkage=mu_shrinkage,
                mu_ewm_span=mu_ewm_span,
                max_weight=max_weight,
                turnover_limit=turnover_limit,
                max_increase=max_increase,
                distance_penalty=distance_penalty,
                risk_free_rate=effective_risk_free_rate,
                eval_frac=eval_frac,
                small_cap_threshold_b=small_cap_threshold_b,
                small_cap_max_weight=(
                    small_cap_max_weight if small_cap_max_weight is not None else 0.05
                ),
                seed=int(random_test_seed),
                benchmark_ticker=benchmark_ticker if use_benchmark else "none",
                include_trials=True,
                progress_callback=_on_random_test_progress,
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Random S&P 500 test failed: {exc}")
            test_result = None
            progress_bar.empty()
            progress_status.empty()

    if test_result is not None:
        progress_bar.progress(1.0, text="Random S&P 500 robustness test complete.")
        progress_status.caption("Detailed per-test results are available below.")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Completed",
            f"{test_result.completed}/{test_result.attempted}",
            delta=f"{test_result.skipped} skipped",
            delta_color="off",
        )
        c2.metric(
            "Sharpe Improvement Rate",
            f"{test_result.improvement_rate:.1%}",
            delta=f"{test_result.improved} improved",
            delta_color="normal",
        )
        c3.metric(
            "Median Sharpe Delta",
            f"{test_result.median_sharpe_delta:+.3f}",
            delta=f"mean {test_result.avg_sharpe_delta:+.3f}",
            delta_color="normal",
        )

        if test_result.completed == 0:
            st.warning(
                "No random portfolios completed. Try shorter lookback, smaller "
                "portfolio size, or disable full-history filtering."
            )

        st.caption(
            "Standalone random test mode: portfolio-specific outputs (weights, "
            "cumulative returns, correlation heatmap) are intentionally hidden."
        )

        if test_result.trials:
            st.subheader("Per-Test Details")
            st.caption(
                "Expand any test below to inspect Sharpe stats, turnover, weights, "
                "and cumulative return paths for current vs proposed portfolios."
            )

            for trial in test_result.trials:
                outcome = "Improved" if trial.improved else "Not improved"
                with st.expander(
                    f"Test {trial.trial_index}: {outcome} | "
                    f"ΔSharpe={trial.sharpe_delta:+.3f} | "
                    f"Assets={len(trial.tickers)} | Mode={trial.mode}",
                    expanded=False,
                ):
                    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                    info_col1.metric("Current Sharpe", f"{trial.current_sharpe:.3f}")
                    info_col2.metric(
                        "Proposed Sharpe",
                        f"{trial.proposed_sharpe:.3f}",
                        delta=f"{trial.sharpe_delta:+.3f}",
                    )
                    info_col3.metric("Turnover", f"{trial.turnover:.2%}")
                    info_col4.metric("Evaluation Mode", trial.mode)

                    st.caption(
                        f"Estimation days: {trial.est_days} | "
                        f"Evaluation days: {trial.eval_days}"
                    )
                    st.caption("Tickers: " + ", ".join(trial.tickers))

                    chart_df = pd.DataFrame(
                        {
                            "Current": trial.cum_current,
                            "Proposed": trial.cum_proposed,
                        },
                        index=pd.to_datetime(trial.dates),
                    )
                    if trial.cum_benchmark is not None and trial.benchmark_label is not None:
                        chart_df[trial.benchmark_label] = trial.cum_benchmark
                    st.line_chart(chart_df, width='stretch')

                    weights_df = pd.DataFrame(
                        {
                            "Ticker": trial.tickers,
                            "Current Weight": trial.current_weights,
                            "Proposed Weight": trial.proposed_weights,
                        }
                    )
                    weights_df["Change"] = (
                        weights_df["Proposed Weight"] - weights_df["Current Weight"]
                    )
                    st.dataframe(
                        weights_df.sort_values("Proposed Weight", ascending=False),
                        width='stretch',
                        hide_index=True,
                    )

elif run_optimize and portfolio_df is not None:
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
    expected_returns = estimate_expected_returns(
        est_returns,
        annualise=True,
        method=mu_method,
        shrinkage=mu_shrinkage,
        ewm_span=mu_ewm_span,
    )

    benchmark_returns: pd.Series | None = None
    benchmark_eval_returns: pd.Series | None = None
    benchmark_label = "S&P 500"
    if use_benchmark and benchmark_ticker and benchmark_ticker.lower() != "none":
        try:
            bench_prices = download_prices([benchmark_ticker], period=period)
            if not bench_prices.empty:
                bench_ret_df = compute_returns(bench_prices)
                bench_series = bench_ret_df.iloc[:, 0]
                benchmark_label = "S&P 500" if benchmark_ticker == "^GSPC" else benchmark_ticker
                benchmark_returns = bench_series.reindex(est_returns.index).dropna()
                if eval_returns is not None:
                    benchmark_eval_returns = bench_series.reindex(eval_returns.index).dropna()

                if benchmark_returns.empty:
                    benchmark_returns = None
                    benchmark_eval_returns = None
                    st.warning(
                        f"Benchmark '{benchmark_ticker}' has no overlapping dates and was skipped."
                    )
            else:
                st.warning(f"No benchmark prices found for '{benchmark_ticker}'.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Benchmark download failed for '{benchmark_ticker}': {exc}")

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
        effective_risk_free_rate = risk_free_rate
        if objective == "max_sharpe" and risk_free_source == "treasury":
            try:
                effective_risk_free_rate, quote_date = download_treasury_risk_free_rate(
                    tenor=risk_free_tenor
                )
                st.info(
                    "Using live Treasury risk-free rate "
                    f"({risk_free_tenor}): {effective_risk_free_rate:.2%} "
                    f"as of {quote_date.date()}."
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(
                    "Treasury risk-free fetch failed "
                    f"({exc}). Falling back to manual rate {risk_free_rate:.2%}."
                )

        rebalanced = rebalance(
            portfolio_filtered,
            cov,
            max_weight=max_weight,
            max_weights=per_asset_max,
            turnover_limit=turnover_limit,
            max_increase=max_increase,
            distance_penalty=distance_penalty,
            objective=objective,
            expected_returns=expected_returns,
            risk_free_rate=effective_risk_free_rate,
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
        stats_summary(
            current_w,
            proposed_w,
            est_returns,
            cov,
            eval_returns=eval_returns,
            benchmark_returns=benchmark_returns,
            benchmark_eval_returns=benchmark_eval_returns,
            benchmark_label=benchmark_label,
        ),
        width='stretch',
    )

    st.divider()

    # ---- Cumulative returns ----
    st.subheader("Cumulative Returns")
    st.plotly_chart(
        fig_cumulative_returns(
            est_returns,
            current_w,
            proposed_w,
            eval_returns=eval_returns,
            benchmark_returns=benchmark_returns,
            benchmark_eval_returns=benchmark_eval_returns,
            benchmark_label=benchmark_label,
        ),
        width='stretch',
    )

    st.divider()
    # ---- Correlation heatmap ----
    st.subheader("Asset Correlation Heatmap")
    st.plotly_chart(fig_correlation_heatmap(est_returns), width='stretch')

elif run_optimize and portfolio_df is None:
    st.warning("Please provide a portfolio before running optimisation.")

else:
    if run_random_test:
        st.info("👈 Click **Run Random S&P 500 Test** to run the standalone robustness check.")
    elif portfolio_df is None:
        st.info("👈 Configure your portfolio in the sidebar, then click **Run Optimisation**.")
    else:
        st.info("👈 Click **Run Optimisation** in the sidebar to see results.")
