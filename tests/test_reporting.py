"""Tests for the reporting module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_rebalance.risk import compute_returns, compute_covariance
from portfolio_rebalance.reporting import (
    weights_table,
    stats_summary,
    fig_weights_bar,
    fig_volatility_gauge,
    fig_correlation_heatmap,
    fig_cumulative_returns,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_rebalanced(n_assets: int = 4, seed: int = 99):
    rng = np.random.default_rng(seed)
    tickers = [f"S{i}" for i in range(n_assets)]
    current = rng.dirichlet(np.ones(n_assets))
    proposed = rng.dirichlet(np.ones(n_assets))
    return pd.DataFrame(
        {"ticker": tickers, "weight": current, "proposed_weight": proposed}
    )


def _make_returns(n_days: int = 300, n_assets: int = 4, seed: int = 99):
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0005, 0.012, size=(n_days, n_assets))
    prices = 100 * np.cumprod(1 + daily_ret, axis=0)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    tickers = [f"S{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# weights_table
# ---------------------------------------------------------------------------


def test_weights_table_columns():
    rebalanced = _make_rebalanced()
    tbl = weights_table(rebalanced)
    assert "Ticker" in tbl.columns
    assert "Current Weight" in tbl.columns
    assert "Proposed Weight" in tbl.columns
    assert "Δ Weight" in tbl.columns


def test_weights_table_row_count():
    rebalanced = _make_rebalanced(n_assets=6)
    tbl = weights_table(rebalanced)
    assert len(tbl) == 6


# ---------------------------------------------------------------------------
# stats_summary
# ---------------------------------------------------------------------------


def test_stats_summary_shape():
    prices = _make_returns()
    ret = compute_returns(prices)
    cov = compute_covariance(ret)
    n = len(cov)
    w = np.full(n, 1.0 / n)
    summary = stats_summary(w, w, ret, cov)
    assert summary.shape[0] == 2  # Current and Proposed rows
    assert "Annualised Volatility" in summary.columns


def test_stats_summary_labels_in_sample():
    """Index labels should contain '(In-Sample)' when no eval_returns is given."""
    prices = _make_returns()
    ret = compute_returns(prices)
    cov = compute_covariance(ret)
    w = np.full(4, 0.25)
    summary = stats_summary(w, w, ret, cov)
    assert all("In-Sample" in idx for idx in summary.index)


def test_stats_summary_labels_oos():
    """Index labels should contain '(OOS)' when eval_returns is provided."""
    from portfolio_rebalance.risk import split_returns

    prices = _make_returns(n_days=500)
    ret = compute_returns(prices)
    est, oos = split_returns(ret, eval_frac=0.2)
    cov = compute_covariance(est)
    w = np.full(4, 0.25)
    summary = stats_summary(w, w, est, cov, eval_returns=oos)
    assert all("OOS" in idx for idx in summary.index)


def test_stats_summary_oos_uses_held_out_period():
    """OOS stats must use data from the held-out period, not the estimation period."""
    from portfolio_rebalance.risk import split_returns

    prices = _make_returns(n_days=600)
    ret = compute_returns(prices)
    est, oos = split_returns(ret, eval_frac=0.25)
    cov = compute_covariance(est)
    w = np.full(4, 0.25)
    is_summary = stats_summary(w, w, est, cov)
    oos_summary = stats_summary(w, w, est, cov, eval_returns=oos)
    # The volatility figures will generally differ between windows (different raw data)
    is_vol = is_summary["Annualised Volatility"].iloc[0]
    oos_vol = oos_summary["Annualised Volatility"].iloc[0]
    # Simply check they are valid percentage strings (format "X.XX%")
    assert is_vol.endswith("%")
    assert oos_vol.endswith("%")
    # The return stat values should also differ between windows
    is_ret = is_summary["Annualised Return"].iloc[0]
    oos_ret = oos_summary["Annualised Return"].iloc[0]
    assert is_ret != oos_ret, "OOS return should use different data from in-sample return"


def test_stats_summary_includes_benchmark_row_when_provided():
    prices = _make_returns()
    ret = compute_returns(prices)
    cov = compute_covariance(ret)
    w = np.full(4, 0.25)
    benchmark = ret.iloc[:, 0]
    summary = stats_summary(
        w,
        w,
        ret,
        cov,
        benchmark_returns=benchmark,
        benchmark_label="S&P 500",
    )
    assert "S&P 500 (In-Sample)" in summary.index


# ---------------------------------------------------------------------------
# Plotly figures (just ensure they are created without error)
# ---------------------------------------------------------------------------


def test_fig_weights_bar():
    rebalanced = _make_rebalanced()
    fig = fig_weights_bar(rebalanced)
    assert fig is not None


def test_fig_volatility_gauge():
    fig = fig_volatility_gauge(0.20, 0.15)
    assert fig is not None


def test_fig_correlation_heatmap():
    prices = _make_returns()
    ret = compute_returns(prices)
    fig = fig_correlation_heatmap(ret)
    assert fig is not None


def test_fig_cumulative_returns():
    prices = _make_returns(n_assets=4)
    ret = compute_returns(prices)
    n = 4
    w = np.full(n, 0.25)
    fig = fig_cumulative_returns(ret, w, w)
    assert fig is not None


def test_fig_cumulative_returns_oos():
    """OOS variant should produce a figure with 4 traces and a vertical line."""
    from portfolio_rebalance.risk import split_returns

    prices = _make_returns(n_days=500, n_assets=4)
    ret = compute_returns(prices)
    est, oos = split_returns(ret, eval_frac=0.2)
    w = np.full(4, 0.25)
    fig = fig_cumulative_returns(est, w, w, eval_returns=oos)
    assert fig is not None
    # Should have 4 scatter traces (2 in-sample + 2 OOS)
    scatter_traces = [t for t in fig.data if t.type == "scatter"]
    assert len(scatter_traces) == 4


def test_fig_cumulative_returns_with_benchmark():
    prices = _make_returns(n_assets=4)
    ret = compute_returns(prices)
    w = np.full(4, 0.25)
    benchmark = ret.iloc[:, 0]
    fig = fig_cumulative_returns(ret, w, w, benchmark_returns=benchmark)
    assert fig is not None
    scatter_traces = [t for t in fig.data if t.type == "scatter"]
    assert len(scatter_traces) == 3


def test_fig_cumulative_returns_oos_with_benchmark():
    from portfolio_rebalance.risk import split_returns

    prices = _make_returns(n_days=500, n_assets=4)
    ret = compute_returns(prices)
    est, oos = split_returns(ret, eval_frac=0.2)
    w = np.full(4, 0.25)
    benchmark_est = est.iloc[:, 0]
    benchmark_oos = oos.iloc[:, 0]
    fig = fig_cumulative_returns(
        est,
        w,
        w,
        eval_returns=oos,
        benchmark_returns=benchmark_est,
        benchmark_eval_returns=benchmark_oos,
        benchmark_label="S&P 500",
    )
    assert fig is not None
    scatter_traces = [t for t in fig.data if t.type == "scatter"]
    assert len(scatter_traces) == 6
