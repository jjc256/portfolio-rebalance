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
