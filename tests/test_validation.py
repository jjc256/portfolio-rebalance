"""Tests for random S&P 500 robustness validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_rebalance.validation import run_random_sp500_sharpe_test


def _fake_prices(tickers: list[str], n_days: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        base = 100.0 + i
        data[t] = base + np.linspace(0.0, 10.0, n_days)
    return pd.DataFrame(data, index=idx)


def test_random_sp500_sharpe_test_counts_improvements(monkeypatch):
    universe = [f"T{i}" for i in range(30)]

    monkeypatch.setattr(
        "portfolio_rebalance.validation.download_sp500_tickers",
        lambda: universe,
    )
    monkeypatch.setattr(
        "portfolio_rebalance.validation.download_prices",
        lambda tickers, period="3y": _fake_prices(tickers),
    )

    def fake_returns(prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change().dropna()

    monkeypatch.setattr("portfolio_rebalance.validation.compute_returns", fake_returns)
    monkeypatch.setattr(
        "portfolio_rebalance.validation.compute_covariance",
        lambda returns, annualise=True, method="sample": pd.DataFrame(
            np.eye(returns.shape[1]),
            index=returns.columns,
            columns=returns.columns,
        ),
    )
    monkeypatch.setattr(
        "portfolio_rebalance.validation.estimate_expected_returns",
        lambda returns, annualise=True, method="shrinkage", shrinkage=0.5, ewm_span=60: pd.Series(
            np.ones(returns.shape[1]),
            index=returns.columns,
        ),
    )

    def fake_rebalance(portfolio: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        out = portfolio.copy()
        out["proposed_weight"] = 0.0
        out.loc[out.index[0], "proposed_weight"] = 1.0
        return out

    monkeypatch.setattr("portfolio_rebalance.validation.rebalance", fake_rebalance)

    def fake_stats(weights, *args, **kwargs):
        return {
            "volatility": 0.1,
            "mean_return": 0.1,
            "sharpe": float(weights[0]),
            "max_drawdown": -0.1,
        }

    monkeypatch.setattr("portfolio_rebalance.validation.portfolio_stats", fake_stats)
    monkeypatch.setattr("portfolio_rebalance.validation.portfolio_stats_realized", fake_stats)

    result = run_random_sp500_sharpe_test(
        n_portfolios=20,
        portfolio_size=8,
        objective="max_sharpe",
        seed=5,
    )

    assert result.attempted == 20
    assert result.completed == 20
    assert result.improved == 20
    assert result.improvement_rate == 1.0


def test_random_sp500_sharpe_test_skips_when_prices_missing(monkeypatch):
    universe = [f"T{i}" for i in range(20)]

    monkeypatch.setattr(
        "portfolio_rebalance.validation.download_sp500_tickers",
        lambda: universe,
    )
    monkeypatch.setattr(
        "portfolio_rebalance.validation.download_prices",
        lambda tickers, period="3y": pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="No price data"):
        run_random_sp500_sharpe_test(n_portfolios=10, portfolio_size=5)
