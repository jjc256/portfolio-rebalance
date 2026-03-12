"""Tests for the optimizer module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_rebalance.risk import compute_returns, compute_covariance, portfolio_volatility
from portfolio_rebalance.optimizer import minimize_variance, rebalance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_prices(n_days: int = 500, n_assets: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0005, 0.015, size=(n_days, n_assets))
    prices = 100 * np.cumprod(1 + daily_ret, axis=0)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _build_cov(n_assets: int = 5, seed: int = 7) -> pd.DataFrame:
    prices = _random_prices(n_assets=n_assets, seed=seed)
    ret = compute_returns(prices)
    return compute_covariance(ret)


# ---------------------------------------------------------------------------
# minimize_variance
# ---------------------------------------------------------------------------


def test_minimize_variance_weights_sum_to_one():
    cov = _build_cov()
    w = minimize_variance(cov)
    assert abs(w.sum() - 1.0) < 1e-6


def test_minimize_variance_long_only():
    cov = _build_cov()
    w = minimize_variance(cov)
    assert np.all(w >= -1e-9)


def test_minimize_variance_reduces_volatility():
    """Optimised portfolio should have <= volatility of equal-weight portfolio."""
    cov = _build_cov()
    n = len(cov)
    w_equal = np.full(n, 1.0 / n)
    w_opt = minimize_variance(cov)
    vol_equal = portfolio_volatility(w_equal, cov)
    vol_opt = portfolio_volatility(w_opt, cov)
    # Allow a tiny tolerance for numerical noise
    assert vol_opt <= vol_equal + 1e-8


def test_minimize_variance_max_weight_respected():
    cov = _build_cov()
    max_w = 0.30
    w = minimize_variance(cov, max_weight=max_w)
    assert np.all(w <= max_w + 1e-6)


def test_minimize_variance_per_asset_max_weights_respected():
    cov = _build_cov(n_assets=4)
    per_asset_max = np.array([0.10, 0.40, 0.40, 0.40])
    w = minimize_variance(cov, max_weights=per_asset_max)
    assert np.all(w <= per_asset_max + 1e-6)


def test_minimize_variance_turnover_respected():
    cov = _build_cov(n_assets=5)
    n = len(cov)
    current = np.full(n, 1.0 / n)
    limit = 0.20
    w = minimize_variance(cov, current_weights=current, turnover_limit=limit)
    turnover = float(np.sum(np.abs(w - current)))
    assert turnover <= limit + 1e-6


def test_minimize_variance_max_increase_respected():
    cov = _build_cov(n_assets=5)
    n = len(cov)
    current = np.full(n, 1.0 / n)
    max_increase = 0.03
    w = minimize_variance(cov, current_weights=current, max_increase=max_increase)
    assert np.all(w <= current + max_increase + 1e-6)


def test_minimize_variance_infeasible_upper_bounds_raises():
    cov = _build_cov(n_assets=5)
    current = np.full(5, 0.20)
    with pytest.raises(ValueError, match="Infeasible constraints"):
        minimize_variance(
            cov,
            current_weights=current,
            max_weight=0.10,
            max_increase=0.0,
        )


def test_minimize_variance_two_assets():
    """With two assets the minimum-variance weight can be verified analytically."""
    # Construct a 2x2 cov matrix
    s1, s2, rho = 0.20, 0.30, 0.5
    cov_arr = np.array(
        [[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]]
    )
    cov = pd.DataFrame(cov_arr, index=["A", "B"], columns=["A", "B"])
    # Analytical min-var weight for asset A:
    #   w1 = (s2^2 - cov12) / (s1^2 + s2^2 - 2*cov12)
    cov12 = rho * s1 * s2
    w1_analytic = (s2**2 - cov12) / (s1**2 + s2**2 - 2 * cov12)

    w = minimize_variance(cov)
    assert abs(w[0] - w1_analytic) < 1e-5


# ---------------------------------------------------------------------------
# rebalance (high-level)
# ---------------------------------------------------------------------------


def test_rebalance_returns_dataframe():
    prices = _random_prices()
    ret = compute_returns(prices)
    cov = compute_covariance(ret)
    n = len(cov)
    portfolio = pd.DataFrame(
        {"ticker": cov.columns.tolist(), "weight": np.full(n, 1.0 / n)}
    )
    result = rebalance(portfolio, cov)
    assert "proposed_weight" in result.columns
    assert abs(result["proposed_weight"].sum() - 1.0) < 1e-6


def test_rebalance_max_weight_column():
    prices = _random_prices()
    ret = compute_returns(prices)
    cov = compute_covariance(ret)
    n = len(cov)
    portfolio = pd.DataFrame(
        {"ticker": cov.columns.tolist(), "weight": np.full(n, 1.0 / n)}
    )
    max_w = 0.25
    result = rebalance(portfolio, cov, max_weight=max_w)
    assert np.all(result["proposed_weight"].values <= max_w + 1e-6)
