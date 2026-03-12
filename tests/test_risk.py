"""Tests for the risk model module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_rebalance.risk import (
    compute_returns,
    compute_covariance,
    portfolio_volatility,
    portfolio_stats,
    portfolio_stats_realized,
    compute_correlation,
    split_returns,
    TRADING_DAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_prices(n_days: int = 500, n_assets: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0005, 0.01, size=(n_days, n_assets))
    prices = 100 * np.cumprod(1 + daily_ret, axis=0)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    tickers = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# compute_returns
# ---------------------------------------------------------------------------


def test_compute_returns_log_shape():
    prices = _random_prices()
    ret = compute_returns(prices, method="log")
    assert ret.shape == (prices.shape[0] - 1, prices.shape[1])


def test_compute_returns_simple_shape():
    prices = _random_prices()
    ret = compute_returns(prices, method="simple")
    assert ret.shape == (prices.shape[0] - 1, prices.shape[1])


def test_compute_returns_invalid_method():
    prices = _random_prices()
    with pytest.raises(ValueError):
        compute_returns(prices, method="unknown")


def test_compute_returns_no_nans():
    prices = _random_prices()
    ret = compute_returns(prices)
    assert not ret.isna().any().any()


# ---------------------------------------------------------------------------
# compute_covariance
# ---------------------------------------------------------------------------


def test_compute_covariance_shape():
    prices = _random_prices(n_assets=4)
    ret = compute_returns(prices)
    cov = compute_covariance(ret)
    assert cov.shape == (4, 4)


def test_compute_covariance_positive_semidefinite():
    prices = _random_prices(n_assets=5)
    ret = compute_returns(prices)
    cov = compute_covariance(ret)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    # All eigenvalues should be >= 0 (allowing tiny floating-point negatives)
    assert np.all(eigenvalues >= -1e-10)


def test_compute_covariance_annualised():
    prices = _random_prices(n_assets=3)
    ret = compute_returns(prices)
    cov_daily = compute_covariance(ret, annualise=False)
    cov_ann = compute_covariance(ret, annualise=True)
    np.testing.assert_allclose(cov_ann.values, cov_daily.values * TRADING_DAYS, rtol=1e-9)


# ---------------------------------------------------------------------------
# portfolio_volatility
# ---------------------------------------------------------------------------


def test_portfolio_volatility_equal_weights():
    prices = _random_prices(n_assets=4)
    ret = compute_returns(prices)
    cov = compute_covariance(ret)
    w = np.full(4, 0.25)
    vol = portfolio_volatility(w, cov)
    assert vol > 0


def test_portfolio_volatility_single_asset():
    """Single-asset portfolio vol should equal asset's own std dev."""
    prices = _random_prices(n_assets=2)
    ret = compute_returns(prices)
    cov = compute_covariance(ret)
    w = np.array([1.0, 0.0])
    vol = portfolio_volatility(w, cov)
    expected = np.sqrt(cov.values[0, 0])
    assert abs(vol - expected) < 1e-12


# ---------------------------------------------------------------------------
# portfolio_stats
# ---------------------------------------------------------------------------


def test_portfolio_stats_keys():
    prices = _random_prices()
    ret = compute_returns(prices)
    cov = compute_covariance(ret)
    w = np.full(4, 0.25)
    stats = portfolio_stats(w, ret, cov)
    assert set(stats.keys()) == {"volatility", "mean_return", "sharpe", "max_drawdown"}


def test_portfolio_stats_max_drawdown_non_positive():
    prices = _random_prices()
    ret = compute_returns(prices)
    cov = compute_covariance(ret)
    w = np.full(4, 0.25)
    stats = portfolio_stats(w, ret, cov)
    assert stats["max_drawdown"] <= 0


# ---------------------------------------------------------------------------
# compute_correlation
# ---------------------------------------------------------------------------


def test_compute_correlation_diagonal_ones():
    prices = _random_prices()
    ret = compute_returns(prices)
    corr = compute_correlation(ret)
    np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-12)


def test_compute_correlation_range():
    prices = _random_prices()
    ret = compute_returns(prices)
    corr = compute_correlation(ret)
    assert corr.values.min() >= -1.0 - 1e-12
    assert corr.values.max() <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# split_returns
# ---------------------------------------------------------------------------


def test_split_returns_no_overlap():
    prices = _random_prices(n_days=500)
    ret = compute_returns(prices)
    est, oos = split_returns(ret, eval_frac=0.2)
    # The two windows must not overlap
    assert est.index[-1] < oos.index[0]


def test_split_returns_size():
    prices = _random_prices(n_days=500)
    ret = compute_returns(prices)
    n = len(ret)
    est, oos = split_returns(ret, eval_frac=0.2)
    # All rows accounted for
    assert len(est) + len(oos) == n
    # OOS size is within 1 row of the requested fraction (integer truncation may vary by 1)
    assert abs(len(oos) - round(n * 0.2)) <= 1


def test_split_returns_columns_preserved():
    prices = _random_prices(n_assets=5)
    ret = compute_returns(prices)
    est, oos = split_returns(ret, eval_frac=0.3)
    assert list(est.columns) == list(ret.columns)
    assert list(oos.columns) == list(ret.columns)


def test_split_returns_invalid_frac():
    prices = _random_prices()
    ret = compute_returns(prices)
    with pytest.raises(ValueError):
        split_returns(ret, eval_frac=0.0)
    with pytest.raises(ValueError):
        split_returns(ret, eval_frac=1.0)


# ---------------------------------------------------------------------------
# portfolio_stats_realized
# ---------------------------------------------------------------------------


def test_portfolio_stats_realized_keys():
    prices = _random_prices()
    ret = compute_returns(prices)
    w = np.full(4, 0.25)
    stats = portfolio_stats_realized(w, ret)
    assert set(stats.keys()) == {"volatility", "mean_return", "sharpe", "max_drawdown"}


def test_portfolio_stats_realized_vol_positive():
    prices = _random_prices()
    ret = compute_returns(prices)
    w = np.full(4, 0.25)
    stats = portfolio_stats_realized(w, ret)
    assert stats["volatility"] > 0


def test_portfolio_stats_realized_max_drawdown_non_positive():
    prices = _random_prices()
    ret = compute_returns(prices)
    w = np.full(4, 0.25)
    stats = portfolio_stats_realized(w, ret)
    assert stats["max_drawdown"] <= 0
