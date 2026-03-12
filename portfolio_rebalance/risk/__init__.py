"""Risk model module.

Computes daily/annualised returns and a covariance matrix from a price DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trading days per year used throughout for annualisation
TRADING_DAYS = 252


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute daily returns from adjusted close prices.

    Parameters
    ----------
    prices:
        DataFrame of adjusted close prices (rows = dates, columns = tickers).
    method:
        ``"log"`` for log returns (default) or ``"simple"`` for simple returns.
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    elif method == "simple":
        returns = prices.pct_change().dropna()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'log' or 'simple'.")
    return returns


def compute_covariance(
    returns: pd.DataFrame,
    annualise: bool = True,
    method: str = "sample",
) -> pd.DataFrame:
    """Compute the covariance matrix of asset returns.

    Parameters
    ----------
    returns:
        Daily returns DataFrame (rows = dates, columns = tickers).
    annualise:
        Multiply by ``TRADING_DAYS`` to get annualised covariance (default True).
    method:
        ``"sample"`` for the standard sample covariance matrix.
        ``"ledoit_wolf"`` for the shrinkage estimator (requires scikit-learn).
    """
    if method == "sample":
        cov = returns.cov()
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf  # optional dependency

        lw = LedoitWolf()
        lw.fit(returns.values)
        cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    else:
        raise ValueError(f"Unknown covariance method '{method}'.")

    if annualise:
        cov = cov * TRADING_DAYS
    return cov


def portfolio_volatility(weights: np.ndarray, cov: pd.DataFrame) -> float:
    """Return annualised portfolio volatility (std dev) given weights and covariance."""
    w = np.asarray(weights, dtype=float)
    sigma2 = float(w @ cov.values @ w)
    return float(np.sqrt(max(sigma2, 0.0)))


def portfolio_stats(
    weights: np.ndarray,
    returns: pd.DataFrame,
    cov: pd.DataFrame,
) -> dict:
    """Compute a set of portfolio statistics.

    Returns a dict with keys:
    - ``volatility``:  annualised portfolio std dev
    - ``mean_return``: annualised mean portfolio return
    - ``sharpe``:      annualised Sharpe ratio (assuming 0 risk-free rate)
    - ``max_drawdown``: maximum drawdown of the equally-weighted portfolio series
    """
    w = np.asarray(weights, dtype=float)
    vol = portfolio_volatility(w, cov)

    # Annualised mean
    port_ret_daily = (returns.values @ w)
    ann_ret = float(np.mean(port_ret_daily) * TRADING_DAYS)

    sharpe = float(ann_ret / vol) if vol > 0 else 0.0

    # Cumulative return series for drawdown
    cum = (1 + pd.Series(port_ret_daily)).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = float(dd.min())

    return {
        "volatility": vol,
        "mean_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def compute_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Return the Pearson correlation matrix of daily returns."""
    return returns.corr()


def split_returns(
    returns: pd.DataFrame,
    eval_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split returns into an estimation period and an out-of-sample evaluation period.

    The estimation (earlier) slice is used to fit the covariance matrix and run
    the optimiser.  The evaluation (later) slice is held out and used only to
    report *realised* out-of-sample performance, preventing look-ahead bias in
    the performance statistics.

    Parameters
    ----------
    returns:
        Full daily returns DataFrame.
    eval_frac:
        Fraction of rows to hold out at the end as the evaluation (OOS) period.
        Must be in (0, 1).  Default ``0.2`` → last 20 % is held out.

    Returns
    -------
    (estimation_returns, eval_returns)
        Two non-overlapping DataFrames with the same columns.
    """
    if not 0.0 < eval_frac < 1.0:
        raise ValueError(f"eval_frac must be strictly in (0, 1), got {eval_frac}.")
    split_idx = int(len(returns) * (1.0 - eval_frac))
    return returns.iloc[:split_idx].copy(), returns.iloc[split_idx:].copy()


def portfolio_stats_realized(
    weights: np.ndarray,
    returns: pd.DataFrame,
) -> dict:
    """Compute portfolio statistics entirely from realised daily returns.

    Unlike :func:`portfolio_stats`, this function derives volatility from the
    *standard deviation of actual portfolio returns* rather than from an
    estimated covariance matrix.  Use this for out-of-sample evaluation to
    avoid conflating the estimation-period covariance with the evaluation-period
    performance.

    Returns a dict with the same keys as :func:`portfolio_stats`:
    ``volatility``, ``mean_return``, ``sharpe``, ``max_drawdown``.
    """
    w = np.asarray(weights, dtype=float)
    port_ret_daily = returns.values @ w

    vol = float(np.std(port_ret_daily, ddof=1) * np.sqrt(TRADING_DAYS))
    ann_ret = float(np.mean(port_ret_daily) * TRADING_DAYS)
    sharpe = float(ann_ret / vol) if vol > 0 else 0.0

    cum = (1 + pd.Series(port_ret_daily)).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = float(dd.min())

    return {
        "volatility": vol,
        "mean_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }
