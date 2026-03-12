"""Optimizer module.

Solves a minimum-variance portfolio optimisation with:
  - Long-only constraints  (w_i >= 0)
  - Budget constraint       (sum w_i = 1)
  - Optional max-position   (w_i <= max_weight)
  - Optional turnover limit (sum |w_new - w_old| <= turnover_limit)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult


def minimize_variance(
    cov: pd.DataFrame,
    current_weights: Optional[np.ndarray] = None,
    max_weight: float = 1.0,
    turnover_limit: Optional[float] = None,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve the minimum-variance optimisation problem.

    Parameters
    ----------
    cov:
        Annualised covariance matrix (tickers × tickers).
    current_weights:
        Current portfolio weights used as the starting point and for the
        optional turnover constraint.  If ``None``, equal weights are used.
    max_weight:
        Maximum weight for any single position (default 1.0 = unconstrained).
    turnover_limit:
        If provided, the sum of absolute weight changes is constrained to be
        at most this value (e.g. 0.5 means ≤50 % one-way turnover).
    tol:
        Solver tolerance.

    Returns
    -------
    np.ndarray
        Optimal weight vector aligned with ``cov.columns``.
    """
    n = len(cov)
    cov_mat = cov.values.astype(float)

    if current_weights is None:
        w0 = np.full(n, 1.0 / n)
    else:
        w0 = np.asarray(current_weights, dtype=float).copy()
        # Normalise starting point just in case
        if w0.sum() > 0:
            w0 /= w0.sum()

    # Objective: minimise portfolio variance w^T Σ w
    def objective(w: np.ndarray) -> float:
        return float(w @ cov_mat @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * cov_mat @ w

    # Bounds: 0 <= w_i <= max_weight
    bounds = [(0.0, max_weight)] * n

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # budget
    ]

    if turnover_limit is not None and current_weights is not None:
        w_old = np.asarray(current_weights, dtype=float)
        constraints.append(
            {
                "type": "ineq",
                # sum|w - w_old| <= turnover_limit  ↔  turnover_limit - sum|…| >= 0
                "fun": lambda w, wo=w_old, lim=turnover_limit: lim - np.sum(np.abs(w - wo)),
            }
        )

    result: OptimizeResult = minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": 1000, "disp": False},
    )

    if not result.success:
        # Fall back to equal weights if solver fails
        import logging

        logging.getLogger(__name__).warning(
            "Optimiser did not converge: %s. Returning equal weights.", result.message
        )
        return np.full(n, 1.0 / n)

    w_opt = result.x
    # Clip tiny negatives from numerical noise and renormalise
    w_opt = np.clip(w_opt, 0.0, None)
    w_opt /= w_opt.sum()
    return w_opt


def rebalance(
    portfolio: pd.DataFrame,
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    turnover_limit: Optional[float] = None,
) -> pd.DataFrame:
    """High-level helper: given a portfolio DataFrame and covariance matrix,
    return a new DataFrame with an added ``proposed_weight`` column.

    Parameters
    ----------
    portfolio:
        DataFrame with columns ``ticker`` and ``weight``.
    cov:
        Covariance matrix indexed/columned by ticker.
    max_weight:
        Maximum per-position weight.
    turnover_limit:
        Optional turnover constraint.
    """
    # Align portfolio and covariance matrix
    tickers = portfolio["ticker"].tolist()
    cov_aligned = cov.loc[tickers, tickers]
    current_w = portfolio["weight"].values

    optimal_w = minimize_variance(
        cov_aligned,
        current_weights=current_w,
        max_weight=max_weight,
        turnover_limit=turnover_limit,
    )

    result = portfolio.copy()
    result["proposed_weight"] = optimal_w
    return result
