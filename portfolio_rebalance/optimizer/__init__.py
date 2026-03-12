"""Optimizer module.

Solves a minimum-variance portfolio optimisation with:
  - Long-only constraints  (w_i >= 0)
  - Budget constraint       (sum w_i = 1)
  - Optional max-position   (w_i <= max_weight)
  - Optional turnover limit (sum |w_new - w_old| <= turnover_limit)
    - Optional max-increase   (w_i <= w_old_i + max_increase)
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
    max_weights: Optional[np.ndarray] = None,
    turnover_limit: Optional[float] = None,
    max_increase: Optional[float] = None,
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
    max_weights:
        Optional per-asset max weights aligned with ``cov.columns``. When
        provided, each position bound is ``min(max_weight, max_weights[i])``.
    turnover_limit:
        If provided, the sum of absolute weight changes is constrained to be
        at most this value (e.g. 0.5 means ≤50 % one-way turnover).
    max_increase:
        If provided (and ``current_weights`` is given), each position weight is
        capped at ``current_weight + max_increase``. This directly limits how
        much a small speculative position can grow in one rebalance.
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
        w_old: Optional[np.ndarray] = None
    else:
        w_old = np.asarray(current_weights, dtype=float).copy()
        # Normalise current weights just in case
        if w_old.sum() > 0:
            w_old /= w_old.sum()
        w0 = w_old.copy()

    # Objective: minimise portfolio variance w^T Σ w
    def objective(w: np.ndarray) -> float:
        return float(w @ cov_mat @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * cov_mat @ w

    # Bounds: 0 <= w_i <= upper_i
    upper = np.full(n, float(max_weight))
    if max_weights is not None:
        per_asset = np.asarray(max_weights, dtype=float)
        if per_asset.shape[0] != n:
            raise ValueError(
                f"max_weights length ({per_asset.shape[0]}) does not match number "
                f"of assets ({n})."
            )
        if np.any(per_asset < 0.0):
            raise ValueError("max_weights values must be non-negative.")
        upper = np.minimum(upper, per_asset)

    if max_increase is not None and w_old is not None:
        upper = np.minimum(upper, w_old + float(max_increase))

    if float(np.sum(upper)) < 1.0 - 1e-12:
        raise ValueError(
            "Infeasible constraints: sum of per-asset upper bounds is below 1. "
            "Relax max_weight and/or max_increase."
        )

    bounds = [(0.0, float(u)) for u in upper]

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # budget
    ]

    if turnover_limit is not None and w_old is not None:
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
    max_weights: Optional[pd.Series | dict | np.ndarray] = None,
    turnover_limit: Optional[float] = None,
    max_increase: Optional[float] = None,
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
    max_weights:
        Optional per-position max weights. Can be:
        - ``np.ndarray`` aligned to ``portfolio`` rows,
        - ``dict`` keyed by ticker,
        - ``pd.Series`` indexed by ticker.
        Missing dict/Series entries fall back to ``max_weight``.
    turnover_limit:
        Optional turnover constraint.
    max_increase:
        Optional cap on per-position increase versus current weight.
    """
    # Align portfolio and covariance matrix
    tickers = portfolio["ticker"].tolist()
    cov_aligned = cov.loc[tickers, tickers]
    current_w = portfolio["weight"].values

    per_asset_max: Optional[np.ndarray] = None
    if max_weights is not None:
        if isinstance(max_weights, np.ndarray):
            per_asset_max = max_weights.astype(float)
        elif isinstance(max_weights, dict):
            per_asset_max = (
                pd.Series(max_weights, dtype=float)
                .reindex(tickers)
                .fillna(float(max_weight))
                .values
            )
        elif isinstance(max_weights, pd.Series):
            per_asset_max = (
                max_weights.astype(float)
                .reindex(tickers)
                .fillna(float(max_weight))
                .values
            )
        else:
            raise TypeError("max_weights must be np.ndarray, dict, or pd.Series.")

    optimal_w = minimize_variance(
        cov_aligned,
        current_weights=current_w,
        max_weight=max_weight,
        max_weights=per_asset_max,
        turnover_limit=turnover_limit,
        max_increase=max_increase,
    )

    result = portfolio.copy()
    result["proposed_weight"] = optimal_w
    return result
