"""Optimizer module.

Solves constrained portfolio optimisation with two objective options:
    - ``min_variance``
    - ``max_sharpe``

Supported constraints:
    - Long-only constraints  (w_i >= 0)
    - Budget constraint      (sum w_i = 1)
    - Optional max-position  (w_i <= max_weight)
    - Optional turnover      (sum |w_new - w_old| <= turnover_limit)
    - Optional max-increase  (w_i <= w_old_i + max_increase)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult


_VALID_OBJECTIVES = {"min_variance", "max_sharpe"}


def _optimise_weights(
    cov: pd.DataFrame,
    objective: str,
    current_weights: Optional[np.ndarray] = None,
    max_weight: float = 1.0,
    max_weights: Optional[np.ndarray] = None,
    turnover_limit: Optional[float] = None,
    max_increase: Optional[float] = None,
    expected_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve the constrained optimization problem for the selected objective."""
    n = len(cov)
    cov_mat = cov.values.astype(float)

    if current_weights is None:
        w0 = np.full(n, 1.0 / n)
        w_old: Optional[np.ndarray] = None
    else:
        w_old = np.asarray(current_weights, dtype=float).copy()
        if w_old.sum() > 0:
            w_old /= w_old.sum()
        w0 = w_old.copy()

    if objective == "min_variance":

        def objective_fn(w: np.ndarray) -> float:
            return float(w @ cov_mat @ w)

        def gradient_fn(w: np.ndarray) -> np.ndarray:
            return 2.0 * cov_mat @ w

    elif objective == "max_sharpe":
        if expected_returns is None:
            raise ValueError("expected_returns is required for objective='max_sharpe'.")

        mu = np.asarray(expected_returns, dtype=float)
        if mu.ndim != 1 or mu.shape[0] != n:
            raise ValueError(
                f"expected_returns length ({mu.shape[0]}) does not match number "
                f"of assets ({n})."
            )

        rf = float(risk_free_rate)
        eps = 1e-12

        def objective_fn(w: np.ndarray) -> float:
            variance = float(w @ cov_mat @ w)
            variance = max(variance, eps)
            vol = float(np.sqrt(variance))
            excess_return = float(w @ mu) - rf
            return float(-excess_return / vol)

        def gradient_fn(w: np.ndarray) -> np.ndarray:
            cov_w = cov_mat @ w
            variance = float(w @ cov_w)
            variance = max(variance, eps)
            vol = float(np.sqrt(variance))
            excess_return = float(w @ mu) - rf
            return (-mu / vol) + (excess_return * cov_w) / (variance * vol)

    else:
        raise ValueError(
            f"Unknown objective '{objective}'. Use one of: {sorted(_VALID_OBJECTIVES)}"
        )

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
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if turnover_limit is not None and w_old is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, wo=w_old, lim=turnover_limit: lim - np.sum(np.abs(w - wo)),
            }
        )

    result: OptimizeResult = minimize(
        objective_fn,
        w0,
        jac=gradient_fn,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": 1000, "disp": False},
    )

    if not result.success:
        import logging

        logging.getLogger(__name__).warning(
            "Optimiser did not converge for %s: %s. Returning equal weights.",
            objective,
            result.message,
        )
        return np.full(n, 1.0 / n)

    w_opt = np.clip(result.x, 0.0, None)
    total = float(w_opt.sum())
    if total <= 0.0:
        return np.full(n, 1.0 / n)
    w_opt /= total
    return w_opt


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
    return _optimise_weights(
        cov,
        objective="min_variance",
        current_weights=current_weights,
        max_weight=max_weight,
        max_weights=max_weights,
        turnover_limit=turnover_limit,
        max_increase=max_increase,
        tol=tol,
    )


def maximize_sharpe(
    cov: pd.DataFrame,
    expected_returns: np.ndarray,
    current_weights: Optional[np.ndarray] = None,
    max_weight: float = 1.0,
    max_weights: Optional[np.ndarray] = None,
    turnover_limit: Optional[float] = None,
    max_increase: Optional[float] = None,
    risk_free_rate: float = 0.0,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve the maximum Sharpe-ratio optimisation problem.

    Parameters
    ----------
    cov:
        Annualised covariance matrix (tickers x tickers).
    expected_returns:
        Annualised expected returns aligned with ``cov.columns``.
    current_weights:
        Current portfolio weights used as the starting point and for optional
        turnover constraints.
    max_weight:
        Maximum weight for any single position.
    max_weights:
        Optional per-asset max weights aligned with ``cov.columns``.
    turnover_limit:
        Optional one-way turnover cap.
    max_increase:
        Optional cap on per-position increase versus current weight.
    risk_free_rate:
        Annual risk-free rate as a decimal (default ``0.0``).
    tol:
        Solver tolerance.

    Returns
    -------
    np.ndarray
        Optimal weight vector aligned with ``cov.columns``.
    """
    return _optimise_weights(
        cov,
        objective="max_sharpe",
        current_weights=current_weights,
        max_weight=max_weight,
        max_weights=max_weights,
        turnover_limit=turnover_limit,
        max_increase=max_increase,
        expected_returns=np.asarray(expected_returns, dtype=float),
        risk_free_rate=risk_free_rate,
        tol=tol,
    )


def rebalance(
    portfolio: pd.DataFrame,
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    max_weights: Optional[pd.Series | dict | np.ndarray] = None,
    turnover_limit: Optional[float] = None,
    max_increase: Optional[float] = None,
    objective: str = "max_sharpe",
    expected_returns: Optional[pd.Series | dict | np.ndarray] = None,
    risk_free_rate: float = 0.0,
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
    objective:
        Optimisation objective: ``"max_sharpe"`` (default) or
        ``"min_variance"``.
    expected_returns:
        Optional annual expected returns used by ``"max_sharpe"``. Can be:
        - ``np.ndarray`` aligned to ``portfolio`` rows,
        - ``dict`` keyed by ticker,
        - ``pd.Series`` indexed by ticker.
    risk_free_rate:
        Annual risk-free rate as a decimal. Used only for ``"max_sharpe"``.
    """
    objective_norm = objective.lower()
    if objective_norm not in _VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown objective '{objective}'. Use one of: {sorted(_VALID_OBJECTIVES)}"
        )

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

    expected_vec: Optional[np.ndarray] = None
    if expected_returns is not None:
        if isinstance(expected_returns, np.ndarray):
            expected_vec = expected_returns.astype(float)
        elif isinstance(expected_returns, dict):
            expected_series = pd.Series(expected_returns, dtype=float).reindex(tickers)
            if expected_series.isna().any():
                missing = sorted(expected_series[expected_series.isna()].index.tolist())
                raise ValueError(
                    "expected_returns is missing values for ticker(s): "
                    + ", ".join(missing)
                )
            expected_vec = expected_series.values
        elif isinstance(expected_returns, pd.Series):
            expected_series = expected_returns.astype(float).reindex(tickers)
            if expected_series.isna().any():
                missing = sorted(expected_series[expected_series.isna()].index.tolist())
                raise ValueError(
                    "expected_returns is missing values for ticker(s): "
                    + ", ".join(missing)
                )
            expected_vec = expected_series.values
        else:
            raise TypeError("expected_returns must be np.ndarray, dict, or pd.Series.")

    if objective_norm == "min_variance":
        optimal_w = minimize_variance(
            cov_aligned,
            current_weights=current_w,
            max_weight=max_weight,
            max_weights=per_asset_max,
            turnover_limit=turnover_limit,
            max_increase=max_increase,
        )
    else:
        if expected_vec is None:
            raise ValueError(
                "expected_returns must be provided when objective='max_sharpe'."
            )

        optimal_w = maximize_sharpe(
            cov_aligned,
            expected_returns=expected_vec,
            current_weights=current_w,
            max_weight=max_weight,
            max_weights=per_asset_max,
            turnover_limit=turnover_limit,
            max_increase=max_increase,
            risk_free_rate=risk_free_rate,
        )

    result = portfolio.copy()
    result["proposed_weight"] = optimal_w
    return result
