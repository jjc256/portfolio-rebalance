"""Validation helpers for robustness testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from portfolio_rebalance.data import download_market_caps, download_prices, download_sp500_tickers
from portfolio_rebalance.optimizer import rebalance
from portfolio_rebalance.risk import (
    compute_covariance,
    compute_returns,
    estimate_expected_returns,
    portfolio_stats,
    portfolio_stats_realized,
    split_returns,
)


@dataclass
class RandomSharpeTestResult:
    """Summary of random S&P 500 portfolio robustness testing."""

    attempted: int
    completed: int
    improved: int
    skipped: int
    improvement_rate: float
    avg_sharpe_delta: float
    median_sharpe_delta: float
    trials: list["RandomSharpeTestTrial"]


@dataclass
class RandomSharpeTestTrial:
    """Per-portfolio random robustness test output for UI inspection."""

    trial_index: int
    tickers: list[str]
    mode: str
    est_days: int
    eval_days: int
    current_sharpe: float
    proposed_sharpe: float
    sharpe_delta: float
    improved: bool
    turnover: float
    current_weights: list[float]
    proposed_weights: list[float]
    dates: list[str]
    cum_current: list[float]
    cum_proposed: list[float]
    cum_benchmark: list[float] | None
    benchmark_label: str | None


def _rebased_cum(ret_arr: np.ndarray) -> np.ndarray:
    """Cumulative return path rebased to 1.0 at the first available date."""
    if ret_arr.size == 0:
        return ret_arr
    cum = np.cumprod(1.0 + ret_arr)
    first = float(cum[0])
    return cum / first if first != 0.0 else cum


def run_random_sp500_sharpe_test(
    n_portfolios: int,
    portfolio_size: int = 15,
    period: str = "3y",
    require_full_history: bool = False,
    cov_method: str = "sample",
    objective: str = "max_sharpe",
    mu_method: str = "shrinkage",
    mu_shrinkage: float = 0.5,
    mu_ewm_span: int = 60,
    max_weight: float = 1.0,
    turnover_limit: float | None = None,
    max_increase: float | None = None,
    distance_penalty: float = 0.0,
    risk_free_rate: float = 0.0,
    eval_frac: float | None = None,
    small_cap_threshold_b: float | None = None,
    small_cap_max_weight: float = 0.05,
    seed: int = 42,
    benchmark_ticker: str = "^GSPC",
    include_trials: bool = False,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> RandomSharpeTestResult:
    """Evaluate how often optimisation improves Sharpe on random S&P 500 portfolios.

    The same optimisation settings are reused for each sampled portfolio.
    """
    if n_portfolios <= 0:
        raise ValueError("n_portfolios must be > 0.")
    if portfolio_size < 2:
        raise ValueError("portfolio_size must be >= 2.")

    universe = download_sp500_tickers()
    if len(universe) < portfolio_size:
        raise ValueError(
            f"S&P 500 universe too small ({len(universe)}) for portfolio_size={portfolio_size}."
        )

    rng = np.random.default_rng(seed)
    sampled_tickers: list[list[str]] = []
    sampled_weights: list[np.ndarray] = []
    for _ in range(n_portfolios):
        tickers = rng.choice(universe, size=portfolio_size, replace=False).tolist()
        w = rng.random(portfolio_size)
        w = w / float(w.sum())
        sampled_tickers.append(tickers)
        sampled_weights.append(w)

    all_tickers = sorted({t for batch in sampled_tickers for t in batch})
    all_prices = download_prices(all_tickers, period=period)
    if all_prices.empty:
        raise ValueError("No price data available for sampled S&P 500 portfolios.")

    benchmark_label = "S&P 500" if benchmark_ticker.upper() == "^GSPC" else benchmark_ticker
    benchmark_returns: pd.Series | None = None
    if include_trials and benchmark_ticker.strip().lower() != "none":
        try:
            bench_prices = download_prices([benchmark_ticker], period=period)
            if not bench_prices.empty:
                bench_ret_df = compute_returns(bench_prices)
                benchmark_returns = bench_ret_df.iloc[:, 0]
        except Exception:  # noqa: BLE001
            benchmark_returns = None

    all_caps: pd.Series | None = None
    if small_cap_threshold_b is not None:
        all_caps = download_market_caps(all_tickers)

    completed = 0
    improved = 0
    sharpe_deltas: list[float] = []
    trials: list[RandomSharpeTestTrial] = []

    for trial_index, (tickers, weights) in enumerate(
        zip(sampled_tickers, sampled_weights),
        start=1,
    ):
        available_tickers = [t for t in tickers if t in all_prices.columns]
        if len(available_tickers) < 2:
            if progress_callback is not None:
                progress_callback(trial_index, n_portfolios, completed, improved)
            continue

        weight_map = {t: float(w) for t, w in zip(tickers, weights)}

        prices = all_prices[available_tickers].copy()
        if require_full_history:
            full_mask = prices.notna().all(axis=0)
            prices = prices.loc[:, full_mask]
            if prices.shape[1] < 2:
                continue

        port = pd.DataFrame(
            {
                "ticker": available_tickers,
                "weight": [weight_map[t] for t in available_tickers],
            }
        )
        port = port[port["ticker"].isin(prices.columns)].copy()
        if port.empty or len(port) < 2:
            if progress_callback is not None:
                progress_callback(trial_index, n_portfolios, completed, improved)
            continue
        port["weight"] = port["weight"] / port["weight"].sum()

        returns = compute_returns(prices)
        if len(returns) < 40:
            if progress_callback is not None:
                progress_callback(trial_index, n_portfolios, completed, improved)
            continue

        eval_returns: pd.DataFrame | None = None
        if eval_frac is not None:
            est_returns, eval_returns = split_returns(returns, eval_frac=eval_frac)
            if len(est_returns) < 20 or len(eval_returns) < 20:
                if progress_callback is not None:
                    progress_callback(trial_index, n_portfolios, completed, improved)
                continue
        else:
            est_returns = returns

        cov = compute_covariance(est_returns, annualise=True, method=cov_method)
        expected_returns = estimate_expected_returns(
            est_returns,
            annualise=True,
            method=mu_method,
            shrinkage=mu_shrinkage,
            ewm_span=mu_ewm_span,
        )

        per_asset_max: pd.Series | None = None
        if all_caps is not None and small_cap_threshold_b is not None:
            cap_threshold_usd = float(small_cap_threshold_b) * 1e9
            per_asset_max = pd.Series(max_weight, index=port["ticker"], dtype=float)
            caps_subset = all_caps.reindex(port["ticker"]).astype(float)
            small_mask = caps_subset < cap_threshold_usd
            if small_mask.any():
                cap_for_small = min(max_weight, small_cap_max_weight)
                per_asset_max.loc[small_mask.index[small_mask]] = cap_for_small

        try:
            rebalanced = rebalance(
                port,
                cov,
                max_weight=max_weight,
                max_weights=per_asset_max,
                turnover_limit=turnover_limit,
                max_increase=max_increase,
                distance_penalty=distance_penalty,
                objective=objective,
                expected_returns=expected_returns,
                risk_free_rate=risk_free_rate,
            )
        except ValueError:
            if progress_callback is not None:
                progress_callback(trial_index, n_portfolios, completed, improved)
            continue

        current_w = rebalanced["weight"].to_numpy(dtype=float)
        proposed_w = rebalanced["proposed_weight"].to_numpy(dtype=float)

        if eval_returns is not None:
            current_stats = portfolio_stats_realized(current_w, eval_returns)
            proposed_stats = portfolio_stats_realized(proposed_w, eval_returns)
        else:
            current_stats = portfolio_stats(current_w, est_returns, cov)
            proposed_stats = portfolio_stats(proposed_w, est_returns, cov)

        delta = float(proposed_stats["sharpe"] - current_stats["sharpe"])
        completed += 1
        if delta > 0.0:
            improved += 1
        sharpe_deltas.append(delta)

        if include_trials:
            score_returns = eval_returns if eval_returns is not None else est_returns
            score_mode = "OOS" if eval_returns is not None else "In-Sample"
            port_ret_current = score_returns.values @ current_w
            port_ret_proposed = score_returns.values @ proposed_w
            cum_current = _rebased_cum(port_ret_current)
            cum_proposed = _rebased_cum(port_ret_proposed)

            cum_benchmark: list[float] | None = None
            if benchmark_returns is not None:
                bench_aligned = benchmark_returns.reindex(score_returns.index)
                bench_valid = bench_aligned.dropna()
                if not bench_valid.empty:
                    bench_rebased = _rebased_cum(bench_valid.values)
                    bench_series = pd.Series(np.nan, index=score_returns.index, dtype=float)
                    bench_series.loc[bench_valid.index] = bench_rebased
                    cum_benchmark = bench_series.tolist()

            trials.append(
                RandomSharpeTestTrial(
                    trial_index=trial_index,
                    tickers=port["ticker"].tolist(),
                    mode=score_mode,
                    est_days=len(est_returns),
                    eval_days=len(score_returns),
                    current_sharpe=float(current_stats["sharpe"]),
                    proposed_sharpe=float(proposed_stats["sharpe"]),
                    sharpe_delta=delta,
                    improved=delta > 0.0,
                    turnover=float(np.sum(np.abs(proposed_w - current_w))),
                    current_weights=current_w.tolist(),
                    proposed_weights=proposed_w.tolist(),
                    dates=[d.strftime("%Y-%m-%d") for d in score_returns.index],
                    cum_current=cum_current.tolist(),
                    cum_proposed=cum_proposed.tolist(),
                    cum_benchmark=cum_benchmark,
                    benchmark_label=benchmark_label if cum_benchmark is not None else None,
                )
            )

        if progress_callback is not None:
            progress_callback(trial_index, n_portfolios, completed, improved)

    skipped = n_portfolios - completed
    if completed == 0:
        return RandomSharpeTestResult(
            attempted=n_portfolios,
            completed=0,
            improved=0,
            skipped=skipped,
            improvement_rate=0.0,
            avg_sharpe_delta=0.0,
            median_sharpe_delta=0.0,
            trials=trials,
        )

    return RandomSharpeTestResult(
        attempted=n_portfolios,
        completed=completed,
        improved=improved,
        skipped=skipped,
        improvement_rate=float(improved / completed),
        avg_sharpe_delta=float(np.mean(sharpe_deltas)),
        median_sharpe_delta=float(np.median(sharpe_deltas)),
        trials=trials,
    )
