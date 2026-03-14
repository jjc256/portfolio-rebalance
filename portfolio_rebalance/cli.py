"""Command-line interface for portfolio-rebalance.

Usage examples
--------------
# From a CSV file (ticker,weight columns)
portfolio-rebalance --csv sample_portfolio.csv

# Inline holdings (ticker=weight pairs)
portfolio-rebalance --holdings AAPL=0.4 MSFT=0.3 GOOGL=0.3

# With constraints
portfolio-rebalance --csv sample_portfolio.csv --max-weight 0.25 --turnover 0.5

# Keep speculative positions from scaling too quickly
portfolio-rebalance --csv sample_portfolio.csv --max-increase 0.05

# Softly discourage large weight moves from current holdings
portfolio-rebalance --csv sample_portfolio.csv --distance-penalty 5.0

# Explicit objective selection
portfolio-rebalance --csv sample_portfolio.csv --objective min_variance

# Sharpe objective with a 3 % annual risk-free rate
portfolio-rebalance --csv sample_portfolio.csv --risk-free-rate 0.03

# Use latest 3-month U.S. Treasury yield for risk-free rate
portfolio-rebalance --csv sample_portfolio.csv --risk-free-source treasury --risk-free-tenor 3m

# Robust expected return estimation for Sharpe objective
portfolio-rebalance --csv sample_portfolio.csv --mu-method shrinkage --mu-shrinkage 0.6

# Robustness test: random S&P 500 portfolios using current settings
portfolio-rebalance --csv sample_portfolio.csv --random-test-n 100 --random-test-size 15

# Cap sub-10B market-cap names at 3 % each
portfolio-rebalance --csv sample_portfolio.csv --small-cap-threshold-b 10 --small-cap-max-weight 0.03

# Compare against S&P 500 (default benchmark is ^GSPC)
portfolio-rebalance --csv sample_portfolio.csv --benchmark ^GSPC

# Use a longer lookback period
portfolio-rebalance --csv sample_portfolio.csv --period 5y
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RISK_FREE_TENORS = ["1m", "2m", "3m", "6m", "1y", "2y", "5y", "10y", "30y"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="portfolio-rebalance",
        description="Sharpe-maximizing portfolio rebalancer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--csv",
        metavar="FILE",
        help="Path to CSV file with 'ticker' and 'weight' (or 'shares') columns.",
    )
    input_group.add_argument(
        "--holdings",
        nargs="+",
        metavar="TICKER=VALUE",
        help="Inline holdings as TICKER=weight pairs, e.g. AAPL=0.4 MSFT=0.6",
    )

    # Data
    parser.add_argument(
        "--period",
        default="3y",
        help="Lookback period for historical data (default: 3y).",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help=(
            "Drop tickers without complete history over the selected lookback "
            "window so the return series spans the full period."
        ),
    )

    # Constraints
    parser.add_argument(
        "--max-weight",
        type=float,
        default=1.0,
        metavar="W",
        help="Maximum weight per position, e.g. 0.25 (default: 1.0 = unconstrained).",
    )
    parser.add_argument(
        "--turnover",
        type=float,
        default=None,
        metavar="T",
        help="Maximum one-way turnover as a fraction, e.g. 0.5 (default: unconstrained).",
    )
    parser.add_argument(
        "--max-increase",
        type=float,
        default=None,
        metavar="D",
        help=(
            "Maximum per-position increase versus current weight, e.g. 0.05 = "
            "a ticker can rise by at most +5 percentage points in one rebalance "
            "(default: unconstrained)."
        ),
    )
    parser.add_argument(
        "--distance-penalty",
        type=float,
        default=0.0,
        metavar="L",
        help=(
            "Soft penalty strength for moving away from current weights. "
            "Adds L * ||w_new - w_old||^2 to the objective (default: 0.0)."
        ),
    )
    parser.add_argument(
        "--small-cap-threshold-b",
        type=float,
        default=None,
        metavar="B",
        help=(
            "If set, tickers with market cap below this threshold (in billions "
            "USD) are capped using --small-cap-max-weight."
        ),
    )
    parser.add_argument(
        "--small-cap-max-weight",
        type=float,
        default=0.05,
        metavar="W",
        help=(
            "Max weight for tickers below --small-cap-threshold-b (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--cov-method",
        choices=["sample", "ledoit_wolf"],
        default="sample",
        help="Covariance estimation method (default: sample).",
    )
    parser.add_argument(
        "--mu-method",
        choices=["sample", "ewma", "shrinkage"],
        default="shrinkage",
        help=(
            "Expected-return estimator used by max_sharpe (default: shrinkage). "
            "sample=plain mean, ewma=recently weighted mean, "
            "shrinkage=mean shrinkage toward grand mean."
        ),
    )
    parser.add_argument(
        "--mu-shrinkage",
        type=float,
        default=0.5,
        metavar="S",
        help=(
            "Shrinkage intensity in [0, 1] for --mu-method=shrinkage "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--mu-ewm-span",
        type=int,
        default=60,
        metavar="N",
        help=(
            "EWMA span (trading days) for --mu-method=ewma (default: 60)."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=["max_sharpe", "min_variance"],
        default="max_sharpe",
        help=(
            "Optimisation objective (default: max_sharpe). "
            "Use min_variance to prioritise volatility reduction only."
        ),
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        metavar="R",
        help=(
            "Annual risk-free rate used by max_sharpe (decimal form, e.g. 0.03). "
            "Default: 0.0"
        ),
    )
    parser.add_argument(
        "--risk-free-source",
        choices=["manual", "treasury"],
        default="manual",
        help=(
            "Source for risk-free rate when objective=max_sharpe. "
            "'manual' uses --risk-free-rate; 'treasury' fetches latest "
            "U.S. Treasury yield online (default: manual)."
        ),
    )
    parser.add_argument(
        "--risk-free-tenor",
        choices=RISK_FREE_TENORS,
        default="3m",
        help=(
            "Treasury tenor used when --risk-free-source=treasury "
            f"(default: 3m; choices: {', '.join(RISK_FREE_TENORS)})."
        ),
    )
    parser.add_argument(
        "--eval-frac",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Hold out the last fraction of data as an out-of-sample evaluation window "
            "(e.g. 0.2 = last 20%%). When omitted, performance stats are shown "
            "in-sample (same data used for covariance estimation and optimisation)."
        ),
    )
    parser.add_argument(
        "--benchmark",
        default="^GSPC",
        metavar="TICKER",
        help=(
            "Benchmark ticker for comparison in stats/charts "
            "(default: ^GSPC, S&P 500 index). Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--random-test-n",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run robustness test on N random S&P 500 portfolios using the same "
            "optimization settings and report how often Sharpe improves "
            "(default: 0 = disabled)."
        ),
    )
    parser.add_argument(
        "--random-test-size",
        type=int,
        default=15,
        metavar="K",
        help=(
            "Number of stocks in each random S&P 500 portfolio for --random-test-n "
            "(default: 15)."
        ),
    )
    parser.add_argument(
        "--random-test-seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for reproducible random S&P 500 test portfolios.",
    )

    # Output
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show debug logging."
    )

    return parser


def _parse_holdings(args_holdings: list[str]) -> dict:
    holdings = {}
    for item in args_holdings:
        if "=" not in item:
            raise ValueError(f"Invalid holding format '{item}'. Expected TICKER=value.")
        ticker, value = item.split("=", 1)
        holdings[ticker.strip().upper()] = float(value)
    return holdings


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not 0.0 < args.max_weight <= 1.0:
        print("ERROR: --max-weight must be in (0, 1].", file=sys.stderr)
        return 1
    if args.turnover is not None and not 0.0 <= args.turnover <= 2.0:
        print("ERROR: --turnover must be between 0 and 2.", file=sys.stderr)
        return 1
    if args.max_increase is not None and not 0.0 <= args.max_increase <= 1.0:
        print("ERROR: --max-increase must be between 0 and 1.", file=sys.stderr)
        return 1
    if args.distance_penalty < 0.0:
        print("ERROR: --distance-penalty must be non-negative.", file=sys.stderr)
        return 1
    if args.small_cap_threshold_b is not None and args.small_cap_threshold_b <= 0.0:
        print("ERROR: --small-cap-threshold-b must be > 0.", file=sys.stderr)
        return 1
    if not 0.0 < args.small_cap_max_weight <= 1.0:
        print("ERROR: --small-cap-max-weight must be in (0, 1].", file=sys.stderr)
        return 1
    if not -1.0 < args.risk_free_rate < 1.0:
        print("ERROR: --risk-free-rate must be in (-1, 1).", file=sys.stderr)
        return 1
    if not 0.0 <= args.mu_shrinkage <= 1.0:
        print("ERROR: --mu-shrinkage must be between 0 and 1.", file=sys.stderr)
        return 1
    if args.mu_ewm_span <= 1:
        print("ERROR: --mu-ewm-span must be > 1.", file=sys.stderr)
        return 1
    if args.random_test_n < 0:
        print("ERROR: --random-test-n must be >= 0.", file=sys.stderr)
        return 1
    if args.random_test_size < 2:
        print("ERROR: --random-test-size must be >= 2.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 1. Load portfolio
    # ------------------------------------------------------------------
    from portfolio_rebalance.data import (
        load_portfolio_csv,
        load_portfolio_dict,
        load_data,
        download_market_caps,
        download_prices,
        download_treasury_risk_free_rate,
    )
    from portfolio_rebalance.risk import (
        compute_returns,
        compute_covariance,
        estimate_expected_returns,
        portfolio_volatility,
        split_returns,
    )
    from portfolio_rebalance.optimizer import rebalance
    from portfolio_rebalance.reporting import weights_table, stats_summary

    if args.csv:
        portfolio = load_portfolio_csv(args.csv)
    else:
        holdings = _parse_holdings(args.holdings)
        portfolio = load_portfolio_dict(holdings)

    print(f"\nPortfolio loaded: {len(portfolio)} tickers")

    # ------------------------------------------------------------------
    # 2. Download price data
    # ------------------------------------------------------------------
    print(f"Downloading historical prices (period={args.period})…")
    try:
        input_tickers = portfolio["ticker"].tolist()
        portfolio, prices = load_data(
            portfolio,
            period=args.period,
            require_full_history=args.full_history,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to download price data: {exc}", file=sys.stderr)
        return 1

    if prices.empty or len(prices.columns) < 2:
        print("ERROR: Not enough price data downloaded.", file=sys.stderr)
        return 1

    kept_tickers = set(portfolio["ticker"].tolist())
    dropped_tickers = sorted(set(input_tickers) - kept_tickers)
    if dropped_tickers:
        reason = "missing full-period history" if args.full_history else "missing price data"
        print(
            f"Dropped {len(dropped_tickers)} ticker(s) ({reason}): "
            f"{', '.join(dropped_tickers)}"
        )

    # ------------------------------------------------------------------
    # 3. Risk model
    # ------------------------------------------------------------------
    returns = compute_returns(prices)

    first_valid = prices.apply(lambda s: s.first_valid_index())
    latest_start = first_valid.max()
    first_price_date = prices.index.min()
    if latest_start > first_price_date:
        drivers = sorted(first_valid[first_valid == latest_start].index.tolist())
        print(
            "Effective common history starts later "
            f"({returns.index.min().date()}) because newer ticker data begins on "
            f"{latest_start.date()} for: {', '.join(drivers)}"
        )
    else:
        print(
            "Return history window: "
            f"{returns.index.min().date()} to {returns.index.max().date()} "
            f"({len(returns)} trading days)"
        )

    # Optionally split into estimation / evaluation windows
    eval_returns: pd.DataFrame | None = None
    if args.eval_frac is not None:
        if not 0.0 < args.eval_frac < 1.0:
            print("ERROR: --eval-frac must be between 0 and 1 (exclusive).", file=sys.stderr)
            return 1
        est_returns, eval_returns = split_returns(returns, eval_frac=args.eval_frac)
        n_est = len(est_returns)
        n_oos = len(eval_returns)
        print(
            f"Data split: {n_est} days estimation / {n_oos} days OOS "
            f"(eval-frac={args.eval_frac:.0%})"
        )
    else:
        est_returns = returns
        print(
            "Note: performance stats are in-sample — the same data used to fit "
            "the covariance matrix and run the optimiser.  Use --eval-frac to "
            "hold out an out-of-sample evaluation window."
        )

    cov = compute_covariance(est_returns, annualise=True, method=args.cov_method)
    expected_returns = estimate_expected_returns(
        est_returns,
        annualise=True,
        method=args.mu_method,
        shrinkage=args.mu_shrinkage,
        ewm_span=args.mu_ewm_span,
    )

    objective_label = (
        "Max Sharpe" if args.objective == "max_sharpe" else "Minimum Variance"
    )

    effective_risk_free_rate = args.risk_free_rate
    if args.objective == "max_sharpe" and args.risk_free_source == "treasury":
        try:
            effective_risk_free_rate, quote_date = download_treasury_risk_free_rate(
                tenor=args.risk_free_tenor
            )
            print(
                "Fetched Treasury risk-free rate "
                f"({args.risk_free_tenor}): {effective_risk_free_rate:.2%} "
                f"as of {quote_date.date()}"
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "Warning: Treasury risk-free fetch failed "
                f"({exc}). Using --risk-free-rate={args.risk_free_rate:.2%}."
            )

    if args.objective == "max_sharpe":
        print(
            f"Objective: {objective_label} (risk-free rate={effective_risk_free_rate:.2%})"
        )
        print(
            "Expected return estimator: "
            f"{args.mu_method} "
            f"(shrinkage={args.mu_shrinkage:.2f}, ewm-span={args.mu_ewm_span})"
        )
    else:
        print(f"Objective: {objective_label}")

    benchmark_returns: pd.Series | None = None
    benchmark_eval_returns: pd.Series | None = None
    benchmark_label = "S&P 500"
    benchmark_ticker = (args.benchmark or "").strip()
    if benchmark_ticker and benchmark_ticker.lower() != "none":
        try:
            bench_prices = download_prices([benchmark_ticker], period=args.period)
            if not bench_prices.empty:
                bench_ret_df = compute_returns(bench_prices)
                bench_series = bench_ret_df.iloc[:, 0]
                benchmark_label = (
                    "S&P 500" if benchmark_ticker.upper() == "^GSPC" else benchmark_ticker
                )

                benchmark_returns = bench_series.reindex(est_returns.index).dropna()
                if eval_returns is not None:
                    benchmark_eval_returns = bench_series.reindex(eval_returns.index).dropna()

                if benchmark_returns.empty:
                    print(
                        f"Warning: benchmark '{benchmark_ticker}' has no overlap with "
                        "portfolio estimation dates; skipping benchmark comparison."
                    )
                    benchmark_returns = None
                    benchmark_eval_returns = None
                else:
                    print(
                        f"Benchmark comparison enabled: {benchmark_ticker} "
                        f"({len(benchmark_returns)} overlapping estimation days)"
                    )
            else:
                print(f"Warning: no benchmark prices for '{benchmark_ticker}'.")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: benchmark download failed for '{benchmark_ticker}': {exc}")

    # Align portfolio to covariance matrix
    portfolio = portfolio[portfolio["ticker"].isin(cov.columns)].reset_index(drop=True)

    # Optional: tighten caps for smaller-cap names (size-risk proxy)
    per_asset_max: pd.Series | None = None
    if args.small_cap_threshold_b is not None:
        market_caps = download_market_caps(portfolio["ticker"].tolist())
        cap_threshold_usd = float(args.small_cap_threshold_b) * 1e9

        per_asset_max = pd.Series(args.max_weight, index=portfolio["ticker"], dtype=float)
        small_mask = market_caps < cap_threshold_usd
        small_tickers = sorted(market_caps[small_mask].index.tolist())
        if small_tickers:
            cap_for_small = min(args.max_weight, args.small_cap_max_weight)
            per_asset_max.loc[small_tickers] = cap_for_small
            print(
                f"Applied small-cap cap: {len(small_tickers)} ticker(s) below "
                f"${args.small_cap_threshold_b:.1f}B limited to {cap_for_small:.2%} max"
            )

        missing_caps = sorted(market_caps[market_caps.isna()].index.tolist())
        if missing_caps:
            print(
                "Warning: market cap unavailable for "
                f"{len(missing_caps)} ticker(s), using global max-weight: "
                f"{', '.join(missing_caps)}"
            )

    # ------------------------------------------------------------------
    # 4. Optimise
    # ------------------------------------------------------------------
    try:
        rebalanced = rebalance(
            portfolio,
            cov,
            max_weight=args.max_weight,
            max_weights=per_asset_max,
            turnover_limit=args.turnover,
            max_increase=args.max_increase,
            distance_penalty=args.distance_penalty,
            objective=args.objective,
            expected_returns=expected_returns,
            risk_free_rate=effective_risk_free_rate,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    current_w = rebalanced["weight"].values
    proposed_w = rebalanced["proposed_weight"].values

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("WEIGHT COMPARISON")
    print("=" * 60)
    tbl = weights_table(rebalanced)
    print(tbl.to_string(index=False))

    print("\n" + "=" * 60)
    print("PERFORMANCE STATISTICS")
    print("=" * 60)
    summary = stats_summary(
        current_w,
        proposed_w,
        est_returns,
        cov,
        eval_returns=eval_returns,
        benchmark_returns=benchmark_returns,
        benchmark_eval_returns=benchmark_eval_returns,
        benchmark_label=benchmark_label,
    )
    print(summary.to_string())

    current_vol = portfolio_volatility(current_w, cov)
    proposed_vol = portfolio_volatility(proposed_w, cov)
    vol_change = proposed_vol - current_vol
    direction = "↓" if vol_change < 0 else "↑"
    print(
        f"\nVolatility change: {current_vol:.2%} → {proposed_vol:.2%} "
        f"({direction}{abs(vol_change):.2%})"
    )

    if args.random_test_n > 0:
        from portfolio_rebalance.validation import run_random_sp500_sharpe_test

        print("\n" + "=" * 60)
        print("RANDOM S&P 500 ROBUSTNESS TEST")
        print("=" * 60)
        print(
            "Running "
            f"{args.random_test_n} random portfolios (size={args.random_test_size}, "
            f"seed={args.random_test_seed})..."
        )

        test_result = run_random_sp500_sharpe_test(
            n_portfolios=args.random_test_n,
            portfolio_size=args.random_test_size,
            period=args.period,
            require_full_history=args.full_history,
            cov_method=args.cov_method,
            objective=args.objective,
            mu_method=args.mu_method,
            mu_shrinkage=args.mu_shrinkage,
            mu_ewm_span=args.mu_ewm_span,
            max_weight=args.max_weight,
            turnover_limit=args.turnover,
            max_increase=args.max_increase,
            distance_penalty=args.distance_penalty,
            risk_free_rate=effective_risk_free_rate,
            eval_frac=args.eval_frac,
            small_cap_threshold_b=args.small_cap_threshold_b,
            small_cap_max_weight=args.small_cap_max_weight,
            seed=args.random_test_seed,
        )

        print(
            f"Completed: {test_result.completed}/{test_result.attempted} "
            f"({test_result.skipped} skipped)"
        )
        if test_result.completed == 0:
            print("No valid random portfolios completed. Try shorter period or smaller size.")
        else:
            print(
                f"Sharpe improved in: {test_result.improved}/{test_result.completed} "
                f"({test_result.improvement_rate:.1%})"
            )
            print(
                "Sharpe delta (proposed - current): "
                f"mean={test_result.avg_sharpe_delta:+.3f}, "
                f"median={test_result.median_sharpe_delta:+.3f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
