"""Command-line interface for portfolio-rebalance.

Usage examples
--------------
# From a CSV file (ticker,weight columns)
portfolio-rebalance --csv sample_portfolio.csv

# Inline holdings (ticker=weight pairs)
portfolio-rebalance --holdings AAPL=0.4 MSFT=0.3 GOOGL=0.3

# With constraints
portfolio-rebalance --csv sample_portfolio.csv --max-weight 0.25 --turnover 0.5

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="portfolio-rebalance",
        description="Minimum-variance portfolio rebalancer",
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
        "--cov-method",
        choices=["sample", "ledoit_wolf"],
        default="sample",
        help="Covariance estimation method (default: sample).",
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

    # ------------------------------------------------------------------
    # 1. Load portfolio
    # ------------------------------------------------------------------
    from portfolio_rebalance.data import load_portfolio_csv, load_portfolio_dict, load_data
    from portfolio_rebalance.risk import (
        compute_returns,
        compute_covariance,
        portfolio_volatility,
        portfolio_stats,
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
        portfolio, prices = load_data(portfolio, period=args.period)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to download price data: {exc}", file=sys.stderr)
        return 1

    if prices.empty or len(prices.columns) < 2:
        print("ERROR: Not enough price data downloaded.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 3. Risk model
    # ------------------------------------------------------------------
    returns = compute_returns(prices)

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

    # Align portfolio to covariance matrix
    portfolio = portfolio[portfolio["ticker"].isin(cov.columns)].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Optimise
    # ------------------------------------------------------------------
    rebalanced = rebalance(
        portfolio,
        cov,
        max_weight=args.max_weight,
        turnover_limit=args.turnover,
    )

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
    summary = stats_summary(current_w, proposed_w, est_returns, cov, eval_returns=eval_returns)
    print(summary.to_string())

    current_vol = portfolio_volatility(current_w, cov)
    proposed_vol = portfolio_volatility(proposed_w, cov)
    vol_change = proposed_vol - current_vol
    direction = "↓" if vol_change < 0 else "↑"
    print(
        f"\nVolatility change: {current_vol:.2%} → {proposed_vol:.2%} "
        f"({direction}{abs(vol_change):.2%})"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
