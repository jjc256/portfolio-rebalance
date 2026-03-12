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

# Cap sub-10B market-cap names at 3 % each
portfolio-rebalance --csv sample_portfolio.csv --small-cap-threshold-b 10 --small-cap-max-weight 0.03

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

    if not 0.0 < args.max_weight <= 1.0:
        print("ERROR: --max-weight must be in (0, 1].", file=sys.stderr)
        return 1
    if args.turnover is not None and not 0.0 <= args.turnover <= 2.0:
        print("ERROR: --turnover must be between 0 and 2.", file=sys.stderr)
        return 1
    if args.max_increase is not None and not 0.0 <= args.max_increase <= 1.0:
        print("ERROR: --max-increase must be between 0 and 1.", file=sys.stderr)
        return 1
    if args.small_cap_threshold_b is not None and args.small_cap_threshold_b <= 0.0:
        print("ERROR: --small-cap-threshold-b must be > 0.", file=sys.stderr)
        return 1
    if not 0.0 < args.small_cap_max_weight <= 1.0:
        print("ERROR: --small-cap-max-weight must be in (0, 1].", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 1. Load portfolio
    # ------------------------------------------------------------------
    from portfolio_rebalance.data import (
        load_portfolio_csv,
        load_portfolio_dict,
        load_data,
        download_market_caps,
    )
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
