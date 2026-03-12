"""Data ingestion module.

Handles loading portfolio holdings and downloading historical price data.
Designed so the download backend can be swapped to a paid API with minimal changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Portfolio loading helpers
# ---------------------------------------------------------------------------


def load_portfolio_csv(path: str | Path) -> pd.DataFrame:
    """Load a portfolio from a CSV file with columns: ticker, weight (or shares).

    Returns a DataFrame with columns ``ticker`` and ``weight`` (normalised to
    sum to 1).  If a ``shares`` column is present instead of ``weight`` the
    weights are derived from equal unit prices (relative shares count).
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    if "ticker" not in df.columns:
        raise ValueError("CSV must contain a 'ticker' column.")

    df["ticker"] = df["ticker"].str.strip().str.upper()

    if "weight" in df.columns:
        weights = df["weight"].astype(float)
    elif "shares" in df.columns:
        shares = df["shares"].astype(float)
        weights = shares / shares.sum()
    else:
        raise ValueError("CSV must contain a 'weight' or 'shares' column.")

    total = weights.sum()
    if total <= 0:
        raise ValueError("Weights/shares must sum to a positive number.")
    weights = weights / total

    return pd.DataFrame({"ticker": df["ticker"], "weight": weights})


def load_portfolio_dict(holdings: Dict[str, float]) -> pd.DataFrame:
    """Create a portfolio DataFrame from a mapping of ticker → weight/shares."""
    tickers = [t.upper() for t in holdings.keys()]
    values = list(holdings.values())
    total = sum(values)
    if total <= 0:
        raise ValueError("Values must sum to a positive number.")
    weights = [v / total for v in values]
    return pd.DataFrame({"ticker": tickers, "weight": weights})


# ---------------------------------------------------------------------------
# Historical price download
# ---------------------------------------------------------------------------


def download_prices(
    tickers: List[str],
    period: str = "3y",
    interval: str = "1d",
    backend: str = "yfinance",
) -> pd.DataFrame:
    """Download adjusted close prices for *tickers*.

    Parameters
    ----------
    tickers:
        List of ticker symbols.
    period:
        Lookback period understood by yfinance (e.g. ``"1y"``, ``"3y"``).
    interval:
        Bar interval (default ``"1d"``).
    backend:
        Currently only ``"yfinance"`` is supported.  Set to a custom value and
        monkey-patch :func:`_download_custom` to integrate a paid data source.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by date with one column per ticker containing
        adjusted close prices.  Columns are aligned and forward-filled.
    """
    if backend == "yfinance":
        return _download_yfinance(tickers, period=period, interval=interval)
    raise NotImplementedError(f"Backend '{backend}' is not implemented.")


def _download_yfinance(
    tickers: List[str],
    period: str = "3y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download via yfinance."""
    import yfinance as yf

    logger.info("Downloading %d tickers via yfinance (period=%s)…", len(tickers), period)
    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker – columns are flat
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.ffill().dropna(how="all")
    # Keep only the tickers we asked for (some may not download)
    available = [t for t in tickers if t in prices.columns]
    missing = set(tickers) - set(available)
    if missing:
        logger.warning("Could not download data for: %s", missing)
    return prices[available]


# ---------------------------------------------------------------------------
# Convenience: load portfolio + prices in one call
# ---------------------------------------------------------------------------


def load_data(
    portfolio: pd.DataFrame,
    period: str = "3y",
    backend: str = "yfinance",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(portfolio_df, prices_df)`` ready for the risk model.

    Filters the portfolio to only tickers for which price data could be
    downloaded.
    """
    tickers = portfolio["ticker"].tolist()
    prices = download_prices(tickers, period=period, backend=backend)

    # Filter portfolio to available tickers and re-normalise
    available = prices.columns.tolist()
    portfolio_filt = portfolio[portfolio["ticker"].isin(available)].copy()
    portfolio_filt["weight"] = (
        portfolio_filt["weight"] / portfolio_filt["weight"].sum()
    )
    return portfolio_filt.reset_index(drop=True), prices
