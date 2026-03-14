"""Data ingestion module.

Handles loading portfolio holdings and downloading historical price data.
Designed so the download backend can be swapped to a paid API with minimal changes.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TREASURY_TENOR_COLUMNS: dict[str, str] = {
    "1m": "1 Mo",
    "2m": "2 Mo",
    "3m": "3 Mo",
    "6m": "6 Mo",
    "1y": "1 Yr",
    "2y": "2 Yr",
    "5y": "5 Yr",
    "10y": "10 Yr",
    "30y": "30 Yr",
}

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CSV_FALLBACK_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

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


def download_market_caps(
    tickers: List[str],
    backend: str = "yfinance",
) -> pd.Series:
    """Download market capitalisation by ticker.

    Returns
    -------
    pd.Series
        Series indexed by ticker with market caps in USD. Missing/unavailable
        values are returned as ``NaN``.
    """
    if backend == "yfinance":
        return _download_market_caps_yfinance(tickers)
    raise NotImplementedError(f"Backend '{backend}' is not implemented.")


def download_treasury_risk_free_rate(
    tenor: str = "3m",
    year: int | None = None,
    timeout: float = 20.0,
) -> tuple[float, pd.Timestamp]:
    """Fetch latest U.S. Treasury yield for *tenor* as a decimal annual rate.

    Parameters
    ----------
    tenor:
        Tenor alias, one of: ``1m``, ``2m``, ``3m``, ``6m``, ``1y``, ``2y``,
        ``5y``, ``10y``, ``30y``.
    year:
        Treasury data year to query. When omitted, current year is tried first
        and previous year is used as a fallback (useful around year boundaries).
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    tuple[float, pd.Timestamp]
        ``(annual_rate_decimal, quote_date)`` where rate is in decimal form,
        e.g. ``0.0371`` for ``3.71%``.
    """
    tenor_key = tenor.strip().lower()
    if tenor_key not in TREASURY_TENOR_COLUMNS:
        supported = ", ".join(sorted(TREASURY_TENOR_COLUMNS))
        raise ValueError(f"Unsupported Treasury tenor '{tenor}'. Choose one of: {supported}.")

    target_column = TREASURY_TENOR_COLUMNS[tenor_key]
    current_year = year or datetime.utcnow().year
    years_to_try = [current_year] if year is not None else [current_year, current_year - 1]

    errors: list[str] = []
    for year_value in years_to_try:
        try:
            df = _download_treasury_yield_curve_csv(year_value, timeout=timeout)
            parsed = _extract_latest_treasury_yield(df, column_name=target_column)
            if parsed is not None:
                return parsed
            errors.append(f"{year_value}: no valid '{target_column}' observations")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Treasury risk-free download failed for %s: %s",
                year_value,
                exc,
            )
            errors.append(f"{year_value}: {exc}")

    detail = "; ".join(errors) if errors else "no response"
    raise ValueError(
        f"Unable to fetch Treasury risk-free rate for tenor '{tenor_key}' ({detail})."
    )


def download_sp500_tickers(
    source: str = "wikipedia",
    timeout: float = 20.0,
) -> list[str]:
    """Return the current S&P 500 ticker universe.

    Parameters
    ----------
    source:
        Data source for the constituent list. Currently only ``"wikipedia"``
        is implemented.
    timeout:
        HTTP timeout in seconds for online sources.
    """
    source_norm = source.strip().lower()
    if source_norm == "wikipedia":
        errors: list[str] = []
        for label, loader in [
            ("wikipedia", _download_sp500_tickers_wikipedia),
            ("csv-fallback", _download_sp500_tickers_csv_fallback),
        ]:
            try:
                tickers = loader(timeout=timeout)
                if len(tickers) >= 400:
                    return tickers
                errors.append(f"{label}: unexpectedly small universe ({len(tickers)})")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{label}: {exc}")

        detail = "; ".join(errors) if errors else "no response"
        raise ValueError(
            "Unable to download S&P 500 constituents from online sources "
            f"({detail})."
        )
    raise NotImplementedError(f"S&P 500 source '{source}' is not implemented.")


def _download_sp500_tickers_wikipedia(timeout: float = 20.0) -> list[str]:
    """Download S&P 500 symbols from the Wikipedia constituents table."""
    payload = _download_text_url(SP500_WIKI_URL, timeout=timeout)

    tables = pd.read_html(io.StringIO(payload))
    if not tables:
        raise ValueError("No tables found on the S&P 500 Wikipedia page.")

    df = tables[0]
    if "Symbol" not in df.columns:
        raise ValueError("S&P 500 constituents table missing 'Symbol' column.")

    # yfinance expects BRK.B/BF.B-style symbols as BRK-B/BF-B.
    tickers = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    tickers = [t for t in tickers.tolist() if t]
    unique = list(dict.fromkeys(tickers))
    return unique


def _download_sp500_tickers_csv_fallback(timeout: float = 20.0) -> list[str]:
    """Fallback S&P 500 universe download from a public CSV mirror."""
    payload = _download_text_url(SP500_CSV_FALLBACK_URL, timeout=timeout)
    df = pd.read_csv(io.StringIO(payload))
    if "Symbol" not in df.columns:
        raise ValueError("S&P 500 fallback CSV missing 'Symbol' column.")

    tickers = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    return [t for t in tickers.tolist() if t]


def _download_text_url(url: str, timeout: float = 20.0) -> str:
    """Download UTF-8 text payload with browser-like headers."""
    req = Request(url, headers=DEFAULT_HTTP_HEADERS)
    with urlopen(req, timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8")


def _download_treasury_yield_curve_csv(year: int, timeout: float = 20.0) -> pd.DataFrame:
    """Download a year of Treasury daily yield-curve data as a DataFrame."""
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        f"daily-treasury-rates.csv/{year}/all"
        "?type=daily_treasury_yield_curve"
        f"&field_tdr_date_value={year}&page&_format=csv"
    )
    payload = _download_text_url(url, timeout=timeout)
    return pd.read_csv(io.StringIO(payload), engine="python")


def _extract_latest_treasury_yield(
    df: pd.DataFrame,
    column_name: str,
) -> tuple[float, pd.Timestamp] | None:
    """Return latest valid Treasury yield from *column_name* in decimal form."""
    if "Date" not in df.columns:
        raise ValueError("Treasury CSV missing 'Date' column.")
    if column_name not in df.columns:
        raise ValueError(f"Treasury CSV missing '{column_name}' column.")

    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], errors="coerce", format="%m/%d/%Y"),
            "yield_pct": pd.to_numeric(df[column_name], errors="coerce"),
        }
    ).dropna()
    if parsed.empty:
        return None

    latest = parsed.sort_values("date").iloc[-1]
    return float(latest["yield_pct"]) / 100.0, pd.Timestamp(latest["date"])


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


def _download_market_caps_yfinance(tickers: List[str]) -> pd.Series:
    """Download market caps via yfinance ticker metadata."""
    import yfinance as yf

    caps: dict[str, float] = {}
    for ticker in tickers:
        cap_value = np.nan
        try:
            tk = yf.Ticker(ticker)

            fast_info = getattr(tk, "fast_info", None)
            if fast_info is not None:
                cap_fast = fast_info.get("market_cap")
                if cap_fast is not None and cap_fast > 0:
                    cap_value = float(cap_fast)

            if np.isnan(cap_value):
                info = getattr(tk, "info", None)
                if isinstance(info, dict):
                    cap_info = info.get("marketCap")
                    if cap_info is not None and cap_info > 0:
                        cap_value = float(cap_info)
        except Exception:  # noqa: BLE001
            logger.warning("Could not fetch market cap for %s", ticker)

        caps[ticker] = cap_value

    return pd.Series(caps, dtype=float)


# ---------------------------------------------------------------------------
# Convenience: load portfolio + prices in one call
# ---------------------------------------------------------------------------


def load_data(
    portfolio: pd.DataFrame,
    period: str = "3y",
    backend: str = "yfinance",
    require_full_history: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(portfolio_df, prices_df)`` ready for the risk model.

    Filters the portfolio to only tickers for which price data could be
    downloaded.

    Parameters
    ----------
    require_full_history:
        When ``True``, keep only tickers with complete price history over the
        selected lookback period.  This preserves the full timeline in the
        multi-asset return matrix at the cost of dropping newer listings.
    """
    tickers = portfolio["ticker"].tolist()
    prices = download_prices(tickers, period=period, backend=backend)

    if prices.empty:
        raise ValueError("No price data downloaded for the provided tickers.")

    available = prices.columns.tolist()
    if not available:
        raise ValueError("No requested tickers had downloadable price data.")

    if require_full_history:
        full_history_mask = prices.notna().all(axis=0)
        full_history_tickers = prices.columns[full_history_mask].tolist()
        dropped = sorted(set(available) - set(full_history_tickers))
        if dropped:
            logger.warning(
                "Dropping %d ticker(s) without full %s history: %s",
                len(dropped),
                period,
                ", ".join(dropped),
            )
        prices = prices[full_history_tickers]
        available = full_history_tickers

    if not available:
        raise ValueError(
            "No tickers remain after applying history filters. "
            "Try a shorter period or disable full-history mode."
        )

    # Filter portfolio to available tickers and re-normalise
    portfolio_filt = portfolio[portfolio["ticker"].isin(available)].copy()
    if portfolio_filt.empty:
        raise ValueError("No portfolio tickers remain after data filtering.")

    portfolio_filt["weight"] = (
        portfolio_filt["weight"] / portfolio_filt["weight"].sum()
    )
    return portfolio_filt.reset_index(drop=True), prices
