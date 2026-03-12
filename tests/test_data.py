"""Tests for data ingestion helpers."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_rebalance.data import (
    load_portfolio_csv,
    load_portfolio_dict,
    load_data,
    download_market_caps,
    download_treasury_risk_free_rate,
)


# ---------------------------------------------------------------------------
# load_portfolio_dict
# ---------------------------------------------------------------------------


def test_load_portfolio_dict_normalises():
    port = load_portfolio_dict({"AAPL": 2.0, "MSFT": 2.0, "GOOGL": 1.0})
    assert len(port) == 3
    assert abs(port["weight"].sum() - 1.0) < 1e-9


def test_load_portfolio_dict_tickers_uppercase():
    port = load_portfolio_dict({"aapl": 1.0, "msft": 1.0})
    assert list(port["ticker"]) == ["AAPL", "MSFT"]


def test_load_portfolio_dict_zero_raises():
    with pytest.raises(ValueError):
        load_portfolio_dict({"AAPL": 0.0, "MSFT": 0.0})


# ---------------------------------------------------------------------------
# load_portfolio_csv
# ---------------------------------------------------------------------------


def _make_csv(content: str, tmp_path: Path) -> Path:
    p = tmp_path / "portfolio.csv"
    p.write_text(textwrap.dedent(content))
    return p


def test_load_portfolio_csv_weights(tmp_path):
    csv = _make_csv(
        """\
        ticker,weight
        AAPL,0.5
        MSFT,0.5
        """,
        tmp_path,
    )
    port = load_portfolio_csv(csv)
    assert list(port["ticker"]) == ["AAPL", "MSFT"]
    assert abs(port["weight"].sum() - 1.0) < 1e-9


def test_load_portfolio_csv_shares(tmp_path):
    csv = _make_csv(
        """\
        ticker,shares
        AAPL,100
        MSFT,200
        """,
        tmp_path,
    )
    port = load_portfolio_csv(csv)
    assert abs(port["weight"].sum() - 1.0) < 1e-9
    # MSFT should have twice the weight of AAPL
    aapl_w = port.loc[port["ticker"] == "AAPL", "weight"].iloc[0]
    msft_w = port.loc[port["ticker"] == "MSFT", "weight"].iloc[0]
    assert abs(msft_w / aapl_w - 2.0) < 1e-9


def test_load_portfolio_csv_no_ticker_col(tmp_path):
    csv = _make_csv("symbol,weight\nAAPL,1.0\n", tmp_path)
    with pytest.raises(ValueError, match="ticker"):
        load_portfolio_csv(csv)


def test_load_portfolio_csv_no_weight_col(tmp_path):
    csv = _make_csv("ticker,qty\nAAPL,10\n", tmp_path)
    with pytest.raises(ValueError, match="weight"):
        load_portfolio_csv(csv)


def test_load_portfolio_csv_normalises_unequal_weights(tmp_path):
    csv = _make_csv(
        """\
        ticker,weight
        AAPL,0.3
        MSFT,0.3
        GOOGL,0.3
        """,
        tmp_path,
    )
    port = load_portfolio_csv(csv)
    assert abs(port["weight"].sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


def test_load_data_require_full_history_drops_late_tickers(monkeypatch):
    portfolio = pd.DataFrame(
        {
            "ticker": ["AAPL", "NEW"],
            "weight": [0.6, 0.4],
        }
    )
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 102.0, 103.0],
            "NEW": [np.nan, 50.0, 51.0, 52.0],
        },
        index=idx,
    )

    monkeypatch.setattr("portfolio_rebalance.data.download_prices", lambda *args, **kwargs: prices)

    filtered_portfolio, filtered_prices = load_data(
        portfolio,
        period="3y",
        require_full_history=True,
    )

    assert list(filtered_prices.columns) == ["AAPL"]
    assert list(filtered_portfolio["ticker"]) == ["AAPL"]
    assert filtered_portfolio["weight"].iloc[0] == pytest.approx(1.0)


def test_load_data_require_full_history_raises_if_empty(monkeypatch):
    portfolio = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "weight": [0.5, 0.5],
        }
    )
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame(
        {
            "A": [np.nan, 1.0, 1.1],
            "B": [np.nan, 2.0, 2.1],
        },
        index=idx,
    )

    monkeypatch.setattr("portfolio_rebalance.data.download_prices", lambda *args, **kwargs: prices)

    with pytest.raises(ValueError, match="No tickers remain"):
        load_data(portfolio, require_full_history=True)


def test_download_market_caps_dispatch(monkeypatch):
    expected = pd.Series({"AAPL": 1.0, "MSFT": 2.0}, dtype=float)
    monkeypatch.setattr(
        "portfolio_rebalance.data._download_market_caps_yfinance",
        lambda tickers: expected,
    )
    out = download_market_caps(["AAPL", "MSFT"])
    pd.testing.assert_series_equal(out, expected)


def test_download_market_caps_unknown_backend_raises():
    with pytest.raises(NotImplementedError):
        download_market_caps(["AAPL"], backend="dummy")


def test_download_treasury_risk_free_rate_latest(monkeypatch):
    df = pd.DataFrame(
        {
            "Date": ["03/10/2026", "03/11/2026"],
            "3 Mo": [3.70, 3.71],
        }
    )
    monkeypatch.setattr(
        "portfolio_rebalance.data._download_treasury_yield_curve_csv",
        lambda year, timeout=10.0: df,
    )

    rate, quote_date = download_treasury_risk_free_rate(tenor="3m", year=2026)

    assert rate == pytest.approx(0.0371)
    assert quote_date == pd.Timestamp("2026-03-11")


def test_download_treasury_risk_free_rate_invalid_tenor():
    with pytest.raises(ValueError, match="Unsupported Treasury tenor"):
        download_treasury_risk_free_rate(tenor="9m", year=2026)


def test_download_treasury_risk_free_rate_missing_column(monkeypatch):
    df = pd.DataFrame(
        {
            "Date": ["03/11/2026"],
            "10 Yr": [4.21],
        }
    )
    monkeypatch.setattr(
        "portfolio_rebalance.data._download_treasury_yield_curve_csv",
        lambda year, timeout=10.0: df,
    )

    with pytest.raises(ValueError, match="Unable to fetch Treasury risk-free rate"):
        download_treasury_risk_free_rate(tenor="3m", year=2026)
