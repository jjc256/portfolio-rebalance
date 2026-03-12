"""Tests for data ingestion helpers."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_rebalance.data import load_portfolio_csv, load_portfolio_dict


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
