"""Tests for the CLI module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_rebalance.cli import _parse_holdings, _build_parser


# ---------------------------------------------------------------------------
# _parse_holdings
# ---------------------------------------------------------------------------


def test_parse_holdings_basic():
    result = _parse_holdings(["AAPL=0.5", "MSFT=0.5"])
    assert result == {"AAPL": 0.5, "MSFT": 0.5}


def test_parse_holdings_uppercase():
    result = _parse_holdings(["aapl=0.3"])
    assert "AAPL" in result


def test_parse_holdings_invalid_format():
    with pytest.raises(ValueError, match="format"):
        _parse_holdings(["AAPL0.5"])


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parser_csv_arg(tmp_path):
    p = tmp_path / "port.csv"
    p.write_text("ticker,weight\nAAPL,1.0\n")
    parser = _build_parser()
    args = parser.parse_args(["--csv", str(p)])
    assert args.csv == str(p)


def test_parser_holdings_arg():
    parser = _build_parser()
    args = parser.parse_args(["--holdings", "AAPL=0.6", "MSFT=0.4"])
    assert args.holdings == ["AAPL=0.6", "MSFT=0.4"]


def test_parser_defaults():
    parser = _build_parser()
    args = parser.parse_args(["--holdings", "AAPL=1.0"])
    assert args.period == "3y"
    assert args.full_history is False
    assert args.max_weight == 1.0
    assert args.turnover is None
    assert args.max_increase is None
    assert args.small_cap_threshold_b is None
    assert args.small_cap_max_weight == 0.05


def test_parser_full_history_flag():
    parser = _build_parser()
    args = parser.parse_args(["--holdings", "AAPL=1.0", "--full-history"])
    assert args.full_history is True


def test_parser_max_increase_arg():
    parser = _build_parser()
    args = parser.parse_args(["--holdings", "AAPL=1.0", "--max-increase", "0.05"])
    assert args.max_increase == 0.05


def test_parser_small_cap_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--holdings",
            "AAPL=1.0",
            "--small-cap-threshold-b",
            "10",
            "--small-cap-max-weight",
            "0.03",
        ]
    )
    assert args.small_cap_threshold_b == 10
    assert args.small_cap_max_weight == 0.03
