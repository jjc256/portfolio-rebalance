# portfolio-rebalance

A Python toolkit for **minimum-variance portfolio rebalancing**.  
Given a set of stocks and their current weights, it downloads historical prices, fits a risk model, and solves a constrained optimisation to reduce portfolio volatility — without changing the investable universe.

---

## Features

| Module | What it does |
|---|---|
| `data` | Load holdings from CSV / dict; download adjusted prices via yfinance (swap-able to any paid API) |
| `risk` | Compute log/simple returns, sample (or Ledoit-Wolf shrinkage) covariance matrix, annualised stats |
| `optimizer` | Long-only, budget-constrained minimum-variance optimisation (SLSQP); optional max-position, turnover, per-position increase, and market-cap-aware limits |
| `reporting` | Weights table, performance stats summary, Plotly charts, benchmark comparison |
| `ui/dashboard` | Streamlit web app with interactive controls |
| `cli` | Argparse-based CLI |

---

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
# or install the package in editable mode:
pip install -e .
```

### Run the CLI

```bash
# Using the bundled sample portfolio (tickers + weights)
portfolio-rebalance --csv sample_portfolio.csv

# Inline holdings
portfolio-rebalance --holdings AAPL=0.4 MSFT=0.3 GOOGL=0.3

# With constraints: max 25 % per stock, max 50 % one-way turnover
portfolio-rebalance --csv sample_portfolio.csv --max-weight 0.25 --turnover 0.5

# Keep small speculative positions from scaling too quickly
portfolio-rebalance --csv sample_portfolio.csv --max-increase 0.05

# Cap smaller-cap names (below $10B) at 3% each
portfolio-rebalance --csv sample_portfolio.csv --small-cap-threshold-b 10 --small-cap-max-weight 0.03

# 5-year lookback, Ledoit-Wolf covariance shrinkage
portfolio-rebalance --csv sample_portfolio.csv --period 5y --cov-method ledoit_wolf

# Out-of-sample evaluation: hold out last 20% of history for testing
portfolio-rebalance --csv sample_portfolio.csv --eval-frac 0.2

# Benchmark comparison (default is S&P 500 via ^GSPC)
portfolio-rebalance --csv sample_portfolio.csv --benchmark ^GSPC

# OOS + constraints
portfolio-rebalance --csv sample_portfolio.csv --max-weight 0.25 --turnover 0.5 --eval-frac 0.2
```

### Test out-of-sample (OOS) performance

By default, summary stats are **in-sample** (the same return history is used to fit
the covariance matrix and evaluate performance). This can be optimistic.

Use `--eval-frac` to split returns into:
- An estimation window (used for covariance estimation + optimisation)
- A held-out OOS window (used only for performance evaluation)

Example:

```bash
portfolio-rebalance --csv sample_portfolio.csv --eval-frac 0.2
```

This uses the first ~80% of observations to build the model and the last ~20%
to report OOS return, Sharpe, drawdown, and realised volatility.

### Launch the web dashboard

```bash
streamlit run portfolio_rebalance/ui/dashboard.py
```

Open `http://localhost:8501` in your browser.

The sidebar lets you:
- Upload a CSV, use the bundled sample portfolio, or enter holdings manually
- Choose a lookback period (1 y – 5 y)
- Set a max-position limit
- Cap how much each ticker can increase versus its current weight
- Add a market-cap-aware cap for smaller companies
- Compare against a benchmark (default S&P 500)
- Enable/disable a one-way turnover constraint
- Enable out-of-sample evaluation and choose a held-out evaluation window

After clicking **Run Optimisation** the dashboard shows:
- KPI metrics (current vs proposed volatility, reduction %, turnover)
- Side-by-side weight bar chart and table
- Performance statistics (annualised return, Sharpe, max drawdown; in-sample or OOS)
- Cumulative return chart
- Asset correlation heatmap

---

## Sample portfolio

`sample_portfolio.csv` contains 8 US large-caps as a demo input:

```
ticker,weight
AAPL,0.20
MSFT,0.20
GOOGL,0.15
AMZN,0.15
NVDA,0.10
JPM,0.08
JNJ,0.07
XOM,0.05
```

Weights need not sum to 1 — they are normalised automatically.  
A `shares` column is also accepted and converted to weights.

---

## Project structure

```
portfolio_rebalance/
├── __init__.py
├── cli.py              ← CLI entry point
├── data/
│   └── __init__.py     ← load_portfolio_csv / dict, download_prices
├── risk/
│   └── __init__.py     ← compute_returns, compute_covariance, stats
├── optimizer/
│   └── __init__.py     ← minimize_variance, rebalance
├── reporting/
│   └── __init__.py     ← tables and Plotly figures
└── ui/
    └── dashboard.py    ← Streamlit app
tests/
├── test_data.py
├── test_risk.py
├── test_optimizer.py
├── test_reporting.py
└── test_cli.py
sample_portfolio.csv
requirements.txt
pyproject.toml
```

---

## Swapping to a paid data source

The data backend is isolated in `portfolio_rebalance/data/__init__.py`.  
Add a new `elif backend == "my_vendor":` branch to `download_prices()` and call it with `--backend my_vendor` or by passing `backend="my_vendor"` in Python.

---

## Running tests

```bash
pytest
```

To run only out-of-sample split tests:

```bash
pytest tests/test_risk.py -k split_returns
```

The test suite covers data loading, risk calculations (including estimation/OOS splitting), the optimiser (including the analytical 2-asset solution), and report generation.
