"""portfolio_rebalance – top-level package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("portfolio-rebalance")
except PackageNotFoundError:
    __version__ = "0.1.0"
