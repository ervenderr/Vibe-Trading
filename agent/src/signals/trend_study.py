"""Monthly trend and momentum backtests on daily closes. Research only; places no orders.

Rules are fixed in scripts/PREREGISTRATION-trend-momentum.md. Signals use the month-end close and
trade at the next day's close, so a position earns returns from the second day after the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EXECUTION_LAG_DAYS = 2  # signal at close t, trade at close t+1, first return earned on day t+2
BLOCK_DAYS = 21


def _month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby([index.year, index.month]).last().values)


def _equal(mask: pd.DataFrame) -> pd.DataFrame:
    count = mask.sum(axis=1).replace(0, np.nan)
    return mask.astype(float).div(count, axis=0).fillna(0.0)


def trend_weights(close: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Equal weight in every asset above its moving average, at month-ends (cash when none)."""
    above = (close > close.rolling(window).mean()) & close.notna()
    return _equal(above).loc[_month_ends(close.index)]


def momentum_weights(close: pd.DataFrame, lookback: int = 252, skip: int = 21, top: int = 3) -> pd.DataFrame:
    """Top ``top`` assets by return from ``lookback`` to ``skip`` days ago, only if that return is positive."""
    score = close.shift(skip) / close.shift(lookback) - 1.0
    ranked = score.rank(axis=1, ascending=False, method="first")
    picked = (ranked <= top) & (score > 0)
    return _equal(picked).loc[_month_ends(close.index)]


def equal_weights(close: pd.DataFrame) -> pd.DataFrame:
    """Benchmark: equal weight in whatever has a price, rebalanced monthly."""
    return _equal(close.notna()).loc[_month_ends(close.index)]


def run_backtest(close: pd.DataFrame, month_end_weights: pd.DataFrame, *, cost_bps: float) -> pd.Series:
    """Daily net returns of the portfolio. Returns a new Series; inputs are untouched."""
    weights = (month_end_weights.reindex(close.index).ffill().shift(EXECUTION_LAG_DAYS).fillna(0.0)
               .where(close.notna(), 0.0))
    returns = close.pct_change().fillna(0.0)
    gross = (weights * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    return gross - turnover * cost_bps / 1e4


def cagr(returns: pd.Series) -> float:
    years = len(returns) / TRADING_DAYS
    return float((1.0 + returns).prod() ** (1.0 / years) - 1.0) if years > 0 else 0.0


def sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=1)
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    return float((1.0 - equity / equity.cummax()).max())


def bootstrap_sharpe_lower(returns: pd.Series, *, level: float = 0.975, n_boot: int = 2000, seed: int = 0) -> float:
    """Lower end of a circular block-bootstrap interval for the Sharpe ratio."""
    values, n = returns.to_numpy(), len(returns)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, -(-n // BLOCK_DAYS)))
    idx = ((starts[:, :, None] + np.arange(BLOCK_DAYS)) % n).reshape(n_boot, -1)[:, :n]
    sample = values[idx]
    std = sample.std(axis=1, ddof=1)
    sharpes = np.where(std > 0, sample.mean(axis=1) / np.where(std > 0, std, 1.0) * np.sqrt(TRADING_DAYS), 0.0)
    return float(np.quantile(sharpes, 1.0 - level))
