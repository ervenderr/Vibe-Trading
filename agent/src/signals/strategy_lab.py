"""Single-asset strategy backtester with fees, slippage and drawdown. Research only; places no orders.

Execution is deliberately conservative: a signal uses data through bar t, the order fills at the close
of bar t+1, and the position first earns the return of bar t+2. Costs are charged on the fill bar,
as basis points of the traded notional, once per side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EXECUTION_LAG_BARS = 2
SECONDS_PER_YEAR = 365.25 * 86400


@dataclass(frozen=True)
class Costs:
    fee_bps: float = 5.0        # exchange or broker fee per side
    slippage_bps: float = 5.0   # price moved against us per side

    @property
    def per_side(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 1e4


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    drawdown: pd.Series
    trades: tuple[dict, ...]
    metrics: dict
    benchmark: dict
    halves: dict


# ---------- signals: 1 = hold the asset, 0 = stay in cash ----------

def sma_cross_signal(close: pd.Series, fast: int, slow: int) -> pd.Series:
    if fast >= slow:
        raise ValueError("fast average must be shorter than the slow average")
    long = close.rolling(fast).mean() > close.rolling(slow).mean()
    return long.astype(int)


def rsi_reversion_signal(close: pd.Series, period: int = 14, low: float = 30.0, high: float = 70.0) -> pd.Series:
    """Buy when RSI drops below ``low``; hold until it rises above ``high``."""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rsi = 100 - 100 / (1 + up / down.replace(0, np.nan))
    rsi = rsi.where(down != 0, 100.0)
    state, out = 0, []
    for value in rsi:
        if not np.isnan(value):
            state = 1 if value < low else 0 if value > high else state
        out.append(state)
    return pd.Series(out, index=close.index)


def breakout_signal(close: pd.Series, window: int = 20) -> pd.Series:
    """Donchian channel: buy a close above the prior ``window`` high, sell below the prior half-window low."""
    prior_high = close.shift(1).rolling(window).max()
    prior_low = close.shift(1).rolling(max(2, window // 2)).min()
    state, out = 0, []
    for c, hi, lo in zip(close, prior_high, prior_low):
        if not np.isnan(hi) and c > hi:
            state = 1
        elif not np.isnan(lo) and c < lo:
            state = 0
        out.append(state)
    return pd.Series(out, index=close.index)


# ---------- engine ----------

def _stats(returns: pd.Series) -> dict:
    if len(returns) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    equity = (1.0 + returns).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    seconds = (returns.index[-1] - returns.index[0]).total_seconds()
    years = max(seconds / SECONDS_PER_YEAR, 1e-9)
    per_year = len(returns) / years
    std = returns.std(ddof=1)
    return {
        "total_return": total,
        "cagr": float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 and years >= 1 / 12 else total,
        "sharpe": float(returns.mean() / std * np.sqrt(per_year)) if std > 0 else 0.0,
        "max_drawdown": float((1.0 - equity / equity.cummax()).max()),
    }


def _trades(close: pd.Series, signal: pd.Series, costs: Costs) -> tuple[dict, ...]:
    entries = [i for i in range(len(signal) - 1) if signal.iloc[i] == 1 and (i == 0 or signal.iloc[i - 1] == 0)]
    trades = []
    for i in entries:
        fill = i + 1
        exit_signal = next((j for j in range(i + 1, len(signal)) if signal.iloc[j] == 0), None)
        exit_fill = exit_signal + 1 if exit_signal is not None and exit_signal + 1 < len(signal) else None
        last = len(signal) - 1
        end = exit_fill if exit_fill is not None else last
        if end <= fill and exit_fill is None:
            continue
        entry_px, exit_px = float(close.iloc[fill]), float(close.iloc[end])
        sides = 2 if exit_fill is not None else 1
        trades.append({
            "entry_time": int(close.index[fill].timestamp()), "exit_time": int(close.index[end].timestamp()),
            "entry_price": entry_px, "exit_price": exit_px, "bars": end - fill,
            "return": exit_px / entry_px - 1.0 - sides * costs.per_side, "open": exit_fill is None,
        })
    return tuple(trades)


def backtest(close: pd.Series, signal: pd.Series, costs: Costs) -> BacktestResult:
    """Run one strategy. Returns a new result object; the inputs are untouched."""
    close = close.astype(float)
    signal = signal.reindex(close.index).fillna(0).astype(int)
    position = signal.shift(EXECUTION_LAG_BARS, fill_value=0)
    asset_returns = close.pct_change().fillna(0.0)
    traded = signal.diff().abs().fillna(signal.abs()).shift(1, fill_value=0)   # charged on the fill bar
    returns = position * asset_returns - traded * costs.per_side
    equity = (1.0 + returns).cumprod()
    drawdown = 1.0 - equity / equity.cummax()

    trades = _trades(close, signal, costs)
    wins = [t["return"] for t in trades if t["return"] > 0]
    losses = [-t["return"] for t in trades if t["return"] < 0]
    metrics = {
        **_stats(returns), "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": (sum(wins) / sum(losses)) if losses else (float("inf") if wins else 0.0),
        "exposure": float((position != 0).mean()),
    }
    bench = asset_returns.copy()
    bench.iloc[0] = -costs.per_side                                            # buy-and-hold pays one entry
    mid = len(returns) // 2
    return BacktestResult(
        returns=returns, equity=equity, drawdown=drawdown, trades=trades, metrics=metrics,
        benchmark=_stats(bench), halves={"first": _stats(returns.iloc[:mid]), "second": _stats(returns.iloc[mid:])},
    )
