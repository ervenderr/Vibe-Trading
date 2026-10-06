"""No-lookahead replay of the verdict engine on intraday bars (paper trading).

This is a simulation for evaluating signals. It never places orders.

Realism rules: a verdict is computed only from information known at a bar's
close, entry is at the *next* bar's open, a bar that touches both stop and
target counts as a stop, costs and slippage are charged on both sides, and
positions are flattened at the end of the day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Sequence

import pandas as pd

from src.signals.technicals import MIN_BARS, TechnicalState, state_at, technical_frame
from src.signals.verdict import Action, Verdict

VerdictFn = Callable[[TechnicalState, datetime], Verdict]
BPS = 1e4


@dataclass(frozen=True)
class ReplayConfig:
    bar_minutes: int = 5
    cost_bps: float = 2.0  # fee per side
    slippage_bps: float = 1.0  # per side
    max_hold_bars: int = 12
    cooldown_bars: int = 3


@dataclass(frozen=True)
class Trade:
    side: str  # "BUY" (long) or "SELL" (short)
    entry_time: datetime
    exit_time: datetime
    entry: float
    exit: float
    exit_reason: str  # target | stop | time | eod
    net_return: float  # fractional, after costs (0.01 = +1%)
    bars_held: int
    news_score: float


@dataclass(frozen=True)
class Stats:
    trades: int
    win_rate: float
    total_net_return: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    avg_bars_held: float


@dataclass(frozen=True)
class _Open:
    side: float
    entry: float
    stop: float
    target: float
    entry_index: int
    entry_time: datetime
    news_score: float


def _same_day(a: pd.Timestamp, b: pd.Timestamp) -> bool:
    return a.date() == b.date()


def _fill(price: float, side: float, slip: float, *, entering: bool) -> float:
    direction = side if entering else -side
    return price * (1.0 + direction * slip)


def _close_trade(pos: _Open, exit_price: float, i: int, ts: datetime, reason: str, cfg: ReplayConfig) -> Trade:
    slip = cfg.slippage_bps / BPS
    entry_fill = pos.entry
    exit_fill = _fill(exit_price, pos.side, slip, entering=False)
    net = pos.side * (exit_fill - entry_fill) / entry_fill - 2.0 * cfg.cost_bps / BPS
    return Trade(
        side="BUY" if pos.side > 0 else "SELL",
        entry_time=pos.entry_time,
        exit_time=ts,
        entry=entry_fill,
        exit=exit_fill,
        exit_reason=reason,
        net_return=net,
        bars_held=i - pos.entry_index + 1,
        news_score=pos.news_score,
    )


def _check_exit(
    pos: _Open, bars: pd.DataFrame, i: int, cfg: ReplayConfig
) -> tuple[float, str] | None:
    open_, high, low, close = (float(bars[c].iloc[i]) for c in ("open", "high", "low", "close"))
    hit_stop = low <= pos.stop if pos.side > 0 else high >= pos.stop
    hit_target = high >= pos.target if pos.side > 0 else low <= pos.target
    if hit_stop:  # conservative: stop first when both are touched in one bar
        # A gap through the stop fills at the (worse) open, not at the stop level.
        return (min(open_, pos.stop) if pos.side > 0 else max(open_, pos.stop)), "stop"
    if hit_target:
        return (max(open_, pos.target) if pos.side > 0 else min(open_, pos.target)), "target"
    last_of_day = i == len(bars) - 1 or not _same_day(bars.index[i], bars.index[i + 1])
    if last_of_day:
        return close, "eod"
    if i - pos.entry_index + 1 >= cfg.max_hold_bars:
        return close, "time"
    return None


def simulate(
    bars: pd.DataFrame, verdict_fn: VerdictFn, config: ReplayConfig | None = None
) -> tuple[Trade, ...]:
    """Replay ``bars`` (tz-aware DatetimeIndex, OHLCV) and return completed trades."""
    cfg = config or ReplayConfig()
    n = len(bars)
    if n <= MIN_BARS:
        return ()
    frame = technical_frame(bars)
    slip = cfg.slippage_bps / BPS
    trades: list[Trade] = []
    position: _Open | None = None
    pending: tuple[float, float, float, float] | None = None  # side, stop_d, target_d, news
    next_allowed = 0

    for i in range(MIN_BARS - 1, n):
        ts = bars.index[i]
        if pending is not None and position is None:
            side, stop_d, target_d, news = pending
            entry = _fill(float(bars["open"].iloc[i]), side, slip, entering=True)
            position = _Open(side, entry, entry - side * stop_d, entry + side * target_d, i, ts, news)
        pending = None

        if position is not None:
            outcome = _check_exit(position, bars, i, cfg)
            if outcome is not None:
                trades.append(_close_trade(position, outcome[0], i, ts, outcome[1], cfg))
                position = None
                next_allowed = i + cfg.cooldown_bars
            continue

        if i < next_allowed or i >= n - 1 or not _same_day(ts, bars.index[i + 1]):
            continue
        state = state_at(frame, i)
        if state is None:
            continue
        verdict = verdict_fn(state, ts + timedelta(minutes=cfg.bar_minutes))
        if verdict.action in (Action.BUY, Action.SELL) and verdict.entry is not None:
            side = 1.0 if verdict.action is Action.BUY else -1.0
            pending = (side, abs(verdict.entry - verdict.stop), abs(verdict.target - verdict.entry), verdict.news_score)
    return tuple(trades)


def summarize(trades: Sequence[Trade]) -> Stats:
    """Aggregate performance statistics over completed trades."""
    if not trades:
        return Stats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    returns = [t.net_return for t in trades]
    wins = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    equity = peak = drawdown = 0.0
    for r in returns:
        equity += r
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    if losses > 0:
        profit_factor = wins / losses
    else:
        profit_factor = float("inf") if wins > 0 else 0.0
    return Stats(
        trades=len(trades),
        win_rate=sum(1 for r in returns if r > 0) / len(returns),
        total_net_return=sum(returns),
        expectancy=sum(returns) / len(returns),
        profit_factor=profit_factor,
        max_drawdown=drawdown,
        avg_bars_held=sum(t.bars_held for t in trades) / len(trades),
    )
