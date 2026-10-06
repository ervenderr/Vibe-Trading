"""Paper-trading track record from the broker's real fills, plus the promotion gate.

Trades are reconstructed by FIFO-matching fills, so exits from stop/target legs and
from an end-of-day flatten are all captured the same way. The promotion criteria are
fixed here, in code, before any results exist: they cannot be tuned after the fact.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from src.signals.event_study import sign_flip_pvalue

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Fill:
    time: datetime
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    price: float
    tag: str = ""  # "flat": an end-of-day/flatten close, which takes the position to zero


@dataclass(frozen=True)
class ClosedTrade:
    symbol: str
    side: str  # "long" | "short"
    qty: float
    entry: float
    exit: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float


@dataclass(frozen=True)
class TrackRecord:
    trades: int
    wins: int
    win_rate: float
    total_pnl: float
    expectancy_pct: float  # mean return per trade (fraction)
    profit_factor: float
    max_drawdown: float  # fraction of peak equity
    p_value: float  # sign-flip test that the mean return is zero
    trading_days: int


@dataclass(frozen=True)
class PromotionCriteria:
    """Fixed up front. Meeting them makes an account *eligible*, nothing more."""

    min_trades: int = 100
    min_days: int = 20
    min_profit_factor: float = 1.3
    max_drawdown: float = 0.05
    max_p_value: float = 0.05


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Promotion:
    eligible: bool
    checks: tuple[Check, ...]
    note: str


def pair_fills(
    fills: Iterable[Fill], *, allow_short: bool = True, intraday_only: bool = False
) -> tuple[ClosedTrade, ...]:
    """Match opposite-side fills per symbol into closed trades.

    * Plain exits are matched FIFO. A scaled exit (one fill closing several entry lots) is
      ONE trade with the quantity-weighted entry.
    * A ``flat`` fill takes the position to zero: it closes the NEWEST lots first and
      discards any older lots, which can only be stale (an entry whose exit was never
      captured) and must not be paired with a later, unrelated exit.
    * ``intraday_only`` expires lots from an earlier New York session.
    * ``allow_short=False`` ignores a sell that has no long to close instead of inventing
      a phantom short.
    """
    lots: dict[str, deque[list]] = defaultdict(deque)  # symbol -> [signed_qty, price, time]
    trades: list[ClosedTrade] = []
    usable = [f for f in fills if f.qty > 0 and f.price > 0 and f.side in ("buy", "sell")]
    for fill in sorted(usable, key=lambda f: f.time):
        sign = 1.0 if fill.side == "buy" else -1.0
        remaining = fill.qty
        book = lots[fill.symbol]
        if intraday_only:
            session = fill.time.astimezone(NY).date()
            while book and book[0][2].astimezone(NY).date() < session:
                book.popleft()
        flat = fill.tag == "flat"
        while remaining > 1e-9 and book and book[-1 if flat else 0][0] * sign < 0:
            lot = book[-1 if flat else 0]
            matched = min(remaining, abs(lot[0]))
            long_trade = lot[0] > 0
            entry, exit_ = lot[1], fill.price
            pnl = (exit_ - entry) * matched if long_trade else (entry - exit_) * matched
            trades.append(ClosedTrade(
                fill.symbol, "long" if long_trade else "short", matched, entry, exit_,
                lot[2], fill.time, pnl, pnl / (entry * matched),
            ))
            lot[0] += sign * matched
            remaining -= matched
            if abs(lot[0]) < 1e-9:
                book.pop() if flat else book.popleft()
        if flat:
            book.clear()  # position is zero now: anything older is stale
        if remaining > 1e-9 and not flat and (allow_short or sign > 0):
            book.append([sign * remaining, fill.price, fill.time])
    return tuple(sorted(_merge_exits(trades), key=lambda t: t.exit_time))


def _merge_exits(trades: Sequence[ClosedTrade]) -> list[ClosedTrade]:
    """Combine the lot pieces that one exit fill produced into a single trade."""
    groups: dict[tuple, list[ClosedTrade]] = defaultdict(list)
    for t in trades:
        groups[(t.symbol, t.side, t.exit_time, t.exit)].append(t)
    merged = []
    for pieces in groups.values():
        qty = sum(t.qty for t in pieces)
        entry = sum(t.entry * t.qty for t in pieces) / qty
        pnl = sum(t.pnl for t in pieces)
        first = pieces[0]
        merged.append(ClosedTrade(first.symbol, first.side, qty, entry, first.exit,
                                  min(t.entry_time for t in pieces), first.exit_time, pnl, pnl / (entry * qty)))
    return merged


def summarize_trades(trades: Sequence[ClosedTrade], start_equity: float, *, seed: int = 0) -> TrackRecord:
    if not trades:
        return TrackRecord(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0)
    pnls = [t.pnl for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
    equity = peak = start_equity
    drawdown = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak if peak > 0 else 0.0)
    returns = np.array([t.return_pct for t in trades])
    return TrackRecord(
        trades=len(trades), wins=wins, win_rate=wins / len(trades), total_pnl=sum(pnls),
        expectancy_pct=float(returns.mean()), profit_factor=profit_factor, max_drawdown=drawdown,
        p_value=sign_flip_pvalue(returns, seed=seed),
        trading_days=len({t.exit_time.astimezone(NY).date() for t in trades}),
    )


def evaluate_promotion(rec: TrackRecord, criteria: PromotionCriteria | None = None) -> Promotion:
    c = criteria or PromotionCriteria()
    checks = (
        Check("closed trades", rec.trades >= c.min_trades, f"{rec.trades} (need {c.min_trades})"),
        Check("trading days", rec.trading_days >= c.min_days, f"{rec.trading_days} (need {c.min_days})"),
        Check("net profit", rec.total_pnl > 0, f"${rec.total_pnl:,.2f}"),
        Check("profit factor", rec.profit_factor >= c.min_profit_factor, f"{rec.profit_factor:.2f} (need {c.min_profit_factor})"),
        Check("max drawdown", rec.max_drawdown <= c.max_drawdown and rec.trades > 0, f"{rec.max_drawdown:.1%} (limit {c.max_drawdown:.0%})"),
        Check("statistical significance", rec.p_value <= c.max_p_value and rec.trades > 0, f"p={rec.p_value:.3f} (need <= {c.max_p_value})"),
    )
    eligible = all(ch.ok for ch in checks)
    note = ("Eligibility only: going live still requires you to commit a mandate manually. "
            "Passing paper trading does not guarantee live profits.")
    return Promotion(eligible, checks, note)
