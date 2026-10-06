"""Pure planning: decide whether a verdict becomes an order, and how big.

No I/O. Every rejection carries a reason so the operator can see why a signal
did not trade. Anything uncertain or inconsistent is a skip, never a guess.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from typing import Mapping, Sequence

from src.execution.models import (
    AccountState,
    DayState,
    ExecutionLimits,
    ExecutionPlan,
    PlanDecision,
)
from src.signals.news_source import is_valid_symbol
from src.signals.service import SignalReport
from src.signals.verdict import Action

ID_WINDOW_SECONDS = 900  # one order per symbol/side per 15-minute window
_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def client_order_id(symbol: str, side: str, at: datetime) -> str:
    """Deterministic id: retries inside a window reuse it, so the broker rejects duplicates."""
    bucket = int(at.timestamp() // ID_WINDOW_SECONDS)
    clean = _ID_SAFE.sub("", symbol.upper())
    digest = hashlib.sha1(symbol.upper().encode()).hexdigest()[:6]  # keeps long symbols distinct
    return f"vt-{clean[:10]}{digest}-{side[:1]}-{bucket}"


def position_size(
    equity: float,
    entry: float,
    stop: float,
    risk_fraction: float,
    max_notional: float,
    buying_power: float,
    buying_power_fraction: float = 0.5,
) -> int:
    """Whole shares: risk budget over stop distance, capped by notional and buying power."""
    distance = abs(entry - stop)
    if not (entry > 0 and distance > 0 and equity > 0):
        return 0
    by_risk = equity * risk_fraction / distance
    by_notional = max_notional / entry
    by_buying_power = buying_power * buying_power_fraction / entry
    shares = math.floor(min(by_risk, by_notional, by_buying_power))
    return max(shares, 0)


def _skip(*reasons: str) -> PlanDecision:
    return PlanDecision("skipped", None, tuple(reasons))


def _num(value: object) -> bool:
    """A real, finite number (bools and numeric strings are not accepted)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _levels_ok(side: str, entry: float, stop: float, target: float) -> bool:
    if side == "buy":
        return stop < entry < target
    return target < entry < stop


def plan_execution(
    report: SignalReport,
    account: AccountState,
    positions: Sequence[Mapping],
    open_orders: Sequence[Mapping],
    day: DayState,
    limits: ExecutionLimits,
    now: datetime,
    live_price: float | None = None,
) -> PlanDecision:
    """Turn a verdict into an :class:`ExecutionPlan`, or explain why not.

    ``live_price`` (when given) re-anchors the entry: levels keep their distances but
    shift to the live price, and a market that moved too far from the signal is refused.
    """
    v = report.verdict
    if v.action not in (Action.BUY, Action.SELL):
        return _skip(f"verdict is {v.action.value}, nothing to trade")
    if report.stale:
        return _skip("price data is stale")
    if now > v.valid_until:
        return _skip("signal expired")
    if not _num(v.confidence) or not v.confidence >= limits.min_confidence:
        return _skip(f"confidence {v.confidence!r} below minimum {limits.min_confidence:.2f}")
    if not all(_num(x) for x in (v.entry, v.stop, v.target)):
        return _skip("signal entry, stop or target is missing or not a finite number")
    if not (_num(account.equity) and account.equity > 0 and _num(account.buying_power) and account.buying_power >= 0):
        return _skip("account equity or buying power is not a usable number")
    if not is_valid_symbol(report.symbol):
        return _skip("invalid symbol")

    side = "buy" if v.action is Action.BUY else "sell"
    if side == "sell" and not limits.allow_shorts:
        return _skip("short selling is disabled")
    if account.trading_blocked:
        return _skip("account trading is blocked")

    symbol = report.symbol.upper()
    if any(str(p.get("symbol", "")).upper() == symbol for p in positions):
        return _skip(f"already holding {symbol}")
    if any(str(o.get("symbol", "")).upper() == symbol for o in open_orders):
        return _skip(f"open order already exists for {symbol}")
    held = {str(p.get("symbol", "")).upper() for p in positions}
    pending = {str(o.get("symbol", "")).upper() for o in open_orders}
    if len(held | pending) >= limits.max_open_positions:
        return _skip(f"max open positions ({limits.max_open_positions}) reached")
    if day.trades >= limits.max_trades_per_day:
        return _skip(f"max trades per day ({limits.max_trades_per_day}) reached")
    if day.start_equity > 0 and (day.start_equity - account.equity) / day.start_equity >= limits.daily_loss_limit:
        return _skip("daily loss limit reached; no new trades today")

    entry, stop, target = float(v.entry), float(v.stop), float(v.target)
    if live_price is not None:
        if not (_num(live_price) and live_price > 0):
            return _skip("live price is not a usable number")
        if abs(live_price / entry - 1.0) > limits.max_price_deviation:
            return _skip(f"price moved {live_price / entry - 1.0:+.1%} since the signal; not chasing")
        delta = live_price - entry
        entry, stop, target = live_price, stop + delta, target + delta
    entry, stop, target = round(entry, 2), round(stop, 2), round(target, 2)
    if entry < limits.min_price:
        return _skip(f"price {entry:.2f} below minimum {limits.min_price:.2f}")
    if not _levels_ok(side, entry, stop, target):
        return _skip("entry/stop/target are inconsistent for this side")

    qty = position_size(
        account.equity, entry, stop, limits.risk_per_trade, limits.max_order_notional,
        account.buying_power, limits.buying_power_fraction,
    )
    if qty < 1:
        return _skip("position size is zero under the risk and notional limits")

    plan = ExecutionPlan(
        symbol=symbol, side=side, qty=qty, entry=entry, stop=stop, target=target,
        client_order_id=client_order_id(symbol, side, now), confidence=v.confidence,
    )
    risk = qty * abs(entry - stop)
    return PlanDecision("planned", plan, (f"{side} {qty} {symbol} @ ~{entry:.2f}, risking ${risk:,.0f}",))
