"""ORB bot: pure rules plus a simulated session with a fake venue."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.execution.orb_bot import Position, breakout, opening_range, pnl, range_ok, run_day, stopped

OPEN = datetime(2026, 10, 26, 13, 30, tzinfo=timezone.utc)
CLOSE = OPEN + timedelta(hours=6, minutes=30)


def bar(minute: int, o, h, l, c):
    return {"time": OPEN + timedelta(minutes=minute), "open": o, "high": h, "low": l, "close": c}


def test_rules():
    assert opening_range([bar(0, 100, 101, 99.5, 100.5), bar(5, 100.5, 100.8, 99.9, 100)]) == (101, 99.5)
    assert range_ok(100.5, 100.0) and not range_ok(100.05, 100.0) and not range_ok(103, 100)
    assert breakout(101, 99, 101.1) == "long" and breakout(101, 99, 98.9) == "short" and breakout(101, 99, 100) is None
    assert stopped(Position("long", 1, 100, 99), 98.9) and not stopped(Position("long", 1, 100, 99), 99.5)
    assert stopped(Position("short", 1, 100, 101), 101.1)
    assert pnl("short", 100, 99, 10) == 10 and pnl("long", 100, 99, 10) == -10


class Clock:
    def __init__(self, start):
        self.t = start

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += timedelta(minutes=5)  # one poll per bar keeps the script short


class Venue:
    def __init__(self, bars_by_symbol, close=CLOSE):
        self.b, self.close, self.orders = bars_by_symbol, close, []

    def session(self):
        return OPEN, self.close

    def bars(self, symbol, start, end):
        return [x for x in self.b[symbol] if start <= x["time"] < end]

    def price(self, symbol):
        return 100.0

    def market(self, symbol, side, qty):
        last = self.b[symbol][-1]["close"]
        self.orders.append((symbol, side, qty))
        return self.fills.pop(0)

    fills: list = []


def session(venue, loss=0.0):
    clock, logs, rec = Clock(OPEN + timedelta(minutes=30)), [], []
    total = run_day(venue, log=logs.append, halted=lambda: False, now_fn=clock.now, sleep=clock.sleep,
                    record=rec.append, symbols=("SPY",), loss_so_far=loss)
    return total, logs, rec


def day(post):
    # OR 100-100.5 (0.5%), then the bars after 14:00
    base = [bar(i * 5, 100.2, 100.5, 100.0, 100.2) for i in range(6)]
    return {"SPY": base + [bar(30 + i * 5, *x) for i, x in enumerate(post)]}


def test_long_breakout_then_stop():
    v = Venue(day([(100.3, 100.7, 100.3, 100.7), (100.6, 100.6, 99.8, 99.9)]))
    v.fills = [100.7, 99.9]
    total, logs, rec = session(v)
    assert v.orders == [("SPY", "buy", 49), ("SPY", "sell", 49)]
    assert rec[0]["why"] == "stop" and total == pytest.approx(-0.8 * 49)


def test_short_breakout_flattened_at_end_of_day():
    v = Venue(day([(100, 100.1, 99.7, 99.8)] + [(99.8, 99.9, 99.7, 99.8)] * 70))
    v.fills = [99.8, 99.5]
    total, _, rec = session(v)
    assert v.orders[0] == ("SPY", "sell", 50) and v.orders[1] == ("SPY", "buy", 50)
    assert rec[0]["why"] == "end of day" and total == pytest.approx(0.3 * 50)


def test_no_breakout_no_trade():
    v = Venue(day([(100.2, 100.4, 100.1, 100.3)] * 10))
    total, _, _ = session(v)
    assert v.orders == [] and total == 0.0


def test_early_close_session_is_skipped():
    v = Venue(day([(100, 100.7, 100, 100.7)]), close=OPEN + timedelta(hours=3.5))
    total, logs, _ = session(v)
    assert v.orders == [] and any("early-close" in m for m in logs)


def test_one_trade_per_symbol_per_day():
    v = Venue(day([(100.3, 100.7, 100.3, 100.7), (100.6, 100.6, 99.8, 99.9), (100.0, 100.9, 100.0, 100.9)]))
    v.fills = [100.7, 99.9, 100.9, 100.9]
    session(v)
    assert len(v.orders) == 2
