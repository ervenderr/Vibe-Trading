"""Managed manual trade: exits, retries, stale data, and the broker-side safety stop."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.execution.manual_trade import BAD_POLLS, decide, run_trade

T0 = datetime(2026, 10, 26, 14, 0, tzinfo=timezone.utc)


class Clock:
    t = T0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += timedelta(seconds=s)


class Venue:
    def __init__(self, clock, closes, *, sell_fails=0, stale=False, safety_fills_at=None):
        self.clock, self.closes, self.sell_fails, self.stale = clock, list(closes), sell_fails, stale
        self.held, self.orders, self.cancelled, self.safety_fills_at, self.i = 0, [], [], safety_fills_at, 0

    def last_bar(self, symbol):
        if not self.closes:
            raise RuntimeError("no bars")
        c = self.closes[min(self.i, len(self.closes) - 1)]
        self.i += 1
        if c is None:
            raise RuntimeError("api blip")
        return (self.clock.now() - timedelta(minutes=10 if self.stale else 0), c)

    def market(self, symbol, side, qty):
        self.orders.append((side, qty))
        if side == "sell":
            if self.sell_fails:
                self.sell_fails -= 1
                raise RuntimeError("rejected")
            self.held = 0
            return self.closes[-1]
        self.held = qty
        return 100.0

    def position_qty(self, symbol):
        if self.safety_fills_at is not None and self.i >= self.safety_fills_at:
            self.held = 0
        return self.held

    def place_stop(self, symbol, qty, price):
        self.stop_price = price
        return "stop-1"

    def cancel(self, order_id):
        self.cancelled.append(order_id)


def go(venue, clock, **kw):
    logs = []
    args = dict(notional=1000.0, stop_pct=0.01, target_pct=0.015, max_hold_s=3600, log=logs.append,
                halted=lambda: False, now_fn=clock.now, sleep=clock.sleep)
    args.update(kw)
    return run_trade(venue, "TSLA", **args), logs


def test_decide():
    assert decide(98.9, 99, 101.5, 10, 100) == "stop" and decide(101.5, 99, 101.5, 10, 100) == "target"
    assert decide(100, 99, 101.5, 100, 100) == "timeout" and decide(100, 99, 101.5, 10, 100) is None


def test_target_exit_cancels_safety_stop_and_sells_real_quantity():
    c = Clock()
    v = Venue(c, [100.0, 100.5, 101.6])
    net, _ = go(v, c)
    assert v.orders == [("buy", 10), ("sell", 10)] and v.cancelled == ["stop-1"]
    assert v.stop_price == pytest.approx(98.7, abs=0.01)  # 0.3% under our own stop at 99
    assert net == pytest.approx(16.0)


def test_failed_sell_is_retried():
    c = Clock()
    v = Venue(c, [100.0, 98.0], sell_fails=2)
    net, logs = go(v, c)
    assert [o[0] for o in v.orders].count("sell") == 3 and net == pytest.approx(-20.0)


def test_sell_that_never_works_is_loud_and_raises():
    c = Clock()
    v = Venue(c, [100.0, 98.0], sell_fails=9)
    with pytest.raises(RuntimeError):
        go(v, c)


def test_one_bad_poll_does_not_end_the_trade():
    c = Clock()
    v = Venue(c, [100.0, None, 100.2, 101.6])
    net, _ = go(v, c)
    assert net == pytest.approx(16.0)


def test_stale_bars_are_never_trusted_and_end_in_a_sell():
    c = Clock()
    v = Venue(c, [100.0] + [90.0] * 10, stale=True)  # a stale 90 must not trigger a "stop"
    net, logs = go(v, c, max_hold_s=10_000)
    assert v.orders[-1][0] == "sell"
    assert v.i <= 1 + BAD_POLLS + 1


def test_position_closed_by_the_broker_stop_is_not_sold_again():
    c = Clock()
    v = Venue(c, [100.0, 99.9, 99.8, 99.7], safety_fills_at=3)
    net, logs = go(v, c, max_hold_s=10_000)
    assert net is None and ("sell", 10) not in v.orders and any("broker-side stop" in m for m in logs)
