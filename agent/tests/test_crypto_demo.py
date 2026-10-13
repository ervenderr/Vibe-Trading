"""Crypto paper demo: runs the real verdict, manages its own stop/target, never trades unless told to."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.execution.crypto_demo import DemoConfig, exit_reason, run_demo
from src.signals.service import SignalReport
from src.signals.verdict import Action, Verdict

T0 = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self) -> None:
        self.t = T0

    def now(self) -> datetime:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += timedelta(seconds=s)


def report(action=Action.BUY, conf=0.7, entry=100.0, stop=99.0, target=101.5, stale=False) -> SignalReport:
    v = Verdict(action, conf, ("because",), T0, 0.0, 0.0, entry, stop, target)
    return SignalReport("BTC-USD", T0, v, (), T0, stale, 0)


class Trader:
    is_paper = True

    def __init__(self, fill=100.0, fail_on_sell=False) -> None:
        self.orders: list[tuple[str, float]] = []
        self.fill, self.fail_on_sell = fill, fail_on_sell
        self.equity = [100_000.0, 100_000.0]

    fee_rate = 0.0

    def buy(self, symbol, qty):
        self.orders.append(("buy", qty))
        return self.fill, qty * (1 - self.fee_rate)  # the exchange takes its fee from the coins bought

    def sell(self, symbol, qty):
        self.orders.append(("sell", qty))
        if self.fail_on_sell:
            raise RuntimeError("sell failed")
        return self.exit_fill

    def equity_now(self) -> float:
        return self.equity.pop(0) if len(self.equity) > 1 else self.equity[0]


def go(trader, signal, quotes, *, halted=lambda: False, cfg=None, **kw):
    clock, logs = Clock(), []
    it = iter(quotes)
    last = [quotes[-1]]

    def quote(_):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    result = run_demo(cfg or DemoConfig(notional=1000.0, wait_minutes=5, max_hold_s=60), trader=trader,
                      signal_fn=signal, quote_fn=quote, log=logs.append, halted=halted,
                      now_fn=clock.now, sleep=clock.sleep, **kw)
    return result, logs


def test_exit_reason_priority_and_none():
    assert exit_reason(100, 99, 101.5, bid=98.9, held_s=1, max_hold_s=60) == "stop"
    assert exit_reason(100, 99, 101.5, bid=101.6, held_s=1, max_hold_s=60) == "target"
    assert exit_reason(100, 99, 101.5, bid=100.2, held_s=61, max_hold_s=60) == "timeout"
    assert exit_reason(100, 99, 101.5, bid=100.2, held_s=5, max_hold_s=60) is None


def test_buy_signal_enters_and_exits_at_target():
    t = Trader(fill=100.0)
    t.exit_fill = 101.6
    r, logs = go(t, lambda s: report(), [{"bid": 99.9, "ask": 100.0}, {"bid": 101.6, "ask": 101.7}])
    assert [o[0] for o in t.orders] == ["buy", "sell"]
    assert t.orders[0][1] == pytest.approx(10.0, rel=1e-3)  # $1000 / 100
    assert r.exit_reason == "target" and r.gross_pnl > 0
    assert any("entered" in l.lower() for l in logs) and any("closed" in l.lower() for l in logs)


def test_stop_exit_loses():
    t = Trader(fill=100.0)
    t.exit_fill = 98.9
    r, _ = go(t, lambda s: report(), [{"bid": 99.9, "ask": 100.0}, {"bid": 98.9, "ask": 99.0}])
    assert r.exit_reason == "stop" and r.gross_pnl < 0


def test_levels_are_rebased_on_the_actual_fill():
    t = Trader(fill=200.0)  # signal said 100 with a 1.0 stop distance: stop must be 199 not 99
    t.exit_fill = 198.9
    r, _ = go(t, lambda s: report(), [{"bid": 199.9, "ask": 200.0}, {"bid": 198.9, "ask": 199.0}])
    assert r.stop == pytest.approx(199.0) and r.target == pytest.approx(201.5)


def test_timeout_exits():
    t = Trader()
    t.exit_fill = 100.1
    r, _ = go(t, lambda s: report(), [{"bid": 100.0, "ask": 100.0}] * 200)
    assert r.exit_reason == "timeout"


@pytest.mark.parametrize("rep", [report(Action.WAIT), report(Action.SELL), report(conf=0.4), report(stale=True)])
def test_no_trade_unless_a_confident_fresh_buy(rep):
    t = Trader()
    r, logs = go(t, lambda s: rep, [{"bid": 100.0, "ask": 100.0}])
    assert t.orders == [] and r.exit_reason is None


def test_wide_spread_blocks_entry():
    t = Trader()
    r, _ = go(t, lambda s: report(), [{"bid": 98.0, "ask": 100.0}])  # 2% spread
    assert t.orders == []


def test_halt_stops_everything_and_non_paper_is_refused():
    t = Trader()
    go(t, lambda s: report(), [{"bid": 100.0, "ask": 100.0}], halted=lambda: True)
    assert t.orders == []
    t.is_paper = False
    with pytest.raises(RuntimeError, match="paper"):
        go(t, lambda s: report(), [{"bid": 100.0, "ask": 100.0}])


def test_failure_while_holding_still_tries_to_sell():
    t = Trader()
    t.exit_fill = 100.0

    def boom(_):
        raise RuntimeError("feed died")

    clock, logs = Clock(), []
    calls = {"n": 0}

    def quote(_):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("feed died")
        return {"bid": 100.0, "ask": 100.0}

    run_demo(DemoConfig(notional=1000.0, wait_minutes=5, max_hold_s=60), trader=t, signal_fn=lambda s: report(),
             quote_fn=quote, log=logs.append, halted=lambda: False, now_fn=clock.now, sleep=clock.sleep)
    assert [o[0] for o in t.orders] == ["buy", "sell"]


def test_sells_the_quantity_actually_held_after_the_fee():
    t = Trader(fill=100.0)
    t.fee_rate = 0.0025
    t.exit_fill = 101.6
    go(t, lambda s: report(), [{"bid": 99.9, "ask": 100.0}, {"bid": 101.6, "ask": 101.7}])
    bought, sold = t.orders[0][1], t.orders[1][1]
    assert sold == pytest.approx(bought * 0.9975)
