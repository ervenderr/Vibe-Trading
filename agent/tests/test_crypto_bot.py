"""Daily crypto trend bot: pre-registered rules, paper-only, one check per day."""

from __future__ import annotations

from datetime import date

import pytest

from src.execution.crypto_bot import BotConfig, desired_long, run_check

D = date(2026, 10, 19)


def test_cross_long_when_fast_above_slow():
    assert desired_long("cross", [1.0] * 150 + [2.0] * 50, False) is True
    assert desired_long("cross", [2.0] * 150 + [1.0] * 50, True) is False
    assert desired_long("cross", [1.0] * 100, False) is False  # not enough history


def test_breakout_enters_on_new_50d_high_and_exits_on_20d_low():
    base = [100.0] * 60
    assert desired_long("breakout", base + [101.0], False) is True
    assert desired_long("breakout", base + [100.0], False) is False  # equal is not a breakout
    assert desired_long("breakout", base + [100.0], True) is True    # holds between signals
    assert desired_long("breakout", base + [99.0], True) is False    # below 20d low
    assert desired_long("breakout", [1.0] * 10, False) is False


class Trader:
    is_paper = True

    def __init__(self) -> None:
        self.orders: list[tuple[str, str, float]] = []

    def buy(self, symbol, qty):
        self.orders.append(("buy", symbol, qty))
        return 100.0, qty * 0.9975

    def sell(self, symbol, qty):
        self.orders.append(("sell", symbol, qty))
        return 110.0


def rising(n=260):
    return [float(i + 1) for i in range(n)]


def run(tmp_path, trader, closes, halted=False, cfg=None):
    logs: list[str] = []
    state = tmp_path / "state.json"
    run_check(cfg or BotConfig(), trader=trader, closes_fn=lambda s: closes, quote_fn=lambda s: {"ask": 100.0},
              log=logs.append, halted=lambda: halted, today=D, state_path=state)
    return logs


def test_buys_once_then_is_idempotent_for_the_day(tmp_path):
    t = Trader()
    run(tmp_path, t, rising())
    buys = [o for o in t.orders if o[0] == "buy"]
    assert len(buys) == 4  # 2 assets x 2 rules
    assert buys[0][2] == pytest.approx(50.0)  # 5000 / 100
    run(tmp_path, t, rising())
    assert len([o for o in t.orders if o[0] == "buy"]) == 4


def test_sells_the_quantity_held_when_signal_turns_flat(tmp_path):
    t = Trader()
    run(tmp_path, t, rising())
    t2 = Trader()
    run_check(BotConfig(), trader=t2, closes_fn=lambda s: [float(260 - i) for i in range(260)], quote_fn=lambda s: {"ask": 100.0},
              log=lambda m: None, halted=lambda: False, today=date(2026, 10, 20), state_path=tmp_path / "state.json")
    sells = [o for o in t2.orders if o[0] == "sell"]
    assert len(sells) == 4 and sells[0][2] == pytest.approx(50.0 * 0.9975)


def test_halt_and_non_paper_trade_nothing(tmp_path):
    t = Trader()
    run(tmp_path, t, rising(), halted=True)
    assert t.orders == []
    t.is_paper = False
    with pytest.raises(RuntimeError):
        run(tmp_path, t, rising())
