"""Tests for the no-lookahead scalping replay simulator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.signals.replay import ReplayConfig, Trade, simulate, summarize
from src.signals.technicals import MIN_BARS, TechnicalState
from src.signals.verdict import Action, Verdict

START = datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc)
ENTRY_SIGNAL_BAR = MIN_BARS + 5
FREE = ReplayConfig(cost_bps=0.0, slippage_bps=0.0, cooldown_bars=0, max_hold_bars=10)


def _bars(n: int = 80, price: float = 100.0, start: datetime = START, step_min: int = 5) -> pd.DataFrame:
    idx = pd.DatetimeIndex([start + timedelta(minutes=step_min * i) for i in range(n)])
    return pd.DataFrame(
        {"open": price, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": 1000.0},
        index=idx,
    )


def _set(bars: pd.DataFrame, i: int, **cols: float) -> pd.DataFrame:
    out = bars.copy()
    for k, v in cols.items():
        out.iloc[i, out.columns.get_loc(k)] = v
    return out


def _verdict(action: Action, state: TechnicalState, now: datetime) -> Verdict:
    side = 1.0 if action is Action.BUY else -1.0
    return Verdict(
        action, 0.8, ("stub",), now + timedelta(minutes=15), 0.5, 0.5,
        entry=state.close, stop=state.close - side * 1.0, target=state.close + side * 1.5,
    )


def _once(action: Action, at: int = ENTRY_SIGNAL_BAR):
    seen: list[datetime] = []

    def fn(state: TechnicalState, now: datetime) -> Verdict:
        seen.append(now)
        i = len(seen) - 1 + (MIN_BARS - 1)
        if i == at:
            return _verdict(action, state, now)
        return Verdict(Action.WAIT, 0.0, ("none",), now, 0.0, 0.0)

    fn.seen = seen  # type: ignore[attr-defined]
    return fn


class TestMechanics:
    def test_enters_at_next_bar_open_not_signal_bar(self) -> None:
        bars = _set(_bars(), ENTRY_SIGNAL_BAR + 1, open=100.2, high=100.6, low=99.8, close=100.2)
        (trade,) = simulate(bars, _once(Action.BUY), FREE)
        assert trade.entry == pytest.approx(100.2)
        assert trade.entry_time == bars.index[ENTRY_SIGNAL_BAR + 1]

    def test_verdict_only_sees_time_at_bar_close(self) -> None:
        fn = _once(Action.BUY)
        bars = _bars()
        simulate(bars, fn, FREE)
        assert fn.seen[0] == bars.index[MIN_BARS - 1] + timedelta(minutes=5)

    def test_long_target_hit(self) -> None:
        bars = _set(_bars(), ENTRY_SIGNAL_BAR + 3, high=103.0)
        (trade,) = simulate(bars, _once(Action.BUY), FREE)
        assert trade.exit_reason == "target"
        assert trade.exit == pytest.approx(101.5)
        assert trade.net_return > 0

    def test_long_stop_hit(self) -> None:
        bars = _set(_bars(), ENTRY_SIGNAL_BAR + 3, low=98.0)
        (trade,) = simulate(bars, _once(Action.BUY), FREE)
        assert trade.exit_reason == "stop"
        assert trade.exit == pytest.approx(99.0)
        assert trade.net_return < 0

    def test_stop_wins_when_both_hit_in_same_bar(self) -> None:
        bars = _set(_bars(), ENTRY_SIGNAL_BAR + 2, high=103.0, low=98.0)
        (trade,) = simulate(bars, _once(Action.BUY), FREE)
        assert trade.exit_reason == "stop"

    def test_short_target_and_stop_are_mirrored(self) -> None:
        win = _set(_bars(), ENTRY_SIGNAL_BAR + 3, low=97.0)
        (t1,) = simulate(win, _once(Action.SELL), FREE)
        assert t1.exit_reason == "target" and t1.net_return > 0
        lose = _set(_bars(), ENTRY_SIGNAL_BAR + 3, high=102.0)
        (t2,) = simulate(lose, _once(Action.SELL), FREE)
        assert t2.exit_reason == "stop" and t2.net_return < 0

    def test_time_exit_after_max_hold(self) -> None:
        (trade,) = simulate(_bars(), _once(Action.BUY), FREE)
        assert trade.exit_reason == "time"
        assert trade.bars_held == FREE.max_hold_bars

    def test_costs_reduce_pnl(self) -> None:
        bars = _set(_bars(), ENTRY_SIGNAL_BAR + 3, high=103.0)
        (free,) = simulate(bars, _once(Action.BUY), FREE)
        (paid,) = simulate(bars, _once(Action.BUY), ReplayConfig(cost_bps=5.0, slippage_bps=2.0, cooldown_bars=0))
        assert paid.net_return < free.net_return

    def test_slippage_worsens_both_fills(self) -> None:
        bars = _set(_bars(), ENTRY_SIGNAL_BAR + 3, high=103.0)
        (paid,) = simulate(bars, _once(Action.BUY), ReplayConfig(cost_bps=0.0, slippage_bps=10.0, cooldown_bars=0))
        assert paid.entry > 100.0

    def test_flattens_at_end_of_day_no_overnight(self) -> None:
        a = _bars(60, start=START)
        b = _bars(20, start=START + timedelta(days=1))
        bars = pd.concat([a, b])
        (trade,) = simulate(bars, _once(Action.BUY, at=len(a) - 4), FREE)
        assert trade.exit_reason == "eod"
        assert trade.exit_time.date() == a.index[-1].date()

    def test_one_position_at_a_time(self) -> None:
        def always(state: TechnicalState, now: datetime) -> Verdict:
            return _verdict(Action.BUY, state, now)

        trades = simulate(_bars(), always, FREE)
        for a, b in zip(trades, trades[1:]):
            assert b.entry_time > a.exit_time or b.entry_time == a.exit_time

    def test_cooldown_blocks_immediate_reentry(self) -> None:
        def always(state: TechnicalState, now: datetime) -> Verdict:
            return _verdict(Action.BUY, state, now)

        bars = _set(_bars(), ENTRY_SIGNAL_BAR + 2, high=103.0)
        loose = simulate(bars, always, ReplayConfig(cost_bps=0, slippage_bps=0, cooldown_bars=0))
        tight = simulate(bars, always, ReplayConfig(cost_bps=0, slippage_bps=0, cooldown_bars=8))
        assert len(tight) < len(loose)

    def test_too_short_history_yields_no_trades(self) -> None:
        assert simulate(_bars(10), _once(Action.BUY), FREE) == ()

    def test_wait_and_avoid_never_trade(self) -> None:
        def waits(state: TechnicalState, now: datetime) -> Verdict:
            return Verdict(Action.AVOID, 0.0, ("x",), now, 0.0, 0.0)

        assert simulate(_bars(), waits, FREE) == ()


def _trade(net: float, exit_reason: str = "target") -> Trade:
    t = START
    return Trade("BUY", t, t + timedelta(minutes=5), 100.0, 100.0, exit_reason, net, 1, 0.0)


class TestSummarize:
    def test_empty(self) -> None:
        s = summarize(())
        assert s.trades == 0 and s.win_rate == 0.0 and s.total_net_return == 0.0

    def test_core_stats(self) -> None:
        s = summarize([_trade(0.02), _trade(0.01), _trade(-0.01), _trade(-0.01)])
        assert s.trades == 4
        assert s.win_rate == pytest.approx(0.5)
        assert s.total_net_return == pytest.approx(0.01)
        assert s.profit_factor == pytest.approx(1.5)
        assert s.expectancy == pytest.approx(0.0025)

    def test_profit_factor_with_no_losses_is_inf(self) -> None:
        assert summarize([_trade(0.01)]).profit_factor == float("inf")

    def test_max_drawdown(self) -> None:
        s = summarize([_trade(0.02), _trade(-0.01), _trade(-0.02), _trade(0.01)])
        assert s.max_drawdown == pytest.approx(0.03)
