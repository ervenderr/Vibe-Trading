"""Crypto timeframe study primitives: no look-ahead, costs charged, fixed candidate list."""

import pandas as pd
import pytest

from src.signals.crypto_study import breakout, hold, max_drawdown, momentum, sma_cross, strategy_returns, variants


def s(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values)))


def test_breakout_enters_on_new_high_and_exits_on_low():
    close = s([10.0] * 10 + [11.0, 11.0, 11.0, 5.0])
    sig = breakout(close, 4)
    assert not sig.iloc[9] and sig.iloc[10] and sig.iloc[12] and not sig.iloc[13]


def test_signal_is_held_over_the_next_bar_only():
    close = s([100.0, 100.0, 110.0, 110.0])
    sig = pd.Series([False, False, True, True], index=close.index)  # fires on the +10% bar itself
    r = strategy_returns(close, sig, cost=0.0)
    assert r.iloc[2] == 0.0  # cannot capture the move that triggered the signal


def test_costs_are_charged_on_each_position_change():
    close = s([100.0] * 5)
    sig = pd.Series([True, True, False, False, False], index=close.index)
    r = strategy_returns(close, sig, cost=0.01)
    assert r.sum() == pytest.approx(-0.02)  # one entry, one exit


def test_mom_and_cross_need_history():
    close = s(range(1, 40))
    assert not momentum(close, 30).iloc[10]
    assert not sma_cross(close, 5, 50).any()


def test_variant_list_is_the_registered_thirteen():
    assert len(variants(365)) == 13


def test_hold_and_drawdown():
    close = s([100.0, 50.0, 100.0])
    assert max_drawdown(hold({"x": close}, cost=0.0)) == pytest.approx(0.5)
