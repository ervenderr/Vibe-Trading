"""Single-asset backtester: signals, no look-ahead, fees and slippage, trades, metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.strategy_lab import (
    Costs, backtest, breakout_signal, rsi_reversion_signal, sma_cross_signal,
)


def series(values, start="2022-01-03") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def trending(n=300, drift=0.002, noise=0.01, seed=3) -> pd.Series:
    rng = np.random.default_rng(seed)
    return series(100 * np.cumprod(1 + drift + rng.normal(0, noise, n)))


def test_sma_cross_is_long_only_when_fast_above_slow():
    close = series(list(range(1, 80)) + list(range(80, 1, -1)))
    sig = sma_cross_signal(close, fast=5, slow=20)
    assert sig.iloc[60] == 1 and sig.iloc[-1] == 0
    assert set(sig.unique()) <= {0, 1}


def test_rsi_reversion_enters_oversold_and_holds_until_overbought():
    close = series([100] * 20 + [95, 90, 85, 80] + [82, 84, 86, 90, 95, 100, 105, 110, 115])
    sig = rsi_reversion_signal(close, period=5, low=30, high=70)
    assert sig.iloc[22] == 1                       # oversold: in
    assert sig.iloc[26] == 1                       # still below the exit level: held
    assert sig.iloc[-1] == 0                       # overbought: out


def test_breakout_enters_on_new_high_and_exits_on_new_low():
    close = series([10] * 25 + [11, 12, 13] + [13] * 5 + [7, 7, 7])
    sig = breakout_signal(close, window=20)
    assert sig.iloc[26] == 1 and sig.iloc[-1] == 0


def test_future_prices_cannot_change_past_results():
    close = trending()
    base = backtest(close, sma_cross_signal(close, 10, 40), Costs(5, 5))
    shocked = close.copy()
    shocked.iloc[-20:] *= 2
    other = backtest(shocked, sma_cross_signal(shocked, 10, 40), Costs(5, 5))
    cut = close.index[-40]
    pd.testing.assert_series_equal(base.returns[:cut], other.returns[:cut])


def test_always_long_pays_exactly_one_entry_cost():
    close = series([100.0] * 10)                   # flat prices: any change in equity is cost
    res = backtest(close, pd.Series(1, index=close.index), Costs(fee_bps=10, slippage_bps=5))
    assert res.returns.sum() == pytest.approx(-15e-4)


def test_costs_lower_returns():
    close = trending()
    sig = sma_cross_signal(close, 10, 40)
    free = backtest(close, sig, Costs(0, 0)).metrics["total_return"]
    paid = backtest(close, sig, Costs(25, 25)).metrics["total_return"]
    assert paid < free


def test_trades_are_segmented_with_net_returns():
    close = series([100, 100, 110, 120, 120, 100, 90, 95, 99, 99], "2023-01-02")
    sig = pd.Series([0, 1, 1, 0, 0, 0, 1, 1, 0, 0], index=close.index)
    res = backtest(close, sig, Costs(0, 0))
    assert len(res.trades) == 2
    first = res.trades[0]  # signal at bar 1 fills at the close of bar 2; exit signal at bar 3 fills at the close of bar 4
    assert first["entry_price"] == 110 and first["exit_price"] == 120 and first["return"] == pytest.approx(120 / 110 - 1)
    assert first["bars"] == 2
    assert res.trades[1]["return"] == pytest.approx(99 / 95 - 1)


def test_open_trade_is_marked_to_the_last_bar():
    close = series([100, 101, 102, 103, 104, 105])
    res = backtest(close, pd.Series([1, 1, 1, 1, 1, 1], index=close.index), Costs(0, 0))
    assert res.trades and res.trades[-1]["open"] is True


def test_metrics_include_drawdown_halves_and_benchmark():
    close = trending(500)
    res = backtest(close, sma_cross_signal(close, 10, 40), Costs(5, 5))
    for key in ("total_return", "cagr", "sharpe", "max_drawdown", "trades", "win_rate", "profit_factor", "exposure"):
        assert key in res.metrics
    assert 0 <= res.metrics["max_drawdown"] <= 1 and 0 <= res.metrics["exposure"] <= 1
    assert "total_return" in res.benchmark and set(res.halves) == {"first", "second"}
    assert len(res.equity) == len(close) and len(res.drawdown) == len(close)


def test_inputs_are_not_mutated():
    close = trending()
    sig = sma_cross_signal(close, 10, 40)
    c0, s0 = close.copy(), sig.copy()
    backtest(close, sig, Costs(5, 5))
    pd.testing.assert_series_equal(close, c0)
    pd.testing.assert_series_equal(sig, s0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_a_naive_bar_by_bar_simulation(seed):
    """Independent re-implementation with explicit cash/shares accounting, no vectorised shifts."""
    close = trending(260, drift=0.0005, noise=0.015, seed=seed)
    signal = rsi_reversion_signal(close, 5, 35, 65)
    costs = Costs(fee_bps=10, slippage_bps=6)
    cash, shares, pending = 1.0, 0.0, []           # pending: (fill_bar, target)
    prices, sig = close.tolist(), signal.tolist()
    for t in range(len(prices)):
        for fill_bar, target in [p for p in pending if p[0] == t]:   # orders fill at the close of their bar
            equity = cash + shares * prices[t]
            want = equity if target == 1 else 0.0
            delta = want - shares * prices[t]
            cash -= delta + abs(delta) * costs.per_side
            shares = want / prices[t]
        pending = [p for p in pending if p[0] > t]
        prev = sig[t - 1] if t else 0
        if sig[t] != prev:
            pending.append((t + 1, sig[t]))
    final = cash + shares * prices[-1]
    got = 1.0 + backtest(close, signal, costs).metrics["total_return"]
    # the naive loop holds from the fill bar's close; the engine earns from the next bar: identical path
    assert got == pytest.approx(final, rel=2e-3)
