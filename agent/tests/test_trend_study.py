"""Monthly trend/momentum backtest: correct picks, no look-ahead, costs charged."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.trend_study import (
    bootstrap_sharpe_lower, equal_weights, max_drawdown, momentum_weights, run_backtest, sharpe, trend_weights, cagr,
)


def prices(n=400, drifts=(0.001, -0.001, 0.0005), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = {f"A{i}": 100 * np.cumprod(1 + d + rng.normal(0, 0.002, n)) for i, d in enumerate(drifts)}
    return pd.DataFrame(cols, index=idx)


def test_trend_holds_only_assets_above_their_average():
    w = trend_weights(prices(), window=100).dropna(how="all")
    last = w.iloc[-1]
    assert last["A0"] > 0 and last["A1"] == 0 and last.sum() == pytest.approx(1.0)  # A1 trends down: excluded


def test_trend_is_cash_when_nothing_qualifies():
    falling = prices(drifts=(-0.001, -0.001))
    assert trend_weights(falling, window=100).iloc[-1].sum() == 0.0


def test_momentum_takes_top_ranked_with_positive_return():
    w = momentum_weights(prices(), lookback=250, skip=20, top=2).iloc[-1]
    assert w["A0"] == pytest.approx(0.5) and w["A2"] == pytest.approx(0.5) and w["A1"] == 0


def test_equal_weights_split_available_assets():
    p = prices()
    p.loc[p.index[:50], "A1"] = np.nan
    w = equal_weights(p)
    assert w.iloc[0].dropna().sum() == pytest.approx(1.0) and w.iloc[-1].sum() == pytest.approx(1.0)


def test_future_prices_cannot_change_past_returns():
    p = prices()
    base = run_backtest(p, trend_weights(p, 100), cost_bps=5)
    p2 = p.copy()
    p2.iloc[-30:] *= 1.5  # shock only the final 30 days
    shocked = run_backtest(p2, trend_weights(p2, 100), cost_bps=5)
    cut = p.index[-60]
    pd.testing.assert_series_equal(base[:cut], shocked[:cut])


def test_costs_reduce_returns_and_zero_cost_matches_gross():
    p = prices()
    w = trend_weights(p, 100)
    assert run_backtest(p, w, cost_bps=0).sum() > run_backtest(p, w, cost_bps=50).sum()


def test_metrics_on_known_series():
    r = pd.Series([0.01, -0.01] * 126)
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(0.5)
    assert cagr(pd.Series([0.0] * 252)) == pytest.approx(0.0)
    assert sharpe(pd.Series([0.001] * 252 + [0.0011])) > 5
    assert abs(sharpe(r)) < 0.5


def test_bootstrap_lower_bound_is_below_point_estimate_and_deterministic():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0006, 0.01, 1000))
    lo = bootstrap_sharpe_lower(r, level=0.975, n_boot=300, seed=0)
    assert lo < sharpe(r) and lo == bootstrap_sharpe_lower(r, level=0.975, n_boot=300, seed=0)
