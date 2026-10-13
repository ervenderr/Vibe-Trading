"""Backtest endpoint: validated parameters, JSON-safe output, bounded size."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.desk.lab import MAX_POINTS, run_lab


def history(symbol, interval, years):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=2000)
    return pd.Series(100 * np.cumprod(1 + 0.0004 + rng.normal(0, 0.01, len(idx))), index=idx)


def ask(**q):
    base = {"symbol": ["SPY"], "strategy": ["sma_cross"], "fast": ["10"], "slow": ["50"]}
    return run_lab({**base, **{k: [str(v)] for k, v in q.items()}}, history_fn=history)


def test_ok_payload_has_metrics_curves_and_trades():
    status, body = ask()
    assert status == 200
    assert {"metrics", "benchmark", "halves", "equity", "drawdown", "trades", "params", "symbol"} <= set(body)
    assert len(body["equity"]) <= MAX_POINTS and body["equity"][0].keys() >= {"time", "strategy", "buy_hold"}
    json.dumps(body, allow_nan=False)


@pytest.mark.parametrize("bad", [
    {"strategy": "martingale"}, {"symbol": "../etc"}, {"fast": 50, "slow": 10}, {"fast": 0}, {"slow": 5000},
    {"fee_bps": -1}, {"slippage_bps": 5000}, {"years": 99}, {"interval": "1Week"}, {"fast": "abc"},
])
def test_rejects_bad_parameters(bad):
    status, body = ask(**bad)
    assert status == 400 and "error" in body


def test_other_strategies_and_crypto_symbol_mapping():
    seen = []
    def hist(symbol, interval, years):
        seen.append(symbol)
        return history(symbol, interval, years)
    for q in ({"strategy": ["rsi_reversion"], "period": ["14"], "low": ["30"], "high": ["70"]}, {"strategy": ["breakout"], "window": ["20"]}):
        status, _ = run_lab({"symbol": ["BTC/USD"], **q}, history_fn=hist)
        assert status == 200
    assert seen and all(s == "BTC/USD" for s in seen)  # mapping to the data vendor happens inside history_fn


def test_data_failure_is_a_safe_502():
    def boom(*a):
        raise RuntimeError("vendor secret detail")
    status, body = run_lab({"symbol": ["SPY"], "strategy": ["sma_cross"]}, history_fn=boom)
    assert status == 502 and "secret" not in json.dumps(body)


def test_too_little_history_is_reported():
    status, body = run_lab({"symbol": ["SPY"], "strategy": ["sma_cross"]}, history_fn=lambda *a: history(*a).iloc[:30])
    assert status == 422 and "history" in body["error"]


def test_thinned_chart_never_hides_the_worst_drawdown():
    def crash(symbol, interval, years):
        idx = pd.bdate_range("2015-01-01", periods=3000)
        px = pd.Series(np.linspace(100, 300, len(idx)), index=idx)
        px.iloc[1501] = px.iloc[1500] * 0.6          # one deep single-day crash that thinning could skip
        return px
    status, body = run_lab({"symbol": ["SPY"], "strategy": ["sma_cross"], "fast": ["5"], "slow": ["20"], "years": ["15"]}, history_fn=crash)
    assert status == 200 and len(body["drawdown"]) <= MAX_POINTS + 1
    assert max(p["value"] for p in body["drawdown"]) == pytest.approx(body["metrics"]["max_drawdown"], abs=1e-9)
