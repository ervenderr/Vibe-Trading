"""Desk server routing: read-only, localhost-only, validated input, no path traversal."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.desk.server import Desk

T = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)


class Broker:
    is_paper = True
    def account(self): raise AssertionError("not needed")
    market_clock = positions = open_orders = fills = account


def make(signal=lambda s: (_ for _ in ()).throw(RuntimeError("boom")), tick=lambda s: {"symbol": s, "bid": 1.0, "ask": 1.2, "last": 1.1, "t": "x"},
         fills=lambda s: [{"time": 5, "side": "buy", "price": 1.0, "qty": 2.0}],
         bars=lambda s, tf: [{"time": 1, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 3}]):
    return Desk(broker_factory=Broker, signal_fn=signal, tick_fn=tick, bars_fn=bars, fills_fn=fills, halt_fn=lambda: False, journal_fn=lambda n: [], now_fn=lambda: T)


def test_serves_index_and_assets_but_not_traversal():
    d = make()
    status, ctype, body = d.handle("GET", "/", {}, host="127.0.0.1:8765")
    assert status == 200 and ctype.startswith("text/html") and b"<html" in body
    assert d.handle("GET", "/../payloads.py", {}, host="127.0.0.1:8765")[0] == 404
    assert d.handle("GET", "/%2e%2e/payloads.py", {}, host="127.0.0.1:8765")[0] == 404


def test_rejects_non_get_and_foreign_host():
    d = make()
    assert d.handle("POST", "/api/state", {}, host="127.0.0.1:8765")[0] == 405
    assert d.handle("GET", "/api/state", {}, host="evil.example")[0] == 403


def test_signal_validates_symbol_and_reports_failure_safely():
    d = make()
    assert d.handle("GET", "/api/signal", {"symbol": ["../x"]}, host="localhost:1")[0] == 400
    status, _, body = d.handle("GET", "/api/signal", {"symbol": ["nvda"]}, host="localhost:1")
    assert status == 502
    assert "boom" not in body.decode()  # internals are not echoed


def test_state_endpoint_survives_broker_exceptions():
    status, _, body = make().handle("GET", "/api/state", {}, host="127.0.0.1:1")
    assert status == 200
    assert json.loads(body)["error"]


def test_tick_accepts_stocks_and_crypto_pairs_only():
    d = make()
    for ok in ("AAPL", "btc/usd", "BRK.B"):
        status, _, body = d.handle("GET", "/api/tick", {"symbol": [ok]}, host="localhost:1")
        assert status == 200 and json.loads(body)["symbol"] == ok.upper()
    for bad in ("", "../x", "BTC/EUR", "A/B/C", "BTC USD"):
        assert d.handle("GET", "/api/tick", {"symbol": [bad]}, host="localhost:1")[0] == 400


def test_tick_failure_is_a_safe_502():
    d = make(tick=lambda s: (_ for _ in ()).throw(RuntimeError("secret detail")))
    status, _, body = d.handle("GET", "/api/tick", {"symbol": ["AAPL"]}, host="localhost:1")
    assert status == 502 and "secret" not in body.decode()


def test_bars_validates_symbol_and_timeframe():
    d = make()
    status, _, body = d.handle("GET", "/api/bars", {"symbol": ["gld"], "tf": ["5Min"]}, host="localhost:1")
    assert status == 200 and json.loads(body)["bars"][0]["close"] == 1.5
    assert d.handle("GET", "/api/bars", {"symbol": ["GLD"], "tf": ["1Week"]}, host="localhost:1")[0] == 400
    assert d.handle("GET", "/api/bars", {"symbol": ["../x"], "tf": ["1Min"]}, host="localhost:1")[0] == 400


def test_bars_failure_is_a_safe_502():
    d = make(bars=lambda s, tf: (_ for _ in ()).throw(RuntimeError("secret detail")))
    status, _, body = d.handle("GET", "/api/bars", {"symbol": ["GLD"], "tf": ["1Min"]}, host="localhost:1")
    assert status == 502 and "secret" not in body.decode()


def test_fills_endpoint_returns_markers_and_validates_symbol():
    d = make()
    status, _, body = d.handle("GET", "/api/fills", {"symbol": ["btc/usd"]}, host="localhost:1")
    assert status == 200 and json.loads(body)["fills"][0]["side"] == "buy"
    assert d.handle("GET", "/api/fills", {"symbol": ["../x"]}, host="localhost:1")[0] == 400
    boom = make(fills=lambda s: (_ for _ in ()).throw(RuntimeError("secret")))
    status, _, body = boom.handle("GET", "/api/fills", {"symbol": ["GLD"]}, host="localhost:1")
    assert status == 502 and "secret" not in body.decode()


def test_backtest_route_validates_and_runs():
    import numpy as np
    import pandas as pd

    def hist(symbol, interval, years):
        rng = np.random.default_rng(0)
        return pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 800)), index=pd.bdate_range("2019-01-01", periods=800))

    d = Desk(broker_factory=Broker, history_fn=hist, halt_fn=lambda: False, journal_fn=lambda n: [], now_fn=lambda: T)
    ok = d.handle("GET", "/api/backtest", {"symbol": ["SPY"], "strategy": ["sma_cross"]}, host="localhost:1")
    assert ok[0] == 200 and "metrics" in json.loads(ok[2])
    assert d.handle("GET", "/api/backtest", {"symbol": ["../x"]}, host="localhost:1")[0] == 400
    assert d.handle("POST", "/api/backtest", {}, host="localhost:1")[0] == 405


def test_agents_routes_list_and_read_reports(tmp_path):
    reports = tmp_path / "NVDA" / "2026-10-13" / "reports"
    reports.mkdir(parents=True)
    (reports / "market_report.md").write_text("hello")
    d = Desk(broker_factory=Broker, agents_base=tmp_path, halt_fn=lambda: False, journal_fn=lambda n: [], now_fn=lambda: T)
    listed = json.loads(d.handle("GET", "/api/agents", {}, host="localhost:1")[2])
    assert listed["runs"][0]["ticker"] == "NVDA"
    ok = d.handle("GET", "/api/agents/run", {"ticker": ["nvda"], "date": ["2026-10-13"]}, host="localhost:1")
    assert ok[0] == 200 and json.loads(ok[2])["reports"][0]["text"] == "hello"
    assert d.handle("GET", "/api/agents/run", {"ticker": ["../x"], "date": ["2026-10-13"]}, host="localhost:1")[0] == 404


def test_snr_route_returns_nearest_levels_with_freshness():
    rows = [(105, 105.5, 103.5, 104), (104, 104.5, 101.5, 102), (102, 102.5, 99.8, 100),
            (101, 102.5, 100.6, 102), (102, 104.5, 101.8, 104), (104, 105.5, 103.8, 105), (105, 106.5, 104.8, 106)]
    bars = [{"time": 1_700_000_000 + 86400 * i, "open": o, "high": h, "low": l, "close": c, "volume": 1} for i, (o, h, l, c) in enumerate(rows)]
    d = Desk(broker_factory=Broker, bars_fn=lambda s, tf: bars, halt_fn=lambda: False, journal_fn=lambda n: [], now_fn=lambda: T)
    status, _, body = d.handle("GET", "/api/snr", {"symbol": ["GLD"], "tf": ["5Min"]}, host="localhost:1")
    levels = json.loads(body)["levels"]
    assert status == 200 and any(l["kind"] == "V" and l["price"] == 100.0 and l["fresh"] for l in levels)
    assert all(set(l) >= {"kind", "price", "fresh", "time"} for l in levels)
    assert d.handle("GET", "/api/snr", {"symbol": ["../x"], "tf": ["5Min"]}, host="localhost:1")[0] == 400
    assert d.handle("GET", "/api/snr", {"symbol": ["GLD"], "tf": ["1Week"]}, host="localhost:1")[0] == 400


def test_warm_imports_is_safe_to_call_repeatedly():
    from src.desk.server import warm_imports
    warm_imports()
    warm_imports()


def test_snr_levels_are_thinned_to_five_per_side_without_near_duplicates():
    from src.desk.server import _spread_levels
    levels = [{"kind": "V", "price": 100 - i * 0.01, "fresh": True} for i in range(10)] + \
             [{"kind": "V", "price": 90 - i * 2, "fresh": True} for i in range(8)] + \
             [{"kind": "A", "price": 110 + i * 2, "fresh": False} for i in range(8)]
    out = _spread_levels(levels, last=101.0)
    below = [l for l in out if l["price"] < 101]
    assert len(below) == 5 and len([l for l in out if l["price"] >= 101]) == 5
    assert sum(1 for l in below if l["price"] > 99.5) == 1            # the ten levels within 0.1 of each other merge into one
