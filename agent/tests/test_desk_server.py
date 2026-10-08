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


def make(signal=lambda s: (_ for _ in ()).throw(RuntimeError("boom"))):
    return Desk(broker_factory=Broker, signal_fn=signal, halt_fn=lambda: False, journal_fn=lambda n: [], now_fn=lambda: T)


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
