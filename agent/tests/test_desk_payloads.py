"""Desk payloads: JSON-safe, read-only views of the paper account and the signal engine."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.desk.payloads import jsonable, state_payload, track_payload
from src.execution.broker import BrokerError
from src.execution.models import AccountState, MarketClock
from src.execution.report import Fill

T = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)


class FakeBroker:
    is_paper = True

    def __init__(self, fail: bool = False, fills=()):
        self.fail, self._fills = fail, fills

    def account(self):
        if self.fail:
            raise BrokerError("down")
        return AccountState(100_000.0, 200_000.0, False, 99_900.0)

    def market_clock(self):
        return MarketClock(True, T, T + timedelta(hours=18), T + timedelta(hours=3))

    def positions(self):
        return [{"symbol": "AAPL", "qty": 1, "unrealized_pl": 1.5}]

    def open_orders(self):
        return []

    def fills(self, since):
        return self._fills


def test_jsonable_handles_nan_inf_datetime_and_dataclass():
    out = jsonable({"a": float("nan"), "b": float("inf"), "t": T, "c": (1, 2)})
    assert out == {"a": None, "b": None, "t": T.isoformat(), "c": [1, 2]}
    json.dumps(out, allow_nan=False)


def test_state_payload_reports_paper_halt_and_account():
    p = state_payload(FakeBroker(), halted=False, journal=[{"t": "x", "msg": "m"}])
    assert p["paper"] is True and p["halted"] is False
    assert p["account"]["equity"] == 100_000.0
    assert p["account"]["day_change"] == 100.0
    assert p["clock"]["is_open"] is True
    assert p["positions"][0]["symbol"] == "AAPL"
    assert p["journal"] == [{"t": "x", "msg": "m"}]
    assert p["error"] is None


def test_state_payload_degrades_when_broker_fails():
    p = state_payload(FakeBroker(fail=True), halted=True, journal=[])
    assert p["error"] and "down" in p["error"]
    assert p["account"] is None and p["halted"] is True


def test_track_payload_empty_has_all_checks_failing():
    p = track_payload(FakeBroker(), since=T - timedelta(days=30))
    assert p["trades"] == [] and p["curve"] == []
    assert p["summary"]["trades"] == 0
    assert p["promotion"]["eligible"] is False
    assert len(p["promotion"]["checks"]) == 6


def test_track_payload_builds_curve_from_fills():
    fills = (
        Fill(T, "AAPL", "buy", 1, 100.0),
        Fill(T + timedelta(minutes=5), "AAPL", "sell", 1, 102.0),
        Fill(T + timedelta(minutes=10), "AAPL", "buy", 1, 100.0),
        Fill(T + timedelta(minutes=15), "AAPL", "sell", 1, 99.0),
    )
    p = track_payload(FakeBroker(fills=fills), since=T - timedelta(days=1))
    assert [round(t["pnl"], 2) for t in p["trades"]] == [2.0, -1.0]
    assert [round(c["cum_pnl"], 2) for c in p["curve"]] == [2.0, 1.0]
    assert p["summary"]["trades"] == 2
    json.dumps(p, allow_nan=False)


def test_signal_payload_flattens_verdict_and_headlines():
    from src.desk.payloads import signal_payload
    from src.signals.jev_analyzer import AnalyzedNews
    from src.signals.news_judgment import NewsJudgment, NewsSignal
    from src.signals.news_source import NewsItem
    from src.signals.service import SignalReport
    from src.signals.verdict import Action, Verdict

    item = NewsItem("NVDA", "<b>Big</b> deal", "s", T, "wire", "javascript:alert(1)", "1")
    judged = AnalyzedNews(item, NewsJudgment(0.9, 0.6, 0.8, "partnership", 0.7, 0.1, 0.9),
                          NewsSignal(0.4, False, "ok"), "jev")
    report = SignalReport("NVDA", T, Verdict(Action.BUY, 0.62, ("price confirms",), T, 0.4, 0.3, 100.0, 98.0, 104.0),
                          (judged,), T, False, 1)
    p = signal_payload(report)
    assert p["symbol"] == "NVDA" and p["action"] == "BUY" and p["confidence"] == 0.62
    assert p["levels"] == {"entry": 100.0, "stop": 98.0, "target": 104.0}
    h = p["headlines"][0]
    assert h["headline"] == "<b>Big</b> deal" and h["event_type"] == "partnership"
    assert h["url"] is None  # non-http(s) links are dropped
    json.dumps(p, allow_nan=False)
