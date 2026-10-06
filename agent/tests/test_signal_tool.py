"""Tests for the trade_signal agent tool and its JSON output."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.signals.jev_analyzer import AnalyzedNews
from src.signals.news_judgment import NewsJudgment, news_signal
from src.signals.news_source import NewsItem
from src.signals.report_format import render_text, report_to_dict
from src.signals.service import SignalReport
from src.signals.verdict import Action, Verdict
from src.tools import signal_tool

T0 = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fresh_signal_cache():
    signal_tool.clear_signal_cache()
    yield
    signal_tool.clear_signal_cache()


def _headline() -> AnalyzedNews:
    j = NewsJudgment(0.95, 0.9, 0.9, "earnings", 0.9, 0.9, 0.95)
    item = NewsItem("NVDA", "NVIDIA beats estimates", "s", T0 - timedelta(minutes=3), "Yahoo", "u", "1")
    return AnalyzedNews(item, j, news_signal(j), "jev")


def _report(action: Action = Action.BUY, **kw) -> SignalReport:
    v = Verdict(action, 0.77, ("news +0.78", "trend +0.76"), T0 + timedelta(minutes=15), 0.78, 0.76,
                entry=100.0, stop=99.0, target=101.5)
    base = dict(symbol="NVDA", generated_at=T0, verdict=v, headlines=(_headline(),),
                bar_time=T0 - timedelta(minutes=5), stale=False, news_count=1)
    return SignalReport(**{**base, **kw})


class TestReportToDict:
    def test_core_fields(self) -> None:
        d = report_to_dict(_report())
        assert d["action"] == "BUY" and d["confidence"] == pytest.approx(0.77)
        assert (d["entry"], d["stop"], d["target"]) == (100.0, 99.0, 101.5)
        assert d["headlines"][0]["headline"] == "NVIDIA beats estimates"
        assert d["headlines"][0]["event_type"] == "earnings"

    def test_is_json_serialisable(self) -> None:
        json.dumps(report_to_dict(_report()))

    def test_states_it_is_advisory_and_places_no_orders(self) -> None:
        assert "no orders" in report_to_dict(_report())["disclaimer"].lower()

    def test_warnings_for_stale_and_news_error(self) -> None:
        d = report_to_dict(_report(stale=True, news_error="news unavailable (X)"))
        assert len(d["warnings"]) == 2

    def test_wait_has_no_levels(self) -> None:
        v = Verdict(Action.WAIT, 0.0, ("signal too weak",), T0, 0.0, 0.0)
        d = report_to_dict(_report(verdict=v))
        assert d["action"] == "WAIT" and d["entry"] is None


class TestRenderText:
    def test_contains_action_levels_and_reasons(self) -> None:
        text = render_text(_report())
        assert "BUY" in text and "NVDA" in text and "99.00" in text and "news +0.78" in text

    def test_wait_text_has_no_stop_line(self) -> None:
        v = Verdict(Action.WAIT, 0.0, ("signal too weak",), T0, 0.0, 0.0)
        assert "stop" not in render_text(_report(verdict=v)).lower()


class TestTool:
    def test_returns_ok_json(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(signal_tool, "run_signal", lambda symbol, **kw: _report())
        monkeypatch.setattr(signal_tool, "DEFAULT_LEDGER", tmp_path / "l.jsonl")
        out = json.loads(signal_tool.TradeSignalTool().execute(symbol="nvda"))
        assert out["status"] == "ok" and out["action"] == "BUY"

    def test_records_actionable_signal_in_ledger(self, monkeypatch, tmp_path) -> None:
        path = tmp_path / "l.jsonl"
        monkeypatch.setattr(signal_tool, "run_signal", lambda symbol, **kw: _report())
        monkeypatch.setattr(signal_tool, "DEFAULT_LEDGER", path)
        signal_tool.TradeSignalTool().execute(symbol="NVDA")
        assert len(path.read_text().splitlines()) == 1

    def test_invalid_symbol_returns_error_not_exception(self) -> None:
        out = json.loads(signal_tool.TradeSignalTool().execute(symbol="../x"))
        assert out["status"] == "error"

    def test_service_failure_returns_error_json(self, monkeypatch) -> None:
        def boom(symbol, **kw):
            raise RuntimeError("secret-token-123")

        monkeypatch.setattr(signal_tool, "run_signal", boom)
        raw = signal_tool.TradeSignalTool().execute(symbol="NVDA")
        assert json.loads(raw)["status"] == "error"
        assert "secret-token-123" not in raw

    def test_tool_is_readonly_and_named(self) -> None:
        tool = signal_tool.TradeSignalTool()
        assert tool.name == "trade_signal" and tool.is_readonly is True
        assert "symbol" in tool.parameters["properties"]

    def test_unavailable_without_keys(self, monkeypatch) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        from src.config.accessor import reset_env_config
        reset_env_config()
        assert signal_tool.TradeSignalTool.check_available() is False
        reset_env_config()
