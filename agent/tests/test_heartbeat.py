"""Heartbeat: every autopilot cycle leaves a trace, so 'running and quiet' differs from 'dead'."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.desk.payloads import state_payload
from src.execution.journal import Journal
from tests.test_autopilot import FakeBroker, FakeExecutor, T, _report
from src.execution.autopilot import Autopilot

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)


def _pilot(beats, *, broker=None, executor=None, symbols=("NVDA", "TSLA"), halted=False):
    return Autopilot(
        list(symbols), runner=lambda s, **k: _report(s), executor=executor or FakeExecutor(),
        broker=broker or FakeBroker(), dry_run=True, interval_s=60.0, halt_check=lambda: halted,
        now_fn=lambda: T, sleep=lambda s: None, out=lambda _: None,
        heartbeat=lambda msg, sleep_s: beats.append((msg, sleep_s)),
    )


def test_trading_cycle_reports_symbols_checked_and_next_wake():
    beats: list = []
    _pilot(beats).cycle()
    assert len(beats) == 1
    assert "checked 2 symbols" in beats[0][0] and "no trades" in beats[0][0]
    assert beats[0][1] > 0


def test_closed_market_and_halt_also_beat():
    beats: list = []
    _pilot(beats, broker=FakeBroker(is_open=False)).cycle()
    assert "market closed" in beats[0][0]
    beats.clear()
    _pilot(beats, halted=True).cycle()
    assert "kill switch" in beats[0][0]


def test_submitted_trades_are_counted():
    beats: list = []
    _pilot(beats, executor=FakeExecutor("submitted")).cycle()
    assert "2 trades" in beats[0][0]


def test_journal_beat_rows_carry_kind_and_next_wake(tmp_path):
    j = Journal(tmp_path / "j.jsonl", now_fn=lambda: NOW)
    j.beat("cycle ok", 120.0)
    row = j.recent(1)[0]
    assert row["k"] == "beat" and row["next_s"] == 120.0 and row["msg"] == "cycle ok"


def test_state_payload_splits_beats_from_the_logbook():
    class B:
        is_paper = True
        def account(self): raise RuntimeError("x")
        market_clock = positions = open_orders = account
    rows = [{"t": "a", "msg": "NVDA: skipped"}, {"t": "b", "msg": "cycle ok", "k": "beat", "next_s": 60.0}]
    p = state_payload(B(), halted=False, journal=rows)
    assert [r["msg"] for r in p["journal"]] == ["NVDA: skipped"]
    assert p["last_beat"]["msg"] == "cycle ok"
    json.dumps(p, allow_nan=False)
