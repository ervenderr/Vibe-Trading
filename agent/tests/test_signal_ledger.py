"""Tests for the signal ledger (track record) and grading."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.signals.ledger import LedgerEntry, SignalLedger, grade_entry
from src.signals.service import SignalReport
from src.signals.verdict import Action, Verdict

T0 = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)


def _report(action: Action = Action.BUY, at: datetime = T0, symbol: str = "NVDA") -> SignalReport:
    v = Verdict(action, 0.7, ("r",), at + timedelta(minutes=15), 0.5, 0.5,
                entry=100.0 if action in (Action.BUY, Action.SELL) else None,
                stop=99.0 if action is Action.BUY else 101.0,
                target=101.5 if action is Action.BUY else 98.5)
    return SignalReport(symbol, at, v, (), at, False, 0)


class TestLedger:
    def test_records_and_loads_round_trip(self, tmp_path) -> None:
        ledger = SignalLedger(tmp_path / "l.jsonl")
        assert ledger.record(_report()) is True
        (entry,) = ledger.load()
        assert entry.symbol == "NVDA" and entry.action == "BUY"
        assert entry.entry == 100.0 and entry.generated_at == T0

    def test_wait_is_not_recorded(self, tmp_path) -> None:
        ledger = SignalLedger(tmp_path / "l.jsonl")
        assert ledger.record(_report(Action.WAIT)) is False
        assert ledger.load() == ()

    def test_duplicate_within_validity_window_is_skipped(self, tmp_path) -> None:
        ledger = SignalLedger(tmp_path / "l.jsonl")
        ledger.record(_report(at=T0))
        assert ledger.record(_report(at=T0 + timedelta(minutes=5))) is False
        assert len(ledger.load()) == 1

    def test_same_symbol_after_window_or_other_side_is_recorded(self, tmp_path) -> None:
        ledger = SignalLedger(tmp_path / "l.jsonl")
        ledger.record(_report(at=T0))
        assert ledger.record(_report(at=T0 + timedelta(minutes=30))) is True
        assert ledger.record(_report(Action.SELL, at=T0 + timedelta(minutes=31))) is True

    def test_other_symbols_are_independent(self, tmp_path) -> None:
        ledger = SignalLedger(tmp_path / "l.jsonl")
        ledger.record(_report(symbol="NVDA"))
        assert ledger.record(_report(symbol="TSLA")) is True

    def test_corrupt_lines_are_skipped(self, tmp_path) -> None:
        path = tmp_path / "l.jsonl"
        ledger = SignalLedger(path)
        ledger.record(_report())
        with path.open("a") as fh:
            fh.write("{not json\n" + json.dumps({"symbol": "x"}) + "\n")
        assert len(ledger.load()) == 1

    def test_missing_file_loads_empty(self, tmp_path) -> None:
        assert SignalLedger(tmp_path / "none.jsonl").load() == ()


def _entry(side: str = "BUY") -> LedgerEntry:
    return LedgerEntry("NVDA", side, 0.7, T0, T0 + timedelta(minutes=15),
                       100.0, 99.0 if side == "BUY" else 101.0, 101.5 if side == "BUY" else 98.5)


def _after(highs: list[float], lows: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([T0 + timedelta(minutes=5 * (i + 1)) for i in range(len(highs))])
    mid = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame({"open": [100.0] * len(highs), "high": highs, "low": lows, "close": mid, "volume": 1.0}, index=idx)


class TestGrade:
    def test_target_hit_is_a_win(self) -> None:
        g = grade_entry(_entry(), _after([100.5, 102.0], [99.6, 100.0]))
        assert g.outcome == "target" and g.gross_return == pytest.approx(0.015)

    def test_stop_hit_is_a_loss(self) -> None:
        g = grade_entry(_entry(), _after([100.5, 100.4], [99.6, 98.5]))
        assert g.outcome == "stop" and g.gross_return == pytest.approx(-0.01)

    def test_stop_wins_a_same_bar_tie(self) -> None:
        assert grade_entry(_entry(), _after([102.0], [98.0])).outcome == "stop"

    def test_short_is_mirrored(self) -> None:
        g = grade_entry(_entry("SELL"), _after([100.4, 100.2], [99.5, 98.0]))
        assert g.outcome == "target" and g.gross_return == pytest.approx(0.015)

    def test_expires_at_validity_end_marked_to_close(self) -> None:
        bars = _after([100.4] * 5, [99.6] * 5)
        g = grade_entry(_entry(), bars)
        assert g.outcome == "expired"
        assert g.gross_return == pytest.approx((100.0 - 100.0) / 100.0)

    def test_bars_before_signal_are_ignored(self) -> None:
        early = _after([200.0], [50.0])
        early.index = early.index - timedelta(hours=1)
        assert grade_entry(_entry(), early).outcome == "pending"

    def test_no_bars_is_pending(self) -> None:
        assert grade_entry(_entry(), _after([], [])).outcome == "pending"


from src.signals.ledger import Grade, summarize_grades


class TestSummarizeGrades:
    def test_counts_resolved_and_pending_separately(self) -> None:
        grades = [Grade("target", 0.015), Grade("stop", -0.01), Grade("expired", 0.002), Grade("pending", 0.0)]
        s = summarize_grades(grades)
        assert s.resolved == 3 and s.pending == 1
        assert s.win_rate == pytest.approx(2 / 3)
        assert s.avg_return == pytest.approx((0.015 - 0.01 + 0.002) / 3)

    def test_empty(self) -> None:
        s = summarize_grades([])
        assert s.resolved == 0 and s.win_rate == 0.0
